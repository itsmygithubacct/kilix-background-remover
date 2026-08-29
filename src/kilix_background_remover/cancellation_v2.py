"""Durable, linearizable cancellation boundary for the v2 provider."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from .contract_v2 import ContractRefusal, ContractRuntime, canonical_bytes

TerminalState = Literal["cancelled", "committed", "failed"]
CrashPoint = Literal["before-commit", "after-commit"]


@dataclass(frozen=True, slots=True)
class TerminalReservation:
    request_id: str
    sequence: int
    state: TerminalState


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


class DurableCancellationGate:
    """SQLite transaction is the one terminal/cancellation serialization point."""

    def __init__(
        self,
        database: Path,
        *,
        runtime: ContractRuntime | None = None,
        clock: Callable[[], str] = _now,
    ) -> None:
        if not database.parent.is_dir() or database.parent.is_symlink():
            raise ValueError("cancellation database directory is unavailable")
        try:
            descriptor = os.open(
                database,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        status = database.lstat()
        if (
            not stat.S_ISREG(status.st_mode)
            or database.is_symlink()
            or status.st_uid != os.getuid()
        ):
            raise ValueError("cancellation database is unsafe")
        if stat.S_IMODE(status.st_mode) & 0o077:
            database.chmod(0o600)
        self._database = database
        self._runtime = runtime or ContractRuntime.load()
        self._clock = clock
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    request_id TEXT PRIMARY KEY,
                    next_sequence INTEGER NOT NULL,
                    terminal_state TEXT,
                    terminal_sequence INTEGER,
                    cancellation_id TEXT,
                    cancel_request BLOB,
                    cancel_outcome BLOB
                )
                """
            )

    def begin(self, request_id: str) -> None:
        if not _request_id_shape(request_id):
            raise ContractRefusal("lifecycle", "LC-REQUEST-ID")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT 1 FROM jobs WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                raise ContractRefusal("lifecycle", "LC-REQUEST-ID-REUSE")
            connection.execute(
                "INSERT INTO jobs(request_id, next_sequence) VALUES (?, 0)",
                (request_id,),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def reserve_terminal(
        self,
        request_id: str,
        state: TerminalState,
        *,
        publish: Callable[[], None] | None = None,
        locked: threading.Event | None = None,
        proceed: threading.Event | None = None,
    ) -> TerminalReservation:
        if state not in {"cancelled", "committed", "failed"}:
            raise ValueError("invalid terminal state")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            _signal_and_wait(locked, proceed)
            row = self._row(connection, request_id)
            if row[1] is not None:
                if row[1] != state:
                    raise ContractRefusal("lifecycle", "LC-TERMINAL-EXCLUSIVE")
                connection.rollback()
                return TerminalReservation(request_id, cast(int, row[2]), state)
            outcome = _outcome_document(row[5])
            if outcome is not None and outcome["outcome"] == "accepted" and state != "cancelled":
                raise ContractRefusal("lifecycle", "LC-ACCEPTED-CANCEL-TERMINAL")
            sequence = cast(int, row[0])
            if publish is not None:
                publish()
            connection.execute(
                """
                UPDATE jobs
                SET next_sequence = ?, terminal_state = ?, terminal_sequence = ?
                WHERE request_id = ?
                """,
                (sequence + 1, state, sequence, request_id),
            )
            connection.commit()
            return TerminalReservation(request_id, sequence, state)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def reserve_progress(self, request_id: str) -> int:
        """Allocate one provider sequence while excluding cancel and terminal reservation."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, request_id)
            if row[1] is not None:
                raise ContractRefusal("lifecycle", "LC-PROGRESS-AFTER-TERMINAL")
            outcome = _outcome_document(row[5])
            if outcome is not None and outcome["outcome"] == "accepted":
                raise ContractRefusal("lifecycle", "LC-PROGRESS-AFTER-CANCEL")
            sequence = cast(int, row[0])
            connection.execute(
                "UPDATE jobs SET next_sequence = ? WHERE request_id = ?",
                (sequence + 1, request_id),
            )
            connection.commit()
            return sequence
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def cancel(
        self,
        request_bytes: bytes,
        *,
        locked: threading.Event | None = None,
        proceed: threading.Event | None = None,
        crash: CrashPoint | None = None,
    ) -> bytes:
        request = self._runtime.accept_wire(request_bytes)
        if request["schema"] != "kilix.media-job.cancel-request/v2":
            raise ContractRefusal("schema", "C-WIRE-IDENTITY")
        request_id = cast(str, request["request_id"])
        cancellation_id = cast(str, request["cancellation_id"])
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            _signal_and_wait(locked, proceed)
            row = self._row(connection, request_id)
            stored_id = cast(str | None, row[3])
            stored_request = cast(bytes | None, row[4])
            stored_outcome = cast(bytes | None, row[5])
            if stored_id is not None:
                if stored_id != cancellation_id:
                    raise ContractRefusal("lifecycle", "LC-SECOND-CANCELLATION-ID")
                if stored_request != request_bytes:
                    raise ContractRefusal("lifecycle", "LC-CHANGED-CANCEL-REPLAY")
                if stored_outcome is None:
                    raise ContractRefusal("lifecycle", "LC-MISSING-CANCEL-OUTCOME")
                connection.rollback()
                return stored_outcome
            sequence = cast(int, row[0])
            terminal_state = cast(str | None, row[1])
            terminal_sequence = cast(int | None, row[2])
            outcome: dict[str, Any] = {
                "cancellation_id": cancellation_id,
                "linearization_sequence": sequence,
                "observed_at": self._clock(),
                "outcome": "terminal-won" if terminal_state is not None else "accepted",
                "request_id": request_id,
                "schema": "kilix.media-job.cancel-outcome/v2",
                "terminal_sequence": terminal_sequence,
                "terminal_state": terminal_state,
            }
            self._runtime.validate_message(outcome)
            encoded = canonical_bytes(outcome)
            if crash == "before-commit":
                os._exit(74)
            connection.execute(
                """
                UPDATE jobs
                SET next_sequence = ?, cancellation_id = ?,
                    cancel_request = ?, cancel_outcome = ?
                WHERE request_id = ?
                """,
                (sequence + 1, cancellation_id, request_bytes, encoded, request_id),
            )
            connection.commit()
            if crash == "after-commit":
                os._exit(75)
            return encoded
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def recorded_outcome(self, request_id: str) -> bytes | None:
        with self._connect() as connection:
            row = self._row(connection, request_id)
            return cast(bytes | None, row[5])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database, timeout=30, isolation_level=None)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _row(connection: sqlite3.Connection, request_id: str) -> tuple[object, ...]:
        row = connection.execute(
            """
            SELECT next_sequence, terminal_state, terminal_sequence,
                   cancellation_id, cancel_request, cancel_outcome
            FROM jobs WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
        if row is None:
            raise ContractRefusal("lifecycle", "LC-FIRST-REQUEST")
        return cast(tuple[object, ...], row)


def _outcome_document(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, bytes):
        raise RuntimeError("stored cancellation outcome is not bytes")
    decoded = json_decode(value)
    if not isinstance(decoded, dict):
        raise RuntimeError("stored cancellation outcome is not an object")
    return cast(dict[str, Any], decoded)


def json_decode(value: bytes) -> Any:
    return ContractRuntime.load().decode(value)


def _signal_and_wait(locked: threading.Event | None, proceed: threading.Event | None) -> None:
    if locked is not None:
        locked.set()
    if proceed is not None and not proceed.wait(timeout=30):
        raise TimeoutError("causal cancellation control timed out")


def _request_id_shape(value: str) -> bool:
    import re

    return (
        re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            value,
        )
        is not None
    )
