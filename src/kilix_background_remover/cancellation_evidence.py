"""Causally force both cancellation races and two durable crash points."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import threading
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, NoReturn, cast

from .cancellation_v2 import CrashPoint, DurableCancellationGate, TerminalReservation
from .contract_v2 import ContractRefusal, ContractRuntime, canonical_bytes

REQUEST_ID = "018f6f65-7c7d-7a8b-8c9d-0123456789ab"
CANCELLATION_ID = "018f6f65-7c7d-7a8b-8c9d-0123456789ac"
OBSERVED_AT = "2026-08-29T12:00:00Z"


def generate_cancellation_evidence(output_directory: Path) -> dict[str, object]:
    runtime = ContractRuntime.load()
    request, committed_result = _base_messages(runtime)
    cancel_request = {
        "cancellation_id": CANCELLATION_ID,
        "client_requested_at": "2026-08-29T11:59:59Z",
        "reason": "user",
        "request_id": REQUEST_ID,
        "schema": "kilix.media-job.cancel-request/v2",
    }
    cancel_bytes = canonical_bytes(cancel_request)
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="f108-r5-cancellation-") as temporary:
        root = Path(temporary)
        records.append(
            _accepted_race(
                root / "accepted.sqlite3", runtime, request, cancel_request, cancel_bytes
            )
        )
        records.append(
            _terminal_race(
                root / "terminal.sqlite3",
                runtime,
                request,
                committed_result,
                cancel_request,
                cancel_bytes,
            )
        )
        records.append(
            _crash_case(
                root / "after.sqlite3",
                runtime,
                request,
                cancel_request,
                cancel_bytes,
                "after-commit",
                75,
            )
        )
        records.append(
            _crash_case(
                root / "before.sqlite3",
                runtime,
                request,
                cancel_request,
                cancel_bytes,
                "before-commit",
                74,
            )
        )
    if not all(record["passed"] is True for record in records):
        raise RuntimeError("a causal cancellation control did not pass")
    summary: dict[str, object] = {
        "schema": "kilix.f108.r5-cancellation-evidence/v1",
        "conditional": True,
        "contract_manifest_sha256": runtime.lock.manifest_sha256,
        "race_outcomes_forced": 2,
        "race_outcomes_total": 2,
        "crash_points_forced": 2,
        "crash_points_total": 2,
        "transcripts_valid": 4,
        "transcripts_total": 4,
        "records": records,
    }
    output_directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    _write(output_directory / "cancellation.json", canonical_bytes(summary))
    return summary


def _accepted_race(
    database: Path,
    runtime: ContractRuntime,
    request: dict[str, Any],
    cancel_request: dict[str, Any],
    cancel_bytes: bytes,
) -> dict[str, object]:
    gate = _gate(database, runtime)
    gate.begin(REQUEST_ID)
    lock_held = threading.Event()
    release = threading.Event()
    terminal_attempted = threading.Event()
    outcome_holder: list[bytes] = []
    terminal_holder: list[TerminalReservation] = []
    commit_refusals: list[str] = []
    committed_publications: list[str] = []

    def cancel_winner() -> None:
        outcome_holder.append(gate.cancel(cancel_bytes, locked=lock_held, proceed=release))

    def terminal_loser() -> None:
        terminal_attempted.set()
        try:
            gate.reserve_terminal(
                REQUEST_ID,
                "committed",
                publish=lambda: committed_publications.append("published"),
            )
        except ContractRefusal as exc:
            commit_refusals.append(f"{exc.stage}:{exc.rule_id}")
        terminal_holder.append(gate.reserve_terminal(REQUEST_ID, "cancelled"))

    first = threading.Thread(target=cancel_winner, name="cancel-winner")
    first.start()
    if not lock_held.wait(timeout=10):
        raise TimeoutError("cancel winner did not acquire the serialization lock")
    second = threading.Thread(target=terminal_loser, name="terminal-loser")
    second.start()
    if not terminal_attempted.wait(timeout=10) or not second.is_alive():
        raise RuntimeError("terminal did not contend while cancellation held the lock")
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)
    if (
        first.is_alive()
        or second.is_alive()
        or len(outcome_holder) != 1
        or len(terminal_holder) != 1
    ):
        raise RuntimeError("accepted race did not terminate exactly once per contender")
    outcome = cast(dict[str, Any], runtime.decode(outcome_holder[0]))
    terminal = terminal_holder[0]
    error = _cancelled_error(terminal)
    transcript = [request, cancel_request, outcome, error]
    runtime.validate_transcript(transcript)
    replay = gate.cancel(cancel_bytes)
    return {
        "case": "cancel-lock-wins",
        "causal_order": [
            "cancel-transaction-lock-acquired",
            "terminal-contender-started-and-blocked",
            "cancel-transaction-released",
            "terminal-reserved-after-cancel",
        ],
        "outcome": outcome,
        "terminal": {"sequence": terminal.sequence, "state": terminal.state},
        "committed_publications": {
            "observed": len(committed_publications),
            "allowed": 0,
            "total": 1,
        },
        "committed_terminal_refusals": {
            "matched": len(commit_refusals),
            "total": 1,
            "refusal": commit_refusals[0] if commit_refusals else None,
        },
        "exact_replay": replay == outcome_holder[0],
        "transcript": transcript,
        "passed": (
            outcome["outcome"] == "accepted"
            and outcome["linearization_sequence"] == 0
            and terminal.sequence == 1
            and commit_refusals == ["lifecycle:LC-ACCEPTED-CANCEL-TERMINAL"]
            and not committed_publications
            and replay == outcome_holder[0]
        ),
    }


def _terminal_race(
    database: Path,
    runtime: ContractRuntime,
    request: dict[str, Any],
    committed_result: dict[str, Any],
    cancel_request: dict[str, Any],
    cancel_bytes: bytes,
) -> dict[str, object]:
    gate = _gate(database, runtime)
    gate.begin(REQUEST_ID)
    lock_held = threading.Event()
    release = threading.Event()
    cancel_attempted = threading.Event()
    terminal_holder: list[TerminalReservation] = []
    outcome_holder: list[bytes] = []
    committed_publications: list[str] = []

    def terminal_winner() -> None:
        terminal_holder.append(
            gate.reserve_terminal(
                REQUEST_ID,
                "committed",
                publish=lambda: committed_publications.append("published"),
                locked=lock_held,
                proceed=release,
            )
        )

    def cancel_loser() -> None:
        cancel_attempted.set()
        outcome_holder.append(gate.cancel(cancel_bytes))

    first = threading.Thread(target=terminal_winner, name="terminal-winner")
    first.start()
    if not lock_held.wait(timeout=10):
        raise TimeoutError("terminal winner did not acquire the serialization lock")
    second = threading.Thread(target=cancel_loser, name="cancel-loser")
    second.start()
    if not cancel_attempted.wait(timeout=10) or not second.is_alive():
        raise RuntimeError("cancellation did not contend while terminal held the lock")
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)
    if (
        first.is_alive()
        or second.is_alive()
        or len(outcome_holder) != 1
        or len(terminal_holder) != 1
    ):
        raise RuntimeError("terminal-won race did not terminate exactly once per contender")
    outcome = cast(dict[str, Any], runtime.decode(outcome_holder[0]))
    terminal = terminal_holder[0]
    result = deepcopy(committed_result)
    result["job"]["sequence"] = terminal.sequence
    transcript = [request, result, cancel_request, outcome]
    runtime.validate_transcript(transcript)
    replay = gate.cancel(cancel_bytes)
    return {
        "case": "terminal-lock-wins",
        "causal_order": [
            "terminal-transaction-lock-acquired",
            "cancel-contender-started-and-blocked",
            "terminal-transaction-released",
            "cancel-linearized-after-terminal",
        ],
        "outcome": outcome,
        "terminal": {"sequence": terminal.sequence, "state": terminal.state},
        "committed_publications": {
            "matched": len(committed_publications),
            "total": 1,
        },
        "exact_replay": replay == outcome_holder[0],
        "transcript": transcript,
        "passed": (
            outcome["outcome"] == "terminal-won"
            and outcome["linearization_sequence"] == 1
            and outcome["terminal_sequence"] == 0
            and outcome["terminal_state"] == "committed"
            and committed_publications == ["published"]
            and replay == outcome_holder[0]
        ),
    }


def _crash_case(
    database: Path,
    runtime: ContractRuntime,
    request: dict[str, Any],
    cancel_request: dict[str, Any],
    cancel_bytes: bytes,
    point: CrashPoint,
    expected_exit: int,
) -> dict[str, object]:
    gate = _gate(database, runtime)
    gate.begin(REQUEST_ID)
    context = mp.get_context("spawn")
    process = context.Process(
        target=_crash_child,
        args=(database, cancel_bytes, point),
        name=f"f108-cancel-{point}",
    )
    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
        raise TimeoutError(f"{point} cancellation crash process did not exit")
    recorded_before_replay = gate.recorded_outcome(REQUEST_ID)
    replay = gate.cancel(cancel_bytes)
    outcome = cast(dict[str, Any], runtime.decode(replay))
    terminal = gate.reserve_terminal(REQUEST_ID, "cancelled")
    transcript = [request, cancel_request, outcome, _cancelled_error(terminal)]
    runtime.validate_transcript(transcript)
    durable_expected = point == "after-commit"
    return {
        "case": f"crash-{point}",
        "exit_code": process.exitcode,
        "caller_received_outcome": False,
        "status_before_replay": "unknown",
        "durable_outcome_before_replay": recorded_before_replay is not None,
        "replay_returned_exact_durable_outcome": (
            recorded_before_replay == replay if recorded_before_replay is not None else True
        ),
        "outcome": outcome,
        "transcript": transcript,
        "passed": (
            process.exitcode == expected_exit
            and (recorded_before_replay is not None) == durable_expected
            and outcome["outcome"] == "accepted"
            and terminal.sequence == 1
        ),
    }


def _crash_child(database: Path, cancel_bytes: bytes, point: CrashPoint) -> NoReturn:
    gate = DurableCancellationGate(database, clock=lambda: OBSERVED_AT)
    gate.cancel(cancel_bytes, crash=point)
    os._exit(76)


def _gate(database: Path, runtime: ContractRuntime) -> DurableCancellationGate:
    return DurableCancellationGate(database, runtime=runtime, clock=lambda: OBSERVED_AT)


def _base_messages(runtime: ContractRuntime) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = cast(
        dict[str, Any],
        runtime.decode(
            (runtime.root / "fixtures" / "valid" / "transparent-result.json").read_bytes()
        ),
    )
    messages = cast(list[dict[str, Any]], fixture["messages"])
    return deepcopy(messages[0]), deepcopy(messages[-1])


def _cancelled_error(reservation: TerminalReservation) -> dict[str, Any]:
    return {
        "job": {
            "category": "cancellation",
            "code": "job.cancelled",
            "diagnostic_reference": "diag-00000000000000000000000000000000",
            "occurred_at": "2026-08-29T12:00:01Z",
            "request_id": reservation.request_id,
            "retryable": False,
            "schema": "kilix.media-job.error/v2",
            "sequence": reservation.sequence,
            "state": "cancelled",
        },
        "phase": "infer",
        "schema": "kilix.background-removal.error/v2",
    }


def _write(destination: Path, payload: bytes) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate_cancellation_evidence(args.output_dir)
    print(
        "PASS F108 cancellation "
        f"race={result['race_outcomes_forced']}/{result['race_outcomes_total']} "
        f"crash={result['crash_points_forced']}/{result['crash_points_total']} "
        f"transcripts={result['transcripts_valid']}/{result['transcripts_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
