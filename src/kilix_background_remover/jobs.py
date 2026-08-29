"""Ordered, resumable batch execution over the shared worker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from .contracts import parse_request, sha256_file
from .worker import JobOutcome, WorkerSupervisor

KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
STATE_SCHEMA = "kilix.background-removal.batch-state/v1"
MAX_STATE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BatchEntry:
    key: str
    request: dict[str, object]


@dataclass(frozen=True, slots=True)
class BatchItemOutcome:
    index: int
    key: str
    disposition: str
    outcome: JobOutcome


def _fingerprint(request: dict[str, object]) -> str:
    stable = json.loads(json.dumps(request))
    job = stable["job"]
    if isinstance(job, dict):
        job.pop("request_id", None)
        job.pop("submitted_at", None)
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _fresh_attempt(request: dict[str, object]) -> dict[str, object]:
    """Give a non-resumed retry a new v2 lifecycle identity."""
    attempt = deepcopy(request)
    job = attempt.get("job")
    if not isinstance(job, dict):
        raise ValueError("validated batch request has no job object")
    job["request_id"] = str(uuid.uuid4())
    job["submitted_at"] = (
        datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )
    return attempt


def _result_files_are_current(result: dict[str, object]) -> bool:
    mask = result.get("mask")
    outputs = result.get("outputs")
    if not isinstance(mask, dict) or not isinstance(outputs, list):
        return False
    records = [mask, *(item for item in outputs if isinstance(item, dict))]
    if len(records) != len(outputs) + 1:
        return False
    for record in records:
        path = record.get("path")
        digest = record.get("sha256")
        size = record.get("bytes")
        if not isinstance(path, str) or not isinstance(digest, str) or not isinstance(size, int):
            return False
        candidate = Path(path)
        try:
            status = candidate.lstat()
        except OSError:
            return False
        if candidate.is_symlink() or not candidate.is_file() or status.st_size != size:
            return False
        if sha256_file(candidate) != digest:
            return False
    return True


def _load_resume(path: Path, fingerprint: str) -> JobOutcome | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_STATE_BYTES:
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("schema") != STATE_SCHEMA or raw.get("request_fingerprint") != fingerprint:
        return None
    result = raw.get("result")
    if not isinstance(result, dict) or not _result_files_are_current(result):
        return None
    return JobOutcome(cast(dict[str, object], result), None, [])


def _persist(path: Path, key: str, fingerprint: str, outcome: JobOutcome) -> None:
    record: dict[str, object] = {
        "schema": STATE_SCHEMA,
        "key": key,
        "request_fingerprint": fingerprint,
        "result": outcome.result,
        "error": outcome.error,
    }
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{key}.", suffix=".stage", dir=path.parent)
    stage = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        stage.unlink(missing_ok=True)
        raise


class BatchRunner:
    """Run caller-ordered entries serially with bounded memory and warm inference."""

    def __init__(self, supervisor: WorkerSupervisor) -> None:
        self._supervisor = supervisor

    def run(
        self,
        entries: list[BatchEntry],
        *,
        state_dir: Path,
        cancel: threading.Event | None = None,
    ) -> list[BatchItemOutcome]:
        if state_dir.is_symlink() or not state_dir.is_dir():
            raise ValueError("batch state directory must be an existing regular directory")
        keys = [entry.key for entry in entries]
        if len(keys) != len(set(keys)):
            raise ValueError("batch entry keys must be unique")
        if any(not KEY_RE.fullmatch(key) for key in keys):
            raise ValueError("batch entry key is outside the fixed local grammar")

        outcomes: list[BatchItemOutcome] = []
        for index, entry in enumerate(entries):
            parse_request(entry.request)
            fingerprint = _fingerprint(entry.request)
            state_path = state_dir / f"{entry.key}.json"
            resumed = _load_resume(state_path, fingerprint)
            if resumed is not None:
                outcomes.append(BatchItemOutcome(index, entry.key, "resumed", resumed))
                continue
            if cancel is not None and cancel.is_set():
                break
            execution_request = (
                _fresh_attempt(entry.request) if state_path.exists() else entry.request
            )
            outcome = self._supervisor.run(execution_request, cancel=cancel)
            _persist(state_path, entry.key, fingerprint, outcome)
            outcomes.append(BatchItemOutcome(index, entry.key, "executed", outcome))
            if outcome.error is not None:
                job = outcome.error.get("job")
                if isinstance(job, dict) and job.get("state") == "cancelled":
                    break
        return outcomes
