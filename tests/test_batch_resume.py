"""Batch and resume gate evidence.

Population taken from the P6 obligations of the scoping plan — stable ordering,
mixed failures, resume, duplicates, large directories, bounded queues, no
committed-output rollback, cancellation, hostile filenames, never overwrite —
minus the four already covered by tests/test_batch_ui_bridge.py (ordering,
mixed failure, resume, non-rolling, duplicate keys).

Mutation table, written before the controls:

====  =================================================  =========================
ID    Control                                            Mutation that must break it
====  =================================================  =========================
B-1   hostile batch keys refused by a fixed grammar      widen KEY_RE to `.*`
B-2   resume refused when an output was tampered with    drop the digest recheck
B-3   resume refused when the request changed            drop the fingerprint check
B-4   a failed item is retried, never resumed as done    accept a non-dict result
B-5   a symlinked state directory is refused             drop the symlink check
====  =================================================  =========================
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from kilix_background_remover.jobs import BatchEntry, BatchRunner
from kilix_background_remover.worker import WorkerSupervisor


def _run(entries: list[BatchEntry], state: Path, cancel: Any = None) -> list[Any]:
    supervisor = WorkerSupervisor()
    try:
        return BatchRunner(supervisor).run(entries, state_dir=state, cancel=cancel)
    finally:
        supervisor.close()


@pytest.mark.parametrize(
    "key",
    [
        "../escape",
        "/absolute",
        "Upper",
        "with space",
        ".leading-dot",
        "a" * 129,
        "",
    ],
)
def test_b1_hostile_batch_keys_are_refused(request_factory: Any, tmp_path: Path, key: str) -> None:
    """The key becomes a state filename, so its grammar is a path-safety
    boundary, not a cosmetic one."""
    state = tmp_path / "state"
    state.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    entry = BatchEntry(key=key, request=request_factory(out))
    with pytest.raises(ValueError, match="fixed local grammar"):
        _run([entry], state)


def test_b2_resume_is_refused_when_a_committed_output_was_tampered_with(
    request_factory: Any, tmp_path: Path
) -> None:
    """A resume record asserts that specific bytes are already on disk. If they
    changed, the record is stale and the item must run again rather than be
    reported as complete."""
    out = tmp_path / "out"
    out.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    entry = BatchEntry(key="case", request=request_factory(out))
    first = _run([entry], state)
    assert first[0].disposition == "executed"
    assert (state / "case.json").is_file()

    record = json.loads((state / "case.json").read_text())
    mask = Path(record["result"]["mask"]["path"])
    assert mask.is_file()
    # Size-preserving, so only the digest recheck can catch it.  Appending
    # bytes would be caught by the cheaper size comparison and would leave the
    # digest check unproven.
    original = mask.read_bytes()
    flipped = bytearray(original)
    flipped[-1] ^= 0xFF
    mask.write_bytes(bytes(flipped))
    assert mask.stat().st_size == len(original)

    second = _run([entry], state)
    assert second[0].disposition == "executed", "a tampered output was resumed as complete"


def test_b3_resume_is_refused_when_the_request_changed(
    request_factory: Any, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    first = _run([BatchEntry(key="case", request=request_factory(out))], state)
    assert first[0].disposition == "executed"

    changed = request_factory(out, output_kinds=["mask"])
    second = _run([BatchEntry(key="case", request=changed)], state)
    assert second[0].disposition == "executed", "a changed request resumed a stale result"


def test_b3_an_unchanged_request_does_resume(request_factory: Any, tmp_path: Path) -> None:
    """The null control for B-2 and B-3: without a mutation, resume must fire,
    otherwise those two prove only that resume never works."""
    out = tmp_path / "out"
    out.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    entry = BatchEntry(key="case", request=request_factory(out))
    assert _run([entry], state)[0].disposition == "executed"
    assert _run([entry], state)[0].disposition == "resumed"


def test_b4_a_failed_item_is_retried_rather_than_resumed(tmp_path: Path) -> None:
    """A persisted failure carries result=None, which the resume loader must
    refuse. Pinning the policy: failures are retried, never reported done."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "case.json").write_text(
        json.dumps(
            {
                "schema": "kilix.background-removal.batch-state/v1",
                "key": "case",
                "request_fingerprint": "0" * 64,
                "result": None,
                "error": {"code": "background.internal"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    from kilix_background_remover.jobs import _load_resume

    assert _load_resume(state / "case.json", "0" * 64) is None


def test_b5_a_symlinked_state_directory_is_refused(request_factory: Any, tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    out = tmp_path / "out"
    out.mkdir()
    entry = BatchEntry(key="case", request=request_factory(out))
    with pytest.raises(ValueError, match="existing regular directory"):
        _run([entry], link)


def test_cancellation_before_the_first_item_processes_nothing(
    request_factory: Any, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    cancel = threading.Event()
    cancel.set()
    entries = [
        BatchEntry(key=f"case{index}", request=request_factory(out, output_key=f"c{index}"))
        for index in range(3)
    ]
    assert _run(entries, state, cancel) == []
    assert not list(state.glob("*.json"))
    assert not list(out.glob("*.png"))
