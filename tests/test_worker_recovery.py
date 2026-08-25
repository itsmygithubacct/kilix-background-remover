"""Supervisor recovery from an abnormal worker death.

The plan requires recovery from inference or encoder failure and process-tree
cancellation. The existing worker test covers cancellation and deadline, both
of which are *orderly* paths the worker itself takes. This covers the disorderly
one: the worker is SIGKILLed mid-job, so it has no chance to clean up after
itself and everything must be done by the supervisor.

Four properties, and the mutation that must break each:

===  ==============================================  ==========================
ID   Property                                        Mutation
===  ==============================================  ==========================
W-1  no committed output at the destination          (structural - see atomic)
W-2  no staging residue survives the kill            drop cleanup on the death path
W-3  the supervisor recovers and the next job runs   drop the hard restart
W-4  no descendant of the dead worker is left        (observed, not mutated)
===  ==============================================  ==========================
"""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from kilix_background_remover.worker import WorkerSupervisor

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/corpus/large-100mp.png"


def _kill_when_staging_appears(
    directory: Path, pid_box: dict[str, int], stop: threading.Event
) -> None:
    """Event-driven rather than timed: kill the instant the worker has started
    writing, so the kill lands mid-job on any machine speed."""
    deadline = time.monotonic() + 60.0
    while not stop.is_set() and time.monotonic() < deadline:
        staged = list(directory.glob(".kilix-f108-*.stage"))
        pid = pid_box.get("pid")
        if staged and pid:
            with __import__("contextlib").suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            return
        time.sleep(0.01)


def _descendants(pid: int) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        parent = stat.rsplit(")", 1)[-1].split()[1]
        if int(parent) == pid:
            found.append(int(entry.name))
    return found


def test_worker_killed_mid_job_leaves_nothing_and_recovers(
    request_factory: Any, tmp_path: Path
) -> None:
    killed_dir = tmp_path / "killed"
    killed_dir.mkdir()
    recovery_dir = tmp_path / "recovery"
    recovery_dir.mkdir()
    big = request_factory(killed_dir, source=FIXTURE)
    small = request_factory(recovery_dir)

    pid_box: dict[str, int] = {}
    stop = threading.Event()
    watcher = threading.Thread(
        target=_kill_when_staging_appears, args=(killed_dir, pid_box, stop), daemon=True
    )

    with WorkerSupervisor() as supervisor:
        original = supervisor.pid
        assert isinstance(original, int)
        pid_box["pid"] = original
        watcher.start()
        try:
            killed = supervisor.run(big)
        finally:
            stop.set()
            watcher.join(timeout=5)

        # W-1: nothing committed, W-2: no staging residue
        assert killed.error is not None, "a SIGKILLed worker reported success"
        leftovers = sorted(p.name for p in killed_dir.iterdir())
        assert leftovers == [], f"residue after an abnormal worker death: {leftovers}"

        # W-4: the dead worker left no descendant behind
        assert _descendants(original) == [], "an orphaned descendant survived"

        # W-3: the supervisor recovered on a fresh worker
        recovered = supervisor.run(small)
        assert recovered.ok, "the supervisor did not recover after a worker death"
        assert supervisor.pid != original, "the dead worker was reused"
        assert any(recovery_dir.iterdir()), "the recovery job produced no output"
