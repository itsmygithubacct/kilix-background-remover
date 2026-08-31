"""Causal controls for explicit process-tree RSS measurement.

R-1 fails if descendants are omitted or unrelated processes are included.
R-2 fails if malformed proc data is silently treated as a measurement.
R-3 fails if peak selection, sample count, or root identity can drift.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from kilix_background_remover.rss import (
    RSS_METRIC,
    ProcessTreeRssMonitor,
    RssSnapshot,
    sample_process_tree_rss,
)
from kilix_background_remover.worker import WorkerSupervisor


def _process(proc_root: Path, pid: int, parent: int, rss_kib: int) -> None:
    root = proc_root / str(pid)
    root.mkdir()
    (root / "stat").write_text(
        f"{pid} (name with ) parenthesis) S {parent} 0 0 0\n",
        encoding="ascii",
    )
    (root / "status").write_text(
        f"Name:\ttest\nState:\tS (sleeping)\nVmRSS:\t{rss_kib} kB\n",
        encoding="ascii",
    )


def test_r1_exact_descendant_tree_is_aggregated_once(tmp_path: Path) -> None:
    _process(tmp_path, 100, 1, 100)
    _process(tmp_path, 101, 100, 200)
    _process(tmp_path, 102, 101, 300)
    _process(tmp_path, 103, 1, 900)

    snapshot = sample_process_tree_rss(100, tmp_path)

    assert snapshot == RssSnapshot(
        root_pid=100,
        aggregate_rss_bytes=600 * 1024,
        process_count=3,
        pids=(100, 101, 102),
    )


def test_r1_child_that_exits_during_status_read_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _process(tmp_path, 200, 1, 100)
    _process(tmp_path, 201, 200, 200)
    original = Path.read_text

    def disappear(path: Path, *args: object, **kwargs: object) -> str:
        if path == tmp_path / "201" / "status":
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", disappear)
    snapshot = sample_process_tree_rss(200, tmp_path)

    assert snapshot.process_count == 1
    assert snapshot.pids == (200,)
    assert snapshot.aggregate_rss_bytes == 100 * 1024


def test_r1_zombie_descendant_is_not_counted_as_resident(tmp_path: Path) -> None:
    _process(tmp_path, 250, 1, 100)
    _process(tmp_path, 251, 250, 200)
    (tmp_path / "251" / "status").write_text(
        "Name:\ttest\nState:\tZ (zombie)\n",
        encoding="ascii",
    )

    snapshot = sample_process_tree_rss(250, tmp_path)

    assert snapshot.process_count == 1
    assert snapshot.pids == (250,)
    assert snapshot.aggregate_rss_bytes == 100 * 1024


def test_r1_live_descendant_without_an_mm_contributes_zero_rss(tmp_path: Path) -> None:
    _process(tmp_path, 275, 1, 100)
    _process(tmp_path, 276, 275, 200)
    (tmp_path / "276" / "status").write_text(
        "Name:\ttest\nState:\tS (sleeping)\n",
        encoding="ascii",
    )

    snapshot = sample_process_tree_rss(275, tmp_path)

    assert snapshot.process_count == 2
    assert snapshot.pids == (275, 276)
    assert snapshot.aggregate_rss_bytes == 100 * 1024


def test_r2_missing_root_and_malformed_live_status_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ProcessLookupError, match="root PID 300"):
        sample_process_tree_rss(300, tmp_path)

    _process(tmp_path, 300, 1, 100)
    (tmp_path / "300" / "status").write_text("Name:\ttest\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="missing VmRSS"):
        sample_process_tree_rss(300, tmp_path)


@pytest.mark.parametrize(
    ("root_pid", "interval"),
    [(0, 0.01), (True, 0.01), (1, 0.0), (1, 1.001), (1, True)],
)
def test_r2_invalid_monitor_bounds_are_refused(root_pid: int, interval: float) -> None:
    with pytest.raises(ValueError):
        ProcessTreeRssMonitor(root_pid, interval_seconds=interval)


def test_r3_monitor_selects_the_exact_peak_and_reports_sampling_identity() -> None:
    values = [100, 900, 300]
    calls = 0
    third_sample = threading.Event()

    def sampler(root_pid: int, _proc_root: Path) -> RssSnapshot:
        nonlocal calls
        value = values[min(calls, len(values) - 1)]
        calls += 1
        if calls >= len(values):
            third_sample.set()
        return RssSnapshot(root_pid, value, 2, (root_pid, root_pid + 1))

    monitor = ProcessTreeRssMonitor(400, interval_seconds=0.001, sampler=sampler)
    monitor.start()
    assert third_sample.wait(timeout=1.0)
    measurement = monitor.stop()

    assert measurement.root_pid == 400
    assert measurement.metric == RSS_METRIC
    assert measurement.peak_rss_bytes == 900
    assert measurement.peak_process_count == 2
    assert measurement.peak_pids == (400, 401)
    assert measurement.samples >= 3
    assert measurement.interval_seconds == 0.001
    assert monitor.measurement == measurement
    assert monitor.stop() == measurement


def test_r3_live_proc_sample_contains_the_calling_process() -> None:
    snapshot = sample_process_tree_rss(os.getpid())
    assert os.getpid() in snapshot.pids
    assert snapshot.process_count >= 1
    assert snapshot.aggregate_rss_bytes > 0


def test_r3_monitor_observes_the_real_provider_process_tree(
    request_factory: object,
    tmp_path: Path,
) -> None:
    request = request_factory(tmp_path)  # type: ignore[operator]
    with (
        ProcessTreeRssMonitor(os.getpid(), interval_seconds=0.001) as monitor,
        WorkerSupervisor(cancellation_database=tmp_path / "cancellation.sqlite3") as supervisor,
    ):
        worker_pid = supervisor.pid
        outcome = supervisor.run(request)

    assert outcome.ok
    assert worker_pid is not None
    assert os.getpid() in monitor.measurement.peak_pids
    assert worker_pid in monitor.measurement.peak_pids
    assert monitor.measurement.peak_process_count >= 2
    assert monitor.measurement.samples >= 2
