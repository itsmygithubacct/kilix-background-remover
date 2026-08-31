"""Explicit process-tree RSS measurement for later release qualification."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PROCFS_ROOT = Path("/proc")
RSS_METRIC = "sampled-aggregate-process-tree-vmrss-bytes"


class _ProcessNotResident(RuntimeError):
    """The proc entry exists but the process is already dead or a zombie."""


@dataclass(frozen=True, slots=True)
class RssSnapshot:
    """One aggregate resident-memory sample for a process tree."""

    root_pid: int
    aggregate_rss_bytes: int
    process_count: int
    pids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PeakRssMeasurement:
    """The largest aggregate sample observed by one monitor run."""

    root_pid: int
    metric: str
    peak_rss_bytes: int
    peak_process_count: int
    peak_pids: tuple[int, ...]
    samples: int
    interval_seconds: float


def _parent_pid(stat_path: Path) -> int:
    payload = stat_path.read_text(encoding="ascii")
    closing = payload.rfind(")")
    if closing < 0:
        raise RuntimeError(f"malformed proc stat: {stat_path}")
    fields = payload[closing + 1 :].split()
    if len(fields) < 2:
        raise RuntimeError(f"truncated proc stat: {stat_path}")
    try:
        return int(fields[1])
    except ValueError as exc:
        raise RuntimeError(f"invalid parent PID in proc stat: {stat_path}") from exc


def _rss_bytes(status_path: Path) -> int:
    memory_field_seen = False
    rss_bytes: int | None = None
    state: str | None = None
    for line in status_path.read_text(encoding="ascii").splitlines():
        if line.startswith("State:"):
            fields = line.split()
            if len(fields) < 2 or fields[0] != "State:":
                raise RuntimeError(f"malformed state in proc status: {status_path}")
            state = fields[1]
            continue
        if line.startswith("Vm"):
            memory_field_seen = True
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[0] != "VmRSS:" or fields[2] != "kB":
            raise RuntimeError(f"malformed VmRSS in proc status: {status_path}")
        try:
            kibibytes = int(fields[1])
        except ValueError as exc:
            raise RuntimeError(f"invalid VmRSS in proc status: {status_path}") from exc
        if kibibytes < 0:
            raise RuntimeError(f"negative VmRSS in proc status: {status_path}")
        rss_bytes = kibibytes * 1024
    if state in {"X", "Z"}:
        raise _ProcessNotResident(f"process is not resident: {status_path}")
    if rss_bytes is not None:
        return rss_bytes
    if state is not None and not memory_field_seen:
        return 0
    raise RuntimeError(f"missing VmRSS in proc status: {status_path}")


def sample_process_tree_rss(
    root_pid: int,
    proc_root: Path = PROCFS_ROOT,
) -> RssSnapshot:
    """Sample aggregate RSS for one root and every observed descendant.

    Processes that exit between the stat and status reads are omitted from the
    current sample. Malformed or unreadable live entries fail closed rather
    than silently producing a partial measurement.
    """

    if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
        raise ValueError("root_pid must be a positive integer")

    children: dict[int, list[int]] = defaultdict(list)
    observed: set[int] = set()
    for entry in proc_root.iterdir():
        if not entry.name.isascii() or not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            parent = _parent_pid(entry / "stat")
        except FileNotFoundError:
            continue
        observed.add(pid)
        children[parent].append(pid)

    if root_pid not in observed:
        raise ProcessLookupError(f"root PID {root_pid} is absent from {proc_root}")

    descendants: set[int] = set()
    pending = deque([root_pid])
    while pending:
        pid = pending.popleft()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children.get(pid, ()))

    resident: dict[int, int] = {}
    for pid in sorted(descendants):
        try:
            resident[pid] = _rss_bytes(proc_root / str(pid) / "status")
        except (FileNotFoundError, _ProcessNotResident):
            if pid == root_pid:
                raise ProcessLookupError(
                    f"root PID {root_pid} exited during the RSS sample"
                ) from None
    pids = tuple(resident)
    return RssSnapshot(
        root_pid=root_pid,
        aggregate_rss_bytes=sum(resident.values()),
        process_count=len(pids),
        pids=pids,
    )


class ProcessTreeRssMonitor:
    """Periodically sample an explicit process tree without choosing a limit."""

    def __init__(
        self,
        root_pid: int,
        *,
        interval_seconds: float = 0.01,
        proc_root: Path = PROCFS_ROOT,
        sampler: Callable[[int, Path], RssSnapshot] = sample_process_tree_rss,
    ) -> None:
        if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
            raise ValueError("root_pid must be a positive integer")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, int | float)
            or not 0.001 <= interval_seconds <= 1.0
        ):
            raise ValueError("interval_seconds must be within 0.001..1.0")
        self._root_pid = root_pid
        self._interval_seconds = float(interval_seconds)
        self._proc_root = proc_root
        self._sampler = sampler
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._samples = 0
        self._peak: RssSnapshot | None = None
        self._measurement: PeakRssMeasurement | None = None

    def _capture(self) -> None:
        snapshot = self._sampler(self._root_pid, self._proc_root)
        if snapshot.root_pid != self._root_pid:
            raise RuntimeError("RSS sampler returned a different root PID")
        if snapshot.process_count != len(snapshot.pids):
            raise RuntimeError("RSS sampler returned an inconsistent process count")
        if snapshot.aggregate_rss_bytes < 0:
            raise RuntimeError("RSS sampler returned a negative aggregate")
        with self._lock:
            self._samples += 1
            if self._peak is None or snapshot.aggregate_rss_bytes > self._peak.aggregate_rss_bytes:
                self._peak = snapshot

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._capture()
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                self._stop.set()
                return

    def start(self) -> None:
        if self._thread is not None or self._measurement is not None:
            raise RuntimeError("RSS monitor instances cannot be restarted")
        self._capture()
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="kilix-background-remover-rss-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> PeakRssMeasurement:
        if self._measurement is not None:
            return self._measurement
        if self._thread is None:
            raise RuntimeError("RSS monitor has not been started")
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("RSS monitor did not stop")
        with self._lock:
            error = self._error
            peak = self._peak
            samples = self._samples
        if error is not None:
            raise RuntimeError("RSS monitor sampling failed") from error
        if peak is None or samples == 0:
            raise RuntimeError("RSS monitor returned no samples")
        self._measurement = PeakRssMeasurement(
            root_pid=self._root_pid,
            metric=RSS_METRIC,
            peak_rss_bytes=peak.aggregate_rss_bytes,
            peak_process_count=peak.process_count,
            peak_pids=peak.pids,
            samples=samples,
            interval_seconds=self._interval_seconds,
        )
        return self._measurement

    @property
    def measurement(self) -> PeakRssMeasurement:
        if self._measurement is None:
            raise RuntimeError("RSS monitor has not completed")
        return self._measurement

    def __enter__(self) -> ProcessTreeRssMonitor:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
