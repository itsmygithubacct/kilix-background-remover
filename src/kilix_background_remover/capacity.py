"""Frozen F100-C0 fixture attestation and measurement windows.

The capacity tier is never accepted from a label alone.  A window first checks
the properties that are observable from inside the frozen fixture, then records
the full Linux load-average line immediately before and after the measured
operation.  This is deliberately a measurement mechanism, not a release
profile, quality floor, or acceptance verdict.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

from .rss import PeakRssMeasurement, ProcessTreeRssMonitor

F100_C0_FREEZE_SHA256 = "89ec10d41fd5dc8bf69472414325f3610dbaa439b5a07dbee027cd16559b6d68"
LOADAVG_PATH = Path("/proc/loadavg")
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_LOADAVG_RE = re.compile(
    r"^(?P<one>[0-9]+(?:\.[0-9]+)?) "
    r"(?P<five>[0-9]+(?:\.[0-9]+)?) "
    r"(?P<fifteen>[0-9]+(?:\.[0-9]+)?) "
    r"(?P<running>[0-9]+)/(?P<entities>[0-9]+) "
    r"(?P<last_pid>[0-9]+)$"
)


class CapacityTier(StrEnum):
    """The three F100-C0 tiers admitted by the 0.2.1 exception."""

    H0 = "h0"
    H1 = "h1"
    H2 = "h2"


@dataclass(frozen=True, slots=True)
class LoadAverage:
    """One complete, strictly parsed ``/proc/loadavg`` observation."""

    raw: str
    one_minute: str
    five_minutes: str
    fifteen_minutes: str
    running_entities: int
    total_entities: int
    most_recent_pid: int

    def wire(self) -> dict[str, object]:
        return {
            "raw": self.raw,
            "one_minute": self.one_minute,
            "five_minutes": self.five_minutes,
            "fifteen_minutes": self.fifteen_minutes,
            "running_entities": self.running_entities,
            "total_entities": self.total_entities,
            "most_recent_pid": self.most_recent_pid,
        }


@dataclass(frozen=True, slots=True)
class NvidiaIdentity:
    """The bounded identity read from the loaded NVIDIA driver."""

    name: str
    driver_version: str

    def wire(self) -> dict[str, object]:
        return {
            "name": self.name,
            "driver_version": self.driver_version,
        }


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    """Safe, hostname-free facts observable from inside a capacity fixture."""

    machine: str
    logical_cpus: int
    memory_total_bytes: int
    storage_free_bytes: int
    block_device_sizes_bytes: tuple[int, ...]
    debian_version: str | None
    debian_version_full: str | None
    qemu_vendor: bool
    q35_machine: bool
    qemu64_cpu: bool
    nvidia_devices: int
    nvidia: NvidiaIdentity | None
    governor_cpus: int
    performance_governors: int
    no_turbo: int | None
    minimum_scaling_max_khz: int | None
    maximum_scaling_max_khz: int | None

    def wire(self) -> dict[str, object]:
        return {
            "machine": self.machine,
            "logical_cpus": self.logical_cpus,
            "memory_total_bytes": self.memory_total_bytes,
            "storage_free_bytes": self.storage_free_bytes,
            "block_device_sizes_bytes": list(self.block_device_sizes_bytes),
            "debian_version": self.debian_version,
            "debian_version_full": self.debian_version_full,
            "qemu_vendor": self.qemu_vendor,
            "q35_machine": self.q35_machine,
            "qemu64_cpu": self.qemu64_cpu,
            "nvidia_devices": self.nvidia_devices,
            "nvidia": None if self.nvidia is None else self.nvidia.wire(),
            "governor_cpus": self.governor_cpus,
            "performance_governors": self.performance_governors,
            "no_turbo": self.no_turbo,
            "minimum_scaling_max_khz": self.minimum_scaling_max_khz,
            "maximum_scaling_max_khz": self.maximum_scaling_max_khz,
        }


@dataclass(frozen=True, slots=True)
class FixtureCheck:
    check_id: str
    passed: bool

    def wire(self) -> dict[str, object]:
        return {"check_id": self.check_id, "passed": self.passed}


@dataclass(frozen=True, slots=True)
class FixtureAttestation:
    tier: CapacityTier
    observation: RuntimeObservation
    checks: tuple[FixtureCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def wire(self) -> dict[str, object]:
        passed = sum(check.passed for check in self.checks)
        return {
            "tier": self.tier.value,
            "checks": {"passed": passed, "total": len(self.checks)},
            "items": [check.wire() for check in self.checks],
            "observation": self.observation.wire(),
        }


@dataclass(frozen=True, slots=True)
class CapacityMeasurement:
    """One completed, fixture-attested process-tree measurement window."""

    tier: CapacityTier
    attestation: FixtureAttestation
    start_loadavg: LoadAverage
    end_loadavg: LoadAverage
    duration_ns: int
    peak_rss: PeakRssMeasurement

    def wire(self) -> dict[str, object]:
        return {
            "schema": "kilix.background-removal.capacity-window/v1",
            "fixture": frozen_fixture(self.tier),
            "runtime_attestation": self.attestation.wire(),
            "loadavg": {
                "published_samples": {"observed": 2, "required": 2},
                "start": self.start_loadavg.wire(),
                "end": self.end_loadavg.wire(),
            },
            "duration_ns": self.duration_ns,
            "peak_rss": {
                "root_pid": self.peak_rss.root_pid,
                "metric": self.peak_rss.metric,
                "peak_rss_bytes": self.peak_rss.peak_rss_bytes,
                "peak_process_count": self.peak_rss.peak_process_count,
                "peak_pids": list(self.peak_rss.peak_pids),
                "samples": self.peak_rss.samples,
                "interval_seconds": self.peak_rss.interval_seconds,
            },
            "release_thresholds_applied": {"observed": 0, "required": 0},
            "release_verdict": None,
        }


class _RssMonitor(Protocol):
    def start(self) -> None: ...

    def stop(self) -> PeakRssMeasurement: ...


def frozen_fixture(tier: CapacityTier | str) -> dict[str, object]:
    """Return a new wire identity for one frozen F100-C0 fixture."""

    selected = CapacityTier(tier)
    common: dict[str, object] = {
        "freeze": "f100-c0-capacity-fixture-r1",
        "freeze_sha256": F100_C0_FREEZE_SHA256,
        "tier": selected.value,
        "h3_excluded": True,
    }
    if selected is CapacityTier.H0:
        return {
            **common,
            "kind": "reproducible-vm",
            "machine": "q35",
            "cpu": "qemu64",
            "logical_cpus": 2,
            "memory_mib": 4096,
            "disk_gib": 40,
            "minimum_free_gib": 32,
            "gpu_passthrough": False,
            "debian_media": "13.5.0-amd64-netinst",
            "apt_snapshot": "20260727T000000Z",
        }
    if selected is CapacityTier.H1:
        return {
            **common,
            "kind": "reproducible-vm",
            "machine": "q35",
            "cpu": "qemu64",
            "logical_cpus": 4,
            "memory_mib": 8192,
            "disk_gib": 100,
            "minimum_free_gib": 80,
            "gpu_passthrough": False,
            "debian_media": "13.5.0-amd64-netinst",
            "apt_snapshot": "20260727T000000Z",
        }
    return {
        **common,
        "kind": "frequency-pinned-bare-metal",
        "logical_cpus": 16,
        "minimum_memory_gib": 16,
        "minimum_free_gib": 120,
        "gpu": "NVIDIA GeForce RTX 3070",
        "vram_mib": 8192,
        "driver": "550.163.01",
        "governor": "performance",
        "governor_cpus": 16,
        "no_turbo": 1,
        "scaling_max_khz": 2_900_000,
        "quiesced": False,
        "loadavg_start_end_required": True,
        "debian_version_full": "13.6",
    }


def capacity_identity() -> dict[str, object]:
    """Describe the complete, policy-neutral fixture measurement mechanism."""

    return {
        "available": True,
        "fixture_freeze_sha256": F100_C0_FREEZE_SHA256,
        "frozen_tiers": {"observed": 3, "required": 3},
        "tiers": [frozen_fixture(tier) for tier in CapacityTier],
        "runtime_attestation": "required-before-window",
        "loadavg_samples_per_window": {"observed": 2, "required": 2},
        "h2_quiesced": False,
        "peak_rss": True,
        "peak_vram": False,
        "release_profile": None,
        "release_corpus": None,
        "release_quality_floor": None,
        "release_qualified": False,
    }


def parse_load_average(payload: bytes | str) -> LoadAverage:
    """Parse exactly one bounded ASCII Linux load-average line."""

    if isinstance(payload, bytes):
        if not payload or len(payload) > 128:
            raise ValueError("loadavg payload is outside its fixed byte bound")
        try:
            raw = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("loadavg payload is not ASCII") from exc
    elif isinstance(payload, str):
        raw = payload
        if not raw or len(raw.encode("utf-8")) > 128:
            raise ValueError("loadavg payload is outside its fixed byte bound")
        if not raw.isascii():
            raise ValueError("loadavg payload is not ASCII")
    else:
        raise TypeError("loadavg payload must be bytes or text")
    if raw.endswith("\n"):
        raw = raw[:-1]
    if "\n" in raw or "\r" in raw:
        raise ValueError("loadavg payload must contain exactly one line")
    match = _LOADAVG_RE.fullmatch(raw)
    if match is None:
        raise ValueError("loadavg payload has an invalid Linux shape")
    running = int(match.group("running"))
    entities = int(match.group("entities"))
    last_pid = int(match.group("last_pid"))
    if entities <= 0 or running > entities or last_pid <= 0:
        raise ValueError("loadavg counters are inconsistent")
    return LoadAverage(
        raw=raw,
        one_minute=match.group("one"),
        five_minutes=match.group("five"),
        fifteen_minutes=match.group("fifteen"),
        running_entities=running,
        total_entities=entities,
        most_recent_pid=last_pid,
    )


def read_load_average(path: Path = LOADAVG_PATH) -> LoadAverage:
    return parse_load_average(path.read_bytes())


def _read_optional(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (FileNotFoundError, PermissionError, UnicodeError, OSError):
        return None
    return value or None


def _os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, UnicodeError, OSError):
        return values
    for line in lines:
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.isascii() and key.replace("_", "").isalnum():
            values[key] = value.strip().strip('"')
    return values


def _memory_total(path: Path = Path("/proc/meminfo")) -> int:
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if fields and fields[0] == "MemTotal:":
            if len(fields) != 3 or fields[2] != "kB":
                break
            value = int(fields[1])
            if value <= 0:
                break
            return value * 1024
    raise RuntimeError("MemTotal is absent or malformed")


def _cpu_model_is_qemu64(path: Path = Path("/proc/cpuinfo")) -> bool:
    models = [
        line.split(":", 1)[1].strip()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.startswith("model name") and ":" in line
    ]
    return bool(models) and all(model.startswith("QEMU Virtual CPU") for model in models)


def _block_device_sizes(root: Path = Path("/sys/class/block")) -> tuple[int, ...]:
    sizes: set[int] = set()
    for entry in root.iterdir():
        if (entry / "partition").exists():
            continue
        sectors = _read_optional(entry / "size")
        if sectors is None or not sectors.isascii() or not sectors.isdecimal():
            continue
        size = int(sectors) * 512
        if size > 0:
            sizes.add(size)
    return tuple(sorted(sizes))


def _nvidia_identity(
    gpu_root: Path = Path("/proc/driver/nvidia/gpus"),
    version_path: Path = Path("/proc/driver/nvidia/version"),
) -> NvidiaIdentity | None:
    information = sorted(gpu_root.glob("*/information"))
    if len(information) != 1:
        return None
    try:
        payload = information[0].read_bytes()
        version_payload = version_path.read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if not payload or len(payload) > 4096 or not version_payload or len(version_payload) > 4096:
        return None
    try:
        lines = payload.decode("ascii").splitlines()
        version = version_payload.decode("ascii")
    except UnicodeDecodeError:
        return None
    models = [line.split(":", 1)[1].strip() for line in lines if line.startswith("Model:")]
    version_match = re.search(r"Kernel Module\s+([0-9]+(?:\.[0-9]+)+)\s", version)
    if len(models) != 1 or not models[0] or version_match is None:
        return None
    return NvidiaIdentity(models[0], version_match.group(1))


def observe_runtime(storage_paths: Sequence[Path]) -> RuntimeObservation:
    """Collect only the facts needed to reject a mismatched fixture."""

    if not storage_paths:
        raise ValueError("at least one measurement storage path is required")
    free_values: list[int] = []
    for path in storage_paths:
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("measurement storage paths must be existing directories")
        free_values.append(shutil.disk_usage(resolved).free)
    governors = [
        value
        for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
        if (value := _read_optional(path)) is not None
    ]
    maximum_frequencies: list[int] = []
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_max_freq"):
        value = _read_optional(path)
        if value is not None and value.isascii() and value.isdecimal():
            maximum_frequencies.append(int(value))
    no_turbo_raw = _read_optional(Path("/sys/devices/system/cpu/intel_pstate/no_turbo"))
    no_turbo = int(no_turbo_raw) if no_turbo_raw in {"0", "1"} else None
    vendor = _read_optional(Path("/sys/class/dmi/id/sys_vendor")) or ""
    product = _read_optional(Path("/sys/class/dmi/id/product_name")) or ""
    nvidia_root = Path("/proc/driver/nvidia/gpus")
    nvidia_devices = len(list(nvidia_root.iterdir())) if nvidia_root.is_dir() else 0
    release = _os_release()
    return RuntimeObservation(
        machine=platform.machine(),
        logical_cpus=os.cpu_count() or 0,
        memory_total_bytes=_memory_total(),
        storage_free_bytes=min(free_values),
        block_device_sizes_bytes=_block_device_sizes(),
        debian_version=release.get("VERSION_ID") if release.get("ID") == "debian" else None,
        debian_version_full=(
            release.get("DEBIAN_VERSION_FULL") if release.get("ID") == "debian" else None
        ),
        qemu_vendor="qemu" in vendor.casefold(),
        q35_machine="q35" in product.casefold(),
        qemu64_cpu=_cpu_model_is_qemu64(),
        nvidia_devices=nvidia_devices,
        nvidia=_nvidia_identity() if nvidia_devices else None,
        governor_cpus=len(governors),
        performance_governors=sum(value == "performance" for value in governors),
        no_turbo=no_turbo,
        minimum_scaling_max_khz=min(maximum_frequencies) if maximum_frequencies else None,
        maximum_scaling_max_khz=max(maximum_frequencies) if maximum_frequencies else None,
    )


def attest_runtime(
    tier: CapacityTier | str,
    observation: RuntimeObservation,
) -> FixtureAttestation:
    """Compare a runtime observation with every locally observable freeze fact."""

    selected = CapacityTier(tier)
    fixture = frozen_fixture(selected)
    checks = [
        FixtureCheck("architecture-x86-64", observation.machine == "x86_64"),
        FixtureCheck("logical-cpus", observation.logical_cpus == fixture["logical_cpus"]),
        FixtureCheck("debian-major-13", observation.debian_version == "13"),
        FixtureCheck(
            "storage-free-floor",
            observation.storage_free_bytes >= cast(int, fixture["minimum_free_gib"]) * _GIB,
        ),
    ]
    if selected in {CapacityTier.H0, CapacityTier.H1}:
        configured_memory = cast(int, fixture["memory_mib"]) * _MIB
        checks.extend(
            [
                FixtureCheck(
                    "configured-memory-envelope",
                    configured_memory - 256 * _MIB
                    <= observation.memory_total_bytes
                    <= configured_memory,
                ),
                FixtureCheck("qemu-vendor", observation.qemu_vendor),
                FixtureCheck("q35-machine", observation.q35_machine),
                FixtureCheck("qemu64-cpu", observation.qemu64_cpu),
                FixtureCheck(
                    "configured-disk-size",
                    cast(int, fixture["disk_gib"]) * _GIB in observation.block_device_sizes_bytes,
                ),
                FixtureCheck("no-gpu-passthrough", observation.nvidia_devices == 0),
                FixtureCheck(
                    "debian-media-family",
                    observation.debian_version_full is None
                    or observation.debian_version_full.startswith("13.5"),
                ),
            ]
        )
    else:
        nvidia = observation.nvidia
        checks.extend(
            [
                FixtureCheck(
                    "memory-floor",
                    observation.memory_total_bytes
                    >= cast(int, fixture["minimum_memory_gib"]) * _GIB,
                ),
                FixtureCheck("one-nvidia-device", observation.nvidia_devices == 1),
                FixtureCheck("nvidia-model", nvidia is not None and nvidia.name == fixture["gpu"]),
                FixtureCheck(
                    "nvidia-driver",
                    nvidia is not None and nvidia.driver_version == fixture["driver"],
                ),
                FixtureCheck(
                    "governor-population",
                    observation.governor_cpus == fixture["governor_cpus"],
                ),
                FixtureCheck(
                    "governor-performance",
                    observation.performance_governors == fixture["governor_cpus"],
                ),
                FixtureCheck("no-turbo", observation.no_turbo == fixture["no_turbo"]),
                FixtureCheck(
                    "frequency-ceiling-minimum",
                    observation.minimum_scaling_max_khz == fixture["scaling_max_khz"],
                ),
                FixtureCheck(
                    "frequency-ceiling-maximum",
                    observation.maximum_scaling_max_khz == fixture["scaling_max_khz"],
                ),
                FixtureCheck(
                    "debian-point-release",
                    observation.debian_version_full == fixture["debian_version_full"],
                ),
            ]
        )
    return FixtureAttestation(selected, observation, tuple(checks))


class CapacityMeasurementWindow:
    """Fail-closed context manager for one frozen-fixture measurement."""

    def __init__(
        self,
        tier: CapacityTier | str,
        *,
        storage_paths: Sequence[Path],
        observation: RuntimeObservation | None = None,
        loadavg_reader: Callable[[], LoadAverage] = read_load_average,
        clock: Callable[[], int] = time.monotonic_ns,
        monitor_factory: Callable[[], _RssMonitor] | None = None,
    ) -> None:
        self._tier = CapacityTier(tier)
        self._storage_paths = tuple(storage_paths)
        self._observation = observation
        self._loadavg_reader = loadavg_reader
        self._clock = clock
        self._monitor_factory = monitor_factory or (lambda: ProcessTreeRssMonitor(os.getpid()))
        self._monitor: _RssMonitor | None = None
        self._attestation: FixtureAttestation | None = None
        self._start_loadavg: LoadAverage | None = None
        self._started_ns: int | None = None
        self._measurement: CapacityMeasurement | None = None
        self._entered = False

    def start(self) -> None:
        if self._entered:
            raise RuntimeError("capacity measurement windows cannot be restarted")
        self._entered = True
        observation = self._observation or observe_runtime(self._storage_paths)
        attestation = attest_runtime(self._tier, observation)
        if not attestation.passed:
            failed = ",".join(check.check_id for check in attestation.checks if not check.passed)
            raise RuntimeError(f"capacity fixture runtime mismatch: {failed}")
        start_loadavg = self._loadavg_reader()
        started_ns = self._clock()
        monitor = self._monitor_factory()
        monitor.start()
        self._attestation = attestation
        self._start_loadavg = start_loadavg
        self._started_ns = started_ns
        self._monitor = monitor

    def stop(self) -> CapacityMeasurement:
        if self._measurement is not None:
            return self._measurement
        if (
            self._monitor is None
            or self._attestation is None
            or self._start_loadavg is None
            or self._started_ns is None
        ):
            raise RuntimeError("capacity measurement window has not started")
        ended_ns = self._clock()
        try:
            end_loadavg = self._loadavg_reader()
        finally:
            peak_rss = self._monitor.stop()
        if ended_ns < self._started_ns:
            raise RuntimeError("monotonic clock moved backwards")
        self._measurement = CapacityMeasurement(
            tier=self._tier,
            attestation=self._attestation,
            start_loadavg=self._start_loadavg,
            end_loadavg=end_loadavg,
            duration_ns=ended_ns - self._started_ns,
            peak_rss=peak_rss,
        )
        return self._measurement

    @property
    def measurement(self) -> CapacityMeasurement:
        if self._measurement is None:
            raise RuntimeError("capacity measurement window has not completed")
        return self._measurement

    def __enter__(self) -> CapacityMeasurementWindow:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
