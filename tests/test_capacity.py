"""Causal controls for the frozen F100-C0 measurement boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

import kilix_background_remover.capacity as capacity_module
import kilix_background_remover.cli as cli
from kilix_background_remover.capacity import (
    F100_C0_FREEZE_SHA256,
    CapacityMeasurementWindow,
    CapacityTier,
    NvidiaIdentity,
    RuntimeObservation,
    attest_runtime,
    capacity_identity,
    frozen_fixture,
    parse_load_average,
)
from kilix_background_remover.contract_v2 import canonical_bytes
from kilix_background_remover.rss import RSS_METRIC, PeakRssMeasurement
from kilix_background_remover.worker import JobOutcome


def _observation(tier: CapacityTier) -> RuntimeObservation:
    if tier is CapacityTier.H0:
        return RuntimeObservation(
            machine="x86_64",
            logical_cpus=2,
            memory_total_bytes=4096 * 1024 * 1024 - 64 * 1024 * 1024,
            storage_free_bytes=33 * 1024**3,
            block_device_sizes_bytes=(40 * 1024**3,),
            debian_version="13",
            debian_version_full="13.5",
            qemu_vendor=True,
            q35_machine=True,
            qemu64_cpu=True,
            nvidia_devices=0,
            nvidia=None,
            governor_cpus=0,
            performance_governors=0,
            no_turbo=None,
            minimum_scaling_max_khz=None,
            maximum_scaling_max_khz=None,
        )
    if tier is CapacityTier.H1:
        return RuntimeObservation(
            machine="x86_64",
            logical_cpus=4,
            memory_total_bytes=8192 * 1024 * 1024 - 64 * 1024 * 1024,
            storage_free_bytes=81 * 1024**3,
            block_device_sizes_bytes=(100 * 1024**3,),
            debian_version="13",
            debian_version_full="13.5",
            qemu_vendor=True,
            q35_machine=True,
            qemu64_cpu=True,
            nvidia_devices=0,
            nvidia=None,
            governor_cpus=0,
            performance_governors=0,
            no_turbo=None,
            minimum_scaling_max_khz=None,
            maximum_scaling_max_khz=None,
        )
    return RuntimeObservation(
        machine="x86_64",
        logical_cpus=16,
        memory_total_bytes=46 * 1024**3,
        storage_free_bytes=273 * 1024**3,
        block_device_sizes_bytes=(2 * 1024**4,),
        debian_version="13",
        debian_version_full="13.6",
        qemu_vendor=False,
        q35_machine=False,
        qemu64_cpu=False,
        nvidia_devices=1,
        nvidia=NvidiaIdentity("NVIDIA GeForce RTX 3070", "550.163.01"),
        governor_cpus=16,
        performance_governors=16,
        no_turbo=1,
        minimum_scaling_max_khz=2_900_000,
        maximum_scaling_max_khz=2_900_000,
    )


class _Monitor:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> PeakRssMeasurement:
        assert self.started
        self.stopped = True
        return PeakRssMeasurement(77, RSS_METRIC, 4096, 2, (77, 78), 5, 0.01)


def test_all_three_frozen_fixture_identities_are_exact_and_unqualified() -> None:
    identity = capacity_identity()

    assert identity["fixture_freeze_sha256"] == F100_C0_FREEZE_SHA256
    assert identity["frozen_tiers"] == {"observed": 3, "required": 3}
    assert [fixture["tier"] for fixture in identity["tiers"]] == ["h0", "h1", "h2"]
    assert frozen_fixture("h0")["logical_cpus"] == 2
    assert frozen_fixture("h1")["logical_cpus"] == 4
    assert frozen_fixture("h2")["loadavg_start_end_required"] is True
    assert identity["release_qualified"] is False


@pytest.mark.parametrize("tier", list(CapacityTier))
def test_each_matching_runtime_attestation_passes_every_check(tier: CapacityTier) -> None:
    attestation = attest_runtime(tier, _observation(tier))

    assert attestation.passed
    assert attestation.wire()["checks"] == {
        "passed": len(attestation.checks),
        "total": len(attestation.checks),
    }


def test_h2_attestation_fails_if_one_cpu_is_not_performance_governed() -> None:
    original = _observation(CapacityTier.H2)
    changed = replace(original, performance_governors=15)

    attestation = attest_runtime(CapacityTier.H2, changed)
    failed = [check.check_id for check in attestation.checks if not check.passed]
    assert failed == ["governor-performance"]


def test_nvidia_identity_uses_bounded_driver_records_without_a_process(tmp_path: Path) -> None:
    # The product's only subprocess authority stays in video.py.  Capacity
    # attestation consumes the already loaded driver's bounded proc records.
    root = tmp_path
    gpu = root / "gpus" / "0000:01:00.0"
    gpu.mkdir(parents=True)
    (gpu / "information").write_text(
        "Model: \t\t NVIDIA GeForce RTX 3070\nBus Type: \t PCIe\n",
        encoding="ascii",
    )
    version = root / "version"
    version.write_text(
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.163.01  build\n",
        encoding="ascii",
    )

    assert capacity_module._nvidia_identity(root / "gpus", version) == NvidiaIdentity(
        "NVIDIA GeForce RTX 3070", "550.163.01"
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"1.0 2.0 3.0 1/2 3\nextra\n",
        b"nan 2.0 3.0 1/2 3\n",
        b"1.0 2.0 3.0 3/2 3\n",
        b"1.0 2.0 3.0 1/0 3\n",
        b"1.0 2.0 3.0 1/2 0\n",
        b"1.0  2.0 3.0 1/2 3\n",
    ],
)
def test_loadavg_parser_rejects_missing_partial_or_inconsistent_lines(payload: bytes) -> None:
    with pytest.raises(ValueError):
        parse_load_average(payload)


def test_h2_window_publishes_the_exact_start_and_end_loadavg_pair() -> None:
    readings: Iterator[bytes] = iter([b"1.70 1.73 1.75 2/100 400\n", b"1.82 1.76 1.74 3/101 401\n"])
    ticks: Iterator[int] = iter([10_000, 25_000])
    monitor = _Monitor()

    with CapacityMeasurementWindow(
        CapacityTier.H2,
        storage_paths=(),
        observation=_observation(CapacityTier.H2),
        loadavg_reader=lambda: parse_load_average(next(readings)),
        clock=lambda: next(ticks),
        monitor_factory=lambda: monitor,
    ) as window:
        pass

    record = window.measurement.wire()
    assert record["loadavg"]["published_samples"] == {"observed": 2, "required": 2}
    assert record["loadavg"]["start"]["raw"] == "1.70 1.73 1.75 2/100 400"
    assert record["loadavg"]["end"]["raw"] == "1.82 1.76 1.74 3/101 401"
    assert record["duration_ns"] == 15_000
    assert record["peak_rss"]["peak_pids"] == [77, 78]
    assert monitor.stopped


def test_mismatched_fixture_refuses_before_reading_loadavg_or_starting_monitor() -> None:
    mismatch = _observation(CapacityTier.H2)
    mismatch = replace(mismatch, logical_cpus=12)
    monitor = _Monitor()
    loadavg_reads = 0

    def loadavg() -> object:
        nonlocal loadavg_reads
        loadavg_reads += 1
        raise AssertionError("loadavg must not be read on the wrong fixture")

    window = CapacityMeasurementWindow(
        CapacityTier.H2,
        storage_paths=(),
        observation=mismatch,
        loadavg_reader=loadavg,  # type: ignore[arg-type]
        monitor_factory=lambda: monitor,
    )

    with pytest.raises(RuntimeError, match="logical-cpus"):
        window.start()
    assert loadavg_reads == 0
    assert not monitor.started


def test_missing_end_loadavg_still_stops_rss_and_publishes_no_partial_record() -> None:
    monitor = _Monitor()
    reads = 0

    def loadavg():  # type: ignore[no-untyped-def]
        nonlocal reads
        reads += 1
        if reads == 1:
            return parse_load_average(b"1.70 1.73 1.75 2/100 400\n")
        raise OSError("injected end read failure")

    window = CapacityMeasurementWindow(
        CapacityTier.H2,
        storage_paths=(),
        observation=_observation(CapacityTier.H2),
        loadavg_reader=loadavg,
        clock=iter([1, 2]).__next__,
        monitor_factory=lambda: monitor,
    )
    window.start()

    with pytest.raises(OSError, match="injected"):
        window.stop()
    assert monitor.stopped
    with pytest.raises(RuntimeError, match="has not completed"):
        _ = window.measurement


def test_capacity_window_cannot_be_restarted() -> None:
    monitor = _Monitor()
    readings = iter(
        [
            parse_load_average(b"1.00 1.00 1.00 1/10 10\n"),
            parse_load_average(b"1.00 1.00 1.00 1/10 11\n"),
        ]
    )
    window = CapacityMeasurementWindow(
        CapacityTier.H0,
        storage_paths=(),
        observation=_observation(CapacityTier.H0),
        loadavg_reader=readings.__next__,
        clock=iter([1, 2]).__next__,
        monitor_factory=lambda: monitor,
    )
    window.start()
    window.stop()

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        window.start()


def test_measure_contract_cli_emits_a_path_free_unqualified_record(
    request_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = request_factory(tmp_path, output_kinds=["mask"])
    request_bytes = canonical_bytes(request)
    request_path = tmp_path / "request.json"
    request_path.write_bytes(request_bytes)
    monkeypatch.setattr(
        capacity_module,
        "observe_runtime",
        lambda _storage_paths: _observation(CapacityTier.H0),
    )
    monkeypatch.setattr(
        cli,
        "_run",
        lambda _raw, _allow_reference: JobOutcome(
            {"schema": "kilix.background-removal.result/v2"}, None, []
        ),
    )

    assert (
        cli.main(
            [
                "measure-contract",
                str(request_path),
                "--fixture-tier",
                "h0",
                "--reference-profile",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)

    assert report["schema"] == "kilix.background-removal.capacity-measurement/v1"
    assert report["window"]["loadavg"]["published_samples"] == {
        "observed": 2,
        "required": 2,
    }
    assert report["workload"]["request_sha256"] == hashlib.sha256(request_bytes).hexdigest()
    assert report["workload"]["measured_images"] == {"observed": 1, "required": 1}
    assert report["quality_metrics"] == {"observed": 0, "required": 4}
    assert report["release_acceptance_credit"] == {"observed": 0, "required": 1}
    assert report["release_qualified"] is False
    encoded = json.dumps(report)
    assert str(tmp_path) not in encoded


def test_measure_contract_cli_still_publishes_h2_load_pair_on_workload_failure(
    request_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical_bytes(request_factory(tmp_path, output_kinds=["mask"])))
    monkeypatch.setattr(
        capacity_module,
        "observe_runtime",
        lambda _storage_paths: _observation(CapacityTier.H2),
    )

    def fail_workload(_raw: object, _allow_reference: bool) -> JobOutcome:
        raise RuntimeError("injected provider failure")

    monkeypatch.setattr(cli, "_run", fail_workload)

    assert (
        cli.main(
            [
                "measure-contract",
                str(request_path),
                "--fixture-tier",
                "h2",
                "--reference-profile",
            ]
        )
        == 3
    )
    report = json.loads(capsys.readouterr().out)

    assert report["window"]["loadavg"]["published_samples"] == {
        "observed": 2,
        "required": 2,
    }
    assert report["outcome"] == {
        "completed": {"observed": 0, "required": 1},
        "kind": "internal-error",
        "terminal_schema": None,
    }
