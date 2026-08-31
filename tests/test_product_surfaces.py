"""Causal controls for the executable multi-surface product boundary."""

from __future__ import annotations

import io
import tomllib
from pathlib import Path
from typing import Any, cast

from kilix_background_remover import provider as provider_module
from kilix_background_remover.app import ContainedAppController
from kilix_background_remover.app_bridge import (
    BRIDGE_REQUEST_SCHEMA_V2,
    BRIDGE_RESPONSE_SCHEMA_V2,
    run_bridge_message,
)
from kilix_background_remover.provider import (
    BackgroundRemovalProvider,
    parse_video_request,
    provider_identity,
    video_request_wire,
)
from kilix_background_remover.provider_port import ProviderPort
from kilix_background_remover.tui import render_provider_header, render_video_progress
from kilix_background_remover.video import (
    VideoEstimate,
    VideoOutputKind,
    VideoRequest,
    VideoResult,
)
from kilix_background_remover.worker import WorkerSupervisor


def _alive(pid: int | None) -> bool:
    return pid is not None and Path(f"/proc/{pid}").exists()


def test_provider_identity_exposes_the_actual_bounded_decode_and_video_policy() -> None:
    identity = provider_identity()
    assert identity["surfaces"] == ["image", "batch", "video", "editable-mask"]
    assert identity["video_output_kinds"] == [kind.value for kind in VideoOutputKind]
    decode = identity["decode"]
    assert decode == {
        "isolation": "spawned-resource-limited-process",
        "wall_seconds": 30.0,
        "cpu_seconds": 30,
        "address_space_bytes": 2 * 1024 * 1024 * 1024,
        "max_status_bytes": 4096,
        "max_input_bytes": 512 * 1024 * 1024,
        "max_decoded_pixels": 100_000_000,
        "max_output_bytes": 1024 * 1024 * 1024,
        "child_to_parent_pixels": "raw-rgba-mode-0600",
        "child_to_parent_pickle": False,
    }
    assert identity["inference"] == {
        "tile_order": "deterministic-row-major-2d",
        "max_working_tile_pixels": 1_048_576,
        "current_overlap_pixels": 0,
        "release_tiling_phases": None,
        "release_seam_policy": None,
        "release_rss_threshold_bytes": None,
        "release_qualified": False,
    }
    assert identity["rss_measurement"] == {
        "available": True,
        "metric": "sampled-aggregate-process-tree-vmrss-bytes",
        "release_scope": None,
        "release_threshold_bytes": None,
        "release_qualified": False,
    }


def test_wheel_declares_all_five_product_executables() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {
        "kilix-background-remover": "kilix_background_remover.cli:main",
        "kilix-background-remover-tui": "kilix_background_remover.tui:main",
        "kilix-background-remover-app": "kilix_background_remover.app:main",
        "kilix-background-remover-app-bridge": "kilix_background_remover.app_bridge:main",
        "kilix-background-remover-provider": "kilix_background_remover.provider_port:main",
    }


def test_one_provider_lends_the_exact_same_supervisor_to_video(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class FakeSupervisor:
        pid = 8142

        def close(self) -> None:
            raise AssertionError("the provider must not close a caller-owned supervisor")

    fake = FakeSupervisor()
    seen: dict[str, object] = {}

    class CapturingMasker:
        def __init__(self, workspace: Path, *, supervisor: object) -> None:
            assert workspace.is_dir()
            seen["supervisor"] = supervisor

        def close(self) -> None:
            seen["closed"] = True

    def fake_video(
        request: VideoRequest,
        masker: object,
        **_kwargs: object,
    ) -> VideoResult:
        seen["masker"] = masker
        return VideoResult(
            destination=request.destination,
            kind=request.output_kind,
            media_type="video/x-matroska",
            sha256="a" * 64,
            bytes=1,
            width=1,
            height=1,
            frame_count=1,
            duration_seconds="1",
            audio_preserved=False,
            raw_frames=False,
            smoothing_radius_frames=1,
            batch_frames=24,
            scene_cut_frames=(0,),
            gif_alpha_threshold_u8=None,
        )

    monkeypatch.setattr(provider_module, "ReferenceFrameMasker", CapturingMasker)
    monkeypatch.setattr(provider_module, "run_video", fake_video)
    request = VideoRequest(
        source=(tmp_path / "source.mkv").absolute(),
        destination=(tmp_path / "result.mkv").absolute(),
        output_kind=VideoOutputKind.MATTE,
    )
    with BackgroundRemovalProvider(
        allow_reference_profile=True,
        supervisor=cast(WorkerSupervisor, fake),
    ) as provider:
        assert provider.supervisor_pid == 8142
        provider.run_video(request)
    assert seen["supervisor"] is fake
    assert seen["closed"] is True


def test_contained_app_uses_one_lazy_worker_and_tears_it_down(
    request_factory: Any, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    request = request_factory(output)
    message = {
        "schema": BRIDGE_REQUEST_SCHEMA_V2,
        "operation": "run-image",
        "request": request,
    }
    controller = ContainedAppController(allow_reference_profile=True)
    assert controller.provider_pid is None
    response = controller.dispatch(message)
    pid = controller.provider_pid
    assert response["schema"] == BRIDGE_RESPONSE_SCHEMA_V2
    assert response["error"] is None
    assert _alive(pid)
    identity = controller.dispatch(
        {
            "schema": BRIDGE_REQUEST_SCHEMA_V2,
            "operation": "discover",
            "request": None,
        }
    )
    assert identity["result"]["surfaces"] == ["image", "batch", "video", "editable-mask"]
    assert controller.provider_pid == pid
    controller.close()
    assert not _alive(pid)


def test_video_surface_request_is_closed_typed_and_round_trips(tmp_path: Path) -> None:
    request = VideoRequest(
        source=(tmp_path / "input.mkv").absolute(),
        destination=(tmp_path / "output.webm").absolute(),
        output_kind=VideoOutputKind.TRANSPARENT_WEBM,
        confirmation_sha256="b" * 64,
        no_audio=True,
        raw_frames=True,
        smoothing_radius_frames=7,
        batch_frames=3,
        scene_cut_threshold_u8=42,
        gif_alpha_threshold_u8=96,
        state_dir=(tmp_path / "state").absolute(),
    )
    wire = video_request_wire(request)
    assert parse_video_request(wire) == request
    for forbidden in ("command", "environment", "model_url", "python_import"):
        with_forbidden = {**wire, forbidden: "forbidden"}
        try:
            parse_video_request(with_forbidden)
        except ValueError as exc:
            assert "unknown" in str(exc)
        else:
            raise AssertionError(f"video request accepted forbidden field {forbidden}")


def test_app_bridge_estimates_and_runs_video_through_its_supplied_provider(
    tmp_path: Path,
) -> None:
    estimate = VideoEstimate(
        source_sha256="a" * 64,
        source_bytes=10,
        width=2,
        height=2,
        frame_count=1,
        duration_seconds="1",
        estimated_wall_seconds="15",
        estimated_temp_bytes=100,
        output_kind="matte",
        source_audio=False,
        preserve_audio=False,
        raw_frames=False,
        smoothing_radius_frames=1,
        batch_frames=24,
        scene_cut_threshold_u8=48,
        gif_alpha_threshold_u8=None,
        gif_hard_edge_disclosure=False,
        background_kind="none",
        background_sha256=None,
        background_bytes=0,
        confirmation_sha256="b" * 64,
    )
    result = VideoResult(
        destination=(tmp_path / "output.mkv").absolute(),
        kind=VideoOutputKind.MATTE,
        media_type="video/x-matroska",
        sha256="c" * 64,
        bytes=20,
        width=2,
        height=2,
        frame_count=1,
        duration_seconds="1",
        audio_preserved=False,
        raw_frames=False,
        smoothing_radius_frames=1,
        batch_frames=24,
        scene_cut_frames=(0,),
        gif_alpha_threshold_u8=None,
    )

    class FakeProvider:
        def estimate_video(
            self, request: VideoRequest, **_kwargs: object
        ) -> tuple[None, VideoEstimate]:
            assert request.output_kind is VideoOutputKind.MATTE
            return None, estimate

        def run_video(self, request: VideoRequest, **kwargs: object) -> VideoResult:
            progress = kwargs["progress"]
            progress("temporal-smooth", 1, 1)
            assert request.confirmation_sha256 == "b" * 64
            return result

    request = VideoRequest(
        source=(tmp_path / "input.mkv").absolute(),
        destination=result.destination,
        output_kind=VideoOutputKind.MATTE,
    )
    estimate_response = run_bridge_message(
        {
            "schema": BRIDGE_REQUEST_SCHEMA_V2,
            "operation": "estimate-video",
            "request": video_request_wire(request),
        },
        allow_reference_profile=False,
        provider=cast(BackgroundRemovalProvider, FakeProvider()),
    )
    assert estimate_response["result"]["confirmation_sha256"] == "b" * 64
    confirmed = VideoRequest(
        source=request.source,
        destination=request.destination,
        output_kind=request.output_kind,
        confirmation_sha256="b" * 64,
    )
    run_response = run_bridge_message(
        {
            "schema": BRIDGE_REQUEST_SCHEMA_V2,
            "operation": "run-video",
            "request": video_request_wire(confirmed),
        },
        allow_reference_profile=False,
        provider=cast(BackgroundRemovalProvider, FakeProvider()),
    )
    assert run_response["progress"] == [
        {
            "schema": "kilix.background-removal.video-progress/v1",
            "phase": "temporal-smooth",
            "frames_completed": 1,
            "frames_total": 1,
        }
    ]
    assert run_response["result"]["sha256"] == "c" * 64


def test_cli_tui_and_app_bridge_have_no_second_video_execution_path() -> None:
    package = Path(provider_module.__file__).parent
    calls = {
        "cli.py": "provider.run_video(",
        "tui.py": "provider.run_video(",
        "app_bridge.py": "active.run_video(",
    }
    for name, call in calls.items():
        source = (package / name).read_text(encoding="utf-8")
        assert call in source
        assert "ReferenceFrameMasker(" not in source


def test_tui_has_bounded_narrow_video_and_provider_views() -> None:
    assert render_video_progress("temporal-smooth", 3, 4, 80) == (
        "video      75.0%  temporal-smooth  frames 3/4"
    )
    assert len(render_video_progress("temporal-smooth", 3, 4, 9)) == 9
    lines = render_provider_header(provider_identity(), 48)
    assert len(lines) == 4
    assert all(len(line) <= 48 for line in lines)
    assert "q/Esc cancel" in lines[3]


def test_f115_provider_port_discovers_exact_installed_surface_and_closes() -> None:
    class FakeProvider:
        @property
        def identity(self) -> dict[str, object]:
            return {
                "schema": "kilix.background-removal.provider-identity/v1",
                "surfaces": ["image", "batch", "video", "editable-mask"],
                "video_output_kinds": [kind.value for kind in VideoOutputKind],
            }

        def close(self) -> None:
            pass

    output = io.BytesIO()
    source = io.BytesIO(b"DISCOVER 0\nCLOSE 0\n")
    port = ProviderPort(cast(BackgroundRemovalProvider, FakeProvider()), output)
    assert port.serve(source) == 0
    payload = output.getvalue()
    assert payload.startswith(b"IDENTITY ")
    assert b"kilix.background-removal.provider-port/v1" in payload
    assert b'"operations":["discover","submit","cancel","close"]' in payload
    assert payload.endswith(b"CLOSED 0\n")
