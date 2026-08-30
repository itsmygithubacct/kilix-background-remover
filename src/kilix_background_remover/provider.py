"""One local provider boundary for every F108 product surface.

The CLI, TUI, contained app, provider port, image batch path, offline-video
path and the in-repository editable-mask consumer all enter through
``BackgroundRemovalProvider``.  The provider owns at most one persistent
``WorkerSupervisor`` and lends that same supervisor to the per-frame video
adapter; no front end creates an inference implementation of its own.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from .contract_v2 import load_candidate_lock
from .contracts import parse_request
from .decode import DEFAULT_DECODE_BUDGET, MAX_DECODE_STATUS_BYTES
from .errors import RemovalFailure
from .frontend import MAX_DECODED_PIXELS, MAX_INPUT_BYTES, MAX_OUTPUT_BYTES
from .jobs import BatchEntry, BatchItemOutcome, BatchRunner
from .video import (
    DEFAULT_VIDEO_LIMITS,
    ReferenceFrameMasker,
    VideoEstimate,
    VideoLimits,
    VideoOutputKind,
    VideoProbe,
    VideoRequest,
    VideoResult,
    estimate_video,
    run_video,
)
from .worker import FALLBACK_REQUEST_ID, JobOutcome, WorkerSupervisor, failure_wire

VideoProgress = Callable[[str, int, int], None]
MAX_SURFACE_JSON_BYTES = 2 * 1024 * 1024


def decode_surface_json(payload: bytes) -> object:
    """Decode one bounded local UI document with a closed JSON domain."""

    if not payload or len(payload) > MAX_SURFACE_JSON_BYTES:
        raise ValueError("surface JSON is outside its fixed byte bound")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("surface JSON contains a duplicate field")
            result[key] = value
        return result

    def no_extensions(_value: str) -> None:
        raise ValueError("surface JSON contains a non-finite number")

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=no_extensions,
    )


def load_video_request(path: Path) -> VideoRequest:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SURFACE_JSON_BYTES:
        raise ValueError("video request must be a bounded regular file")
    return parse_video_request(decode_surface_json(path.read_bytes()))


def profile_failure_wire(request_id: str) -> dict[str, object]:
    return failure_wire(
        request_id,
        RemovalFailure(
            "background.profile-unavailable",
            "No release-qualified model profile is installed.",
            "provider",
            "resolve-profile",
        ),
    )


def _runtime_root() -> Path:
    base = Path(os.environ.get("XDG_RUNTIME_DIR", "/var/tmp"))
    root = base / f"kilix-background-remover-{os.getuid()}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = root.lstat()
    if root.is_symlink() or not root.is_dir() or status.st_uid != os.getuid():
        raise RuntimeError("the provider runtime directory is unsafe")
    os.chmod(root, 0o700)
    return root


def provider_identity() -> dict[str, object]:
    """Return immutable installed-provider and resource-limit identity."""

    candidate = load_candidate_lock()
    budget = DEFAULT_DECODE_BUDGET
    return {
        "schema": "kilix.background-removal.provider-identity/v1",
        "distribution": "kilix-background-remover",
        "version": version("kilix-background-remover"),
        "provider_api": 1,
        "release_qualified": False,
        "candidate_manifest_sha256": candidate.manifest_sha256,
        "surfaces": ["image", "batch", "video", "editable-mask"],
        "video_output_kinds": [kind.value for kind in VideoOutputKind],
        "decode": {
            "isolation": "spawned-resource-limited-process",
            "wall_seconds": budget.wall_seconds,
            "cpu_seconds": budget.cpu_seconds,
            "address_space_bytes": budget.address_space_bytes,
            "max_status_bytes": MAX_DECODE_STATUS_BYTES,
            "max_input_bytes": MAX_INPUT_BYTES,
            "max_decoded_pixels": MAX_DECODED_PIXELS,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            "child_to_parent_pixels": "raw-rgba-mode-0600",
            "child_to_parent_pickle": False,
        },
        "video": {
            "ffmpeg": "/usr/bin/ffmpeg",
            "ffprobe": "/usr/bin/ffprobe",
            "temporal_smoothing": "centered-integer-mean-scene-isolated",
            "raw_frame_mode": True,
            "atomic_publication": "verified-no-replace",
        },
    }


def _path(value: object, field: str, *, optional: bool = False) -> Path | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a nonempty typed path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute")
    return path


def _integer(value: object, field: str, low: int, high: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{field} is outside its closed integer bound")
    return value


def parse_video_request(value: object) -> VideoRequest:
    """Parse the command-free, fixed-field video surface request."""

    if not isinstance(value, Mapping):
        raise ValueError("video request must be an object")
    required = {
        "background_image",
        "background_video",
        "batch_frames",
        "confirmation_sha256",
        "destination",
        "gif_alpha_threshold_u8",
        "no_audio",
        "output_kind",
        "raw_frames",
        "scene_cut_threshold_u8",
        "schema",
        "smoothing_radius_frames",
        "source",
        "state_dir",
    }
    if set(value) != required:
        raise ValueError("video request has missing or unknown fields")
    if value.get("schema") != "kilix.background-removal.video-request/v1":
        raise ValueError("video request schema is unsupported")
    try:
        kind = VideoOutputKind(cast(str, value["output_kind"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("video output kind is unsupported") from exc
    confirmation = value["confirmation_sha256"]
    if confirmation is not None and (
        not isinstance(confirmation, str)
        or len(confirmation) != 64
        or any(character not in "0123456789abcdef" for character in confirmation)
    ):
        raise ValueError("video confirmation digest is invalid")
    no_audio = value["no_audio"]
    raw_frames = value["raw_frames"]
    if not isinstance(no_audio, bool) or not isinstance(raw_frames, bool):
        raise ValueError("video boolean settings are invalid")
    source = _path(value["source"], "source")
    destination = _path(value["destination"], "destination")
    assert source is not None and destination is not None
    return VideoRequest(
        source=source,
        destination=destination,
        output_kind=kind,
        confirmation_sha256=confirmation,
        no_audio=no_audio,
        raw_frames=raw_frames,
        smoothing_radius_frames=_integer(
            value["smoothing_radius_frames"], "smoothing_radius_frames", 0, 32
        ),
        batch_frames=_integer(value["batch_frames"], "batch_frames", 1, 1_000_000),
        scene_cut_threshold_u8=_integer(
            value["scene_cut_threshold_u8"], "scene_cut_threshold_u8", 0, 255
        ),
        gif_alpha_threshold_u8=_integer(
            value["gif_alpha_threshold_u8"], "gif_alpha_threshold_u8", 0, 255
        ),
        background_image=_path(value["background_image"], "background_image", optional=True),
        background_video=_path(value["background_video"], "background_video", optional=True),
        state_dir=_path(value["state_dir"], "state_dir", optional=True),
    )


def video_request_wire(request: VideoRequest) -> dict[str, object]:
    return {
        "schema": "kilix.background-removal.video-request/v1",
        "source": str(request.source.absolute()),
        "destination": str(request.destination.absolute()),
        "output_kind": request.output_kind.value,
        "confirmation_sha256": request.confirmation_sha256,
        "no_audio": request.no_audio,
        "raw_frames": request.raw_frames,
        "smoothing_radius_frames": request.smoothing_radius_frames,
        "batch_frames": request.batch_frames,
        "scene_cut_threshold_u8": request.scene_cut_threshold_u8,
        "gif_alpha_threshold_u8": request.gif_alpha_threshold_u8,
        "background_image": (
            str(request.background_image.absolute()) if request.background_image else None
        ),
        "background_video": (
            str(request.background_video.absolute()) if request.background_video else None
        ),
        "state_dir": str(request.state_dir.absolute()) if request.state_dir else None,
    }


def video_estimate_wire(estimate: VideoEstimate) -> dict[str, object]:
    return {
        "schema": "kilix.background-removal.video-estimate/v1",
        **estimate.wire(),
    }


def video_result_wire(result: VideoResult) -> dict[str, object]:
    record = asdict(result)
    record["destination"] = str(result.destination)
    record["kind"] = result.kind.value
    record["scene_cut_frames"] = list(result.scene_cut_frames)
    return {
        "schema": "kilix.background-removal.video-result/v1",
        **cast(dict[str, object], record),
    }


class BackgroundRemovalProvider:
    """Own one supervised worker and expose all admitted local operations."""

    def __init__(
        self,
        *,
        allow_reference_profile: bool = False,
        supervisor: WorkerSupervisor | None = None,
        video_limits: VideoLimits = DEFAULT_VIDEO_LIMITS,
    ) -> None:
        if supervisor is not None and not allow_reference_profile:
            raise ValueError("a reference supervisor requires explicit reference-profile authority")
        self._allow_reference_profile = allow_reference_profile
        self._supervisor = supervisor
        self._owns_supervisor = supervisor is None
        self._video_limits = video_limits
        self._lock = threading.Lock()
        self._closed = False
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        if allow_reference_profile and self._supervisor is not None:
            self._temporary = tempfile.TemporaryDirectory(prefix="provider-", dir=_runtime_root())
            os.chmod(self._temporary.name, 0o700)

    @property
    def supervisor_pid(self) -> int | None:
        return None if self._supervisor is None else self._supervisor.pid

    @property
    def identity(self) -> dict[str, object]:
        return provider_identity()

    def _active_supervisor(self) -> WorkerSupervisor:
        if self._closed:
            raise RuntimeError("provider is closed")
        if not self._allow_reference_profile or self._supervisor is None:
            if self._allow_reference_profile and self._supervisor is None:
                self._supervisor = WorkerSupervisor()
                self._temporary = tempfile.TemporaryDirectory(
                    prefix="provider-", dir=_runtime_root()
                )
                os.chmod(self._temporary.name, 0o700)
                return self._supervisor
            raise RemovalFailure(
                "background.profile-unavailable",
                "No release-qualified model profile is installed.",
                "provider",
                "resolve-profile",
            )
        return self._supervisor

    def run(
        self,
        request: object,
        *,
        cancel: threading.Event | None = None,
        on_progress: Callable[[dict[str, object]], None] | None = None,
    ) -> JobOutcome:
        parsed = parse_request(request)
        if not self._allow_reference_profile:
            return JobOutcome(None, profile_failure_wire(parsed.request_id), [])
        with self._lock:
            return self._active_supervisor().run(
                request,
                cancel=cancel,
                on_progress=on_progress,
            )

    def run_batch(
        self,
        entries: Sequence[BatchEntry],
        *,
        state_dir: Path,
        cancel: threading.Event | None = None,
    ) -> list[BatchItemOutcome]:
        with self._lock:
            return BatchRunner(self._active_supervisor()).run(
                list(entries),
                state_dir=state_dir,
                cancel=cancel,
            )

    def estimate_video(
        self,
        request: VideoRequest,
        *,
        cancel: threading.Event | None = None,
    ) -> tuple[VideoProbe, VideoEstimate]:
        if self._closed:
            raise RuntimeError("provider is closed")
        return estimate_video(request, limits=self._video_limits, cancel=cancel)

    def run_video(
        self,
        request: VideoRequest,
        *,
        cancel: threading.Event | None = None,
        progress: VideoProgress | None = None,
    ) -> VideoResult:
        with self._lock:
            supervisor = self._active_supervisor()
            if self._temporary is None:
                raise RuntimeError("provider video workspace is unavailable")
            masker = ReferenceFrameMasker(
                Path(self._temporary.name),
                supervisor=supervisor,
            )
            try:
                return run_video(
                    request,
                    masker,
                    limits=self._video_limits,
                    cancel=cancel,
                    progress=progress,
                )
            finally:
                masker.close()

    def cancel(self, request_bytes: bytes) -> bytes:
        return self._active_supervisor().cancel(request_bytes)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_supervisor and self._supervisor is not None:
            self._supervisor.close()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> BackgroundRemovalProvider:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def profile_failure_outcome() -> JobOutcome:
    """Return the common failure for callers without a parsed image request."""

    return JobOutcome(None, profile_failure_wire(FALLBACK_REQUEST_ID), [])
