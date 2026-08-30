"""Bounded offline-video removal through one fixed FFmpeg adapter.

The video protocol is product-local preparation: it intentionally does not
define or consume the pending shared image-job contract.  It provides all six
owner-required render profiles, an estimate/confirmation join, bounded process
I/O, temporal smoothing with batch overlap, audio policy, and the same atomic
final commit used by image output.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from PIL import Image

from .atomic import allocate_staging_path, commit_staged, finalize_staged_file
from .errors import RemovalFailure

FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")
STILL_IMAGE_CODECS = frozenset({"mjpeg", "png", "tiff", "webp"})
SMOOTH_CHUNK_BYTES = 1024 * 1024
_PROCESS_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "AV_LOG_FORCE_NOCOLOR": "1",
}


class VideoOutputKind(StrEnum):
    TRANSPARENT_MOV = "transparent-mov"
    TRANSPARENT_WEBM = "transparent-webm"
    MATTE = "matte"
    COMPOSITE_IMAGE = "composite-image"
    COMPOSITE_VIDEO = "composite-video"
    GIF = "gif"


@dataclass(frozen=True, slots=True)
class VideoLimits:
    max_input_bytes: int = 64 * 1024 * 1024 * 1024
    max_output_bytes: int = 64 * 1024 * 1024 * 1024
    max_temp_bytes: int = 256 * 1024 * 1024 * 1024
    max_frames: int = 1_000_000
    max_dimension: int = 16_384
    max_duration_seconds: Decimal = Decimal(365 * 24 * 60 * 60)
    max_probe_bytes: int = 64 * 1024 * 1024
    max_log_bytes: int = 64 * 1024
    probe_timeout_seconds: float = 60.0
    job_timeout_seconds: float = 7 * 24 * 60 * 60
    measured_frame_seconds: Decimal = Decimal("15")


DEFAULT_VIDEO_LIMITS = VideoLimits()


@dataclass(frozen=True, slots=True)
class VideoIdentity:
    path: Path
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VideoProbe:
    identity: VideoIdentity
    width: int
    height: int
    frame_timestamps: tuple[Decimal, ...]
    frame_durations: tuple[Decimal, ...]
    duration_seconds: Decimal
    rotation_degrees: int
    variable_frame_rate: bool
    video_codec: str
    pixel_format: str
    format_names: frozenset[str]
    has_alpha: bool
    audio_codec: str | None

    @property
    def frame_count(self) -> int:
        return len(self.frame_timestamps)


@dataclass(frozen=True, slots=True)
class VideoEstimate:
    source_sha256: str
    source_bytes: int
    width: int
    height: int
    frame_count: int
    duration_seconds: str
    estimated_wall_seconds: str
    estimated_temp_bytes: int
    output_kind: str
    source_audio: bool
    preserve_audio: bool
    raw_frames: bool
    smoothing_radius_frames: int
    batch_frames: int
    scene_cut_threshold_u8: int
    gif_alpha_threshold_u8: int | None
    gif_hard_edge_disclosure: bool
    background_kind: str
    background_sha256: str | None
    background_bytes: int
    confirmation_sha256: str

    def wire(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class VideoRequest:
    source: Path
    destination: Path
    output_kind: VideoOutputKind
    confirmation_sha256: str | None = None
    no_audio: bool = False
    raw_frames: bool = False
    smoothing_radius_frames: int = 1
    batch_frames: int = 24
    scene_cut_threshold_u8: int = 48
    gif_alpha_threshold_u8: int = 128
    background_image: Path | None = None
    background_video: Path | None = None
    state_dir: Path | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.smoothing_radius_frames <= 32:
            raise ValueError("smoothing radius must be within 0..32 frames")
        if self.batch_frames <= 0:
            raise ValueError("batch size must be positive")
        if not 0 <= self.scene_cut_threshold_u8 <= 255:
            raise ValueError("scene-cut threshold must be within 0..255")
        if not 0 <= self.gif_alpha_threshold_u8 <= 255:
            raise ValueError("GIF alpha threshold must be within 0..255")
        if self.output_kind is VideoOutputKind.COMPOSITE_IMAGE:
            if self.background_image is None or self.background_video is not None:
                raise ValueError("composite-image requires exactly one background image")
        elif self.output_kind is VideoOutputKind.COMPOSITE_VIDEO:
            if self.background_video is None or self.background_image is not None:
                raise ValueError("composite-video requires exactly one background video")
        elif self.background_image is not None or self.background_video is not None:
            raise ValueError("the selected output does not accept a background")


@dataclass(frozen=True, slots=True)
class VideoResult:
    destination: Path
    kind: VideoOutputKind
    media_type: str
    sha256: str
    bytes: int
    width: int
    height: int
    frame_count: int
    duration_seconds: str
    audio_preserved: bool
    raw_frames: bool
    smoothing_radius_frames: int
    batch_frames: int
    scene_cut_frames: tuple[int, ...]
    gif_alpha_threshold_u8: int | None


FrameMasker = Callable[[Image.Image, int, threading.Event | None], Image.Image]
VideoProgress = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class FFmpegCapabilities:
    encoders: frozenset[str]
    muxers: frozenset[str]

    def supports(self, kind: VideoOutputKind) -> bool:
        profile = _PROFILES[kind]
        return profile.video_encoder in self.encoders and profile.muxer in self.muxers


@dataclass(frozen=True, slots=True)
class _OutputProfile:
    video_encoder: str
    muxer: str
    media_type: str
    codec_name: str
    audio_encoder: str | None
    pixel_format: str


_PROFILES = {
    VideoOutputKind.TRANSPARENT_MOV: _OutputProfile(
        "prores_ks", "mov", "video/quicktime", "prores", "pcm_s16le", "yuva444p10le"
    ),
    VideoOutputKind.TRANSPARENT_WEBM: _OutputProfile(
        "libvpx-vp9", "webm", "video/webm", "vp9", "libopus", "yuva420p"
    ),
    VideoOutputKind.MATTE: _OutputProfile(
        "ffv1", "matroska", "video/x-matroska", "ffv1", "flac", "gray"
    ),
    VideoOutputKind.COMPOSITE_IMAGE: _OutputProfile(
        "ffv1", "matroska", "video/x-matroska", "ffv1", "flac", "rgb24"
    ),
    VideoOutputKind.COMPOSITE_VIDEO: _OutputProfile(
        "ffv1", "matroska", "video/x-matroska", "ffv1", "flac", "rgb24"
    ),
    VideoOutputKind.GIF: _OutputProfile("gif", "gif", "image/gif", "gif", None, "pal8"),
}


@dataclass(frozen=True, slots=True)
class _ProcessOutput:
    stdout: bytes
    stderr_tail: bytes


def probe_capabilities(
    *,
    limits: VideoLimits = DEFAULT_VIDEO_LIMITS,
    cancel: threading.Event | None = None,
) -> FFmpegCapabilities:
    deadline = time.monotonic() + limits.probe_timeout_seconds
    encoders = _run_capture(
        [str(FFMPEG), "-nostdin", "-hide_banner", "-encoders"],
        deadline=deadline,
        max_stdout=limits.max_probe_bytes,
        max_stderr=limits.max_log_bytes,
        cancel=cancel,
        phase="probe-capabilities",
    ).stdout.decode("utf-8", errors="replace")
    muxers = _run_capture(
        [str(FFMPEG), "-nostdin", "-hide_banner", "-muxers"],
        deadline=deadline,
        max_stdout=limits.max_probe_bytes,
        max_stderr=limits.max_log_bytes,
        cancel=cancel,
        phase="probe-capabilities",
    ).stdout.decode("utf-8", errors="replace")
    encoder_names = frozenset(
        match.group(1) for match in re.finditer(r"^\s*[A-Z.]{6}\s+(\S+)", encoders, re.MULTILINE)
    )
    muxer_names: set[str] = set()
    for line in muxers.splitlines():
        fields = line.split()
        if len(fields) >= 2 and "E" in fields[0]:
            muxer_names.update(fields[1].split(","))
    return FFmpegCapabilities(encoder_names, frozenset(muxer_names))


def probe_video(
    path: Path,
    *,
    limits: VideoLimits = DEFAULT_VIDEO_LIMITS,
    cancel: threading.Event | None = None,
) -> VideoProbe:
    identity = _video_identity(path, limits, cancel)
    deadline = time.monotonic() + limits.probe_timeout_seconds
    stream_document = _json_capture(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(identity.path),
        ],
        deadline,
        limits,
        cancel,
    )
    frame_document = _json_capture(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,pkt_duration_time,width,height",
            "-of",
            "json",
            str(identity.path),
        ],
        deadline,
        limits,
        cancel,
    )
    return _parse_probe(identity, stream_document, frame_document, limits)


def estimate_video(
    request: VideoRequest,
    *,
    limits: VideoLimits = DEFAULT_VIDEO_LIMITS,
    cancel: threading.Event | None = None,
) -> tuple[VideoProbe, VideoEstimate]:
    probe = probe_video(request.source, limits=limits, cancel=cancel)
    if request.output_kind is VideoOutputKind.GIF and probe.audio_codec and not request.no_audio:
        raise RemovalFailure(
            "background.invalid-request",
            "GIF cannot preserve source audio; select no-audio explicitly.",
            "input",
            "estimate",
        )
    preserve_audio = probe.audio_codec is not None and not request.no_audio
    rendered_planes = {
        VideoOutputKind.MATTE: 1,
        VideoOutputKind.COMPOSITE_IMAGE: 3,
        VideoOutputKind.COMPOSITE_VIDEO: 3,
        VideoOutputKind.TRANSPARENT_MOV: 4,
        VideoOutputKind.TRANSPARENT_WEBM: 4,
        VideoOutputKind.GIF: 4,
    }[request.output_kind]
    # Source RGBA + raw mask + final mask + the profile's rendered raster.
    # Keep a distinct encoded-carrier reservation because it coexists with all
    # private inputs at the atomic publication boundary.
    planes = 6 + rendered_planes
    if request.output_kind is VideoOutputKind.COMPOSITE_VIDEO:
        planes += 4
    raster_pixels = probe.width * probe.height * probe.frame_count
    encoded_reserve = min(
        limits.max_output_bytes,
        raster_pixels * 8 + probe.identity.bytes + 2 * 1024 * 1024,
    )
    estimated_temp = (
        probe.identity.bytes
        + raster_pixels * planes
        + encoded_reserve
        + raster_pixels // 64
        + probe.frame_count * 4096
        + 2 * 1024 * 1024
    )
    background_kind = "none"
    background_sha256: str | None = None
    background_bytes = 0
    if request.background_video is not None:
        background_probe = probe_video(request.background_video, limits=limits, cancel=cancel)
        _require_aligned_background(probe, background_probe)
        background_kind = "video"
        background_sha256 = background_probe.identity.sha256
        background_bytes = background_probe.identity.bytes
        estimated_temp += background_bytes
    if request.background_image is not None:
        background_identity = _probe_still_image(
            request.background_image,
            expected_geometry=(probe.width, probe.height),
            limits=limits,
            cancel=cancel,
        )
        background_kind = "image"
        background_sha256 = background_identity.sha256
        background_bytes = background_identity.bytes
        estimated_temp += background_bytes
    if estimated_temp > limits.max_temp_bytes:
        raise RemovalFailure(
            "background.input-limit",
            "The video temporary-space estimate exceeds its configured limit.",
            "resource",
            "estimate",
        )
    fields: dict[str, object] = {
        "source_sha256": probe.identity.sha256,
        "source_bytes": probe.identity.bytes,
        "width": probe.width,
        "height": probe.height,
        "frame_count": probe.frame_count,
        "duration_seconds": _decimal_text(probe.duration_seconds),
        "estimated_wall_seconds": _decimal_text(limits.measured_frame_seconds * probe.frame_count),
        "estimated_temp_bytes": estimated_temp,
        "output_kind": request.output_kind.value,
        "source_audio": probe.audio_codec is not None,
        "preserve_audio": preserve_audio,
        "raw_frames": request.raw_frames,
        "smoothing_radius_frames": 0 if request.raw_frames else request.smoothing_radius_frames,
        "batch_frames": request.batch_frames,
        "scene_cut_threshold_u8": request.scene_cut_threshold_u8,
        "gif_alpha_threshold_u8": (
            request.gif_alpha_threshold_u8 if request.output_kind is VideoOutputKind.GIF else None
        ),
        "gif_hard_edge_disclosure": request.output_kind is VideoOutputKind.GIF,
        "background_kind": background_kind,
        "background_sha256": background_sha256,
        "background_bytes": background_bytes,
    }
    confirmation = hashlib.sha256(_canonical_json(fields)).hexdigest()
    estimate = VideoEstimate(**fields, confirmation_sha256=confirmation)  # type: ignore[arg-type]
    return probe, estimate


def temporal_smooth_masks(
    masks: Sequence[bytes],
    width: int,
    height: int,
    *,
    radius: int,
    scene_cut_frames: Sequence[int],
    batch_frames: int,
) -> list[bytes]:
    """Centered integer temporal mean; batches commit interiors only."""

    if width <= 0 or height <= 0 or radius < 0 or batch_frames <= 0:
        raise ValueError("invalid temporal smoothing geometry")
    frame_bytes = width * height
    if any(len(mask) != frame_bytes for mask in masks):
        raise ValueError("mask geometry mismatch")
    cuts = sorted(set(scene_cut_frames) | {0, len(masks)})
    if (
        cuts[0] != 0
        or cuts[-1] != len(masks)
        or any(left >= right for left, right in pairwise(cuts))
    ):
        raise ValueError("invalid scene-cut frames")
    segments = [0] * len(masks)
    for segment, (start, end) in enumerate(pairwise(cuts)):
        for index in range(start, end):
            segments[index] = segment
    output: list[bytes | None] = [None] * len(masks)
    for interior_start in range(0, len(masks), batch_frames):
        interior_end = min(len(masks), interior_start + batch_frames)
        overlap_start = max(0, interior_start - radius)
        overlap_end = min(len(masks), interior_end + radius)
        available = masks[overlap_start:overlap_end]
        for index in range(interior_start, interior_end):
            first = max(overlap_start, index - radius)
            last = min(overlap_end, index + radius + 1)
            neighbours = [
                available[candidate - overlap_start]
                for candidate in range(first, last)
                if segments[candidate] == segments[index]
            ]
            divisor = len(neighbours)
            rounding = divisor // 2
            output[index] = bytes(
                (sum(neighbour[pixel] for neighbour in neighbours) + rounding) // divisor
                for pixel in range(frame_bytes)
            )
    if any(mask is None for mask in output):
        raise RuntimeError("a smoothing batch did not commit its interior")
    return [cast(bytes, mask) for mask in output]


def run_video(
    request: VideoRequest,
    masker: FrameMasker,
    *,
    limits: VideoLimits = DEFAULT_VIDEO_LIMITS,
    cancel: threading.Event | None = None,
    progress: VideoProgress | None = None,
) -> VideoResult:
    """Execute one confirmed offline job and atomically publish its output."""

    initial_probe, estimate = estimate_video(request, limits=limits, cancel=cancel)
    if request.confirmation_sha256 != estimate.confirmation_sha256:
        raise RemovalFailure(
            "background.invalid-request",
            "The exact video time/frame/disk estimate has not been confirmed.",
            "input",
            "confirm-estimate",
        )
    capabilities = probe_capabilities(limits=limits, cancel=cancel)
    profile = _PROFILES[request.output_kind]
    required_encoders = {profile.video_encoder}
    if estimate.preserve_audio and profile.audio_encoder is not None:
        required_encoders.add(profile.audio_encoder)
    if not required_encoders <= capabilities.encoders or profile.muxer not in capabilities.muxers:
        raise RemovalFailure(
            "background.backend-unavailable",
            "The qualified FFmpeg codec/container profile is unavailable.",
            "provider",
            "probe-capabilities",
        )
    deadline = time.monotonic() + limits.job_timeout_seconds
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if request.state_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="kilix-f108-video-")
        workspace = Path(temporary.name)
    else:
        workspace = _persistent_workspace(request.state_dir, estimate.confirmation_sha256)
    try:
        _initialize_workspace(workspace, request, estimate)
        free_bytes = shutil.disk_usage(workspace).free
        if estimate.estimated_temp_bytes > min(limits.max_temp_bytes, free_bytes):
            raise RemovalFailure(
                "background.input-limit",
                "The confirmed video job does not fit in available temporary space.",
                "resource",
                "confirm-estimate",
            )
        source_copy = workspace / "source.media"
        _copy_verified(initial_probe.identity, source_copy, cancel)
        source_probe = probe_video(source_copy, limits=limits, cancel=cancel)
        _require_same_probe(initial_probe, source_probe)

        background_image: bytes | None = None
        background_frames: Path | None = None
        if request.background_image is not None:
            background_identity = _probe_still_image(
                request.background_image,
                expected_geometry=(source_probe.width, source_probe.height),
                limits=limits,
                cancel=cancel,
            )
            copied = workspace / "background-image.media"
            _copy_verified(background_identity, copied, cancel)
            copied_identity = _probe_still_image(
                copied,
                expected_geometry=(source_probe.width, source_probe.height),
                limits=limits,
                cancel=cancel,
            )
            if (
                copied_identity.bytes != background_identity.bytes
                or copied_identity.sha256 != background_identity.sha256
            ):
                raise RemovalFailure(
                    "background.input-unreadable",
                    "The background image changed after confirmation.",
                    "input",
                    "confirm-estimate",
                )
            background_image = _decode_still_rgba(
                copied,
                source_probe.width,
                source_probe.height,
                limits,
                deadline,
                cancel,
            )
        elif request.background_video is not None:
            background_identity = _video_identity(request.background_video, limits, cancel)
            copied = workspace / "background-video.media"
            _copy_verified(background_identity, copied, cancel)
            background_probe = probe_video(copied, limits=limits, cancel=cancel)
            _require_aligned_background(source_probe, background_probe)
            background_frames = workspace / "background-video"
            _ensure_private_directory(background_frames)
            _decode_raw_frames(
                copied,
                background_probe,
                background_frames,
                limits,
                deadline,
                cancel,
                None,
                progress=None,
            )

        frames = workspace / "frames"
        _ensure_private_directory(frames)
        cuts = _decode_raw_frames(
            source_copy,
            source_probe,
            frames,
            limits,
            deadline,
            cancel,
            request.scene_cut_threshold_u8,
            progress=progress,
        )
        masks = workspace / "masks"
        _ensure_private_directory(masks)
        _produce_masks(
            frames,
            masks,
            source_probe,
            request,
            masker,
            cancel,
            progress,
        )
        final_masks = workspace / "final-masks"
        _ensure_private_directory(final_masks)
        _smooth_mask_files(
            masks,
            final_masks,
            source_probe,
            request,
            cuts,
            cancel,
            progress,
        )
        rendered = workspace / "rendered"
        _ensure_private_directory(rendered)
        _render_frames(
            frames,
            final_masks,
            rendered,
            source_probe,
            request,
            background_image,
            background_frames,
            limits,
            cancel,
            progress,
        )
        manifest = _write_concat_manifest(rendered, source_probe.frame_durations)
        reserved = allocate_staging_path(
            request.destination, staging_token=estimate.confirmation_sha256[:32]
        )
        try:
            _encode_video(
                manifest,
                rendered,
                source_copy,
                reserved.stage,
                source_probe,
                request,
                profile,
                estimate.preserve_audio,
                limits,
                deadline,
                cancel,
            )
            staged = finalize_staged_file(
                reserved,
                media_type=profile.media_type,
                kind=request.output_kind.value,
                max_output_bytes=limits.max_output_bytes,
                verify=lambda path: _verify_encoded(
                    path,
                    rendered,
                    source_probe,
                    request,
                    profile,
                    estimate.preserve_audio,
                    limits,
                    deadline,
                    cancel,
                ),
            )
            commit_staged([staged])
        except Exception:
            reserved.close()
            reserved.stage.unlink(missing_ok=True)
            raise
        if progress is not None:
            progress("committed", source_probe.frame_count, source_probe.frame_count)
        return VideoResult(
            destination=staged.destination,
            kind=request.output_kind,
            media_type=staged.media_type,
            sha256=staged.sha256,
            bytes=staged.bytes,
            width=source_probe.width,
            height=source_probe.height,
            frame_count=source_probe.frame_count,
            duration_seconds=_decimal_text(source_probe.duration_seconds),
            audio_preserved=estimate.preserve_audio,
            raw_frames=request.raw_frames,
            smoothing_radius_frames=0 if request.raw_frames else request.smoothing_radius_frames,
            batch_frames=request.batch_frames,
            scene_cut_frames=tuple(cuts),
            gif_alpha_threshold_u8=(
                request.gif_alpha_threshold_u8
                if request.output_kind is VideoOutputKind.GIF
                else None
            ),
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


def _persistent_workspace(state_dir: Path, confirmation: str) -> Path:
    if state_dir.is_symlink():
        raise ValueError("video state directory cannot be a symlink")
    state_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
    if not state_dir.is_dir():
        raise ValueError("video state directory is unavailable")
    workspace = state_dir / f"kilix-f108-video-{confirmation[:32]}"
    workspace.mkdir(mode=0o700, exist_ok=True)
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("video workspace is unavailable")
    os.chmod(workspace, 0o700)
    return workspace


def _initialize_workspace(workspace: Path, request: VideoRequest, estimate: VideoEstimate) -> None:
    document = {
        "estimate": estimate.wire(),
        "destination": str(request.destination),
        "background_image": str(request.background_image) if request.background_image else None,
        "background_video": str(request.background_video) if request.background_video else None,
        "scene_cut_threshold_u8": request.scene_cut_threshold_u8,
        "gif_alpha_threshold_u8": request.gif_alpha_threshold_u8,
    }
    expected = _canonical_json(document)
    manifest = workspace / "job.json"
    if manifest.exists():
        if manifest.is_symlink() or manifest.read_bytes() != expected:
            raise RemovalFailure(
                "background.invalid-request",
                "The video resume state does not match the confirmed job.",
                "input",
                "resume",
            )
    else:
        _write_private_atomic(manifest, expected)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RemovalFailure(
            "background.output-failed",
            "The private video workspace is unavailable.",
            "output",
            "write-output",
        )
    os.chmod(path, 0o700)


def _copy_verified(
    identity: VideoIdentity, destination: Path, cancel: threading.Event | None
) -> None:
    if destination.exists():
        existing = _video_identity(
            destination,
            VideoLimits(max_input_bytes=max(identity.bytes, 1)),
            cancel,
        )
        if existing.bytes == identity.bytes and existing.sha256 == identity.sha256:
            return
        raise RemovalFailure(
            "background.input-unreadable",
            "The private video input copy does not match its identity.",
            "input",
            "resume",
        )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink(missing_ok=True)
    source_fd = os.open(identity.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    destination_fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    digest = hashlib.sha256()
    copied = 0
    try:
        while True:
            _check_cancel(cancel, "copy-input")
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            copied += len(block)
            _write_all(destination_fd, block)
        os.fsync(destination_fd)
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    if copied != identity.bytes or digest.hexdigest() != identity.sha256:
        temporary.unlink(missing_ok=True)
        raise RemovalFailure(
            "background.input-unreadable",
            "The video input changed before execution.",
            "input",
            "copy-input",
        )
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _probe_still_image(
    path: Path,
    *,
    expected_geometry: tuple[int, int],
    limits: VideoLimits,
    cancel: threading.Event | None,
) -> VideoIdentity:
    """Identify one admitted still image without parsing it in this process."""

    identity = _video_identity(path, limits, cancel)
    document = _json_capture(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(identity.path),
        ],
        time.monotonic() + limits.probe_timeout_seconds,
        limits,
        cancel,
    )
    raw_streams = document.get("streams")
    if not isinstance(raw_streams, list) or len(raw_streams) != 1:
        raise RemovalFailure(
            "background.invalid-request",
            "The background image must contain exactly one image stream.",
            "input",
            "probe",
        )
    stream = raw_streams[0]
    if not isinstance(stream, dict) or stream.get("codec_type") != "video":
        raise RemovalFailure(
            "background.invalid-request",
            "The background image stream is invalid.",
            "input",
            "probe",
        )
    codec = stream.get("codec_name")
    frames = stream.get("nb_read_frames")
    if codec not in STILL_IMAGE_CODECS or frames != "1":
        raise RemovalFailure(
            "background.invalid-request",
            "The background must be one admitted still image.",
            "input",
            "probe",
        )
    width = _probe_int(stream.get("width"), "background width")
    height = _probe_int(stream.get("height"), "background height")
    if _rotation(stream) != 0 or (width, height) != expected_geometry:
        raise RemovalFailure(
            "background.invalid-request",
            "The background image must already match the oriented source geometry.",
            "input",
            "probe",
        )
    return identity


def _decode_still_rgba(
    path: Path,
    width: int,
    height: int,
    limits: VideoLimits,
    deadline: float,
    cancel: threading.Event | None,
) -> bytes:
    """Decode exactly one verified still through the bounded FFmpeg process."""

    expected = width * height * 4
    if expected > limits.max_temp_bytes:
        raise RemovalFailure(
            "background.input-limit",
            "The background image exceeds the temporary-space limit.",
            "resource",
            "decode",
        )
    output = _run_capture(
        [
            str(FFMPEG),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-frames:v",
            "2",
            "-an",
            "-sn",
            "-dn",
            "-pix_fmt",
            "rgba",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        deadline=deadline,
        max_stdout=expected * 2,
        max_stderr=limits.max_log_bytes,
        cancel=cancel,
        phase="decode",
    ).stdout
    if len(output) != expected:
        raise RemovalFailure(
            "background.invalid-request",
            "The background must decode to exactly one frame at source geometry.",
            "input",
            "decode",
        )
    return output


def _require_same_probe(expected: VideoProbe, actual: VideoProbe) -> None:
    fields = (
        "width",
        "height",
        "frame_timestamps",
        "frame_durations",
        "duration_seconds",
        "rotation_degrees",
        "video_codec",
        "pixel_format",
        "format_names",
        "has_alpha",
        "audio_codec",
    )
    if expected.identity.sha256 != actual.identity.sha256 or any(
        getattr(expected, field) != getattr(actual, field) for field in fields
    ):
        raise RemovalFailure(
            "background.input-unreadable",
            "The video changed after its estimate was confirmed.",
            "input",
            "confirm-estimate",
        )


def _require_aligned_background(foreground: VideoProbe, background: VideoProbe) -> None:
    if (
        (foreground.width, foreground.height) != (background.width, background.height)
        or foreground.frame_timestamps != background.frame_timestamps
        or foreground.frame_durations != background.frame_durations
    ):
        raise RemovalFailure(
            "background.invalid-request",
            "The background video must match source geometry and frame timestamps.",
            "input",
            "decode",
        )


def _decode_raw_frames(
    source: Path,
    probe: VideoProbe,
    output: Path,
    limits: VideoLimits,
    deadline: float,
    cancel: threading.Event | None,
    scene_threshold: int | None,
    *,
    progress: VideoProgress | None,
) -> list[int]:
    complete = output / "decode.done"
    cuts_file = output / "scene-cuts.json"
    expected_frame_bytes = probe.width * probe.height * 4
    if complete.exists() and all(
        _valid_private_file(_frame_path(output, index), expected_frame_bytes)
        for index in range(probe.frame_count)
    ):
        if scene_threshold is None:
            return []
        try:
            cuts = json.loads(cuts_file.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise _video_error("The video resume scene state is invalid.", "resume") from exc
        if not isinstance(cuts, list) or not all(isinstance(item, int) for item in cuts):
            raise _video_error("The video resume scene state is invalid.", "resume")
        return cast(list[int], cuts)

    for entry in output.glob("frame-*.rgba"):
        entry.unlink(missing_ok=True)
    complete.unlink(missing_ok=True)
    cuts_file.unlink(missing_ok=True)
    argv = [
        str(FFMPEG),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-pix_fmt",
        "rgba",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    cuts = _capture_raw_frames(
        argv,
        output,
        probe,
        limits,
        deadline,
        cancel,
        scene_threshold,
        progress,
    )
    if scene_threshold is not None:
        _write_private_atomic(cuts_file, (json.dumps(cuts, separators=(",", ":")) + "\n").encode())
    _write_private_atomic(complete, b"complete\n")
    return cuts


def _capture_raw_frames(
    argv: list[str],
    output: Path,
    probe: VideoProbe,
    limits: VideoLimits,
    deadline: float,
    cancel: threading.Event | None,
    scene_threshold: int | None,
    progress: VideoProgress | None,
) -> list[int]:
    frame_bytes = probe.width * probe.height * 4
    if frame_bytes * probe.frame_count > limits.max_temp_bytes:
        raise RemovalFailure(
            "background.input-limit",
            "Decoded video frames exceed the temporary-space limit.",
            "resource",
            "decode",
        )
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_PROCESS_ENV,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise RemovalFailure(
            "background.backend-unavailable",
            "The qualified FFmpeg decoder is unavailable.",
            "provider",
            "decode",
        ) from exc
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    pending = bytearray()
    stderr = bytearray()
    frame_index = 0
    cuts: list[int] = []
    previous_thumbnail: bytes | None = None
    try:
        while selector.get_map():
            _check_process_deadline(process, deadline, cancel, "decode")
            for key, _events in selector.select(timeout=0.05):
                block = os.read(key.fd, 1024 * 1024)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stderr":
                    stderr.extend(block)
                    if len(stderr) > limits.max_log_bytes:
                        del stderr[: len(stderr) - limits.max_log_bytes]
                    continue
                pending.extend(block)
                while len(pending) >= frame_bytes:
                    if frame_index >= probe.frame_count:
                        _terminate_group(process)
                        raise _video_error(
                            "The decoder produced more frames than the immutable probe.",
                            "decode",
                        )
                    payload = bytes(pending[:frame_bytes])
                    del pending[:frame_bytes]
                    _write_private_atomic(_frame_path(output, frame_index), payload)
                    if scene_threshold is not None:
                        thumbnail = _thumbnail(payload, probe.width, probe.height)
                        if (
                            previous_thumbnail is not None
                            and _scene_score(previous_thumbnail, thumbnail) >= scene_threshold
                        ):
                            cuts.append(frame_index)
                        previous_thumbnail = thumbnail
                    frame_index += 1
                    if progress is not None:
                        progress("decode", frame_index, probe.frame_count)
        returncode = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired as exc:
        _terminate_group(process)
        raise _video_error("The FFmpeg decoder did not terminate safely.", "decode") from exc
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            _terminate_group(process)
    if returncode != 0 or pending or frame_index != probe.frame_count:
        raise _video_error(
            "The decoded frame count does not match the immutable probe.",
            "decode",
        )
    return cuts


def _thumbnail(payload: bytes, width: int, height: int) -> bytes:
    image = Image.frombytes("RGBA", (width, height), payload)
    reduced = image.convert("L").resize((min(32, width), min(32, height)))
    return reduced.tobytes()


def _scene_score(previous: bytes, current: bytes) -> int:
    if len(previous) != len(current) or not previous:
        raise ValueError("scene thumbnails must share nonzero geometry")
    total = sum(abs(left - right) for left, right in zip(previous, current, strict=True))
    return (total + len(previous) // 2) // len(previous)


def _produce_masks(
    frames: Path,
    masks: Path,
    probe: VideoProbe,
    request: VideoRequest,
    masker: FrameMasker,
    cancel: threading.Event | None,
    progress: VideoProgress | None,
) -> None:
    frame_bytes = probe.width * probe.height * 4
    mask_bytes = probe.width * probe.height
    radius = 0 if request.raw_frames else request.smoothing_radius_frames
    for start in range(0, probe.frame_count, request.batch_frames):
        end = min(probe.frame_count, start + request.batch_frames)
        marker = masks / f"batch-{start:09}-{end:09}.done"
        if marker.exists() and all(
            _valid_private_file(_mask_path(masks, index), mask_bytes) for index in range(start, end)
        ):
            if progress is not None:
                progress("infer", end, probe.frame_count)
            continue
        # A cancelled batch has no marker. Resume deliberately reprocesses the
        # entire interior and the temporal overlap on both sides.
        work_start = max(0, start - radius)
        work_end = min(probe.frame_count, end + radius)
        for index in range(work_start, work_end):
            _check_cancel(cancel, "infer")
            rgba = _read_exact(_frame_path(frames, index), frame_bytes)
            image = Image.frombytes("RGBA", (probe.width, probe.height), rgba)
            mask = masker(image, index, cancel)
            if mask.mode != "L" or mask.size != image.size:
                raise RemovalFailure(
                    "background.inference-failed",
                    "The frame provider returned an invalid gray8 mask.",
                    "provider",
                    "infer",
                )
            payload = mask.tobytes()
            if len(payload) != mask_bytes:
                raise RemovalFailure(
                    "background.inference-failed",
                    "The frame provider returned an invalid mask geometry.",
                    "provider",
                    "infer",
                )
            _write_private_atomic(_mask_path(masks, index), payload)
        _write_private_atomic(marker, b"complete\n")
        if progress is not None:
            progress("infer", end, probe.frame_count)


def _smooth_mask_files(
    masks: Path,
    output: Path,
    probe: VideoProbe,
    request: VideoRequest,
    cuts: Sequence[int],
    cancel: threading.Event | None,
    progress: VideoProgress | None,
) -> None:
    frame_bytes = probe.width * probe.height
    radius = 0 if request.raw_frames else request.smoothing_radius_frames
    boundaries = sorted(set(cuts) | {0, probe.frame_count})
    segment = [0] * probe.frame_count
    for number, (start, end) in enumerate(pairwise(boundaries)):
        for index in range(start, end):
            segment[index] = number
    for start in range(0, probe.frame_count, request.batch_frames):
        end = min(probe.frame_count, start + request.batch_frames)
        marker = output / f"batch-{start:09}-{end:09}.done"
        if marker.exists() and all(
            _valid_private_file(_mask_path(output, index), frame_bytes)
            for index in range(start, end)
        ):
            if progress is not None:
                progress("temporal-smooth", end, probe.frame_count)
            continue
        for index in range(start, end):
            _check_cancel(cancel, "temporal-smooth")
            first = max(0, index - radius)
            last = min(probe.frame_count, index + radius + 1)
            neighbours = [
                _mask_path(masks, candidate)
                for candidate in range(first, last)
                if segment[candidate] == segment[index]
            ]
            _write_smoothed_mask_atomic(
                _mask_path(output, index),
                neighbours,
                frame_bytes,
                cancel,
            )
        _write_private_atomic(marker, b"complete\n")
        if progress is not None:
            progress("temporal-smooth", end, probe.frame_count)


def _write_smoothed_mask_atomic(
    destination: Path,
    neighbours: Sequence[Path],
    expected_bytes: int,
    cancel: threading.Event | None,
) -> None:
    """Reduce any temporal radius with a fixed-size in-memory accumulator."""

    import numpy as np

    if not neighbours or expected_bytes <= 0:
        raise ValueError("a nonempty smoothing window and geometry are required")
    descriptors: list[int] = []
    snapshots: list[os.stat_result] = []
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    output_descriptor = -1
    try:
        for path in neighbours:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_size != expected_bytes:
                os.close(descriptor)
                raise _video_error("A temporal mask has invalid geometry.", "resume")
            descriptors.append(descriptor)
            snapshots.append(status)
        output_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        divisor = len(descriptors)
        rounding = divisor // 2
        remaining = expected_bytes
        while remaining:
            _check_cancel(cancel, "temporal-smooth")
            block_bytes = min(SMOOTH_CHUNK_BYTES, remaining)
            accumulator = np.zeros(block_bytes, dtype=np.uint32)
            for descriptor in descriptors:
                _check_cancel(cancel, "temporal-smooth")
                payload = _read_descriptor_exact(descriptor, block_bytes)
                accumulator += np.frombuffer(payload, dtype=np.uint8)
            accumulator += rounding
            accumulator //= divisor
            _write_all(output_descriptor, accumulator.astype(np.uint8).tobytes())
            remaining -= block_bytes
        for descriptor, before in zip(descriptors, snapshots, strict=True):
            after = os.fstat(descriptor)
            if (
                after.st_dev != before.st_dev
                or after.st_ino != before.st_ino
                or after.st_size != before.st_size
            ):
                raise _video_error("A temporal mask changed during smoothing.", "resume")
        os.fsync(output_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
        if output_descriptor >= 0:
            os.close(output_descriptor)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _read_descriptor_exact(descriptor: int, expected_bytes: int) -> bytearray:
    payload = bytearray()
    while len(payload) < expected_bytes:
        block = os.read(descriptor, expected_bytes - len(payload))
        if not block:
            break
        payload.extend(block)
    if len(payload) != expected_bytes:
        raise _video_error("A temporal mask is truncated.", "resume")
    return payload


def _render_frames(
    frames: Path,
    masks: Path,
    output: Path,
    probe: VideoProbe,
    request: VideoRequest,
    background_image: bytes | None,
    background_frames: Path | None,
    limits: VideoLimits,
    cancel: threading.Event | None,
    progress: VideoProgress | None,
) -> None:
    rgba_bytes = probe.width * probe.height * 4
    mask_bytes = probe.width * probe.height
    encoded_total = 0
    for start in range(0, probe.frame_count, request.batch_frames):
        end = min(probe.frame_count, start + request.batch_frames)
        marker = output / f"batch-{start:09}-{end:09}.done"
        if marker.exists() and all(
            _render_path(output, index).is_file() for index in range(start, end)
        ):
            if progress is not None:
                progress("render", end, probe.frame_count)
            continue
        for index in range(start, end):
            _check_cancel(cancel, "render")
            foreground = Image.frombytes(
                "RGBA",
                (probe.width, probe.height),
                _read_exact(_frame_path(frames, index), rgba_bytes),
            )
            mask = Image.frombytes(
                "L",
                (probe.width, probe.height),
                _read_exact(_mask_path(masks, index), mask_bytes),
            )
            if request.output_kind is VideoOutputKind.MATTE:
                rendered = mask
            elif request.output_kind is VideoOutputKind.COMPOSITE_IMAGE:
                if background_image is None:
                    raise RuntimeError("validated background image is unavailable")
                background = Image.frombytes("RGBA", (probe.width, probe.height), background_image)
                rendered = Image.composite(foreground, background, mask).convert("RGB")
            elif request.output_kind is VideoOutputKind.COMPOSITE_VIDEO:
                if background_frames is None:
                    raise RuntimeError("validated background video is unavailable")
                background = Image.frombytes(
                    "RGBA",
                    (probe.width, probe.height),
                    _read_exact(_frame_path(background_frames, index), rgba_bytes),
                )
                rendered = Image.composite(foreground, background, mask).convert("RGB")
            else:
                rendered = foreground.copy()
                output_mask = mask
                if request.output_kind is VideoOutputKind.GIF:
                    threshold = request.gif_alpha_threshold_u8
                    output_mask = mask.point([0] * threshold + [255] * (256 - threshold))
                rendered.putalpha(output_mask)
            frame_path = _render_path(output, index)
            _save_png_private(rendered, frame_path)
            encoded_total += frame_path.stat().st_size
            if encoded_total > limits.max_temp_bytes:
                raise RemovalFailure(
                    "background.output-failed",
                    "Rendered video frames exceed the temporary-space limit.",
                    "resource",
                    "render",
                )
        _write_private_atomic(marker, b"complete\n")
        if progress is not None:
            progress("render", end, probe.frame_count)


def _save_png_private(image: Image.Image, destination: Path) -> None:
    temporary = destination.with_suffix(".png.partial")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            image.save(
                stream,
                format="PNG",
                compress_level=1,
                optimize=False,
                icc_profile=None,
                exif=b"",
                pnginfo=None,
            )
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _write_concat_manifest(rendered: Path, durations: Sequence[Decimal]) -> Path:
    lines = ["ffconcat version 1.0"]
    for index, duration in enumerate(durations):
        lines.append(f"file 'frame-{index:09}.png'")
        # Prevent image2's default 1/25 time base from rounding VFR timestamps.
        # The manifest is generated here from fixed basenames, never user text.
        if len(durations) == 1:
            rate = (Fraction(1, 1) / Fraction(duration)).limit_denominator(2_147_483_647)
            lines.append(f"option framerate {rate.numerator}/{rate.denominator}")
        else:
            lines.append("option framerate 1000000")
        lines.append(f"duration {_decimal_text(duration)}")
    if durations:
        # The concat demuxer needs a following timestamp to retain the final
        # packet duration. The encoder is capped at the original frame count.
        lines.append(f"file 'frame-{len(durations) - 1:09}.png'")
        lines.append("option framerate 1000000")
    manifest = rendered / "frames.ffconcat"
    _write_private_atomic(manifest, ("\n".join(lines) + "\n").encode("ascii"))
    return manifest


def _encode_video(
    manifest: Path,
    rendered: Path,
    source: Path,
    stage: Path,
    probe: VideoProbe,
    request: VideoRequest,
    profile: _OutputProfile,
    preserve_audio: bool,
    limits: VideoLimits,
    deadline: float,
    cancel: threading.Event | None,
) -> None:
    argv = [
        str(FFMPEG),
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        # `option framerate` is rejected in safe mode. Every manifest basename
        # is generated above, and the process is already confined to `rendered`.
        "0",
        "-i",
        manifest.name,
    ]
    if preserve_audio:
        argv.extend(["-i", str(source)])
    if request.output_kind is VideoOutputKind.GIF:
        threshold = request.gif_alpha_threshold_u8
        argv.extend(
            [
                "-filter_complex",
                "[0:v]split[v0][v1];"
                "[v0]palettegen=reserve_transparent=1:stats_mode=full[p];"
                f"[v1][p]paletteuse=alpha_threshold={threshold}:dither=sierra2_4a[v]",
                "-map",
                "[v]",
            ]
        )
    else:
        argv.extend(["-map", "0:v:0"])
    if preserve_audio:
        argv.extend(["-map", "1:a:0"])
    argv.extend(["-map_metadata", "-1", "-map_chapters", "-1", "-fps_mode", "vfr"])
    if request.output_kind is VideoOutputKind.TRANSPARENT_MOV:
        argv.extend(
            [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4",
                "-pix_fmt",
                "yuva444p10le",
                "-vendor",
                "apl0",
            ]
        )
    elif request.output_kind is VideoOutputKind.TRANSPARENT_WEBM:
        argv.extend(
            [
                "-c:v",
                "libvpx-vp9",
                "-lossless",
                "1",
                "-pix_fmt",
                "yuva420p",
                "-metadata:s:v:0",
                "alpha_mode=1",
            ]
        )
    elif request.output_kind in {
        VideoOutputKind.MATTE,
        VideoOutputKind.COMPOSITE_IMAGE,
        VideoOutputKind.COMPOSITE_VIDEO,
    }:
        argv.extend(
            [
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-coder",
                "1",
                "-context",
                "1",
                "-pix_fmt",
                profile.pixel_format,
            ]
        )
    else:
        argv.extend(["-c:v", "gif", "-pix_fmt", "pal8"])
    if preserve_audio:
        if profile.audio_encoder is None:
            raise RuntimeError("validated audio profile is unavailable")
        argv.extend(["-c:a", profile.audio_encoder, "-shortest"])
    argv.extend(["-frames:v", str(probe.frame_count)])
    argv.extend(["-f", profile.muxer, str(stage)])
    _run_capture(
        argv,
        deadline=deadline,
        max_stdout=1,
        max_stderr=limits.max_log_bytes,
        cancel=cancel,
        phase="encode",
        cwd=rendered,
    )


def _verify_encoded(
    path: Path,
    rendered: Path,
    source: VideoProbe,
    request: VideoRequest,
    profile: _OutputProfile,
    preserve_audio: bool,
    limits: VideoLimits,
    deadline: float,
    cancel: threading.Event | None,
) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RemovalFailure(
            "background.deadline",
            "The video job exceeded its deadline.",
            "deadline",
            "verify-output",
        )
    verification_limits = VideoLimits(
        max_input_bytes=max(limits.max_input_bytes, limits.max_output_bytes),
        max_output_bytes=limits.max_output_bytes,
        max_temp_bytes=limits.max_temp_bytes,
        max_frames=limits.max_frames,
        max_dimension=limits.max_dimension,
        max_duration_seconds=limits.max_duration_seconds,
        max_probe_bytes=limits.max_probe_bytes,
        max_log_bytes=limits.max_log_bytes,
        probe_timeout_seconds=min(limits.probe_timeout_seconds, remaining),
        job_timeout_seconds=limits.job_timeout_seconds,
        measured_frame_seconds=limits.measured_frame_seconds,
    )
    actual = probe_video(path, limits=verification_limits, cancel=cancel)
    expected_formats = {
        VideoOutputKind.TRANSPARENT_MOV: {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
        VideoOutputKind.TRANSPARENT_WEBM: {"matroska", "webm"},
        VideoOutputKind.MATTE: {"matroska", "webm"},
        VideoOutputKind.COMPOSITE_IMAGE: {"matroska", "webm"},
        VideoOutputKind.COMPOSITE_VIDEO: {"matroska", "webm"},
        VideoOutputKind.GIF: {"gif"},
    }[request.output_kind]
    if not actual.format_names & expected_formats:
        raise _encoded_video_error("The output container probe does not match its fixed profile.")
    if actual.video_codec != profile.codec_name:
        raise _encoded_video_error("The output codec probe does not match its fixed profile.")
    if (actual.width, actual.height) != (source.width, source.height):
        raise _encoded_video_error("The output video geometry does not match the source.")
    if actual.frame_count != source.frame_count:
        raise _encoded_video_error("The output video frame count does not match the source.")
    if (actual.audio_codec is not None) != preserve_audio:
        raise _encoded_video_error("The output audio policy does not match the confirmed job.")
    if (
        request.output_kind
        in {
            VideoOutputKind.TRANSPARENT_MOV,
            VideoOutputKind.TRANSPARENT_WEBM,
        }
        and not actual.has_alpha
    ):
        raise _encoded_video_error("The transparent output does not advertise an alpha plane.")
    expected_relative = [
        timestamp - source.frame_timestamps[0] for timestamp in source.frame_timestamps
    ]
    actual_relative = [
        timestamp - actual.frame_timestamps[0] for timestamp in actual.frame_timestamps
    ]
    tolerance = Decimal("0.002")
    if any(
        abs(expected - observed) > tolerance
        for expected, observed in zip(expected_relative, actual_relative, strict=True)
    ):
        raise _encoded_video_error("The output video timestamps do not match the source.")
    duration_tolerance = max(source.frame_durations) + tolerance
    if abs(actual.duration_seconds - source.duration_seconds) > duration_tolerance:
        raise _encoded_video_error("The output duration does not match the source.")
    _verify_rendered_pixels(path, rendered, source, request, limits, deadline, cancel)


def _verify_rendered_pixels(
    path: Path,
    rendered: Path,
    source: VideoProbe,
    request: VideoRequest,
    limits: VideoLimits,
    deadline: float,
    cancel: threading.Event | None,
) -> None:
    """Decode the staged carrier and compare its authoritative pixel plane."""

    alpha_output = request.output_kind in {
        VideoOutputKind.TRANSPARENT_MOV,
        VideoOutputKind.TRANSPARENT_WEBM,
        VideoOutputKind.GIF,
    }
    if request.output_kind is VideoOutputKind.MATTE or alpha_output:
        pixel_format = "gray"
        frame_bytes = source.width * source.height
    else:
        pixel_format = "rgb24"
        frame_bytes = source.width * source.height * 3
    argv = [
        str(FFMPEG),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    # FFmpeg's native VP9 decoder does not expose WebM alpha.  The fixed
    # encoder/decoder pair is therefore part of this output profile.
    if request.output_kind is VideoOutputKind.TRANSPARENT_WEBM:
        argv.extend(["-c:v", "libvpx-vp9"])
    argv.extend(
        [
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
        ]
    )
    if alpha_output:
        argv.extend(["-vf", "format=rgba,alphaextract,format=gray"])
    argv.extend(
        [
            "-pix_fmt",
            pixel_format,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_PROCESS_ENV,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise RemovalFailure(
            "background.backend-unavailable",
            "The qualified FFmpeg decoder is unavailable.",
            "provider",
            "verify-output",
        ) from exc
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    pending = bytearray()
    stderr = bytearray()
    frame_index = 0
    try:
        while selector.get_map():
            _check_process_deadline(process, deadline, cancel, "verify-output")
            for key, _events in selector.select(timeout=0.05):
                block = os.read(key.fd, 1024 * 1024)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stderr":
                    stderr.extend(block)
                    if len(stderr) > limits.max_log_bytes:
                        del stderr[: len(stderr) - limits.max_log_bytes]
                    continue
                pending.extend(block)
                while len(pending) >= frame_bytes:
                    if frame_index >= source.frame_count:
                        _terminate_group(process)
                        raise _encoded_video_error(
                            "The output decoder produced more frames than were rendered."
                        )
                    actual = bytes(pending[:frame_bytes])
                    del pending[:frame_bytes]
                    expected = _rendered_samples(
                        _render_path(rendered, frame_index),
                        source.width,
                        source.height,
                        request.output_kind,
                    )
                    if not _rendered_samples_match(actual, expected, request.output_kind):
                        _terminate_group(process)
                        raise _encoded_video_error(
                            "The encoded output pixels do not match the rendered frames."
                        )
                    frame_index += 1
        returncode = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired as exc:
        _terminate_group(process)
        raise _encoded_video_error("The output decoder did not terminate safely.") from exc
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            _terminate_group(process)
    if returncode != 0 or pending or frame_index != source.frame_count:
        raise _encoded_video_error(
            "The decoded output frame count does not match the rendered frames."
        )


def _rendered_samples_match(
    actual: bytes,
    expected: bytes,
    output_kind: VideoOutputKind,
) -> bool:
    if len(actual) != len(expected):
        return False
    if output_kind not in {
        VideoOutputKind.TRANSPARENT_MOV,
        VideoOutputKind.TRANSPARENT_WEBM,
    }:
        return actual == expected
    # Both alpha carriers rescale an 8-bit plane through their coded alpha
    # depth.  FFmpeg's defined round trip differs by at most one u8 unit; all
    # lossless matte/composite carriers and thresholded GIF remain exact.
    return all(
        abs(observed - rendered) <= 1 for observed, rendered in zip(actual, expected, strict=True)
    )


def _rendered_samples(
    path: Path,
    width: int,
    height: int,
    output_kind: VideoOutputKind,
) -> bytes:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
            raise OSError("rendered frame is not a regular file")
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with stream, Image.open(stream) as opened:
            opened.load()
            if opened.size != (width, height):
                raise OSError("rendered frame geometry changed")
            if output_kind is VideoOutputKind.MATTE:
                if opened.mode != "L":
                    raise OSError("rendered matte has an invalid mode")
                return cast(bytes, opened.tobytes())
            if output_kind in {
                VideoOutputKind.COMPOSITE_IMAGE,
                VideoOutputKind.COMPOSITE_VIDEO,
            }:
                if opened.mode != "RGB":
                    raise OSError("rendered composite has an invalid mode")
                return cast(bytes, opened.tobytes())
            if opened.mode != "RGBA":
                raise OSError("rendered alpha frame has an invalid mode")
            return cast(bytes, opened.getchannel("A").tobytes())
    except (OSError, ValueError) as exc:
        raise _encoded_video_error(
            "A rendered frame is invalid during output verification."
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _frame_path(root: Path, index: int) -> Path:
    return root / f"frame-{index:09}.rgba"


def _mask_path(root: Path, index: int) -> Path:
    return root / f"mask-{index:09}.gray"


def _render_path(root: Path, index: int) -> Path:
    return root / f"frame-{index:09}.png"


def _valid_private_file(path: Path, expected_bytes: int) -> bool:
    try:
        status = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(status.st_mode) and status.st_size == expected_bytes


def _read_exact(path: Path, expected_bytes: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise _video_error("A private video frame is unavailable.", "resume") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size != expected_bytes:
            raise _video_error("A private video frame has invalid geometry.", "resume")
        payload = bytearray()
        while len(payload) < expected_bytes:
            block = os.read(descriptor, min(1024 * 1024, expected_bytes - len(payload)))
            if not block:
                break
            payload.extend(block)
        if len(payload) != expected_bytes:
            raise _video_error("A private video frame is truncated.", "resume")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _write_private_atomic(destination: Path, payload: bytes) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _write_all(descriptor: int, payload: bytes) -> None:
    position = 0
    while position < len(payload):
        written = os.write(descriptor, payload[position:])
        if written <= 0:
            raise OSError("short write")
        position += written


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ReferenceFrameMasker:
    """Adapter that reuses the persistent image worker for offline frames."""

    def __init__(
        self,
        workspace: Path,
        *,
        deadline_ms: int = 120_000,
        supervisor: Any | None = None,
    ) -> None:
        from .worker import WorkerSupervisor

        self._root = workspace / "frame-provider"
        _ensure_private_directory(self._root)
        self._deadline_ms = deadline_ms
        self._supervisor = supervisor or WorkerSupervisor()
        self._owns_supervisor = supervisor is None

    def __call__(
        self, image: Image.Image, index: int, cancel: threading.Event | None
    ) -> Image.Image:
        from .frontend import describe_image, make_request

        input_path = self._root / f"input-{index:09}.png"
        output_path = self._root / f"frame-{index:09}.mask.png"
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        _save_png_private(image.convert("RGBA"), input_path)
        try:
            request = make_request(
                describe_image(input_path),
                output_dir=self._root,
                output_key=f"frame-{index:09}",
                output_kinds=["mask"],
                deadline_ms=self._deadline_ms,
            )
            outcome = self._supervisor.run(request, cancel=cancel)
            if not outcome.ok or outcome.result is None:
                raise _failure_from_outcome(outcome.error)
            mask_record = outcome.result.get("mask")
            if not isinstance(mask_record, dict) or mask_record.get("path") != str(output_path):
                raise RemovalFailure(
                    "background.inference-failed",
                    "The frame provider returned an invalid mask descriptor.",
                    "provider",
                    "infer",
                )
            with Image.open(output_path) as opened:
                opened.load()
                if opened.mode != "L" or opened.size != image.size:
                    raise RemovalFailure(
                        "background.inference-failed",
                        "The frame provider returned an invalid mask geometry.",
                        "provider",
                        "infer",
                    )
                return cast(Image.Image, opened.copy())
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._owns_supervisor:
            self._supervisor.close()

    def __enter__(self) -> ReferenceFrameMasker:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _failure_from_outcome(error: dict[str, object] | None) -> RemovalFailure:
    if isinstance(error, dict):
        job = error.get("job")
        phase = error.get("phase")
        if isinstance(job, dict):
            code = job.get("code")
            category = job.get("category")
            safe_message = job.get("safe_message")
            retryable = job.get("retryable", False)
            if (
                isinstance(code, str)
                and isinstance(category, str)
                and isinstance(safe_message, str)
                and isinstance(phase, str)
                and isinstance(retryable, bool)
            ):
                return RemovalFailure(code, safe_message, category, phase, retryable)
    return RemovalFailure(
        "background.inference-failed",
        "The frame provider failed safely.",
        "provider",
        "infer",
    )


def _video_identity(
    path: Path, limits: VideoLimits, cancel: threading.Event | None
) -> VideoIdentity:
    if not path.is_absolute():
        path = path.absolute()
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise RemovalFailure(
            "background.input-unreadable",
            "The video input cannot be read.",
            "input",
            "probe",
        ) from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or not 1 <= status.st_size <= limits.max_input_bytes:
            raise RemovalFailure(
                "background.input-limit",
                "The video input exceeds its byte or file-type limit.",
                "resource",
                "probe",
            )
        digest = hashlib.sha256()
        while True:
            _check_cancel(cancel, "probe")
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            after.st_dev != status.st_dev
            or after.st_ino != status.st_ino
            or after.st_size != status.st_size
        ):
            raise RemovalFailure(
                "background.input-unreadable",
                "The video input changed while it was identified.",
                "input",
                "probe",
            )
        return VideoIdentity(path, status.st_size, digest.hexdigest())
    finally:
        os.close(descriptor)


def _parse_probe(
    identity: VideoIdentity,
    stream_document: Mapping[str, object],
    frame_document: Mapping[str, object],
    limits: VideoLimits,
) -> VideoProbe:
    streams_raw = stream_document.get("streams")
    frames_raw = frame_document.get("frames")
    if not isinstance(streams_raw, list) or not isinstance(frames_raw, list):
        raise _video_error("The media probe returned an invalid document.", "probe")
    streams = [item for item in streams_raw if isinstance(item, dict)]
    frames = [item for item in frames_raw if isinstance(item, dict)]
    videos = [item for item in streams if item.get("codec_type") == "video"]
    audios = [item for item in streams if item.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) > 1 or len(frames) < 1:
        raise _video_error(
            "The input must contain one video stream and at most one audio stream.",
            "probe",
        )
    if len(frames) > limits.max_frames:
        raise RemovalFailure(
            "background.input-limit",
            "The video frame limit is exceeded.",
            "resource",
            "probe",
        )
    video = videos[0]
    coded_width = _probe_int(video.get("width"), "width")
    coded_height = _probe_int(video.get("height"), "height")
    rotation = _rotation(video)
    width, height = (
        (coded_height, coded_width) if rotation in {90, 270} else (coded_width, coded_height)
    )
    if width > limits.max_dimension or height > limits.max_dimension:
        raise RemovalFailure(
            "background.input-limit",
            "The video dimension limit is exceeded.",
            "resource",
            "probe",
        )
    timestamps: list[Decimal] = []
    packet_durations: list[Decimal | None] = []
    for frame in frames:
        frame_width = _probe_int(frame.get("width"), "frame width")
        frame_height = _probe_int(frame.get("height"), "frame height")
        if (frame_width, frame_height) != (coded_width, coded_height):
            raise _video_error("The video changes geometry between frames.", "probe")
        timestamps.append(_probe_decimal(frame.get("best_effort_timestamp_time"), "timestamp"))
        raw_duration = frame.get("pkt_duration_time")
        packet_durations.append(
            _probe_decimal(raw_duration, "frame duration") if raw_duration is not None else None
        )
    if any(current <= previous for previous, current in pairwise(timestamps)):
        raise _video_error("The video frame timestamps are not strictly increasing.", "probe")
    durations: list[Decimal] = []
    for index in range(len(timestamps) - 1):
        durations.append(timestamps[index + 1] - timestamps[index])
    last_packet_duration = packet_durations[-1]
    if last_packet_duration is not None and last_packet_duration > 0:
        durations.append(last_packet_duration)
    elif durations:
        durations.append(durations[-1])
    else:
        stream_duration = _optional_decimal(video.get("duration"))
        format_record = stream_document.get("format")
        format_duration = (
            _optional_decimal(format_record.get("duration"))
            if isinstance(format_record, dict)
            else None
        )
        duration = stream_duration or format_duration
        if duration is None or duration <= 0:
            raise _video_error("The video frame duration is unavailable.", "probe")
        durations.append(duration)
    if any(duration <= 0 for duration in durations):
        raise _video_error("The video contains a non-positive frame duration.", "probe")
    total_duration = timestamps[-1] - timestamps[0] + durations[-1]
    if total_duration > limits.max_duration_seconds:
        raise RemovalFailure(
            "background.input-limit",
            "The configured video duration limit is exceeded.",
            "resource",
            "probe",
        )
    video_codec = video.get("codec_name")
    pixel_format = video.get("pix_fmt")
    if not isinstance(video_codec, str) or not isinstance(pixel_format, str):
        raise _video_error("The video codec profile is unavailable.", "probe")
    audio_codec: str | None = None
    if audios:
        raw_audio_codec = audios[0].get("codec_name")
        if not isinstance(raw_audio_codec, str):
            raise _video_error("The source audio codec is unavailable.", "probe")
        audio_codec = raw_audio_codec
    format_record = stream_document.get("format")
    format_name = format_record.get("format_name") if isinstance(format_record, dict) else None
    if not isinstance(format_name, str) or not format_name:
        raise _video_error("The video container profile is unavailable.", "probe")
    tags = video.get("tags")
    alpha_tag = None
    if isinstance(tags, dict):
        alpha_tag = next(
            (value for key, value in tags.items() if str(key).lower() == "alpha_mode"),
            None,
        )
    return VideoProbe(
        identity=identity,
        width=width,
        height=height,
        frame_timestamps=tuple(timestamps),
        frame_durations=tuple(durations),
        duration_seconds=total_duration,
        rotation_degrees=rotation,
        variable_frame_rate=len(set(durations)) > 1,
        video_codec=video_codec,
        pixel_format=pixel_format,
        format_names=frozenset(format_name.split(",")),
        has_alpha=pixel_format.startswith("yuva") or alpha_tag == "1",
        audio_codec=audio_codec,
    )


def _rotation(stream: Mapping[str, object]) -> int:
    candidate: object | None = None
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and "rotation" in item:
                candidate = item["rotation"]
                break
    tags = stream.get("tags")
    if candidate is None and isinstance(tags, dict):
        candidate = tags.get("rotate")
    if candidate is None:
        return 0
    try:
        value = round(float(cast(Any, candidate))) % 360
    except (TypeError, ValueError, OverflowError) as exc:
        raise _video_error("The video rotation metadata is invalid.", "probe") from exc
    if value not in {0, 90, 180, 270}:
        raise _video_error("The video rotation is not a right angle.", "probe")
    return value


def _probe_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _video_error(f"The probed {name} is invalid.", "probe")
    return value


def _probe_decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, str):
        raise _video_error(f"The probed {name} is invalid.", "probe")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise _video_error(f"The probed {name} is invalid.", "probe") from exc
    if not parsed.is_finite():
        raise _video_error(f"The probed {name} is invalid.", "probe")
    return parsed


def _optional_decimal(value: object) -> Decimal | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _json_capture(
    argv: list[str],
    deadline: float,
    limits: VideoLimits,
    cancel: threading.Event | None,
) -> dict[str, object]:
    output = _run_capture(
        argv,
        deadline=deadline,
        max_stdout=limits.max_probe_bytes,
        max_stderr=limits.max_log_bytes,
        cancel=cancel,
        phase="probe",
    )
    try:
        decoded = json.loads(output.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _video_error("The media probe returned invalid JSON.", "probe") from exc
    if not isinstance(decoded, dict):
        raise _video_error("The media probe returned an invalid document.", "probe")
    return cast(dict[str, object], decoded)


def _run_capture(
    argv: list[str],
    *,
    deadline: float,
    max_stdout: int,
    max_stderr: int,
    cancel: threading.Event | None,
    phase: str,
    cwd: Path | None = None,
) -> _ProcessOutput:
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=_PROCESS_ENV,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise RemovalFailure(
            "background.backend-unavailable",
            "The qualified FFmpeg adapter is unavailable.",
            "provider",
            phase,
        ) from exc
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    stdout = bytearray()
    stderr = bytearray()
    try:
        while selector.get_map():
            _check_process_deadline(process, deadline, cancel, phase)
            for key, _events in selector.select(timeout=0.05):
                block = os.read(key.fd, 64 * 1024)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    stdout.extend(block)
                    if len(stdout) > max_stdout:
                        _terminate_group(process)
                        raise RemovalFailure(
                            "background.input-limit",
                            "The media probe exceeded its output limit.",
                            "resource",
                            phase,
                        )
                else:
                    stderr.extend(block)
                    if len(stderr) > max_stderr:
                        del stderr[: len(stderr) - max_stderr]
        returncode = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired as exc:
        _terminate_group(process)
        raise _video_error("The FFmpeg process did not terminate safely.", phase) from exc
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            _terminate_group(process)
    if returncode != 0:
        raise _video_error("The FFmpeg process rejected the media input.", phase)
    return _ProcessOutput(bytes(stdout), bytes(stderr))


def _check_process_deadline(
    process: subprocess.Popen[bytes],
    deadline: float,
    cancel: threading.Event | None,
    phase: str,
) -> None:
    if cancel is not None and cancel.is_set():
        _terminate_group(process)
        raise RemovalFailure(
            "job.cancelled",
            "The job was cancelled.",
            "cancellation",
            phase,
        )
    if time.monotonic() >= deadline:
        _terminate_group(process)
        raise RemovalFailure(
            "background.deadline",
            "The video job exceeded its deadline.",
            "deadline",
            phase,
        )


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=0.75)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1.0)


def _check_cancel(cancel: threading.Event | None, phase: str) -> None:
    if cancel is not None and cancel.is_set():
        raise RemovalFailure(
            "job.cancelled",
            "The job was cancelled.",
            "cancellation",
            phase,
        )


def _canonical_json(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _video_error(message: str, phase: str) -> RemovalFailure:
    return RemovalFailure(
        "background.input-unreadable",
        message,
        "input",
        phase,
    )


def _encoded_video_error(message: str) -> RemovalFailure:
    return RemovalFailure(
        "background.output-failed",
        message,
        "output",
        "verify-output",
    )
