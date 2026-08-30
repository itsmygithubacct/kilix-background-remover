"""Executable controls for the required bounded offline-video pipeline."""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import kilix_background_remover.video as video
from kilix_background_remover.errors import RemovalFailure
from kilix_background_remover.video import (
    DEFAULT_VIDEO_LIMITS,
    VideoOutputKind,
    VideoRequest,
    estimate_video,
    probe_capabilities,
    probe_video,
    run_video,
    temporal_smooth_masks,
)


@dataclass(frozen=True)
class MediaSet:
    root: Path
    source: Path
    background_video: Path
    background_image: Path
    vfr: Path
    rotated: Path


def _ffmpeg(*args: str, cwd: Path | None = None) -> None:
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "AV_LOG_FORCE_NOCOLOR": "1",
    }
    completed = subprocess.run(
        [
            "/usr/bin/ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            *args,
        ],
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> MediaSet:
    root = tmp_path_factory.mktemp("video-media")
    source = root / "source.mkv"
    background_video = root / "background.mkv"
    background_image = root / "background.png"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=16x12:rate=4:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=8000:duration=1",
        "-shortest",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "bgra",
        "-c:a",
        "flac",
        str(source),
    )
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:size=16x12:rate=4:duration=1",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "bgra",
        str(background_video),
    )
    Image.new("RGBA", (16, 12), (0, 255, 0, 255)).save(background_image)

    vfr_frames = root / "vfr-frames"
    vfr_frames.mkdir()
    for index, value in enumerate((20, 100, 220)):
        Image.new("RGBA", (16, 12), (value, 10, 255 - value, 255)).save(
            vfr_frames / f"frame-{index}.png"
        )
    manifest = vfr_frames / "frames.ffconcat"
    manifest.write_text(
        "ffconcat version 1.0\n"
        "file 'frame-0.png'\noption framerate 1000000\nduration 0.1\n"
        "file 'frame-1.png'\noption framerate 1000000\nduration 0.35\n"
        "file 'frame-2.png'\noption framerate 1000000\nduration 0.2\n",
        encoding="ascii",
    )
    vfr = root / "vfr.mkv"
    _ffmpeg(
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        manifest.name,
        "-map",
        "0:v:0",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "bgra",
        "-fps_mode",
        "vfr",
        str(vfr),
        cwd=vfr_frames,
    )

    rotation_base = root / "rotation-base.mp4"
    rotated = root / "rotated.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=16x12:rate=4:duration=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(rotation_base),
    )
    _ffmpeg(
        "-display_rotation:v:0",
        "90",
        "-i",
        str(rotation_base),
        "-map",
        "0",
        "-c",
        "copy",
        str(rotated),
    )
    return MediaSet(root, source, background_video, background_image, vfr, rotated)


def _masker(image: Image.Image, _index: int, _cancel: threading.Event | None) -> Image.Image:
    return image.getchannel("R")


def _request(media: MediaSet, destination: Path, kind: VideoOutputKind) -> VideoRequest:
    values: dict[str, Any] = {}
    if kind is VideoOutputKind.COMPOSITE_IMAGE:
        values["background_image"] = media.background_image
    elif kind is VideoOutputKind.COMPOSITE_VIDEO:
        values["background_video"] = media.background_video
    elif kind is VideoOutputKind.GIF:
        values["no_audio"] = True
    return VideoRequest(
        source=media.source,
        destination=destination,
        output_kind=kind,
        **values,
    )


@pytest.mark.parametrize("kind", list(VideoOutputKind))
def test_all_required_outputs_are_encoded_and_container_probed(
    media: MediaSet,
    tmp_path: Path,
    kind: VideoOutputKind,
) -> None:
    request = _request(media, tmp_path / f"{kind.value}.media", kind)
    if kind is VideoOutputKind.GIF:
        quantized_source = tmp_path / "gif-centisecond-source.mkv"
        _ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=16x12:rate=3:duration=1",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgra",
            str(quantized_source),
        )
        request = replace(request, source=quantized_source)
    source_probe, estimate = estimate_video(request)
    confirmed = replace(request, confirmation_sha256=estimate.confirmation_sha256)

    result = run_video(confirmed, _masker)
    output_probe = probe_video(result.destination)

    assert result.kind is kind
    assert result.frame_count == source_probe.frame_count
    assert (result.width, result.height) == (source_probe.width, source_probe.height)
    assert result.sha256 == hashlib.sha256(result.destination.read_bytes()).hexdigest()
    assert result.audio_preserved is (kind is not VideoOutputKind.GIF)
    assert (output_probe.audio_codec is not None) is (kind is not VideoOutputKind.GIF)
    assert estimate.gif_hard_edge_disclosure is (kind is VideoOutputKind.GIF)
    assert estimate.gif_alpha_threshold_u8 == (
        request.gif_alpha_threshold_u8 if kind is VideoOutputKind.GIF else None
    )
    if kind is VideoOutputKind.GIF:
        source_relative = [
            timestamp - source_probe.frame_timestamps[0]
            for timestamp in source_probe.frame_timestamps
        ]
        output_relative = [
            timestamp - output_probe.frame_timestamps[0]
            for timestamp in output_probe.frame_timestamps
        ]
        timestamp_delta = max(
            abs(expected - actual)
            for expected, actual in zip(source_relative, output_relative, strict=True)
        )
        assert Decimal("0.002") < timestamp_delta <= Decimal("0.005")


def test_capability_probe_covers_every_required_profile() -> None:
    capabilities = probe_capabilities()

    assert {kind for kind in VideoOutputKind if capabilities.supports(kind)} == set(VideoOutputKind)


def test_temporal_batches_match_one_pass_and_scene_cuts_do_not_bleed() -> None:
    masks = [bytes([0]), bytes([90]), bytes([180]), bytes([255])]
    one_pass = temporal_smooth_masks(
        masks,
        1,
        1,
        radius=1,
        scene_cut_frames=[2],
        batch_frames=len(masks),
    )
    single_frame_batches = temporal_smooth_masks(
        masks,
        1,
        1,
        radius=1,
        scene_cut_frames=[2],
        batch_frames=1,
    )

    assert single_frame_batches == one_pass
    assert one_pass == [bytes([45]), bytes([45]), bytes([218]), bytes([218])]
    assert (
        temporal_smooth_masks(
            masks,
            1,
            1,
            radius=0,
            scene_cut_frames=[],
            batch_frames=1,
        )
        == masks
    )


def test_production_temporal_reducer_uses_a_fixed_memory_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    byte_count = video.SMOOTH_CHUNK_BYTES + 257
    neighbours: list[Path] = []
    for index, value in enumerate((0, 90, 255)):
        path = tmp_path / f"mask-{index}.gray"
        path.write_bytes(bytes([value]) * byte_count)
        neighbours.append(path)
    observed_reads: list[int] = []
    read = os.read

    def bounded_read(descriptor: int, requested: int) -> bytes:
        observed_reads.append(requested)
        return read(descriptor, requested)

    monkeypatch.setattr(video.os, "read", bounded_read)
    destination = tmp_path / "smoothed.gray"

    video._write_smoothed_mask_atomic(destination, neighbours, byte_count, None)

    assert observed_reads
    assert max(observed_reads) <= video.SMOOTH_CHUNK_BYTES
    assert destination.read_bytes() == bytes([115]) * byte_count


def test_confirmed_estimate_binds_gif_disclosure_and_processing_settings(
    media: MediaSet,
    tmp_path: Path,
) -> None:
    first = VideoRequest(
        source=media.source,
        destination=tmp_path / "first.gif",
        output_kind=VideoOutputKind.GIF,
        no_audio=True,
        gif_alpha_threshold_u8=120,
        scene_cut_threshold_u8=40,
    )
    second = replace(first, gif_alpha_threshold_u8=121)
    _probe, first_estimate = estimate_video(first)
    _probe, second_estimate = estimate_video(second)

    assert first_estimate.confirmation_sha256 != second_estimate.confirmation_sha256
    assert first_estimate.gif_hard_edge_disclosure
    assert first_estimate.gif_alpha_threshold_u8 == 120
    assert first_estimate.scene_cut_threshold_u8 == 40

    with pytest.raises(RemovalFailure, match="has not been confirmed"):
        run_video(first, _masker)
    assert not first.destination.exists()


def test_audio_is_never_silently_dropped_for_gif(media: MediaSet, tmp_path: Path) -> None:
    request = VideoRequest(
        source=media.source,
        destination=tmp_path / "audio.gif",
        output_kind=VideoOutputKind.GIF,
    )

    with pytest.raises(RemovalFailure, match="select no-audio explicitly"):
        estimate_video(request)

    assert not request.destination.exists()


@pytest.mark.parametrize("source_name", ["vfr", "rotated"])
def test_vfr_and_rotation_survive_the_container_round_trip(
    media: MediaSet,
    tmp_path: Path,
    source_name: str,
) -> None:
    source = getattr(media, source_name)
    request = VideoRequest(
        source=source,
        destination=tmp_path / f"{source_name}.mkv",
        output_kind=VideoOutputKind.MATTE,
        no_audio=True,
    )
    source_probe, estimate = estimate_video(request)
    result = run_video(replace(request, confirmation_sha256=estimate.confirmation_sha256), _masker)
    output_probe = probe_video(result.destination)

    if source_name == "vfr":
        assert source_probe.variable_frame_rate
    else:
        assert source_probe.rotation_degrees == 90
        assert (source_probe.width, source_probe.height) == (12, 16)
    assert output_probe.frame_count == source_probe.frame_count
    assert (output_probe.width, output_probe.height) == (
        source_probe.width,
        source_probe.height,
    )


def test_cancelled_batch_is_reprocessed_with_its_overlap_on_resume(
    media: MediaSet,
    tmp_path: Path,
) -> None:
    cancellation = threading.Event()
    calls: list[int] = []

    def interrupted(
        image: Image.Image,
        index: int,
        _cancel: threading.Event | None,
    ) -> Image.Image:
        calls.append(index)
        if index == 1:
            cancellation.set()
        return image.getchannel("R")

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    request = VideoRequest(
        source=media.source,
        destination=tmp_path / "resumed.mkv",
        output_kind=VideoOutputKind.MATTE,
        no_audio=True,
        smoothing_radius_frames=1,
        batch_frames=2,
        state_dir=state_dir,
    )
    _probe, estimate = estimate_video(request)
    confirmed = replace(request, confirmation_sha256=estimate.confirmation_sha256)

    with pytest.raises(RemovalFailure) as caught:
        run_video(confirmed, interrupted, cancel=cancellation)
    assert caught.value.code == "job.cancelled"
    assert not request.destination.exists()

    cancellation.clear()
    run_video(confirmed, lambda image, index, cancel: _record_mask(calls, image, index))

    assert calls == [0, 1, 0, 1, 2, 1, 2, 3]
    assert request.destination.is_file()


def _record_mask(calls: list[int], image: Image.Image, index: int) -> Image.Image:
    calls.append(index)
    return image.getchannel("R")


def test_encoder_failure_leaves_no_destination_or_sibling_stage(
    media: MediaSet,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = VideoRequest(
        source=media.source,
        destination=tmp_path / "dead-encoder.mkv",
        output_kind=VideoOutputKind.MATTE,
        no_audio=True,
    )
    _probe, estimate = estimate_video(request)
    confirmed = replace(request, confirmation_sha256=estimate.confirmation_sha256)

    def fail_encoder(*args: object, **kwargs: object) -> None:
        stage = args[3]
        assert isinstance(stage, Path)
        stage.write_bytes(b"partial")
        raise RemovalFailure(
            "background.output-failed",
            "The encoder died.",
            "output",
            "encode",
        )

    monkeypatch.setattr(video, "_encode_video", fail_encoder)
    with pytest.raises(RemovalFailure, match="encoder died"):
        run_video(confirmed, _masker)

    assert not request.destination.exists()
    assert not list(tmp_path.glob(".kilix-f108-*.stage"))


def test_metadata_valid_wrong_pixel_carrier_is_refused_before_atomic_commit(
    media: MediaSet,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = VideoRequest(
        source=media.source,
        destination=tmp_path / "wrong-pixels.mkv",
        output_kind=VideoOutputKind.MATTE,
        no_audio=True,
    )
    _probe, estimate = estimate_video(request)
    confirmed = replace(request, confirmation_sha256=estimate.confirmation_sha256)
    encode_video = video._encode_video

    def encode_then_replace_pixels(*args: object, **kwargs: object) -> None:
        encode_video(*args, **kwargs)
        stage = args[3]
        probe = args[4]
        assert isinstance(stage, Path)
        assert isinstance(probe, video.VideoProbe)
        wrong = stage.with_name(stage.name + ".wrong.mkv")
        _ffmpeg(
            "-i",
            str(stage),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "negate",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-coder",
            "1",
            "-context",
            "1",
            "-pix_fmt",
            "gray",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(probe.frame_count),
            "-f",
            "matroska",
            str(wrong),
        )
        stage.write_bytes(wrong.read_bytes())
        wrong.unlink()

    monkeypatch.setattr(video, "_encode_video", encode_then_replace_pixels)
    with pytest.raises(RemovalFailure, match="pixels do not match") as caught:
        run_video(confirmed, _masker)

    assert caught.value.code == "background.output-failed"
    assert caught.value.phase == "verify-output"
    assert not request.destination.exists()
    assert not list(tmp_path.glob(".kilix-f108-*.stage"))


def test_disk_estimate_and_corrupt_input_fail_before_output(
    media: MediaSet,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "too-large.mkv"
    request = VideoRequest(
        source=media.source,
        destination=destination,
        output_kind=VideoOutputKind.MATTE,
        no_audio=True,
    )
    tiny = replace(DEFAULT_VIDEO_LIMITS, max_temp_bytes=1)
    with pytest.raises(RemovalFailure, match="temporary-space estimate"):
        estimate_video(request, limits=tiny)
    assert not destination.exists()

    corrupt = tmp_path / "corrupt.mkv"
    corrupt.write_bytes(os.urandom(256))
    corrupt_request = replace(request, source=corrupt, destination=tmp_path / "corrupt.out")
    with pytest.raises(RemovalFailure):
        estimate_video(corrupt_request)
    assert not corrupt_request.destination.exists()


def test_batched_and_single_pass_video_have_identical_decoded_frames(
    media: MediaSet,
    tmp_path: Path,
) -> None:
    outputs: list[bytes] = []
    for batch in (1, 64):
        request = VideoRequest(
            source=media.source,
            destination=tmp_path / f"batch-{batch}.mkv",
            output_kind=VideoOutputKind.MATTE,
            no_audio=True,
            smoothing_radius_frames=1,
            batch_frames=batch,
        )
        _probe, estimate = estimate_video(request)
        result = run_video(
            replace(request, confirmation_sha256=estimate.confirmation_sha256),
            _masker,
        )
        outputs.append(_decoded_video_bytes(result.destination))

    assert outputs[0] == outputs[1]


def _decoded_video_bytes(path: Path) -> bytes:
    completed = subprocess.run(
        [
            "/usr/bin/ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return completed.stdout
