"""Command-line front end for the shared F108 worker."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import cast

from .contracts import parse_request, sha256_file
from .errors import RemovalFailure
from .frontend import describe_image, load_json_document, make_request, stable_output_key
from .jobs import BatchEntry
from .provider import (
    BackgroundRemovalProvider,
    profile_failure_wire,
    provider_identity,
    video_result_wire,
)
from .video import VideoOutputKind, VideoRequest
from .worker import FALLBACK_REQUEST_ID, JobOutcome, failure_wire, reference_identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kilix-background-remover")
    subcommands = parser.add_subparsers(dest="command", required=True)

    contract = subcommands.add_parser(
        "run-contract", help="run a conditional candidate-R5 v2 request"
    )
    contract.add_argument("request", type=Path)
    contract.add_argument("--reference-profile", action="store_true")

    image = subcommands.add_parser("image", help="remove one image background")
    _add_image_options(image)
    image.add_argument("input", type=Path)

    batch = subcommands.add_parser("batch", help="run an ordered directory batch")
    _add_image_options(batch)
    batch.add_argument("input_dir", type=Path)
    batch.add_argument("--state-dir", type=Path)

    video = subcommands.add_parser(
        "video",
        help="estimate or run one confirmed bounded offline-video job",
    )
    video.add_argument("input", type=Path)
    video.add_argument("output", type=Path)
    video.add_argument(
        "--output-kind",
        choices=[kind.value for kind in VideoOutputKind],
        required=True,
    )
    video.add_argument("--confirm-estimate", metavar="SHA256")
    video.add_argument("--reference-profile", action="store_true")
    video.add_argument("--no-audio", action="store_true")
    video.add_argument("--raw-frames", action="store_true")
    video.add_argument("--smoothing-radius-frames", type=int, default=1)
    video.add_argument("--batch-frames", type=int, default=24)
    video.add_argument("--scene-cut-threshold-u8", type=int, default=48)
    video.add_argument("--gif-alpha-threshold-u8", type=int, default=128)
    video.add_argument("--state-dir", type=Path)
    video_background = video.add_mutually_exclusive_group()
    video_background.add_argument("--background-image", type=Path)
    video_background.add_argument("--background-video", type=Path)

    doctor = subcommands.add_parser("doctor", help="report local reference readiness")
    doctor.add_argument("--json", action="store_true")
    return parser


def _add_image_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-profile", action="store_true")
    parser.add_argument("--deadline-ms", type=int, default=120_000)
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--webp", action="store_true")
    backgrounds = parser.add_mutually_exclusive_group()
    backgrounds.add_argument("--composite-color", metavar="R,G,B,A")
    backgrounds.add_argument("--composite-image", type=Path)


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _profile_failure(request_id: str) -> dict[str, object]:
    return profile_failure_wire(request_id)


def _run(raw: object, allow_reference: bool) -> JobOutcome:
    parse_request(raw)
    with BackgroundRemovalProvider(allow_reference_profile=allow_reference) as provider:
        return provider.run(raw)


def _outcome_wire(outcome: JobOutcome) -> dict[str, object]:
    return {
        "progress": outcome.progress,
        "result": outcome.result,
        "error": outcome.error,
    }


def _output_policy(args: argparse.Namespace) -> tuple[list[str], dict[str, object]]:
    kinds = ["mask"]
    if not args.mask_only:
        kinds.append("cutout-png")
    if args.webp:
        kinds.append("cutout-webp")
    background: dict[str, object] = {"mode": "transparent"}
    if args.composite_color is not None:
        try:
            rgba = [float(value) for value in args.composite_color.split(",")]
        except ValueError as exc:
            raise ValueError("composite color must contain four numeric channels") from exc
        if len(rgba) != 4 or any(not 0.0 <= value <= 1.0 for value in rgba):
            raise ValueError("composite color channels must be within 0..1")
        background = {"mode": "color", "rgba": rgba}
        kinds.append("composite")
    elif args.composite_image is not None:
        background = {"mode": "image", "image": describe_image(args.composite_image)}
        kinds.append("composite")
    return kinds, background


def _image_command(args: argparse.Namespace) -> int:
    image = describe_image(args.input)
    kinds, background = _output_policy(args)
    key = stable_output_key(Path(Path(cast(str, image["path"])).name))
    request = make_request(
        image,
        output_dir=args.output_dir,
        output_key=key,
        output_kinds=kinds,
        background=background,
        deadline_ms=args.deadline_ms,
    )
    outcome = _run(request, args.reference_profile)
    print(_json(_outcome_wire(outcome)), end="")
    return 0 if outcome.ok else 3


def _discover_images(root: Path) -> list[Path]:
    absolute = root.resolve(strict=True)
    if root.is_symlink() or not absolute.is_dir():
        raise ValueError("input directory must be an existing regular directory")
    candidates: list[Path] = []
    for current, directories, filenames in os.walk(absolute, followlinks=False):
        directories[:] = sorted(
            (name for name in directories if not (Path(current) / name).is_symlink()),
            key=os.fsencode,
        )
        for filename in sorted(filenames, key=os.fsencode):
            path = Path(current) / filename
            if path.is_symlink():
                continue
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
                candidates.append(path)
    return sorted(candidates, key=lambda path: os.fsencode(str(path.relative_to(absolute))))


def _batch_command(args: argparse.Namespace) -> int:
    if not args.reference_profile:
        print(_json({"error": _profile_failure(FALLBACK_REQUEST_ID)}), end="")
        return 3
    kinds, background = _output_policy(args)
    input_root = args.input_dir.resolve(strict=True)
    entries: list[BatchEntry] = []
    for path in _discover_images(args.input_dir):
        relative = path.relative_to(input_root)
        key = stable_output_key(relative)
        request = make_request(
            describe_image(path),
            output_dir=args.output_dir,
            output_key=key,
            output_kinds=kinds,
            background=background,
            deadline_ms=args.deadline_ms,
        )
        entries.append(BatchEntry(key, request))
    state_dir = args.state_dir or (args.output_dir / ".kilix-background-remover-state")
    state_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
    with BackgroundRemovalProvider(allow_reference_profile=True) as provider:
        outcomes = provider.run_batch(entries, state_dir=state_dir)
    wire = {
        "schema": "kilix.background-removal.batch-report/v1",
        "items": [
            {
                "index": item.index,
                "key": item.key,
                "disposition": item.disposition,
                **_outcome_wire(item.outcome),
            }
            for item in outcomes
        ],
    }
    print(_json(wire), end="")
    return 0 if all(item.outcome.ok for item in outcomes) else 3


def _video_command(args: argparse.Namespace) -> int:
    request = VideoRequest(
        source=args.input,
        destination=args.output.absolute(),
        output_kind=VideoOutputKind(args.output_kind),
        confirmation_sha256=args.confirm_estimate,
        no_audio=args.no_audio,
        raw_frames=args.raw_frames,
        smoothing_radius_frames=args.smoothing_radius_frames,
        batch_frames=args.batch_frames,
        scene_cut_threshold_u8=args.scene_cut_threshold_u8,
        gif_alpha_threshold_u8=args.gif_alpha_threshold_u8,
        background_image=args.background_image,
        background_video=args.background_video,
        state_dir=args.state_dir,
    )
    with BackgroundRemovalProvider(allow_reference_profile=args.reference_profile) as provider:
        _probe, estimate = provider.estimate_video(request)
        if args.confirm_estimate is None:
            print(
                _json(
                    {
                        "status": "confirmation-required",
                        "estimate": estimate.wire(),
                    }
                ),
                end="",
            )
            return 0
        if args.confirm_estimate != estimate.confirmation_sha256:
            raise RemovalFailure(
                "background.invalid-request",
                "The exact video time/frame/disk estimate has not been confirmed.",
                "input",
                "confirm-estimate",
            )
        if not args.reference_profile:
            print(_json({"error": _profile_failure(FALLBACK_REQUEST_ID)}), end="")
            return 3
        result = provider.run_video(request)
    print(
        _json(
            {
                "status": "committed",
                "result": {
                    key: value
                    for key, value in video_result_wire(result).items()
                    if key != "schema"
                },
            }
        ),
        end="",
    )
    return 0


def _doctor(json_output: bool) -> int:
    profile_id, digest = reference_identity()
    package_model = Path(__file__).with_name("reference_luma.onnx")
    report = {
        "schema": "kilix.background-removal.doctor/v1",
        "version": version("kilix-background-remover"),
        "release_qualified": False,
        "reference_profile": profile_id,
        "artifact_verified": package_model.is_file() and sha256_file(package_model) == digest,
        "onnxruntime_available": importlib.util.find_spec("onnxruntime") is not None,
        "network_listener": False,
        "provider": provider_identity(),
    }
    if json_output:
        print(_json(report), end="")
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0 if report["artifact_verified"] and report["onnxruntime_available"] else 3


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run-contract":
            raw = load_json_document(args.request)
            outcome = _run(raw, args.reference_profile)
            print(_json(_outcome_wire(outcome)), end="")
            return 0 if outcome.ok else 3
        if args.command == "image":
            return _image_command(args)
        if args.command == "batch":
            return _batch_command(args)
        if args.command == "video":
            return _video_command(args)
        if args.command == "doctor":
            return _doctor(args.json)
    except RemovalFailure as failure:
        print(_json({"error": failure_wire(FALLBACK_REQUEST_ID, failure)}), end="")
        return 2
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        internal = RemovalFailure(
            "background.invalid-request",
            "The local request could not be read or validated.",
            "input",
            "accepted",
        )
        print(_json({"error": failure_wire(FALLBACK_REQUEST_ID, internal)}), end="")
        return 2
    except Exception:
        internal = RemovalFailure(
            "background.internal",
            "The command failed safely.",
            "internal",
            "accepted",
        )
        print(_json({"error": failure_wire(FALLBACK_REQUEST_ID, internal)}), end="")
        return 3
    raise AssertionError("unreachable command")
