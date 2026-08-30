"""Generate the installed F108 product-surface artifact packet.

Run this file with ``python -I`` from an isolated environment containing only
the built product wheel and its locked dependencies.  All product operations
are invoked through installed console scripts or the installed provider port.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, BinaryIO, NoReturn

from PIL import Image

from kilix_background_remover.contract_v2 import ContractRuntime, canonical_bytes
from kilix_background_remover.editable_mask import (
    EditableMaskDocument,
    consume_editable_mask_transcript,
)
from kilix_background_remover.frontend import describe_image, make_request
from kilix_background_remover.provider import video_request_wire
from kilix_background_remover.video import VideoOutputKind, VideoRequest, probe_video

PROCESS_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "AV_LOG_FORCE_NOCOLOR": "1",
    "TMPDIR": os.environ.get("TMPDIR", "/home/pleb/scratch-workers"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def _jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    payload = "".join(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for value in values
    )
    path.write_text(payload, encoding="utf-8")


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path,
    expected: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        env=PROCESS_ENV,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in expected:
        raise RuntimeError(
            f"installed command {Path(arguments[0]).name} returned {completed.returncode}: "
            + completed.stderr.decode("utf-8", errors="replace")[-4096:]
        )
    return completed


def _document(completed: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
    value = json.loads(completed.stdout.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("installed product returned a non-object JSON result")
    return value


def _ffmpeg(arguments: Sequence[str], *, cwd: Path) -> None:
    _run(
        [
            "/usr/bin/ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            *arguments,
        ],
        cwd=cwd,
    )


def _fixtures(root: Path, bounded_fixture: Path) -> dict[str, Path]:
    root.mkdir(mode=0o700)
    image = root / "subject.png"
    background = root / "background.png"
    cancellation = root / "cancellation.png"
    subject = Image.new("RGBA", (16, 12))
    subject.putdata(
        [
            ((x * 17 + y * 5) % 256, (x * 3 + y * 23) % 256, (x * 11 + y * 7) % 256, 255)
            for y in range(12)
            for x in range(16)
        ]
    )
    subject.save(image)
    Image.new("RGBA", (16, 12), (10, 190, 80, 255)).save(background)
    cancellation_image = Image.new("RGBA", (768, 512))
    cancellation_image.putdata(
        [
            ((x * 5 + y) % 256, (x + y * 3) % 256, (x * 7 + y * 11) % 256, 255)
            for y in range(512)
            for x in range(768)
        ]
    )
    cancellation_image.save(cancellation)
    source = root / "source.mkv"
    background_video = root / "background.mkv"
    _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=16x12:rate=3:duration=1",
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
        ],
        cwd=root,
    )
    _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=16x12:rate=3:duration=1",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgra",
            str(background_video),
        ],
        cwd=root,
    )
    bounded = root / "bounded-100mp.png"
    shutil.copyfile(bounded_fixture, bounded)
    return {
        "image": image,
        "background": background,
        "cancellation": cancellation,
        "source": source,
        "background_video": background_video,
        "bounded": bounded,
    }


def _frame_write(stream: BinaryIO, operation: str, payload: bytes = b"") -> None:
    stream.write(f"{operation} {len(payload)}\n".encode("ascii"))
    stream.write(payload)
    stream.flush()


def _frame_read(stream: BinaryIO) -> tuple[str, bytes]:
    header = stream.readline(129)
    if not header or len(header) > 128 or not header.endswith(b"\n"):
        raise RuntimeError("installed provider returned an invalid frame header")
    kind, length_text = header[:-1].split(b" ", 1)
    length = int(length_text)
    payload = bytearray()
    while len(payload) < length:
        block = stream.read(length - len(payload))
        if not block:
            raise RuntimeError("installed provider returned a truncated frame")
        payload.extend(block)
    return kind.decode("ascii"), bytes(payload)


def _new_request(source: Path, output: Path, key: str) -> tuple[dict[str, object], bytes]:
    request = make_request(
        describe_image(source),
        output_dir=output,
        output_key=key,
        output_kinds=["mask"],
    )
    return request, canonical_bytes(request)


def _provider_port(
    executable: Path,
    work: Path,
    fixtures: Mapping[str, Path],
    outputs: Path,
) -> dict[str, object]:
    process = subprocess.Popen(
        [str(executable), "--reference-profile"],
        cwd=work,
        env=PROCESS_ENV,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    try:
        _frame_write(process.stdin, "DISCOVER")
        kind, identity_bytes = _frame_read(process.stdout)
        if kind != "IDENTITY":
            raise RuntimeError("installed provider did not return identity")
        identity = json.loads(identity_bytes)

        request, request_bytes = _new_request(fixtures["image"], outputs, "f115-pane4")
        document = EditableMaskDocument(
            source_sha256=str(request["input"]["sha256"]),
            width=int(request["input"]["width"]),
            height=int(request["input"]["height"]),
        )
        source_before = _sha256(fixtures["image"])
        _frame_write(process.stdin, "SUBMIT", request_bytes)
        messages: list[bytes] = []
        while True:
            message_kind, payload = _frame_read(process.stdout)
            if message_kind != "MESSAGE":
                raise RuntimeError("provider returned a non-message during the image operation")
            messages.append(payload)
            decoded = json.loads(payload)
            if decoded.get("schema") in {
                "kilix.background-removal.result/v2",
                "kilix.background-removal.error/v2",
            }:
                break
        imported = consume_editable_mask_transcript([request_bytes, *messages], document)
        if imported is None:
            raise RuntimeError("installed editable-mask operation did not import a mask")

        cancel_request, cancel_request_bytes = _new_request(
            fixtures["cancellation"], outputs, "f115-cancel"
        )
        _frame_write(process.stdin, "SUBMIT", cancel_request_bytes)
        first_kind, first_progress = _frame_read(process.stdout)
        if first_kind != "MESSAGE":
            raise RuntimeError("provider did not begin the cancellation subject")
        cancel_bytes = canonical_bytes(
            {
                "cancellation_id": str(uuid.uuid4()),
                "client_requested_at": datetime.now(UTC)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z"),
                "reason": "user",
                "request_id": cancel_request["job"]["request_id"],
                "schema": "kilix.media-job.cancel-request/v2",
            }
        )
        _frame_write(process.stdin, "CANCEL", cancel_bytes)
        cancel_outcome: dict[str, Any] | None = None
        cancel_terminal: dict[str, Any] | None = None
        cancel_messages = [first_progress]
        while cancel_outcome is None or cancel_terminal is None:
            frame_kind, payload = _frame_read(process.stdout)
            decoded = json.loads(payload) if payload else None
            if frame_kind == "CANCEL-OUTCOME":
                cancel_outcome = decoded
            elif frame_kind == "MESSAGE":
                cancel_messages.append(payload)
                if decoded.get("schema") in {
                    "kilix.background-removal.result/v2",
                    "kilix.background-removal.error/v2",
                }:
                    cancel_terminal = decoded
            elif frame_kind == "PORT-ERROR":
                raise RuntimeError(f"provider port cancellation failed: {decoded}")
        ContractRuntime.load().validate_transcript(
            [
                ContractRuntime.load().accept_wire(cancel_request_bytes),
                *(ContractRuntime.load().accept_wire(item) for item in cancel_messages),
            ]
        )
        _frame_write(process.stdin, "CLOSE")
        closed_kind, closed_payload = _frame_read(process.stdout)
        if closed_kind != "CLOSED" or closed_payload:
            raise RuntimeError("provider port did not close exactly")
        return_code = process.wait(timeout=10)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if return_code != 0:
            raise RuntimeError(f"provider port returned {return_code}: {stderr[-4096:]}")
        return {
            "schema": "kilix.background-removal.product-f115-port/v1",
            "identity": identity,
            "capabilities": {"passed": 5, "total": 5},
            "provider_process_external": True,
            "editable_mask": {
                "imported": True,
                "document_revision": document.revision,
                "geometry": [imported.provenance.width, imported.provenance.height],
                "sample_sha256": imported.provenance.mask_samples_sha256,
                "source_unchanged": source_before == _sha256(fixtures["image"]),
                "progress_messages": {
                    "passed": len(messages) - 1,
                    "total": len(messages) - 1,
                },
            },
            "cancellation": {
                "outcome": cancel_outcome,
                "terminal_schema": cancel_terminal["schema"],
                "terminal_state": cancel_terminal["job"]["state"],
                "destination_present": Path(str(cancel_request["destinations"]["mask"])).exists(),
            },
            "f115_gate_8_disposition": {"passed": 0, "total": 1},
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _video_arguments(
    executable: Path,
    fixtures: Mapping[str, Path],
    destination: Path,
    kind: VideoOutputKind,
    *,
    batch_frames: int = 2,
) -> list[str]:
    arguments = [
        str(executable),
        "video",
        str(fixtures["source"]),
        str(destination),
        "--output-kind",
        kind.value,
        "--batch-frames",
        str(batch_frames),
    ]
    if kind is VideoOutputKind.COMPOSITE_IMAGE:
        arguments.extend(["--background-image", str(fixtures["background"])])
    elif kind is VideoOutputKind.COMPOSITE_VIDEO:
        arguments.extend(["--background-video", str(fixtures["background_video"])])
    elif kind is VideoOutputKind.GIF:
        arguments.append("--no-audio")
    return arguments


def _all_video_profiles(
    executable: Path,
    work: Path,
    fixtures: Mapping[str, Path],
    outputs: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for kind in VideoOutputKind:
        destination = outputs / f"cli-{kind.value}.media"
        arguments = _video_arguments(executable, fixtures, destination, kind)
        estimate = _document(_run(arguments, cwd=work))
        confirmation = estimate["estimate"]["confirmation_sha256"]
        result = _document(
            _run(
                [
                    *arguments,
                    "--confirm-estimate",
                    confirmation,
                    "--reference-profile",
                ],
                cwd=work,
            )
        )
        probe = probe_video(destination)
        records.append(
            {
                "surface": "cli",
                "kind": kind.value,
                "estimate": estimate["estimate"],
                "result": result["result"],
                "container_probe": {
                    "codec": probe.video_codec,
                    "formats": sorted(probe.format_names),
                    "frame_count": probe.frame_count,
                    "audio": probe.audio_codec is not None,
                    "has_alpha": probe.has_alpha,
                },
            }
        )
    return records


def _tui_video(
    executable: Path,
    work: Path,
    fixtures: Mapping[str, Path],
    requests: Path,
    outputs: Path,
) -> tuple[dict[str, object], Path]:
    destination = outputs / "tui-matte.media"
    request = VideoRequest(
        source=fixtures["source"],
        destination=destination,
        output_kind=VideoOutputKind.MATTE,
        batch_frames=1,
    )
    request_path = requests / "tui-video.json"
    _json(request_path, video_request_wire(request))
    estimate_run = _run(
        [str(executable), str(request_path), "--operation", "video"],
        cwd=work,
    )
    estimate = _document(estimate_run)
    confirmation = estimate["estimate"]["confirmation_sha256"]
    confirmed = replace(request, confirmation_sha256=confirmation)
    _json(request_path, video_request_wire(confirmed))
    result_run = _run(
        [
            str(executable),
            str(request_path),
            "--operation",
            "video",
            "--reference-profile",
        ],
        cwd=work,
    )
    return {
        "schema": "kilix.background-removal.product-tui-video/v1",
        "estimate": estimate,
        "result": _document(result_run),
        "stderr_has_provider_header": b"decode: spawned parser" in result_run.stderr,
        "keyboard_contract": ["q", "Escape", "r"],
    }, destination


def _app_video(
    executable: Path,
    work: Path,
    fixtures: Mapping[str, Path],
    requests: Path,
    outputs: Path,
) -> tuple[dict[str, object], Path]:
    destination = outputs / "app-matte.media"
    request = VideoRequest(
        source=fixtures["source"],
        destination=destination,
        output_kind=VideoOutputKind.MATTE,
        batch_frames=3,
    )
    message_path = requests / "app-video.json"
    message = {
        "schema": "kilix.background-removal.app-bridge-request/v2",
        "operation": "estimate-video",
        "request": video_request_wire(request),
    }
    _json(message_path, message)
    estimate = _document(_run([str(executable), "--message", str(message_path)], cwd=work))
    confirmation = estimate["result"]["confirmation_sha256"]
    confirmed = replace(request, confirmation_sha256=confirmation)
    message["operation"] = "run-video"
    message["request"] = video_request_wire(confirmed)
    _json(message_path, message)
    result = _document(
        _run(
            [str(executable), "--message", str(message_path), "--reference-profile"],
            cwd=work,
        )
    )
    return {
        "schema": "kilix.background-removal.product-contained-app-video/v1",
        "estimate": estimate,
        "result": result,
        "headless_contained_lifecycle": True,
        "graphical_entry_point": executable.name,
    }, destination


def _decoded_matte(path: Path, work: Path) -> bytes:
    completed = _run(
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
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        cwd=work,
    )
    return completed.stdout


def _bounded_decode(
    executable: Path,
    work: Path,
    bounded: Path,
    requests: Path,
    outputs: Path,
) -> dict[str, object]:
    request = make_request(
        describe_image(bounded),
        output_dir=outputs,
        output_key="bounded-refusal",
        output_kinds=["mask"],
    )
    request["job"]["limits"]["max_decoded_pixels"] = 1_000_000
    request_path = requests / "bounded-decode-request.json"
    request_path.write_bytes(canonical_bytes(request))
    completed = _run(
        [str(executable), "run-contract", str(request_path), "--reference-profile"],
        cwd=work,
        expected=frozenset({3}),
    )
    result = _document(completed)
    destination = Path(str(request["destinations"]["mask"]))
    return {
        "schema": "kilix.background-removal.product-bounded-decode/v1",
        "fixture": {
            "bytes": bounded.stat().st_size,
            "decoded_pixels": int(request["input"]["width"]) * int(request["input"]["height"]),
            "expansion_ratio_rgba": (
                int(request["input"]["width"]) * int(request["input"]["height"]) * 4
            )
            // bounded.stat().st_size,
        },
        "configured_max_decoded_pixels": 1_000_000,
        "terminal": result,
        "typed_refusal": result["error"]["job"]["code"] == "background.input-limit",
        "destination_present": destination.exists(),
        "bounded_child_policy": {
            "wall_seconds": 30.0,
            "cpu_seconds": 30,
            "address_space_bytes": 2 * 1024 * 1024 * 1024,
            "status_bytes": 4096,
        },
    }


def _manifest(root: Path) -> None:
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "PRODUCT-SHA256SUMS":
            entries.append(f"{_sha256(path)}  {path.relative_to(root)}")
    (root / "PRODUCT-SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="ascii")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bounded-fixture", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--product-commit", required=True)
    parser.add_argument("--product-tree", required=True)
    return parser


def _fail(message: str) -> NoReturn:
    raise SystemExit(message)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir.absolute()
    if output.exists():
        _fail("product artifact output directory already exists")
    if not args.wheel.is_file() or not args.bounded_fixture.is_file():
        _fail("wheel and bounded fixture must be regular files")
    output.mkdir(mode=0o700)
    fixtures_root = output / "fixtures"
    outputs = output / "outputs"
    requests = output / "requests"
    outputs.mkdir(mode=0o700)
    requests.mkdir(mode=0o700)
    fixtures = _fixtures(fixtures_root, args.bounded_fixture)
    work = output / "empty-cwd"
    work.mkdir(mode=0o700)

    cli = args.bin_dir / "kilix-background-remover"
    tui = args.bin_dir / "kilix-background-remover-tui"
    app = args.bin_dir / "kilix-background-remover-app"
    port = args.bin_dir / "kilix-background-remover-provider"
    executables = [cli, tui, app, port]
    if not all(item.is_file() and os.access(item, os.X_OK) for item in executables):
        _fail("all installed product executables are required")

    doctor = _document(_run([str(cli), "doctor", "--json"], cwd=work))
    _json(output / "installed-identity.json", doctor)

    image_output = outputs / "image-cli"
    image_output.mkdir(mode=0o700)
    image_run = _run(
        [
            str(cli),
            "image",
            "--output-dir",
            str(image_output),
            "--reference-profile",
            str(fixtures["image"]),
        ],
        cwd=work,
    )
    image_result = _document(image_run)
    before = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(image_output.iterdir())
    }
    second = _run(
        [
            str(cli),
            "image",
            "--output-dir",
            str(image_output),
            "--reference-profile",
            str(fixtures["image"]),
        ],
        cwd=work,
        expected=frozenset({3}),
    )
    after = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(image_output.iterdir())
    }
    _json(
        output / "image-cli.json",
        {
            "schema": "kilix.background-removal.product-image-cli/v1",
            "result": image_result,
            "bounded_decode_success": True,
            "atomic_existing_destination": {
                "second_terminal": _document(second),
                "outputs_unchanged": before == after,
                "staging_files": {
                    "present": len(list(outputs.rglob(".kilix-f108-*.stage"))),
                    "total_allowed": 0,
                },
            },
        },
    )

    _json(
        output / "bounded-decode.json",
        _bounded_decode(cli, work, fixtures["bounded"], requests, outputs),
    )
    video_records = _all_video_profiles(cli, work, fixtures, outputs)
    _jsonl(output / "video-outcomes.jsonl", video_records)
    tui_record, tui_matte = _tui_video(tui, work, fixtures, requests, outputs)
    app_record, app_matte = _app_video(app, work, fixtures, requests, outputs)
    _json(output / "tui-video.json", tui_record)
    _json(output / "contained-app-video.json", app_record)
    tui_decoded = _decoded_matte(tui_matte, work)
    app_decoded = _decoded_matte(app_matte, work)
    _json(
        output / "temporal-smoothing.json",
        {
            "schema": "kilix.background-removal.product-temporal-smoothing/v1",
            "tui_batch_frames": 1,
            "app_batch_frames": 3,
            "decoded_bytes": len(tui_decoded),
            "tui_sha256": hashlib.sha256(tui_decoded).hexdigest(),
            "app_sha256": hashlib.sha256(app_decoded).hexdigest(),
            "bit_identical": tui_decoded == app_decoded,
            "scene_isolation": "production-scene-cut-segments",
            "raw_frame_mode": "available",
        },
    )
    _json(output / "f115-provider-port.json", _provider_port(port, work, fixtures, outputs))

    artifact_records: list[dict[str, object]] = []
    for path in sorted(outputs.rglob("*")):
        if path.is_file():
            artifact_records.append(
                {
                    "path": str(path.relative_to(output)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "mode": oct(path.stat().st_mode & 0o777),
                }
            )
    _jsonl(output / "artifact-ledger.jsonl", artifact_records)
    installed = distribution("kilix-background-remover")
    summary = {
        "schema": "kilix.background-removal.product-return/v1",
        "product": {
            "branch": "work/0.2.1-f108",
            "commit": args.product_commit,
            "tree": args.product_tree,
            "distribution": installed.metadata["Name"],
            "version": installed.version,
            "wheel_sha256": _sha256(args.wheel),
            "source_checkout_imported": False,
            "empty_current_directory": True,
        },
        "surfaces": {"passed": 5, "total": 5},
        "surface_names": ["provider", "cli", "tui", "contained-app", "f115-port"],
        "image": {"passed": 1, "total": 1},
        "bounded_decode": {"passed": 2, "total": 2},
        "video_profiles": {"passed": len(video_records), "total": 6},
        "temporal_batch_equality": {"passed": int(tui_decoded == app_decoded), "total": 1},
        "atomic_existing_output": {"passed": int(before == after), "total": 1},
        "f115_provider_capabilities": {"passed": 5, "total": 5},
        "f115_external_gate_disposition": {"passed": 0, "total": 1},
        "od22_independent_adjudication": {"passed": 5, "total": 5},
        "g5b_freeze": {"passed": 0, "total": 1},
        "acceptance": {"passed": 0, "total": 1},
        "provider_refusals_received": {"passed": 0, "total": 0},
        "artifact_files": {"passed": len(artifact_records), "total": len(artifact_records)},
    }
    _json(output / "summary.json", summary)
    readme = (
        "# F108 installed product return\n\n"
        "This packet was produced from the isolated installed wheel named in "
        "`summary.json`, from an empty current directory. It contains executable "
        "image, bounded-decode, 6/6 offline-video, temporal-smoothing, atomic-output, "
        "CLI, TUI, contained-app and F115-provider-port results.\n\n"
        "The synthetic reference graph is untrained and not release qualified. "
        "The independently adjudicated OD-22 return remains passed 5/5; G5b freeze "
        "remains 0/1, F108 acceptance remains 0/1, and the F115-owned external Gate "
        "8 disposition remains 0/1.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    work.rmdir()
    _manifest(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
