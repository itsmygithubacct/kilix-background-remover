"""Verify closure and causal claims in the installed F108 product packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

MANIFEST = "PRODUCT-SHA256SUMS"
PRODUCT_COMMIT = "d8c74998f8fd8b00ee7da34e3107fb42a21286ea"
PRODUCT_TREE = "6919b345534c8e81577066cc6adeb8dc67907cf2"
PRODUCT_WHEEL = "43726d2ccb907ef34bc7d8a3f390aefc7e727f1a1ac94cbb62886d8359233627"
CANDIDATE_MANIFEST = "803a5661a708b366b1d26884a4cf52d45c71dac58926e8216eb69aa902cbd25c"

EXPECTED_FILES = frozenset(
    {
        MANIFEST,
        "README.md",
        "artifact-ledger.jsonl",
        "bounded-decode.json",
        "contained-app-video.json",
        "f115-provider-port.json",
        "fixtures/background.mkv",
        "fixtures/background.png",
        "fixtures/bounded-over-cap.png",
        "fixtures/cancellation.png",
        "fixtures/source.mkv",
        "fixtures/subject.png",
        "image-cli.json",
        "installed-identity.json",
        "outputs/app-matte.media",
        "outputs/cli-composite-image.media",
        "outputs/cli-composite-video.media",
        "outputs/cli-gif.media",
        "outputs/cli-matte.media",
        "outputs/cli-transparent-mov.media",
        "outputs/cli-transparent-webm.media",
        "outputs/f115-pane4.mask.png",
        "outputs/image-cli/subject-png-103ab4c01ac9f831.cutout.png",
        "outputs/image-cli/subject-png-103ab4c01ac9f831.mask.png",
        "outputs/tui-matte.media",
        "requests/app-video.json",
        "requests/tui-video.json",
        "summary.json",
        "temporal-smoothing.json",
        "tui-video.json",
        "video-outcomes.jsonl",
    }
)

VIDEO_PROFILES: Mapping[str, tuple[str, bool, bool]] = {
    "transparent-mov": ("prores", True, True),
    "transparent-webm": ("vp9", True, True),
    "matte": ("ffv1", True, False),
    "composite-image": ("ffv1", True, False),
    "composite-video": ("ffv1", True, False),
    # GIF transparency is verified by decoding the authoritative alpha plane;
    # pal8 does not advertise a separate alpha plane through ffprobe.
    "gif": ("gif", False, False),
}


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"FAIL F108 installed product packet: {message}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} is not an object")
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        _require(isinstance(value, dict), f"{path.name} line {number} is not an object")
        records.append(value)
    return records


def _count(value: object, passed: int, total: int, label: str) -> None:
    _require(value == {"passed": passed, "total": total}, f"{label} count differs")


def _verify_manifest(root: Path) -> None:
    pattern = re.compile(r"([0-9a-f]{64})  ([^\n]+)")
    entries: dict[str, str] = {}
    for line in (root / MANIFEST).read_text(encoding="ascii").splitlines():
        match = pattern.fullmatch(line)
        _require(match is not None, "checksum manifest has an invalid row")
        digest, relative = match.groups()
        _require(relative not in entries, "checksum manifest has a duplicate path")
        entries[relative] = digest
    _require(set(entries) == EXPECTED_FILES - {MANIFEST}, "checksum closure differs")
    for relative, digest in entries.items():
        _require(_sha256(root / relative) == digest, f"checksum differs for {relative}")


def _verify_summary(root: Path) -> None:
    summary = _object(root / "summary.json")
    product = summary.get("product")
    _require(isinstance(product, dict), "product identity is absent")
    _require(product.get("commit") == PRODUCT_COMMIT, "product commit differs")
    _require(product.get("tree") == PRODUCT_TREE, "product tree differs")
    _require(product.get("wheel_sha256") == PRODUCT_WHEEL, "product wheel differs")
    _require(product.get("source_checkout_imported") is False, "checkout was imported")
    _require(product.get("empty_current_directory") is True, "current directory was not empty")
    _count(summary.get("surfaces"), 5, 5, "surface")
    _count(summary.get("image"), 1, 1, "image")
    _count(summary.get("bounded_decode"), 2, 2, "bounded decode")
    _count(summary.get("video_profiles"), 6, 6, "video profile")
    _count(summary.get("temporal_batch_equality"), 1, 1, "temporal equality")
    _count(summary.get("atomic_existing_output"), 1, 1, "atomic output")
    _count(summary.get("f115_provider_capabilities"), 5, 5, "F115 capability")
    _count(summary.get("f115_external_gate_disposition"), 0, 1, "F115 external gate")
    _count(summary.get("od22_independent_adjudication"), 5, 5, "OD-22 adjudication")
    _count(
        summary.get("od22_successor_product_byte_coverage"),
        0,
        1,
        "OD-22 successor coverage",
    )
    _count(summary.get("g5b_freeze"), 0, 1, "G5b freeze")
    _count(summary.get("acceptance"), 0, 1, "acceptance")
    _count(summary.get("provider_refusals_received"), 0, 0, "provider refusal")
    _count(summary.get("artifact_files"), 11, 11, "artifact")


def _verify_video(root: Path) -> None:
    records = _records(root / "video-outcomes.jsonl")
    by_kind = {str(record.get("kind")): record for record in records}
    _require(len(records) == len(by_kind) == 6, "video outcomes are not unique 6/6")
    _require(set(by_kind) == set(VIDEO_PROFILES), "video profile names differ")
    for kind, (codec, audio, advertised_alpha) in VIDEO_PROFILES.items():
        record = by_kind[kind]
        probe = record.get("container_probe")
        estimate = record.get("estimate")
        result = record.get("result")
        _require(record.get("surface") == "cli", f"{kind} did not use the CLI")
        _require(isinstance(probe, dict), f"{kind} probe is absent")
        _require(isinstance(estimate, dict), f"{kind} estimate is absent")
        _require(isinstance(result, dict), f"{kind} result is absent")
        _require(probe.get("codec") == codec, f"{kind} codec differs")
        _require(probe.get("audio") is audio, f"{kind} audio disposition differs")
        _require(probe.get("has_alpha") is advertised_alpha, f"{kind} alpha tag differs")
        _require(
            probe.get("frame_count") == result.get("frame_count") == 3,
            f"{kind} frame count differs",
        )
        _require(
            result.get("kind") == estimate.get("output_kind") == kind, f"{kind} result differs"
        )
        if kind == "gif":
            _require(estimate.get("gif_hard_edge_disclosure") is True, "GIF disclosure is absent")
            _require(result.get("gif_alpha_threshold_u8") == 128, "GIF threshold differs")


def _verify_bounded_atomic_temporal(root: Path) -> None:
    bounded = _object(root / "bounded-decode.json")
    fixture = bounded.get("fixture")
    terminal = bounded.get("terminal")
    _require(
        isinstance(fixture, dict) and isinstance(terminal, dict), "bounded proof is incomplete"
    )
    _require(fixture.get("decoded_pixels") == 100_010_000, "bounded geometry differs")
    _require(bounded.get("configured_max_decoded_pixels") == 100_000_000, "decode cap differs")
    _require(bounded.get("typed_refusal") is True, "decode refusal is untyped")
    _require(bounded.get("destination_present") is False, "decode refusal published output")
    error = terminal.get("error")
    _require(isinstance(error, dict), "bounded terminal error is absent")
    job = error.get("job")
    _require(
        isinstance(job, dict) and job.get("code") == "background.input-limit",
        "bounded error code differs",
    )

    image = _object(root / "image-cli.json")
    atomic = image.get("atomic_existing_destination")
    _require(isinstance(atomic, dict), "atomic image proof is absent")
    _require(atomic.get("outputs_unchanged") is True, "existing image output changed")
    stages = atomic.get("staging_files")
    _require(stages == {"present": 0, "total_allowed": 0}, "image staging residue exists")
    second = atomic.get("second_terminal")
    _require(
        isinstance(second, dict) and second.get("result") is None, "second image run committed"
    )

    temporal = _object(root / "temporal-smoothing.json")
    _require(temporal.get("tui_batch_frames") == 1, "single-pass batch differs")
    _require(temporal.get("app_batch_frames") == 3, "batched run differs")
    _require(temporal.get("bit_identical") is True, "temporal batches differ")
    _require(temporal.get("tui_sha256") == temporal.get("app_sha256"), "temporal digests differ")


def _verify_surfaces(root: Path) -> None:
    tui = _object(root / "tui-video.json")
    _require(tui.get("keyboard_contract") == ["q", "Escape", "r"], "TUI keys differ")
    _require(tui.get("stderr_has_provider_header") is True, "TUI provider identity is absent")
    _require(
        tui.get("estimate", {}).get("status") == "confirmation-required", "TUI estimate differs"
    )
    _require(tui.get("result", {}).get("status") == "committed", "TUI result differs")

    app = _object(root / "contained-app-video.json")
    _require(
        app.get("graphical_entry_point") == "kilix-background-remover-app", "app entry differs"
    )
    _require(app.get("headless_contained_lifecycle") is True, "app lifecycle proof is absent")
    _require(app.get("estimate", {}).get("operation") == "estimate-video", "app estimate differs")
    _require(app.get("result", {}).get("operation") == "run-video", "app result differs")

    port = _object(root / "f115-provider-port.json")
    _count(port.get("capabilities"), 5, 5, "provider-port capability")
    _require(port.get("provider_process_external") is True, "provider port was not external")
    editable = port.get("editable_mask")
    cancellation = port.get("cancellation")
    identity = port.get("identity")
    _require(isinstance(editable, dict), "editable-mask proof is absent")
    _require(editable.get("imported") is True, "editable mask was not imported")
    _require(editable.get("source_unchanged") is True, "editable-mask import changed source pixels")
    _count(editable.get("progress_messages"), 10, 10, "editable-mask progress")
    _require(isinstance(cancellation, dict), "cancellation proof is absent")
    _require(cancellation.get("destination_present") is False, "cancelled output exists")
    _require(cancellation.get("terminal_state") == "cancelled", "cancel terminal differs")
    outcome = cancellation.get("outcome")
    _require(
        isinstance(outcome, dict) and outcome.get("outcome") == "accepted",
        "cancel was not accepted",
    )
    _require(
        isinstance(identity, dict) and identity.get("transport") == "length-framed-stdio",
        "port transport differs",
    )
    provider = identity.get("provider")
    _require(isinstance(provider, dict), "provider identity is absent")
    _require(
        provider.get("candidate_manifest_sha256") == CANDIDATE_MANIFEST, "candidate pin differs"
    )
    _require(provider.get("release_qualified") is False, "reference provider claims qualification")


def _verify_artifacts(root: Path) -> None:
    records = _records(root / "artifact-ledger.jsonl")
    expected = {
        str(path.relative_to(root)) for path in (root / "outputs").rglob("*") if path.is_file()
    }
    by_path = {str(record.get("path")): record for record in records}
    _require(len(records) == len(by_path) == 11, "artifact ledger is not unique 11/11")
    _require(set(by_path) == expected, "artifact ledger closure differs")
    for relative, record in by_path.items():
        path = root / relative
        _require(record.get("sha256") == _sha256(path), f"artifact digest differs for {relative}")
        _require(
            record.get("bytes") == path.stat().st_size, f"artifact size differs for {relative}"
        )
        _require(record.get("mode") == "0o600", f"artifact mode differs for {relative}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    root = _parser().parse_args(argv).packet.absolute()
    _require(root.is_dir(), "packet directory is absent")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    _require(actual == EXPECTED_FILES, "packet file closure differs")
    _verify_manifest(root)
    _verify_summary(root)
    _verify_video(root)
    _verify_bounded_atomic_temporal(root)
    _verify_surfaces(root)
    _verify_artifacts(root)
    print(
        "PASS F108 installed product packet "
        "files=31/31 checksums=30/30 artifacts=11/11 surfaces=5/5 "
        "video=6/6 bounded=2/2 temporal=1/1 atomic=1/1 f115=5/5 "
        "od22-parent=5/5 od22-successor-coverage=0/1 g5b=0/1 acceptance=0/1 "
        "provider-refusals=0/0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
