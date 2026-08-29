"""Generate the conditional F108 outcome ledger for all public R5 cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

from .contract_v2 import (
    ContractRefusal,
    ContractRuntime,
    canonical_bytes,
    decode_gray8_png,
    ensure_closed_graph,
    load_closed_registry,
    mutable_documents,
    png_profile_accepted,
    process_foreground_plane,
    strict_decode,
    verify_manifest_tree,
)


@dataclass(frozen=True, slots=True)
class Expected:
    outcome: str
    stage: str | None = None
    rule_id: str | None = None


@dataclass(frozen=True, slots=True)
class Actual:
    outcome: str
    stage: str | None = None
    rule_id: str | None = None


def generate_ledger(output_directory: Path) -> dict[str, object]:
    runtime = ContractRuntime.load()
    outcomes: list[dict[str, object]] = []
    _run_transcripts(runtime, outcomes)
    _run_pixels(runtime, outcomes)
    _run_registry(runtime, outcomes)
    _run_manifests(runtime, outcomes)
    identifiers = [cast(str, item["case_id"]) for item in outcomes]
    if len(outcomes) != 168 or len(set(identifiers)) != 168:
        raise RuntimeError(
            f"F108 R5 population actual={len(outcomes)}/168 unique={len(set(identifiers))}/168"
        )
    if not all(item["matched"] is True for item in outcomes):
        failures = [item["case_id"] for item in outcomes if item["matched"] is not True]
        raise RuntimeError(f"F108 R5 outcome mismatches={len(failures)}/168: {failures}")
    categories = Counter(cast(str, item["category"]) for item in outcomes)
    actual_outcomes = Counter(
        cast(dict[str, object], item["actual"])["outcome"] for item in outcomes
    )
    expected_categories = {
        "transcript-accept": 23,
        "transcript-refuse": 81,
        "pixel-png": 41,
        "registry": 8,
        "manifest": 15,
    }
    if categories != expected_categories:
        raise RuntimeError(f"F108 R5 category population differs: {dict(categories)}")
    summary: dict[str, object] = {
        "schema": "kilix.f108.r5-return-summary/v1",
        "conditional": True,
        "candidate_status": "candidate-only-not-frozen-not-accepted",
        "contract_manifest_sha256": runtime.lock.manifest_sha256,
        "contract_manifest_bytes": runtime.lock.manifest_bytes,
        "registry_sha256": runtime.lock.registry_sha256,
        "carrier": {
            "distribution": runtime.lock.distribution,
            "version": runtime.lock.version,
            "wheel_sha256": runtime.lock.wheel_sha256,
            "sdist_sha256": runtime.lock.sdist_sha256,
        },
        "population": {
            "matched": 168,
            "total": 168,
            "unique": 168,
            "unique_total": 168,
            "transcript_accepts": categories["transcript-accept"],
            "transcript_accepts_total": 23,
            "transcript_refusals": categories["transcript-refuse"],
            "transcript_refusals_total": 81,
            "pixel_png": categories["pixel-png"],
            "pixel_png_total": 41,
            "registry": categories["registry"],
            "registry_total": 8,
            "manifest": categories["manifest"],
            "manifest_total": 15,
            "actual_accepts": actual_outcomes["accepted"],
            "actual_accepts_population": 168,
            "actual_refusals": actual_outcomes["refused"],
            "actual_refusals_population": 168,
        },
        "rerun_if_manifest_changes": True,
    }
    output_directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    ledger = b"".join(canonical_bytes(item) for item in outcomes)
    _atomic_write(output_directory / "outcomes.jsonl", ledger)
    summary["outcomes_sha256"] = hashlib.sha256(ledger).hexdigest()
    summary["outcomes_bytes"] = len(ledger)
    _atomic_write(output_directory / "summary.json", canonical_bytes(summary))
    return summary


def _run_transcripts(runtime: ContractRuntime, outcomes: list[dict[str, object]]) -> None:
    valid_root = runtime.root / "fixtures" / "valid"
    invalid_root = runtime.root / "fixtures" / "invalid"
    for path in sorted(valid_root.glob("*.json")):
        fixture = _document(path)
        messages = _fixture_messages(runtime, fixture)
        expected = _expected(cast(Mapping[str, Any], fixture["expected"]))
        _append_case(
            outcomes,
            category="transcript-accept" if expected.outcome == "accepted" else "transcript-refuse",
            case_id=f"valid/{path.name}::{fixture['case_id']}",
            source=f"fixtures/valid/{path.name}",
            expected=expected,
            action=partial(runtime.validate_transcript, messages),
        )
    for path in sorted(invalid_root.glob("*.json")):
        fixture = _document(path)
        if path.name.startswith("raw-"):
            _run_raw_fixture(path.name, fixture, outcomes)
        elif path.name == "diagnostic-cases.json":
            _run_diagnostics(runtime, path.name, fixture, outcomes)
        elif fixture["fixture_schema"] == "kilix.contract-mutation-set/v2":
            base = _load_named_fixture(runtime, cast(str, fixture["base"]))
            for untyped_case in cast(list[object], fixture["cases"]):
                case = cast(dict[str, Any], untyped_case)
                messages = _pointer_set(base, case["pointer"], case["value"])
                expected = Expected("refused", case["stage"], case["rule_id"])
                _append_case(
                    outcomes,
                    category="transcript-refuse",
                    case_id=f"invalid/{path.name}::{case['case_id']}",
                    source=f"fixtures/invalid/{path.name}",
                    expected=expected,
                    action=partial(runtime.validate_transcript, messages),
                )
        elif fixture["fixture_schema"] == "kilix.contract-case-set/v2":
            for untyped_case in cast(list[object], fixture["cases"]):
                case = cast(dict[str, Any], untyped_case)
                base_name = cast(str, case.get("base", fixture["base"]))
                messages = _apply_operations(
                    runtime,
                    _load_named_fixture(runtime, base_name),
                    cast(list[dict[str, Any]], case["operations"]),
                )
                expected = _expected(cast(Mapping[str, Any], case["expected"]))
                _append_case(
                    outcomes,
                    category=(
                        "transcript-accept"
                        if expected.outcome == "accepted"
                        else "transcript-refuse"
                    ),
                    case_id=f"invalid/{path.name}::{case['case_id']}",
                    source=f"fixtures/invalid/{path.name}",
                    expected=expected,
                    action=partial(runtime.validate_transcript, messages),
                )
        else:
            raise RuntimeError(f"unknown public fixture family: {path.name}")


def _run_raw_fixture(
    name: str, fixture: Mapping[str, Any], outcomes: list[dict[str, object]]
) -> None:
    base = canonical_bytes(fixture["document"])
    for untyped_case in cast(list[object], fixture["cases"]):
        case = cast(dict[str, Any], untyped_case)
        wire = _raw_mutation(cast(str, case["mutation"]), base)
        expected = Expected(
            cast(str, case["outcome"]),
            cast(str | None, case.get("stage")),
            cast(str | None, case.get("rule_id")),
        )
        _append_case(
            outcomes,
            category=(
                "transcript-accept" if expected.outcome == "accepted" else "transcript-refuse"
            ),
            case_id=f"invalid/{name}::{case['case_id']}",
            source=f"fixtures/invalid/{name}",
            expected=expected,
            action=partial(strict_decode, wire),
        )


def _run_diagnostics(
    runtime: ContractRuntime,
    name: str,
    fixture: Mapping[str, Any],
    outcomes: list[dict[str, object]],
) -> None:
    request_id = "018f6f65-7c7d-7a8b-8c9d-0123456789ab"
    for untyped_case in cast(list[object], fixture["valid_errors"]):
        case = cast(dict[str, Any], untyped_case)
        message = _diagnostic_message(request_id, case)
        _append_case(
            outcomes,
            category="transcript-accept",
            case_id=f"invalid/{name}::valid-error::{case['code']}",
            source=f"fixtures/invalid/{name}",
            expected=Expected("accepted"),
            action=partial(runtime.validate_message, message),
        )
    for code in cast(list[str], fixture["valid_warnings"]):
        message = _result_with_warning(request_id, {"code": code})
        _append_case(
            outcomes,
            category="transcript-accept",
            case_id=f"invalid/{name}::valid-warning::{code}",
            source=f"fixtures/invalid/{name}",
            expected=Expected("accepted"),
            action=partial(runtime.validate_message, message),
        )
    for untyped_case in cast(list[object], fixture["invalid_errors"]):
        case = cast(dict[str, Any], untyped_case)
        message = _diagnostic_message(request_id, case)
        message.update(cast(dict[str, Any], case.get("extra", {})))
        _append_case(
            outcomes,
            category="transcript-refuse",
            case_id=f"invalid/{name}::invalid-error::{case['case_id']}",
            source=f"fixtures/invalid/{name}",
            expected=Expected("refused", "schema", "C-SHAPE"),
            action=partial(runtime.validate_message, message),
        )
    for untyped_case in cast(list[object], fixture["invalid_warnings"]):
        case = cast(dict[str, Any], untyped_case)
        message = _result_with_warning(request_id, cast(dict[str, Any], case["warning"]))
        _append_case(
            outcomes,
            category="transcript-refuse",
            case_id=f"invalid/{name}::invalid-warning::{case['case_id']}",
            source=f"fixtures/invalid/{name}",
            expected=Expected("refused", "schema", "C-SHAPE"),
            action=partial(runtime.validate_message, message),
        )


def _run_pixels(runtime: ContractRuntime, outcomes: list[dict[str, object]]) -> None:
    pixel_root = runtime.root / "fixtures" / "pixels"
    for filename in ("foreground-alpha-v2.json", "edge-r4-v2.json"):
        fixture = _document(pixel_root / filename)
        for untyped_case in cast(list[object], fixture["vectors"]):
            case = cast(dict[str, Any], untyped_case)
            expected = (
                Expected("refused", "pixel", cast(str, case["refusal"]))
                if "refusal" in case
                else Expected("accepted")
            )

            def vector_action(case: dict[str, Any] = case) -> None:
                settings = {
                    "feather_radius_px": case["feather_radius_px"],
                    "matting_mode": case["matting_mode"],
                    "preserve_source_alpha": case["preserve_source_alpha"],
                    "threshold_u8": case["threshold_u8"],
                }
                if canonical_bytes(settings) != canonical_bytes(case["effective_settings"]):
                    raise ContractRefusal("pixel", "PX-EFFECTIVE-SETTINGS")
                if "expected_window_samples" in case:
                    samples = (2 * case["feather_radius_px"] + 1) ** 2
                    if (
                        samples != case["expected_window_samples"]
                        or samples * 255 != case["expected_max_accumulator"]
                    ):
                        raise ContractRefusal("pixel", "PX-ACCUMULATOR")
                actual = process_foreground_plane(case)
                if actual != case.get("expected"):
                    raise ContractRefusal("pixel", "PX-PIXELS")

            _append_case(
                outcomes,
                category="pixel-png",
                case_id=f"pixels::{case['case_id']}",
                source=f"fixtures/pixels/{filename}",
                expected=expected,
                action=vector_action,
            )
        if filename == "foreground-alpha-v2.json":
            for untyped_case in cast(list[object], fixture["png_profiles"]):
                case = cast(dict[str, Any], untyped_case)
                accepted = cast(bool, case["accepted"])

                def profile_action(case: dict[str, Any] = case) -> None:
                    if not png_profile_accepted(case):
                        raise ContractRefusal("pixel", "PX-PNG-PROFILE")

                _append_case(
                    outcomes,
                    category="pixel-png",
                    case_id=f"pixels::{case['case_id']}",
                    source=f"fixtures/pixels/{filename}",
                    expected=(
                        Expected("accepted")
                        if accepted
                        else Expected("refused", "pixel", "PX-PNG-PROFILE")
                    ),
                    action=profile_action,
                )
    png_fixture = _document(pixel_root / "png-byte-cases-v2.json")
    for untyped_case in cast(list[object], png_fixture["cases"]):
        case = cast(dict[str, Any], untyped_case)
        expected = Expected(
            cast(str, case["outcome"]),
            cast(str | None, case.get("stage")),
            cast(str | None, case.get("rule_id")),
        )

        def png_action(case: dict[str, Any] = case) -> None:
            payload = (runtime.root / "fixtures" / cast(str, case["path"])).read_bytes()
            if hashlib.sha256(payload).hexdigest() != case["sha256"]:
                raise ContractRefusal("pixel", "PX-PNG-DIGEST")
            width, height, pixels = decode_gray8_png(payload)
            expected_pixels = case["expected"]
            if (width, height) != (expected_pixels["width"], expected_pixels["height"]):
                raise ContractRefusal("pixel", "PX-GEOMETRY")
            if pixels != expected_pixels["pixels"]:
                raise ContractRefusal("pixel", "PX-PIXELS")

        _append_case(
            outcomes,
            category="pixel-png",
            case_id=f"pixels::{case['case_id']}",
            source="fixtures/pixels/png-byte-cases-v2.json",
            expected=expected,
            action=png_action,
        )


def _run_registry(runtime: ContractRuntime, outcomes: list[dict[str, object]]) -> None:
    path = runtime.root / "fixtures" / "registry" / "registry-controls-v2.json"
    fixture = _document(path)
    for untyped_case in cast(list[object], fixture["cases"]):
        case = cast(dict[str, Any], untyped_case)
        expected = Expected(
            cast(str, case["outcome"]),
            cast(str | None, case.get("stage")),
            cast(str | None, case.get("rule_id")),
        )

        def registry_action(mutation: str = cast(str, case["mutation"])) -> None:
            if mutation == "format-disabled":
                load_closed_registry(runtime.root, None)
            elif mutation == "unknown-resource":
                try:
                    runtime.registry.get_or_retrieve("urn:kilix:schema:missing:v2")
                except Exception as exc:
                    raise ContractRefusal("schema", "C-REF-CLOSED") from exc
                raise RuntimeError("unknown registry identity resolved")
            elif mutation == "none":
                identities = list(runtime.documents)
                if len(identities) != 12 or len(set(identities)) != 12:
                    raise ContractRefusal("schema", "C-REGISTRY-ID")
                serialized = [json.dumps(item) for item in runtime.documents.values()]
                if any("/v1" in item or ":v1" in item for item in serialized):
                    raise ContractRefusal("schema", "C-REF-CLOSED")
            elif mutation == "duplicate-registry-key":
                strict_decode(
                    b'{"registry_schema":"kilix.contract-registry/v2","resources":{},'
                    b'"resources":{}}\n'
                )
            else:
                changed = mutable_documents(runtime)
                target = changed["urn:kilix:schema:kilix.media-job.request:v2"]["properties"][
                    "limits"
                ]
                references = {
                    "unresolved-reference": "urn:kilix:schema:missing:v2#/$defs/limits",
                    "remote-reference": "https://example.invalid/remote.schema.json",
                    "wrong-version-reference": (
                        "urn:kilix:schema:kilix.media-job.types:v1#/$defs/limits"
                    ),
                    "cyclic-reference": ("urn:kilix:schema:kilix.background-removal.request:v2"),
                }
                target["$ref"] = references[mutation]
                ensure_closed_graph(changed)

        _append_case(
            outcomes,
            category="registry",
            case_id=f"registry::{case['case_id']}",
            source="fixtures/registry/registry-controls-v2.json",
            expected=expected,
            action=registry_action,
        )


def _run_manifests(runtime: ContractRuntime, outcomes: list[dict[str, object]]) -> None:
    fixture = _document(runtime.root / "fixtures" / "manifest" / "manifest-controls-v2.json")
    for untyped_case in cast(list[object], fixture["cases"]):
        case = cast(dict[str, Any], untyped_case)
        expected = Expected(
            cast(str, case["outcome"]),
            cast(str | None, case.get("stage")),
            cast(str | None, case.get("rule_id")),
        )

        def manifest_action(mutation: str = cast(str, case["mutation"])) -> None:
            with tempfile.TemporaryDirectory(prefix="f108-r5-manifest-") as temporary:
                root = Path(temporary) / "contract-v2"
                shutil.copytree(runtime.root, root, symlinks=True)
                manifest = root / "SHA256SUMS"
                _mutate_manifest_tree(mutation, root, manifest)
                verify_manifest_tree(root, manifest)

        _append_case(
            outcomes,
            category="manifest",
            case_id=f"manifest::{case['case_id']}",
            source="fixtures/manifest/manifest-controls-v2.json",
            expected=expected,
            action=manifest_action,
        )


def _append_case(
    outcomes: list[dict[str, object]],
    *,
    category: str,
    case_id: str,
    source: str,
    expected: Expected,
    action: Callable[[], object],
) -> None:
    try:
        action()
    except ContractRefusal as refusal:
        actual = Actual("refused", refusal.stage, refusal.rule_id)
    else:
        actual = Actual("accepted")
    matched = actual.outcome == expected.outcome
    if expected.stage is not None:
        matched = matched and actual.stage == expected.stage
    if expected.rule_id is not None:
        matched = matched and actual.rule_id == expected.rule_id
    outcomes.append(
        {
            "schema": "kilix.f108.r5-case-outcome/v1",
            "case_id": case_id,
            "source": source,
            "category": category,
            "expected": _outcome_wire(expected),
            "actual": _outcome_wire(actual),
            "matched": matched,
        }
    )


def _outcome_wire(value: Expected | Actual) -> dict[str, object]:
    result: dict[str, object] = {"outcome": value.outcome}
    if value.stage is not None:
        result["stage"] = value.stage
    if value.rule_id is not None:
        result["rule_id"] = value.rule_id
    return result


def _document(path: Path) -> dict[str, Any]:
    value = strict_decode(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"public fixture is not an object: {path.name}")
    return cast(dict[str, Any], value)


def _expected(value: Mapping[str, Any]) -> Expected:
    return Expected(value["outcome"], value.get("stage"), value.get("rule_id"))


def _fixture_messages(runtime: ContractRuntime, fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    if fixture["fixture_schema"] == "kilix.contract-derived-transcript/v2":
        return _apply_operations(
            runtime,
            _load_named_fixture(runtime, cast(str, fixture["base"])),
            cast(list[dict[str, Any]], fixture["operations"]),
        )
    return deepcopy(cast(list[dict[str, Any]], fixture["messages"]))


def _load_named_fixture(runtime: ContractRuntime, name: str) -> list[dict[str, Any]]:
    return _fixture_messages(runtime, _document(runtime.root / "fixtures" / "valid" / name))


def _pointer_set(value: Any, pointer: str, replacement: Any) -> Any:
    result = deepcopy(value)
    target = result
    parts = pointer.strip("/").split("/") if pointer else []
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    final = parts[-1]
    if isinstance(target, list):
        target[int(final)] = replacement
    else:
        target[final] = replacement
    return result


def _pointer_delete(value: Any, pointer: str) -> Any:
    result = deepcopy(value)
    target = result
    parts = pointer.strip("/").split("/")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    final = parts[-1]
    if isinstance(target, list):
        del target[int(final)]
    else:
        del target[final]
    return result


def _apply_operations(
    runtime: ContractRuntime,
    messages: list[dict[str, Any]],
    operations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = deepcopy(messages)
    for operation in operations:
        kind = operation["op"]
        if kind == "set":
            result = _pointer_set(result, operation["pointer"], operation["value"])
        elif kind == "delete":
            result = _pointer_delete(result, operation["pointer"])
        elif kind == "insert":
            result.insert(operation["index"], deepcopy(operation["value"]))
        elif kind == "append":
            result.append(deepcopy(operation["value"]))
        elif kind in {"insert-clone", "append-clone"}:
            source = _load_named_fixture(runtime, operation["fixture"])[operation["message_index"]]
            if kind == "insert-clone":
                result.insert(operation["index"], source)
            else:
                result.append(source)
        else:
            raise RuntimeError(f"unknown public fixture operation: {kind}")
    return result


def _raw_mutation(name: str, canonical: bytes) -> bytes:
    mutations = {
        "missing-lf": canonical[:-1],
        "extra-lf": canonical + b"\n",
        "leading-space": b" " + canonical,
        "alternate-order": b'{"z":0,"a":1}\n',
        "duplicate-top": b'{"a":1,"a":1}\n',
        "duplicate-nested": b'{"a":{"b":1,"b":1}}\n',
        "duplicate-third-depth": b'{"a":{"b":{"c":1,"c":1}}}\n',
        "nan": b'{"a":NaN}\n',
        "infinity": b'{"a":Infinity}\n',
        "negative-infinity": b'{"a":-Infinity}\n',
        "negative-zero": b'{"a":-0}\n',
        "one-point-zero": b'{"a":1.0}\n',
        "exponent-one": b'{"a":1e0}\n',
        "escaped-unicode": b'{"a":"\\u0061"}\n',
        "literal-unicode": '{"a":"é"}\n'.encode(),
        "lone-surrogate": b'{"a":"\\ud800"}\n',
        "comment": b'{"a":1/*comment*/}\n',
        "bom": b"\xef\xbb\xbf" + canonical,
        "invalid-utf8": b'{"a":"\xff"}\n',
        "trailing-json": canonical + b"{}",
        "two-to-53": b'{"a":9007199254740992}\n',
        "largest-uint32": b'{"a":4294967295}\n',
    }
    try:
        return mutations[name]
    except KeyError as exc:
        raise RuntimeError(f"unknown public raw mutation: {name}") from exc


def _diagnostic_message(request_id: str, case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "category": case["category"],
        "code": case["code"],
        "diagnostic_reference": "diag-00000000000000000000000000000000",
        "occurred_at": "2026-08-29T00:00:00Z",
        "request_id": request_id,
        "retryable": case["retryable"],
        "schema": "kilix.media-job.error/v2",
        "sequence": 0,
        "state": case["state"],
    }


def _result_with_warning(request_id: str, warning: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "committed_at": "2026-08-29T00:00:00Z",
        "diagnostic_reference": "diag-00000000000000000000000000000000",
        "elapsed_ms": 0,
        "request_id": request_id,
        "schema": "kilix.media-job.result/v2",
        "sequence": 0,
        "state": "committed",
        "warnings": [dict(warning)],
    }


def _mutate_manifest_tree(mutation: str, root: Path, manifest: Path) -> None:
    target = root / "schemas" / "kilix.media-job.request-v2.schema.json"
    if mutation == "changed-byte":
        target.write_bytes(target.read_bytes() + b" ")
    elif mutation == "changed-byte-count":
        target.write_bytes(target.read_bytes()[:-1])
    elif mutation == "extra-file":
        (root / "EXTRA").write_bytes(b"extra\n")
    elif mutation == "missing-file":
        (root / "LICENSES" / "README.md").unlink()
    elif mutation == "renamed-resource":
        target.rename(target.with_name("renamed.schema.json"))
    elif mutation == "symlink":
        (root / "link").symlink_to("LICENSES/README.md")
    elif mutation == "special-file":
        os.mkfifo(root / "named-pipe")
    elif mutation == "duplicate-entry":
        first = manifest.read_bytes().splitlines()[0]
        manifest.write_bytes(manifest.read_bytes() + first + b"\n")
    elif mutation == "traversal-path":
        manifest.write_bytes(b"0" * 64 + b"  ../escape\n")
    elif mutation == "absolute-path":
        manifest.write_bytes(b"0" * 64 + b"  /escape\n")
    elif mutation == "dot-segment":
        manifest.write_bytes(b"0" * 64 + b"  schemas/./escape\n")
    elif mutation == "empty-segment":
        manifest.write_bytes(b"0" * 64 + b"  schemas//escape\n")
    elif mutation == "backslash":
        manifest.write_bytes(b"0" * 64 + b"  schemas\\escape\n")
    elif mutation == "control-character":
        manifest.write_bytes(b"0" * 64 + b"  bad\tname\n")
    elif mutation == "private-evidence-packaged":
        evidence = root / "evidence"
        evidence.mkdir()
        (evidence / "FREEZE-RECORD.md").write_bytes(b"private\n")
    else:
        raise RuntimeError(f"unknown public manifest mutation: {mutation}")


def _atomic_write(destination: Path, payload: bytes) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = generate_ledger(args.output_dir)
    population = cast(dict[str, object], summary["population"])
    print(
        "PASS F108 conditional R5 outcomes "
        f"matched={population['matched']}/{population['total']} "
        f"unique={population['unique']}/{population['total']} "
        f"manifest={summary['contract_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
