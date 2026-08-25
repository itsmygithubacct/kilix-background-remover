from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from conftest import ROOT, assert_valid_message
from jsonschema import Draft202012Validator

from kilix_background_remover.contracts import parse_request
from kilix_background_remover.errors import RemovalFailure

EXPECTED_AUTHORITY_MANIFEST = "7fedfa2d504fb4e27a538db54fcfcaaca33e90972748cb04b1592eed0c68f846"


def _verify_manifest(root: Path, manifest: Path) -> None:
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        candidate = root / relative
        assert candidate.is_file(), relative
        assert hashlib.sha256(candidate.read_bytes()).hexdigest() == digest


def _semantic_lifecycle_errors(messages: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    request = messages[0]
    request_job = request["job"]
    assert isinstance(request_job, dict)
    last_sequence = -1
    last_progress = -1.0
    states = {
        None: {"queued"},
        "queued": {"cancelled", "failed", "loading", "queued", "running"},
        "loading": {"cancelled", "failed", "loading", "running"},
        "running": {"cancelled", "encoding", "failed", "running"},
        "encoding": {"cancelled", "committed", "encoding", "failed"},
    }
    state: str | None = None
    for message in messages[1:]:
        if message["schema"] == "kilix.media-job.cancel/v1":
            continue
        job = message["job"]
        assert isinstance(job, dict)
        sequence = job["sequence"]
        assert isinstance(sequence, int)
        if sequence <= last_sequence:
            errors.append("sequence")
        last_sequence = sequence
        next_state = job["state"]
        assert isinstance(next_state, str)
        if next_state not in states.get(state, {state}):
            errors.append("transition")
        state = next_state
        if message["schema"] == "kilix.background-removal.progress/v1":
            progress = job["progress"]
            assert isinstance(progress, int | float)
            if progress < last_progress:
                errors.append("progress")
            last_progress = float(progress)
        if message["schema"] == "kilix.background-removal.result/v1":
            source = request["input"]
            model = request["model"]
            assert isinstance(source, dict)
            if message["model"] != model:
                errors.append("model")
            expected_source = {
                "sha256": source["sha256"],
                "width": source["width"],
                "height": source["height"],
            }
            if message["source"] != expected_source:
                errors.append("source")
            mask = message["mask"]
            assert isinstance(mask, dict)
            if (mask["width"], mask["height"]) != (source["width"], source["height"]):
                errors.append("mask-geometry")
    return errors


def _mutated_messages(fixture: dict[str, object]) -> list[dict[str, object]]:
    base_name = fixture["base"]
    assert isinstance(base_name, str)
    base_path = ROOT / "contracts" / "fixtures" / "valid" / base_name
    lifecycle = json.loads(base_path.read_text(encoding="utf-8"))
    mutation = fixture["mutation"]
    assert isinstance(mutation, dict)
    pointer = mutation["path"]
    assert isinstance(pointer, str)
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]
    target: object = lifecycle
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]  # type: ignore[index]
    final = parts[-1]
    if isinstance(target, list):
        target[int(final)] = mutation["value"]
    else:
        assert isinstance(target, dict)
        target[final] = mutation["value"]
    return lifecycle["messages"]


def test_frozen_contract_bytes_and_manifests() -> None:
    contract_root = ROOT / "contracts"
    authority = contract_root / "AUTHORITY-SHA256SUMS"
    assert hashlib.sha256(authority.read_bytes()).hexdigest() == EXPECTED_AUTHORITY_MANIFEST
    _verify_manifest(contract_root, contract_root / "SHA256SUMS")
    authority_records = {
        relative: digest
        for digest, relative in (
            line.split("  ", 1) for line in authority.read_text(encoding="utf-8").splitlines()
        )
    }
    for line in (contract_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert authority_records[relative] == digest


def test_valid_authority_lifecycles(
    validators: dict[str, Draft202012Validator],
) -> None:
    for path in sorted((ROOT / "contracts" / "fixtures" / "valid").glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        messages = fixture["messages"]
        for message in messages:
            assert_valid_message(validators, message)
        assert not _semantic_lifecycle_errors(messages), path.name


def test_every_invalid_authority_lifecycle_is_rejected(
    validators: dict[str, Draft202012Validator],
) -> None:
    for path in sorted((ROOT / "contracts" / "fixtures" / "invalid").glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        messages = _mutated_messages(fixture)
        schema_errors = []
        for message in messages:
            identity = message.get("schema")
            if identity in validators:
                schema_errors.extend(validators[identity].iter_errors(message))
            else:
                schema_errors.append(identity)
        request_error = False
        try:
            parse_request(messages[0])
        except RemovalFailure:
            request_error = True
        assert schema_errors or request_error or _semantic_lifecycle_errors(messages), path.name


def test_runtime_decoder_rejects_unknown_minor_field(
    request_factory: object, tmp_path: Path
) -> None:
    request = request_factory(tmp_path)  # type: ignore[operator]
    mutated = copy.deepcopy(request)
    mutated["future_minor_field"] = True
    with pytest.raises(RemovalFailure, match="missing or unknown"):
        parse_request(mutated)
