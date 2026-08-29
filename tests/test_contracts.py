from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
from conftest import ROOT

from kilix_background_remover.contract_v2 import (
    WIRE_TO_SCHEMA,
    ContractRefusal,
    ContractRuntime,
    canonical_bytes,
)
from kilix_background_remover.contracts import parse_request
from kilix_background_remover.errors import RemovalFailure
from kilix_background_remover.r5_return import generate_ledger
from kilix_background_remover.return_controls import (
    G5A_MANIFEST_SHA256,
    g5a_rejection_result,
    load_g5a_request,
)


def test_installed_candidate_r5_identity_and_inventory() -> None:
    runtime = ContractRuntime.load()
    assert runtime.lock.manifest_sha256 == (
        "803a5661a708b366b1d26884a4cf52d45c71dac58926e8216eb69aa902cbd25c"
    )
    assert runtime.lock.version == "0.2.1.dev5"
    assert len(runtime.manifest_entries) == 46
    assert len(runtime.documents) == 12
    assert len(WIRE_TO_SCHEMA) == 10
    assert all(wire.endswith("/v2") for wire in WIRE_TO_SCHEMA)


def test_independent_product_ledger_matches_complete_public_population(tmp_path: Path) -> None:
    summary = generate_ledger(tmp_path)
    population = summary["population"]
    assert isinstance(population, dict)
    assert population["matched"] == population["total"] == 168
    assert population["unique"] == population["total"] == 168
    assert (tmp_path / "outcomes.jsonl").read_bytes().count(b"\n") == 168


def test_historical_g5a_is_immutable_and_rejected_by_both_production_entries() -> None:
    authority = ROOT / "contracts" / "AUTHORITY-SHA256SUMS"
    assert hashlib.sha256(authority.read_bytes()).hexdigest() == G5A_MANIFEST_SHA256
    fixture = ROOT / "contracts" / "fixtures" / "valid" / "f108-reference-mask-lifecycle.json"
    result = g5a_rejection_result(load_g5a_request(fixture))
    assert result["production_rejections"] == {"matched": 2, "total": 2}
    assert result["negotiation_or_fallback_paths"] == {
        "observed": 0,
        "allowed": 0,
        "total": 1,
    }


def test_runtime_decoder_rejects_unknown_v2_field(request_factory: object, tmp_path: Path) -> None:
    request = request_factory(tmp_path)  # type: ignore[operator]
    mutated = copy.deepcopy(request)
    mutated["future_minor_field"] = True
    with pytest.raises(RemovalFailure, match="conditional R5") as caught:
        parse_request(mutated)
    assert isinstance(caught.value.__cause__, ContractRefusal)
    assert str(caught.value.__cause__) == "schema:C-SHAPE"


def test_runtime_accepts_canonical_v2_request(request_factory: object, tmp_path: Path) -> None:
    request = request_factory(tmp_path)  # type: ignore[operator]
    runtime = ContractRuntime.load()
    assert runtime.accept_wire(canonical_bytes(request)) == request
    assert parse_request(request).wire == request
