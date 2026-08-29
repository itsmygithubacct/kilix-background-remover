"""Independent product controls for the OD-22 conditional R5 return."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import kilix_f108_f115_contracts

from .contract_v2 import (
    WIRE_TO_SCHEMA,
    ContractRefusal,
    ContractRuntime,
    canonical_bytes,
)
from .contracts import parse_request
from .errors import RemovalFailure

G5A_MANIFEST_SHA256 = "7fedfa2d504fb4e27a538db54fcfcaaca33e90972748cb04b1592eed0c68f846"
R5_SOURCE_MANIFEST_SHA256 = "5172ada83f789104b41be4f652c12e56fe2f84ec339e832639e78518852a71ec"
EXPECTED_PUBLIC_RESOURCES = 46
EXPECTED_SCHEMAS = 12
EXPECTED_WIRE_IDENTITIES = 10


def pins_result(candidate_source: Path) -> dict[str, object]:
    """Verify and return every candidate and immutable-carrier identity."""
    runtime = ContractRuntime.load()
    source_manifest = candidate_source / "SOURCE-SHA256SUMS"
    public_manifest = candidate_source / "contract-v2" / "SHA256SUMS"
    wheel = (
        candidate_source
        / "evidence"
        / "package-builds"
        / "build-1"
        / "kilix_f108_f115_contracts-0.2.1.dev5-py3-none-any.whl"
    )
    sdist = (
        candidate_source
        / "evidence"
        / "package-builds"
        / "build-1"
        / "kilix_f108_f115_contracts-0.2.1.dev5.tar.gz"
    )
    observed = {
        "source_manifest": _sha256(source_manifest),
        "public_manifest": _sha256(public_manifest),
        "wheel": _sha256(wheel),
        "sdist": _sha256(sdist),
    }
    expected = {
        "source_manifest": R5_SOURCE_MANIFEST_SHA256,
        "public_manifest": runtime.lock.manifest_sha256,
        "wheel": runtime.lock.wheel_sha256,
        "sdist": runtime.lock.sdist_sha256,
    }
    if observed != expected:
        raise RuntimeError(f"candidate R5 pin mismatch: observed={observed!r}")
    if len(runtime.manifest_entries) != EXPECTED_PUBLIC_RESOURCES:
        raise RuntimeError("installed R5 public-resource population differs")
    return {
        "schema": "kilix.f108.r5-pins/v1",
        "conditional": True,
        "candidate_status": "candidate-only-not-frozen-not-accepted",
        "rerun_if_manifest_changes": True,
        "source_manifest": {
            "bytes": source_manifest.stat().st_size,
            "sha256": observed["source_manifest"],
        },
        "public_manifest": {
            "bytes": runtime.lock.manifest_bytes,
            "sha256": observed["public_manifest"],
        },
        "registry_sha256": runtime.lock.registry_sha256,
        "carrier": {
            "distribution": runtime.lock.distribution,
            "version": runtime.lock.version,
            "wheel": {
                "bytes": wheel.stat().st_size,
                "sha256": observed["wheel"],
            },
            "sdist": {
                "bytes": sdist.stat().st_size,
                "sha256": observed["sdist"],
            },
        },
        "installed_public_resources": {
            "matched": EXPECTED_PUBLIC_RESOURCES,
            "total": EXPECTED_PUBLIC_RESOURCES,
        },
        "installed_schemas": {
            "matched": len(runtime.documents),
            "total": EXPECTED_SCHEMAS,
        },
    }


def installed_carrier_result(candidate_source: Path) -> dict[str, object]:
    """Prove resolution from installed packages in an isolated empty cwd."""
    runtime = ContractRuntime.load()
    product_distribution = metadata.distribution("kilix-background-remover")
    carrier_distribution = metadata.distribution(runtime.lock.distribution)
    product_install_root = Path(str(product_distribution.locate_file(""))).resolve()
    carrier_install_root = Path(str(carrier_distribution.locate_file(""))).resolve()
    product_module = Path(__file__).resolve()
    carrier_module_value = kilix_f108_f115_contracts.__file__
    if carrier_module_value is None:
        raise RuntimeError("installed carrier has no module origin")
    carrier_module = Path(carrier_module_value).resolve()
    contract_root = runtime.root.resolve()
    candidate_root = candidate_source.resolve()
    current_directory = Path.cwd().resolve()
    search_roots = _search_roots(current_directory)

    product_is_installed = product_module.is_relative_to(product_install_root)
    carrier_is_installed = carrier_module.is_relative_to(
        carrier_install_root
    ) and contract_root.is_relative_to(carrier_install_root)
    source_dependency = (
        contract_root.is_relative_to(candidate_root)
        or carrier_module.is_relative_to(candidate_root)
        or candidate_root in search_roots
    )
    current_directory_dependency = (
        current_directory in search_roots
        or product_module.is_relative_to(current_directory)
        or carrier_module.is_relative_to(current_directory)
    )
    current_directory_entries = list(current_directory.iterdir())
    distribution_files = carrier_distribution.files or []
    private_paths = [
        str(path)
        for path in distribution_files
        if "evidence" in {part.lower() for part in Path(str(path)).parts}
    ]
    public_manifest = contract_root / "SHA256SUMS"
    checks = {
        "isolated_python": sys.flags.isolated == 1,
        "empty_current_directory": not current_directory_entries,
        "product_is_installed": product_is_installed,
        "carrier_is_installed": carrier_is_installed,
        "candidate_source_tree_dependency": source_dependency,
        "current_directory_dependency": current_directory_dependency,
        "private_evidence_exposed": bool(private_paths),
        "public_manifest_present": public_manifest.is_file(),
    }
    required = {
        "isolated_python": True,
        "empty_current_directory": True,
        "product_is_installed": True,
        "carrier_is_installed": True,
        "candidate_source_tree_dependency": False,
        "current_directory_dependency": False,
        "private_evidence_exposed": False,
        "public_manifest_present": True,
    }
    if checks != required:
        raise RuntimeError(f"installed-carrier isolation control failed: {checks!r}")
    if len(runtime.manifest_entries) != EXPECTED_PUBLIC_RESOURCES:
        raise RuntimeError("installed public-resource population differs")
    if len(runtime.documents) != EXPECTED_SCHEMAS:
        raise RuntimeError("installed schema population differs")
    return {
        "schema": "kilix.f108.r5-installed-carrier/v1",
        "conditional": True,
        "contract_manifest_sha256": runtime.lock.manifest_sha256,
        "distribution": runtime.lock.distribution,
        "version": runtime.lock.version,
        "execution": {
            "isolated_python": {"matched": 1, "total": 1},
            "empty_current_directory": {"matched": 1, "total": 1},
            "installed_product": {"matched": 1, "total": 1},
            "installed_carrier": {"matched": 1, "total": 1},
        },
        "resolution_dependencies": {
            "dependencies_observed": 0,
            "dependencies_allowed": 0,
            "checks_passed": 2,
            "checks_total": 2,
            "candidate_source_tree": False,
            "current_directory": False,
        },
        "public_resources": {
            "matched": len(runtime.manifest_entries),
            "total": EXPECTED_PUBLIC_RESOURCES,
        },
        "schemas": {"matched": len(runtime.documents), "total": EXPECTED_SCHEMAS},
        "public_manifest": {"matched": 1, "total": 1},
        "private_evidence": {
            "paths_observed": 0,
            "paths_allowed": 0,
            "checks_passed": 1,
            "checks_total": 1,
        },
        "passed": True,
    }


def g5a_rejection_result(request: Mapping[str, Any]) -> dict[str, object]:
    """Exercise a historical G5a request through both production entry points."""
    runtime = ContractRuntime.load()
    encoded = canonical_bytes(dict(request))
    direct = _direct_refusal(runtime, encoded)
    parsed = _parse_refusal(request)
    production_entries = [direct, parsed]
    v1_dispatch_entries = [wire for wire in WIRE_TO_SCHEMA if wire.endswith("/v1")]
    if not all(entry["refusal"] == "schema:C-WIRE-IDENTITY" for entry in production_entries):
        raise RuntimeError(f"G5a production refusal differs: {production_entries!r}")
    if v1_dispatch_entries:
        raise RuntimeError(f"G5a identities remain in production dispatch: {v1_dispatch_entries!r}")
    return {
        "schema": "kilix.f108.g5a-production-rejection/v1",
        "conditional_r5_manifest_sha256": runtime.lock.manifest_sha256,
        "g5a_manifest_sha256": G5A_MANIFEST_SHA256,
        "g5a_request_schema": request.get("schema"),
        "production_entry_results": production_entries,
        "production_rejections": {"matched": 2, "total": 2},
        "v2_dispatch_identities": {
            "matched": len(WIRE_TO_SCHEMA),
            "total": EXPECTED_WIRE_IDENTITIES,
        },
        "v1_dispatch_identities": {"observed": 0, "total_dispatch_identities": 10},
        "negotiation_or_fallback_paths": {"observed": 0, "allowed": 0, "total": 1},
        "passed": True,
    }


def write_record(destination: Path, value: Mapping[str, object]) -> None:
    destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    payload = canonical_bytes(dict(value))
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


def load_g5a_request(fixture: Path) -> dict[str, Any]:
    """Read the digest-pinned historical fixture without creating a converter."""
    authority = fixture.parents[2] / "AUTHORITY-SHA256SUMS"
    if _sha256(authority) != G5A_MANIFEST_SHA256:
        raise RuntimeError("historical G5a authority manifest differs")
    try:
        value = json.loads(fixture.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("historical G5a fixture is unreadable") from exc
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise RuntimeError("historical G5a fixture has the wrong shape")
    messages = cast(list[object], value["messages"])
    if not messages or not isinstance(messages[0], dict):
        raise RuntimeError("historical G5a fixture has no request")
    return cast(dict[str, Any], messages[0])


def _direct_refusal(runtime: ContractRuntime, encoded: bytes) -> dict[str, object]:
    try:
        runtime.accept_wire(encoded)
    except ContractRefusal as exc:
        return {
            "entry": "ContractRuntime.accept_wire",
            "accepted": False,
            "refusal": f"{exc.stage}:{exc.rule_id}",
        }
    raise RuntimeError("ContractRuntime accepted a historical G5a request")


def _parse_refusal(request: Mapping[str, Any]) -> dict[str, object]:
    try:
        parse_request(dict(request))
    except RemovalFailure as exc:
        cause = exc.__cause__
        if not isinstance(cause, ContractRefusal):
            raise RuntimeError("production parser did not retain the contract refusal") from exc
        return {
            "entry": "contracts.parse_request",
            "accepted": False,
            "removal_failure_code": exc.code,
            "refusal": f"{cause.stage}:{cause.rule_id}",
        }
    raise RuntimeError("production parser accepted a historical G5a request")


def _search_roots(current_directory: Path) -> set[Path]:
    roots: set[Path] = set()
    for entry in sys.path:
        try:
            roots.add(Path(entry).resolve() if entry else current_directory)
        except OSError:
            continue
    return roots


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
