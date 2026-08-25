"""Length-bounded, command-free stdin/stdout bridge for the contained app."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .contracts import parse_request
from .errors import RemovalFailure
from .frontend import MAX_JSON_BYTES
from .worker import FALLBACK_REQUEST_ID, WorkerSupervisor, failure_wire

BRIDGE_REQUEST_SCHEMA = "kilix.background-removal.app-bridge-request/v1"
BRIDGE_RESPONSE_SCHEMA = "kilix.background-removal.app-bridge-response/v1"


def _decode(payload: bytes) -> object:
    if not payload or len(payload) > MAX_JSON_BYTES:
        raise ValueError("bridge payload is outside the fixed length bound")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate bridge field")
            result[key] = value
        return result

    return json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates)


def run_bridge_message(
    value: object,
    *,
    allow_reference_profile: bool,
    supervisor: WorkerSupervisor | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"operation", "request", "schema"}:
        raise ValueError("bridge message has missing or unknown fields")
    if value.get("schema") != BRIDGE_REQUEST_SCHEMA or value.get("operation") != "run":
        raise ValueError("bridge message identity or operation is unsupported")
    request = value["request"]
    parsed = parse_request(request)
    if not allow_reference_profile:
        failure = RemovalFailure(
            "background.profile-unavailable",
            "No release-qualified model profile is installed.",
            "provider",
            "resolve-profile",
        )
        return {
            "schema": BRIDGE_RESPONSE_SCHEMA,
            "progress": [],
            "result": None,
            "error": failure_wire(parsed.request_id, failure),
        }
    owns_supervisor = supervisor is None
    active = supervisor or WorkerSupervisor()
    try:
        outcome = active.run(request)
    finally:
        if owns_supervisor:
            active.close()
    return {
        "schema": BRIDGE_RESPONSE_SCHEMA,
        "progress": outcome.progress,
        "result": outcome.result,
        "error": outcome.error,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kilix-background-remover-app-bridge")
    parser.add_argument("--reference-profile", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = sys.stdin.buffer.read(MAX_JSON_BYTES + 1)
        response = run_bridge_message(
            _decode(payload), allow_reference_profile=args.reference_profile
        )
    except (UnicodeError, ValueError, json.JSONDecodeError, RemovalFailure):
        failure = RemovalFailure(
            "background.invalid-request",
            "The app bridge rejected an invalid local message.",
            "input",
            "accepted",
        )
        response = {
            "schema": BRIDGE_RESPONSE_SCHEMA,
            "progress": [],
            "result": None,
            "error": failure_wire(FALLBACK_REQUEST_ID, failure),
        }
    except Exception:
        failure = RemovalFailure(
            "background.internal",
            "The contained app bridge failed safely.",
            "internal",
            "accepted",
        )
        response = {
            "schema": BRIDGE_RESPONSE_SCHEMA,
            "progress": [],
            "result": None,
            "error": failure_wire(FALLBACK_REQUEST_ID, failure),
        }
    sys.stdout.write(json.dumps(response, indent=2, sort_keys=True) + "\n")
    return 0 if response["result"] is not None else 3
