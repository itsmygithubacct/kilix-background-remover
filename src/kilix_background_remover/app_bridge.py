"""Length-bounded, command-free stdin/stdout bridge for the contained app."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Sequence
from typing import Any

from .contracts import parse_request
from .errors import RemovalFailure
from .frontend import MAX_JSON_BYTES
from .provider import (
    BackgroundRemovalProvider,
    parse_video_request,
    video_estimate_wire,
    video_result_wire,
)
from .worker import FALLBACK_REQUEST_ID, WorkerSupervisor, failure_wire

BRIDGE_REQUEST_SCHEMA = "kilix.background-removal.app-bridge-request/v1"
BRIDGE_RESPONSE_SCHEMA = "kilix.background-removal.app-bridge-response/v1"
BRIDGE_REQUEST_SCHEMA_V2 = "kilix.background-removal.app-bridge-request/v2"
BRIDGE_RESPONSE_SCHEMA_V2 = "kilix.background-removal.app-bridge-response/v2"


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
    provider: BackgroundRemovalProvider | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"operation", "request", "schema"}:
        raise ValueError("bridge message has missing or unknown fields")
    schema = value.get("schema")
    operation = value.get("operation")
    if schema not in {BRIDGE_REQUEST_SCHEMA, BRIDGE_REQUEST_SCHEMA_V2}:
        raise ValueError("bridge message identity or operation is unsupported")
    if provider is not None and supervisor is not None:
        raise ValueError("the bridge accepts one provider authority")
    owns_provider = provider is None
    active = provider or BackgroundRemovalProvider(
        allow_reference_profile=allow_reference_profile,
        supervisor=supervisor,
    )
    try:
        if schema == BRIDGE_REQUEST_SCHEMA:
            if operation != "run":
                raise ValueError("bridge message identity or operation is unsupported")
            request = value["request"]
            parse_request(request)
            outcome = active.run(request, cancel=cancel)
            return {
                "schema": BRIDGE_RESPONSE_SCHEMA,
                "progress": outcome.progress,
                "result": outcome.result,
                "error": outcome.error,
            }
        if operation == "discover":
            if value["request"] is not None:
                raise ValueError("bridge discovery accepts no request")
            return {
                "schema": BRIDGE_RESPONSE_SCHEMA_V2,
                "operation": "discover",
                "progress": [],
                "result": active.identity,
                "error": None,
            }
        if operation == "run-image":
            request = value["request"]
            parse_request(request)
            outcome = active.run(request, cancel=cancel)
            return {
                "schema": BRIDGE_RESPONSE_SCHEMA_V2,
                "operation": "run-image",
                "progress": outcome.progress,
                "result": outcome.result,
                "error": outcome.error,
            }
        if operation not in {"estimate-video", "run-video"}:
            raise ValueError("bridge message identity or operation is unsupported")
        video_request = parse_video_request(value["request"])
        if operation == "estimate-video":
            _probe, estimate = active.estimate_video(video_request, cancel=cancel)
            return {
                "schema": BRIDGE_RESPONSE_SCHEMA_V2,
                "operation": "estimate-video",
                "progress": [],
                "result": video_estimate_wire(estimate),
                "error": None,
            }
        progress: list[dict[str, object]] = []

        def collect(phase: str, completed: int, total: int) -> None:
            progress.append(
                {
                    "schema": "kilix.background-removal.video-progress/v1",
                    "phase": phase,
                    "frames_completed": completed,
                    "frames_total": total,
                }
            )

        result = active.run_video(video_request, cancel=cancel, progress=collect)
        return {
            "schema": BRIDGE_RESPONSE_SCHEMA_V2,
            "operation": "run-video",
            "progress": progress,
            "result": video_result_wire(result),
            "error": None,
        }
    except RemovalFailure as failure:
        if schema == BRIDGE_REQUEST_SCHEMA:
            raise
        return {
            "schema": BRIDGE_RESPONSE_SCHEMA_V2,
            "operation": operation,
            "progress": [],
            "result": None,
            "error": {
                "schema": "kilix.background-removal.video-error/v1",
                "code": failure.code,
                "category": failure.category,
                "phase": failure.phase,
                "retryable": failure.retryable,
            },
        }
    finally:
        if owns_provider:
            active.close()


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
