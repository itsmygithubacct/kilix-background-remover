"""One persistent, supervised ONNX worker for image, batch and UI consumers."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast

from PIL import Image

from .atomic import (
    StagedImage,
    cleanup_staging_files,
    commit_staged,
    discard_staged,
    stage_image,
)
from .cancellation_v2 import DurableCancellationGate, TerminalState
from .contract_v2 import ContractRefusal, canonical_bytes, strict_decode
from .contracts import RemovalRequest, parse_request, sha256_file
from .decode import decode_image
from .errors import RemovalFailure, diagnostic_reference
from .postprocess import (
    apply_edge_policy,
    render_color_composite,
    render_cutout,
    render_image_composite,
)

FALLBACK_REQUEST_ID = "00000000-0000-4000-8000-000000000000"
INFERENCE_STRIP_PIXELS = 1_048_576
ABORT_GRACE_SECONDS = 0.75
ERROR_POLICY = {
    "background.artifact-invalid": ("provider", False, "failed"),
    "background.backend-unavailable": ("provider", True, "failed"),
    "background.deadline": ("deadline", True, "failed"),
    "background.inference-failed": ("provider", True, "failed"),
    "background.input-limit": ("input", False, "failed"),
    "background.input-unreadable": ("input", False, "failed"),
    "background.internal": ("internal", False, "failed"),
    "background.invalid-request": ("input", False, "failed"),
    "background.output-failed": ("output", True, "failed"),
    "background.profile-unavailable": ("provider", True, "failed"),
    "job.cancelled": ("cancellation", False, "cancelled"),
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _default_cancellation_database() -> Path:
    base = Path(os.environ.get("XDG_RUNTIME_DIR", "/var/tmp"))
    root = base / f"kilix-background-remover-{os.getuid()}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = root.lstat()
    if root.is_symlink() or not root.is_dir() or status.st_uid != os.getuid():
        raise RuntimeError("the cancellation state directory is unsafe")
    return root / "cancellation-v2.sqlite3"


def _profile() -> tuple[dict[str, Any], Path]:
    package = resources.files("kilix_background_remover")
    profile = json.loads(package.joinpath("reference_profile.json").read_text(encoding="utf-8"))
    model = Path(str(package.joinpath("reference_luma.onnx")))
    return profile, model


def reference_identity() -> tuple[str, str]:
    profile, _ = _profile()
    return str(profile["profile_id"]), str(profile["artifact_sha256"])


def _check_cancel(cancel: Any, phase: str) -> None:
    if cancel.is_set():
        raise RemovalFailure(
            "job.cancelled",
            "The job was cancelled.",
            "cancellation",
            phase,
        )


def _progress(
    request_id: str,
    sequence: int,
    state: str,
    progress: float,
    phase: str,
    started: float,
) -> dict[str, object]:
    return {
        "schema": "kilix.background-removal.progress/v2",
        "job": {
            "schema": "kilix.media-job.progress/v2",
            "request_id": request_id,
            "sequence": sequence,
            "state": state,
            "progress": progress,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "estimated_remaining_ms": None,
            "updated_at": _now(),
        },
        "phase": phase,
    }


def _error_wire(request_id: str, sequence: int, failure: RemovalFailure) -> dict[str, object]:
    category, retryable, state = ERROR_POLICY.get(failure.code, ("internal", False, "failed"))
    return {
        "schema": "kilix.background-removal.error/v2",
        "job": {
            "schema": "kilix.media-job.error/v2",
            "request_id": request_id,
            "sequence": sequence,
            "state": state,
            "code": failure.code,
            "category": category,
            "retryable": retryable,
            "diagnostic_reference": diagnostic_reference(request_id, failure.code),
            "occurred_at": _now(),
        },
        "phase": failure.phase,
    }


def failure_wire(
    request_id: str, failure: RemovalFailure, *, sequence: int = 0
) -> dict[str, object]:
    """Build a frozen, path-free terminal error for local front ends."""

    return _error_wire(request_id, sequence, failure)


def _request_id(raw: object) -> str:
    if not isinstance(raw, dict):
        return FALLBACK_REQUEST_ID
    job = raw.get("job")
    if not isinstance(job, dict) or not isinstance(job.get("request_id"), str):
        return FALLBACK_REQUEST_ID
    candidate = cast(str, job["request_id"])
    try:
        parsed = uuid.UUID(candidate)
    except ValueError:
        return FALLBACK_REQUEST_ID
    if str(parsed) != candidate or parsed.variant != uuid.RFC_4122:
        return FALLBACK_REQUEST_ID
    return candidate


def _run_onnx_mask(
    session: Any, image: Image.Image, cancel: Any
) -> tuple[Image.Image, list[dict[str, str]]]:
    """Infer full-resolution output in bounded horizontal strips."""

    try:
        import numpy as np

        width, height = image.size
        strip_height = max(1, min(height, INFERENCE_STRIP_PIXELS // width))
        payload = bytearray(width * height)
        low = float("inf")
        high = float("-inf")
        for top in range(0, height, strip_height):
            _check_cancel(cancel, "infer")
            bottom = min(height, top + strip_height)
            strip = image.crop((0, top, width, bottom)).convert("RGB")
            rgb = np.asarray(strip, dtype=np.float32) / np.float32(255.0)
            tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
            raw_output = session.run(["mask"], {"image": tensor})[0]
            output = np.asarray(raw_output, dtype=np.float32)
            expected = width * (bottom - top)
            if output.size != expected:
                raise RemovalFailure(
                    "background.inference-failed",
                    "The model returned an invalid mask geometry.",
                    "provider",
                    "infer",
                )
            flat = output.reshape(-1)
            if not bool(np.isfinite(flat).all()):
                raise RemovalFailure(
                    "background.inference-failed",
                    "The model returned a non-finite mask.",
                    "provider",
                    "infer",
                )
            low = min(low, float(flat.min()))
            high = max(high, float(flat.max()))
            encoded = np.floor(
                np.clip(flat, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)
            ).astype(np.uint8)
            start = top * width
            payload[start : start + expected] = encoded.tobytes()
        if high == low:
            if high <= 0.0:
                return Image.new("L", (width, height), 0), []
            raise RemovalFailure(
                "background.inference-failed",
                "The model returned a constant nonzero mask.",
                "provider",
                "infer",
            )
        return Image.frombytes("L", (width, height), bytes(payload)), []
    except RemovalFailure:
        raise
    except Exception as exc:
        raise RemovalFailure(
            "background.inference-failed",
            "The ONNX inference operation failed.",
            "provider",
            "infer",
        ) from exc


def _load_session(request: RemovalRequest, model_path: Path, cached: dict[str, Any]) -> Any:
    profile, _ = _profile()
    if (
        request.model.profile_id != profile["profile_id"]
        or request.model.artifact_sha256 != profile["artifact_sha256"]
    ):
        raise RemovalFailure(
            "background.profile-unavailable",
            "The selected model profile is not installed and qualified.",
            "provider",
            "resolve-profile",
        )
    if sha256_file(model_path) != request.model.artifact_sha256:
        raise RemovalFailure(
            "background.artifact-invalid",
            "The installed model artifact failed verification.",
            "provider",
            "load-model",
        )
    if request.model.artifact_sha256 not in cached:
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            cached[request.model.artifact_sha256] = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RemovalFailure(
                "background.backend-unavailable",
                "The qualified ONNX Runtime CPU backend is unavailable.",
                "provider",
                "load-model",
            ) from exc
    return cached[request.model.artifact_sha256]


def _prepared(items: list[StagedImage]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in items:
        status = item.stage.stat()
        records.append(
            {
                "stage": str(item.stage),
                "destination": str(item.destination),
                "device": status.st_dev,
                "inode": status.st_ino,
            }
        )
    return records


def _execute(
    raw: object,
    cancel: Any,
    send: Any,
    model_path: Path,
    cached: dict[str, Any],
    staging_token: str,
    cancellation_gate: DurableCancellationGate,
) -> dict[str, object]:
    request = parse_request(raw)
    started = time.monotonic()

    def emit(state: str, fraction: float, phase: str) -> None:
        _check_cancel(cancel, phase)
        try:
            sequence = cancellation_gate.reserve_progress(request.request_id)
        except ContractRefusal as exc:
            if exc.rule_id == "LC-PROGRESS-AFTER-CANCEL":
                raise RemovalFailure(
                    "job.cancelled",
                    "The job was cancelled.",
                    "cancellation",
                    phase,
                ) from exc
            raise
        send(
            {
                "kind": "progress",
                "payload": _progress(
                    request.request_id,
                    sequence,
                    state,
                    fraction,
                    phase,
                    started,
                ),
            }
        )

    emit("queued", 0.0, "accepted")
    emit("loading", 0.05, "resolve-profile")
    session = _load_session(request, model_path, cached)
    emit("loading", 0.15, "load-model")
    decoded = decode_image(request.input, request.limits)
    emit("running", 0.30, "decode")
    emit("running", 0.40, "preprocess")
    mask, warnings = _run_onnx_mask(session, decoded.image, cancel)
    emit("running", 0.60, "infer")
    mask = apply_edge_policy(mask, decoded.source_alpha, request.edge)
    emit("running", 0.72, "postprocess")

    background_image: Image.Image | None = None
    if request.background_image is not None:
        background_image = decode_image(request.background_image, request.limits).image

    staged: list[StagedImage] = []
    try:
        staged.append(
            stage_image(
                mask,
                request.destinations["mask"],
                image_format="PNG",
                media_type="image/png",
                kind="mask",
                max_output_bytes=request.limits.max_output_bytes,
                staging_token=staging_token,
            )
        )
        cutout: Image.Image | None = None
        for kind in request.output_kinds:
            if kind == "mask":
                continue
            if kind in {"cutout-png", "cutout-webp"}:
                if cutout is None:
                    cutout = render_cutout(decoded.image, mask)
                destination_key = "cutout_png" if kind == "cutout-png" else "cutout_webp"
                staged.append(
                    stage_image(
                        cutout,
                        request.destinations[destination_key],
                        image_format="PNG" if kind == "cutout-png" else "WEBP",
                        media_type="image/png" if kind == "cutout-png" else "image/webp",
                        kind=kind,
                        max_output_bytes=request.limits.max_output_bytes,
                        staging_token=staging_token,
                    )
                )
            elif kind == "composite":
                if request.background.get("mode") == "color":
                    rgba = cast(list[float], request.background["rgba"])
                    composite = render_color_composite(decoded.image, mask, rgba)
                elif request.background.get("mode") == "image" and background_image is not None:
                    composite = render_image_composite(decoded.image, mask, background_image)
                else:
                    raise RemovalFailure(
                        "background.invalid-request",
                        "Composite output requires a frozen color or image background.",
                        "input",
                        "postprocess",
                    )
                staged.append(
                    stage_image(
                        composite,
                        request.destinations["composite"],
                        image_format="PNG",
                        media_type="image/png",
                        kind="composite",
                        max_output_bytes=request.limits.max_output_bytes,
                        staging_token=staging_token,
                    )
                )
        if sum(item.bytes for item in staged) > request.limits.max_output_bytes:
            raise RemovalFailure(
                "background.output-failed",
                "The encoded outputs exceed the combined byte limit.",
                "resource",
                "verify-output",
            )
        emit("encoding", 0.82, "write-output")
        emit("encoding", 0.92, "verify-output")
        emit("encoding", 0.98, "commit")
        send({"kind": "prepared", "payload": _prepared(staged)})
        _check_cancel(cancel, "commit")
        try:
            terminal = cancellation_gate.reserve_terminal(
                request.request_id,
                "committed",
                publish=lambda: commit_staged(staged),
            )
        except ContractRefusal as exc:
            if exc.rule_id == "LC-ACCEPTED-CANCEL-TERMINAL":
                raise RemovalFailure(
                    "job.cancelled",
                    "The job was cancelled.",
                    "cancellation",
                    "commit",
                ) from exc
            raise
    except Exception:
        discard_staged(staged)
        raise

    mask_item = next(item for item in staged if item.kind == "mask")
    outputs = [
        {
            "kind": item.kind,
            "path": str(item.destination),
            "media_type": item.media_type,
            "bytes": item.bytes,
            "sha256": item.sha256,
            "width": item.width,
            "height": item.height,
        }
        for item in staged
        if item.kind != "mask"
    ]
    return {
        "schema": "kilix.background-removal.result/v2",
        "request_schema": "kilix.background-removal.request/v2",
        "job": {
            "schema": "kilix.media-job.result/v2",
            "request_id": request.request_id,
            "sequence": terminal.sequence,
            "state": "committed",
            "committed_at": _now(),
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "warnings": warnings,
            "diagnostic_reference": diagnostic_reference(request.request_id, "committed"),
        },
        "source": {
            "sha256": request.input.sha256,
            "width": request.input.width,
            "height": request.input.height,
        },
        "mask": {
            "path": str(mask_item.destination),
            "media_type": "image/png",
            "encoding": "gray8",
            "semantics": "foreground-alpha",
            "pixel_contract": "kilix.foreground-alpha-gray8/v2",
            "bytes": mask_item.bytes,
            "sha256": mask_item.sha256,
            "width": mask_item.width,
            "height": mask_item.height,
        },
        "outputs": outputs,
        "model": {
            "profile_id": request.model.profile_id,
            "artifact_sha256": request.model.artifact_sha256,
        },
        "backend": "onnxruntime-cpu",
        "settings": {
            "edge": request.wire["edge"],
            "background": _background_provenance(request.wire["background"]),
        },
    }


def _background_provenance(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("validated background is not an object")
    mode = value.get("mode")
    if mode == "transparent":
        return {"mode": "transparent"}
    if mode == "color":
        return {"mode": "color", "rgba": value["rgba"]}
    image = value.get("image")
    if mode == "image" and isinstance(image, dict):
        return {
            "mode": "image",
            "image": {
                "sha256": image["sha256"],
                "width": image["width"],
                "height": image["height"],
            },
        }
    raise RuntimeError("validated background mode is unavailable")


def _worker_loop(
    connection: Connection,
    cancel: Any,
    model_path: str,
    cancellation_database: str,
) -> None:
    cached: dict[str, Any] = {}
    cancellation_gate = DurableCancellationGate(Path(cancellation_database))
    while True:
        try:
            message = connection.recv()
        except EOFError:
            return
        if message.get("op") == "shutdown":
            return
        if message.get("op") != "run":
            continue
        root = message.get("request")
        request_id = _request_id(root)
        staging_token = message.get("staging_token")
        if not isinstance(staging_token, str):
            staging_token = "invalid"
        sequence = 0

        def tracked_send(outbound: dict[str, object]) -> None:
            nonlocal sequence
            if outbound.get("kind") == "progress":
                payload = outbound.get("payload")
                if isinstance(payload, dict):
                    job = payload.get("job")
                    if isinstance(job, dict) and isinstance(job.get("sequence"), int):
                        sequence = max(sequence, job["sequence"] + 1)
            connection.send(outbound)

        try:
            result = _execute(
                root,
                cancel,
                tracked_send,
                Path(model_path),
                cached,
                staging_token,
                cancellation_gate,
            )
            connection.send({"kind": "result", "payload": result})
        except RemovalFailure as caught:
            connection.send({"kind": "error", "payload": _error_wire(request_id, sequence, caught)})
        except Exception:
            internal = RemovalFailure(
                "background.internal",
                "The supervised worker failed safely.",
                "internal",
                "accepted",
            )
            connection.send(
                {"kind": "error", "payload": _error_wire(request_id, sequence, internal)}
            )


@dataclass(slots=True)
class JobOutcome:
    result: dict[str, object] | None
    error: dict[str, object] | None
    progress: list[dict[str, object]]

    @property
    def ok(self) -> bool:
        return self.result is not None


def _failure_phase(progress: list[dict[str, object]]) -> str:
    if progress and isinstance(progress[-1].get("phase"), str):
        return cast(str, progress[-1]["phase"])
    return "accepted"


def _rollback_prepared(records: list[dict[str, object]]) -> None:
    for record in records:
        stage = Path(cast(str, record["stage"]))
        destination = Path(cast(str, record["destination"]))
        device = cast(int, record["device"])
        inode = cast(int, record["inode"])
        try:
            status = destination.stat(follow_symlinks=False)
            if status.st_dev == device and status.st_ino == inode:
                destination.unlink()
        except OSError:
            pass
        with suppress(OSError):
            stage.unlink(missing_ok=True)


class WorkerSupervisor:
    """Own exactly one persistent worker and one warm ORT session cache."""

    def __init__(self, *, cancellation_database: Path | None = None) -> None:
        _, model = _profile()
        self._context: Any = mp.get_context("spawn")
        self._model_path = str(model)
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._active_request_id: str | None = None
        self._cancellation_database = cancellation_database or _default_cancellation_database()
        self._cancellation_gate = DurableCancellationGate(self._cancellation_database)
        self._parent: Connection
        self._cancel: Any
        self._process: Any
        self._start_worker()

    def _start_worker(self) -> None:
        self._parent, child = self._context.Pipe(duplex=True)
        self._cancel = self._context.Event()
        self._process = self._context.Process(
            target=_worker_loop,
            args=(
                child,
                self._cancel,
                self._model_path,
                str(self._cancellation_database),
            ),
            name="kilix-background-remover-worker",
            daemon=True,
        )
        self._process.start()
        child.close()

    def _hard_restart(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=2.0)
        self._parent.close()
        self._start_worker()

    @property
    def pid(self) -> int | None:
        return cast(int | None, self._process.pid)

    def cancel(self, request_bytes: bytes) -> bytes:
        """Linearize a canonical v2 cancel request without taking the run lock."""
        outcome_bytes = self._cancellation_gate.cancel(request_bytes)
        outcome = strict_decode(outcome_bytes)
        if not isinstance(outcome, dict):
            raise RuntimeError("validated cancellation outcome is not an object")
        request_id = outcome.get("request_id")
        if outcome.get("outcome") == "accepted" and isinstance(request_id, str):
            with self._state_lock:
                if self._active_request_id == request_id:
                    self._cancel.set()
        return outcome_bytes

    def _finish(self, request_id: str, outcome: JobOutcome) -> JobOutcome:
        with self._state_lock:
            if self._active_request_id == request_id:
                self._active_request_id = None
        return outcome

    def _terminal_error(
        self,
        request_id: str,
        *,
        failure: RemovalFailure | None = None,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if failure is None and payload is None:
            raise ValueError("a terminal failure or payload is required")
        state: TerminalState = "failed"
        if failure is not None and failure.code == "job.cancelled":
            state = "cancelled"
        elif payload is not None:
            job = payload.get("job")
            if isinstance(job, dict) and job.get("state") == "cancelled":
                state = "cancelled"
        try:
            terminal = self._cancellation_gate.reserve_terminal(request_id, state)
        except ContractRefusal as exc:
            if exc.rule_id != "LC-ACCEPTED-CANCEL-TERMINAL":
                raise
            failure = RemovalFailure(
                "job.cancelled",
                "The job was cancelled.",
                "cancellation",
                failure.phase if failure is not None else "accepted",
            )
            terminal = self._cancellation_gate.reserve_terminal(request_id, "cancelled")
        if failure is not None:
            return _error_wire(request_id, terminal.sequence, failure)
        assert payload is not None
        job = payload.get("job")
        if not isinstance(job, dict):
            raise RuntimeError("worker terminal error has no job object")
        job["sequence"] = terminal.sequence
        return payload

    def run(
        self,
        request: object,
        *,
        cancel: threading.Event | None = None,
        on_progress: Callable[[dict[str, object]], None] | None = None,
    ) -> JobOutcome:
        with self._lock:
            if self._closed:
                raise RuntimeError("supervisor is closed")
            if not self._process.is_alive():
                self._hard_restart()
            parsed = parse_request(request)
            self._cancellation_gate.begin(parsed.request_id)
            with self._state_lock:
                self._active_request_id = parsed.request_id
            staging_token = secrets.token_hex(16)
            self._cancel.clear()
            self._parent.send(
                {
                    "op": "run",
                    "request": request,
                    "staging_token": staging_token,
                }
            )
            progress: list[dict[str, object]] = []
            prepared: list[dict[str, object]] = []
            deadline = time.monotonic() + parsed.limits.deadline_ms / 1000.0
            abort_failure: RemovalFailure | None = None
            abort_deadline = 0.0
            local_cancel_requested = False
            while True:
                now = time.monotonic()
                if abort_failure is None:
                    if cancel is not None and cancel.is_set() and not local_cancel_requested:
                        local_cancel_requested = True
                        cancel_request = {
                            "cancellation_id": str(uuid.uuid4()),
                            "client_requested_at": _now(),
                            "reason": "user",
                            "request_id": parsed.request_id,
                            "schema": "kilix.media-job.cancel-request/v2",
                        }
                        cancel_outcome = strict_decode(self.cancel(canonical_bytes(cancel_request)))
                        if not isinstance(cancel_outcome, dict):
                            raise RuntimeError("local cancellation outcome is not an object")
                        if cancel_outcome.get("outcome") == "accepted":
                            abort_failure = RemovalFailure(
                                "job.cancelled",
                                "The job was cancelled.",
                                "cancellation",
                                _failure_phase(progress),
                            )
                    elif self._cancel.is_set():
                        abort_failure = RemovalFailure(
                            "job.cancelled",
                            "The job was cancelled.",
                            "cancellation",
                            _failure_phase(progress),
                        )
                    elif now >= deadline:
                        abort_failure = RemovalFailure(
                            "background.deadline",
                            "The job exceeded its deadline.",
                            "deadline",
                            _failure_phase(progress),
                        )
                    if abort_failure is not None:
                        self._cancel.set()
                        abort_deadline = now + ABORT_GRACE_SECONDS

                if abort_failure is not None and now >= abort_deadline:
                    self._hard_restart()
                    _rollback_prepared(prepared)
                    cleanup_staging_files(list(parsed.destinations.values()), staging_token)
                    return self._finish(
                        parsed.request_id,
                        JobOutcome(
                            None,
                            self._terminal_error(parsed.request_id, failure=abort_failure),
                            progress,
                        ),
                    )

                wait_until = abort_deadline if abort_failure is not None else deadline
                wait_seconds = max(0.0, min(0.05, wait_until - now))
                if self._parent.poll(wait_seconds):
                    try:
                        message = self._parent.recv()
                    except EOFError:
                        message = {"kind": "worker-exited"}
                    kind = message.get("kind")
                    payload = message.get("payload")
                    if kind == "progress" and isinstance(payload, dict):
                        progress.append(payload)
                        if on_progress is not None:
                            on_progress(payload)
                    elif kind == "prepared" and isinstance(payload, list):
                        prepared = [item for item in payload if isinstance(item, dict)]
                    elif kind == "result" and isinstance(payload, dict):
                        terminal = self._cancellation_gate.reserve_terminal(
                            parsed.request_id, "committed"
                        )
                        job = payload.get("job")
                        if not isinstance(job, dict) or job.get("sequence") != terminal.sequence:
                            raise RuntimeError("worker result does not match terminal reservation")
                        return self._finish(
                            parsed.request_id,
                            JobOutcome(payload, None, progress),
                        )
                    elif kind == "error" and isinstance(payload, dict):
                        cleanup_staging_files(list(parsed.destinations.values()), staging_token)
                        if abort_failure is not None:
                            return self._finish(
                                parsed.request_id,
                                JobOutcome(
                                    None,
                                    self._terminal_error(
                                        parsed.request_id,
                                        failure=abort_failure,
                                    ),
                                    progress,
                                ),
                            )
                        return self._finish(
                            parsed.request_id,
                            JobOutcome(
                                None,
                                self._terminal_error(parsed.request_id, payload=payload),
                                progress,
                            ),
                        )

                if not self._process.is_alive():
                    _rollback_prepared(prepared)
                    cleanup_staging_files(list(parsed.destinations.values()), staging_token)
                    self._hard_restart()
                    failure = abort_failure or RemovalFailure(
                        "background.internal",
                        "The supervised worker exited unexpectedly.",
                        "internal",
                        _failure_phase(progress),
                    )
                    return self._finish(
                        parsed.request_id,
                        JobOutcome(
                            None,
                            self._terminal_error(parsed.request_id, failure=failure),
                            progress,
                        ),
                    )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._process.is_alive():
                with suppress(BrokenPipeError, EOFError, OSError):
                    self._parent.send({"op": "shutdown"})
                self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._parent.close()
            self._closed = True

    def __enter__(self) -> WorkerSupervisor:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
