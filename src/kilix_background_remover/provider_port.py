"""Length-framed stdio provider port for external F115 consumers.

The port transports exact candidate-R5 canonical bytes.  It has no listener,
shell, URL, environment, import-name or arbitrary-command field.  One process
owns one :class:`BackgroundRemovalProvider` and therefore one supervised ONNX
worker for the complete session.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import BinaryIO

from .contract_v2 import ContractRefusal, ContractRuntime, canonical_bytes
from .errors import RemovalFailure
from .provider import MAX_SURFACE_JSON_BYTES, BackgroundRemovalProvider

MAX_HEADER_BYTES = 128
PORT_IDENTITY = {
    "schema": "kilix.background-removal.provider-port/v1",
    "transport": "length-framed-stdio",
    "operations": ["discover", "submit", "cancel", "close"],
    "max_payload_bytes": MAX_SURFACE_JSON_BYTES,
    "concurrent_requests": 1,
}


@dataclass(frozen=True, slots=True)
class _Frame:
    operation: str
    payload: bytes


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    payload = bytearray()
    while len(payload) < length:
        block = stream.read(length - len(payload))
        if not block:
            raise EOFError("provider port payload ended early")
        payload.extend(block)
    return bytes(payload)


def _read_frame(stream: BinaryIO) -> _Frame | None:
    header = stream.readline(MAX_HEADER_BYTES + 1)
    if not header:
        return None
    if len(header) > MAX_HEADER_BYTES or not header.endswith(b"\n"):
        raise ValueError("provider port header is outside its fixed bound")
    try:
        operation_bytes, length_bytes = header[:-1].split(b" ", 1)
        operation = operation_bytes.decode("ascii")
        if not length_bytes or not length_bytes.isdigit():
            raise ValueError
        length = int(length_bytes)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("provider port header is invalid") from exc
    if operation not in {"DISCOVER", "SUBMIT", "CANCEL", "CLOSE"}:
        raise ValueError("provider port operation is unsupported")
    if length > MAX_SURFACE_JSON_BYTES:
        raise ValueError("provider port payload exceeds its fixed bound")
    if operation in {"DISCOVER", "CLOSE"} and length != 0:
        raise ValueError("provider port control operation accepts no payload")
    if operation in {"SUBMIT", "CANCEL"} and length == 0:
        raise ValueError("provider port data operation requires a payload")
    return _Frame(operation, _read_exact(stream, length))


class ProviderPort:
    """Run the fixed 4/4-operation byte port over caller-owned streams."""

    def __init__(
        self,
        provider: BackgroundRemovalProvider,
        output: BinaryIO,
    ) -> None:
        self._provider = provider
        self._output = output
        self._output_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active: threading.Thread | None = None
        self._cancel = threading.Event()
        self._runtime = ContractRuntime.load()

    def _write(self, kind: str, payload: bytes = b"") -> None:
        with self._output_lock:
            self._output.write(f"{kind} {len(payload)}\n".encode("ascii"))
            self._output.write(payload)
            self._output.flush()

    def _error(self, code: str) -> None:
        self._write(
            "PORT-ERROR",
            canonical_bytes(
                {
                    "schema": "kilix.background-removal.provider-port-error/v1",
                    "code": code,
                }
            ),
        )

    def discover(self) -> None:
        identity = {**PORT_IDENTITY, "provider": self._provider.identity}
        self._write("IDENTITY", canonical_bytes(identity))

    def submit(self, payload: bytes) -> None:
        try:
            request = self._runtime.accept_wire(payload)
        except ContractRefusal:
            self._error("invalid-canonical-request")
            return
        with self._state_lock:
            if self._active is not None and self._active.is_alive():
                self._error("request-active")
                return
            self._cancel.clear()
            ready = threading.Event()
            done = threading.Event()

            def progress(message: dict[str, object]) -> None:
                ready.set()
                self._write("MESSAGE", canonical_bytes(message))

            def run() -> None:
                try:
                    outcome = self._provider.run(
                        request,
                        cancel=self._cancel,
                        on_progress=progress,
                    )
                    terminal = outcome.result if outcome.result is not None else outcome.error
                    if terminal is None:
                        self._error("provider-terminal-missing")
                    else:
                        self._write("MESSAGE", canonical_bytes(terminal))
                except Exception:
                    self._error("provider-failed-safely")
                finally:
                    done.set()

            self._active = threading.Thread(
                target=run,
                name="kilix-background-remover-provider-request",
            )
            self._active.start()
        deadline = time.monotonic() + 5.0
        while not ready.is_set() and not done.is_set() and time.monotonic() < deadline:
            ready.wait(timeout=0.01)
        if not ready.is_set() and not done.is_set():
            self._error("provider-start-timeout")
            self._cancel.set()

    def cancel(self, payload: bytes) -> None:
        try:
            outcome = self._provider.cancel(payload)
        except (ContractRefusal, RemovalFailure, RuntimeError, ValueError):
            self._error("invalid-canonical-cancel")
            return
        self._write("CANCEL-OUTCOME", outcome)

    def close(self) -> None:
        self._cancel.set()
        active = self._active
        if active is not None:
            active.join(timeout=5.0)
        self._provider.close()
        self._write("CLOSED")

    def serve(self, source: BinaryIO) -> int:
        try:
            while True:
                frame = _read_frame(source)
                if frame is None:
                    self.close()
                    return 0
                if frame.operation == "DISCOVER":
                    self.discover()
                elif frame.operation == "SUBMIT":
                    self.submit(frame.payload)
                elif frame.operation == "CANCEL":
                    self.cancel(frame.payload)
                else:
                    self.close()
                    return 0
        except (EOFError, OSError, ValueError):
            self._error("invalid-frame")
            self.close()
            return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kilix-background-remover-provider")
    parser.add_argument("--reference-profile", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    provider = BackgroundRemovalProvider(allow_reference_profile=args.reference_profile)
    return ProviderPort(provider, sys.stdout.buffer).serve(sys.stdin.buffer)
