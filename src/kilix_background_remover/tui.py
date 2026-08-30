"""Keyboard-driven terminal client over the single F108 provider boundary."""

from __future__ import annotations

import argparse
import json
import selectors
import shutil
import signal
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .contracts import parse_request
from .errors import RemovalFailure
from .frontend import load_json_document
from .provider import (
    BackgroundRemovalProvider,
    load_video_request,
    video_estimate_wire,
    video_result_wire,
)
from .worker import FALLBACK_REQUEST_ID, failure_wire


def render_progress(message: dict[str, object], width: int) -> str:
    job = message.get("job")
    if not isinstance(job, dict):
        return "invalid progress"[: max(1, width)]
    state = str(job.get("state", "unknown"))
    phase = str(message.get("phase", "unknown"))
    raw_fraction = job.get("progress", 0.0)
    fraction = float(raw_fraction) if isinstance(raw_fraction, int | float) else 0.0
    text = f"{state:8s} {fraction * 100:6.1f}%  {phase}"
    return text[: max(1, width)]


def render_video_progress(phase: str, completed: int, total: int, width: int) -> str:
    fraction = 0.0 if total <= 0 else completed / total
    text = f"video    {fraction * 100:6.1f}%  {phase}  frames {completed}/{total}"
    return text[: max(1, width)]


def render_provider_header(identity: dict[str, object], width: int) -> tuple[str, ...]:
    decode = identity.get("decode")
    if not isinstance(decode, dict):
        decode = {}
    lines = (
        "Kilix Background Remover 0.2.1",
        "profile: reference-unqualified  backend: CPUExecutionProvider",
        (
            "decode: spawned parser  "
            f"pixels<=100000000  rss<={decode.get('address_space_bytes', 'unknown')}"
        ),
        "keys: q/Esc cancel  r retry after a failed job",
    )
    return tuple(line[: max(1, width)] for line in lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kilix-background-remover-tui")
    parser.add_argument("request", type=Path)
    parser.add_argument(
        "--operation",
        choices=("image", "video"),
        default="image",
        help="interpret REQUEST as a candidate-R5 image request or fixed video request",
    )
    parser.add_argument("--reference-profile", action="store_true")
    parser.add_argument("--retry", type=int, default=0, metavar="COUNT")
    return parser


def _keyboard(cancelled: threading.Event, retry: threading.Event, stop: threading.Event) -> None:
    selector = selectors.DefaultSelector()
    try:
        selector.register(sys.stdin, selectors.EVENT_READ)
        while not stop.is_set():
            if not selector.select(timeout=0.1):
                continue
            key = sys.stdin.read(1)
            if key in {"q", "Q", "\x1b"}:
                cancelled.set()
            elif key in {"r", "R"}:
                retry.set()
    finally:
        selector.close()


def _install_signals(cancelled: threading.Event) -> dict[signal.Signals, Any]:
    previous: dict[signal.Signals, Any] = {}

    def cancel_job(_signum: int, _frame: object) -> None:
        cancelled.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, cancel_job)
    return previous


def _restore_signals(previous: dict[signal.Signals, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _emit(text: str) -> None:
    ending = "\r" if sys.stderr.isatty() else "\n"
    print(text, file=sys.stderr, end=ending, flush=True)


def _image(
    provider: BackgroundRemovalProvider,
    raw: object,
    cancelled: threading.Event,
) -> tuple[int, dict[str, object]]:
    parse_request(raw)

    def show(message: dict[str, object]) -> None:
        _emit(render_progress(message, shutil.get_terminal_size((80, 24)).columns))

    outcome = provider.run(raw, cancel=cancelled, on_progress=show)
    terminal = outcome.result if outcome.result is not None else outcome.error
    assert terminal is not None
    return (0 if outcome.ok else 3), terminal


def _video(
    provider: BackgroundRemovalProvider,
    request_path: Path,
    cancelled: threading.Event,
) -> tuple[int, dict[str, object]]:
    request = load_video_request(request_path)
    _probe, estimate = provider.estimate_video(request, cancel=cancelled)
    if request.confirmation_sha256 is None:
        return 0, {
            "schema": "kilix.background-removal.tui-result/v1",
            "status": "confirmation-required",
            "estimate": video_estimate_wire(estimate),
        }
    if request.confirmation_sha256 != estimate.confirmation_sha256:
        raise RemovalFailure(
            "background.invalid-request",
            "The exact video time/frame/disk estimate has not been confirmed.",
            "input",
            "confirm-estimate",
        )

    def show(phase: str, completed: int, total: int) -> None:
        width = shutil.get_terminal_size((80, 24)).columns
        _emit(render_video_progress(phase, completed, total, width))

    result = provider.run_video(request, cancel=cancelled, progress=show)
    return 0, {
        "schema": "kilix.background-removal.tui-result/v1",
        "status": "committed",
        "result": video_result_wire(result),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.retry < 0 or args.retry > 16:
        raise SystemExit("kilix-background-remover-tui: retry count must be within 0..16")
    request_id = FALLBACK_REQUEST_ID
    cancelled = threading.Event()
    retry_key = threading.Event()
    stop_keyboard = threading.Event()
    keyboard: threading.Thread | None = None
    previous = _install_signals(cancelled)
    try:
        raw: object | None = None
        if args.operation == "image":
            raw = load_json_document(args.request)
            request_id = parse_request(raw).request_id
        with BackgroundRemovalProvider(allow_reference_profile=args.reference_profile) as provider:
            width = shutil.get_terminal_size((80, 24)).columns
            for line in render_provider_header(provider.identity, width):
                print(line, file=sys.stderr)
            if sys.stdin.isatty():
                keyboard = threading.Thread(
                    target=_keyboard,
                    args=(cancelled, retry_key, stop_keyboard),
                    name="kilix-background-remover-tui-keys",
                    daemon=True,
                )
                keyboard.start()
            remaining = args.retry
            while True:
                try:
                    if args.operation == "image":
                        assert raw is not None
                        code, terminal = _image(provider, raw, cancelled)
                    else:
                        code, terminal = _video(provider, args.request, cancelled)
                except RemovalFailure as caught_failure:
                    terminal = failure_wire(request_id, caught_failure)
                    code = 3
                if code == 0 or cancelled.is_set():
                    break
                if remaining > 0:
                    remaining -= 1
                    cancelled.clear()
                    continue
                if not sys.stdin.isatty():
                    break
                print("\nfailed; press r to retry or q to exit", file=sys.stderr)
                while not retry_key.wait(timeout=0.1):
                    if cancelled.is_set():
                        break
                if cancelled.is_set():
                    break
                retry_key.clear()
        if sys.stderr.isatty():
            print(file=sys.stderr)
        print(json.dumps(terminal, indent=2, sort_keys=True))
        return code
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        failure = RemovalFailure(
            "background.invalid-request",
            "The local request could not be read or validated.",
            "input",
            "accepted",
        )
        print(json.dumps(failure_wire(request_id, failure), sort_keys=True))
        return 2
    finally:
        stop_keyboard.set()
        if keyboard is not None:
            keyboard.join(timeout=1.0)
        _restore_signals(previous)
