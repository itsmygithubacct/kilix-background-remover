"""Keyboard-driven terminal status view over the shared worker contract."""

from __future__ import annotations

import argparse
import json
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
from .worker import WorkerSupervisor, failure_wire


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kilix-background-remover-tui")
    parser.add_argument("request", type=Path)
    parser.add_argument("--reference-profile", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request_id = "00000000-0000-4000-8000-000000000000"
    try:
        raw = load_json_document(args.request)
        parsed = parse_request(raw)
        request_id = parsed.request_id
        if not args.reference_profile:
            failure = RemovalFailure(
                "background.profile-unavailable",
                "No release-qualified model profile is installed.",
                "provider",
                "resolve-profile",
            )
            print(json.dumps(failure_wire(request_id, failure), sort_keys=True))
            return 3

        cancelled = threading.Event()
        previous: dict[signal.Signals, Any] = {}

        def cancel_job(_signum: int, _frame: object) -> None:
            cancelled.set()

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, cancel_job)

        def show(message: dict[str, object]) -> None:
            width = shutil.get_terminal_size((80, 24)).columns
            text = render_progress(message, width)
            ending = "\r" if sys.stderr.isatty() else "\n"
            print(text, file=sys.stderr, end=ending, flush=True)

        try:
            with WorkerSupervisor() as supervisor:
                outcome = supervisor.run(raw, cancel=cancelled, on_progress=show)
        finally:
            for restore_signal, handler in previous.items():
                signal.signal(restore_signal, handler)
        if sys.stderr.isatty():
            print(file=sys.stderr)
        terminal = outcome.result if outcome.result is not None else outcome.error
        print(json.dumps(terminal, indent=2, sort_keys=True))
        return 0 if outcome.ok else 3
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        failure = RemovalFailure(
            "background.invalid-request",
            "The local request could not be read or validated.",
            "input",
            "accepted",
        )
        print(json.dumps(failure_wire(request_id, failure), sort_keys=True))
        return 2
