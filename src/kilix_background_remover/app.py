"""Contained graphical F108 app over the command-free local provider.

The normal entry point opens a small Tk-contained window.  ``--message`` is a
headless lifecycle mode used by package qualification; it exercises the same
controller and provider as the window without requiring a display server.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

from .app_bridge import BRIDGE_REQUEST_SCHEMA_V2, run_bridge_message
from .frontend import describe_image, make_request, stable_output_key
from .provider import (
    MAX_SURFACE_JSON_BYTES,
    BackgroundRemovalProvider,
    decode_surface_json,
    video_request_wire,
)
from .video import VideoOutputKind, VideoRequest


class ContainedAppController:
    """Own one provider for the full contained-tab lifecycle."""

    def __init__(self, *, allow_reference_profile: bool = False) -> None:
        self._provider = BackgroundRemovalProvider(allow_reference_profile=allow_reference_profile)
        self._closed = False
        self._cancel = threading.Event()

    @property
    def provider_pid(self) -> int | None:
        return self._provider.supervisor_pid

    def dispatch(self, message: object) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("contained app is closed")
        self._cancel.clear()
        return run_bridge_message(
            message,
            allow_reference_profile=False,
            provider=self._provider,
            cancel=self._cancel,
        )

    def cancel(self, request_bytes: bytes) -> bytes:
        if self._closed:
            raise RuntimeError("contained app is closed")
        return self._provider.cancel(request_bytes)

    def request_cancel(self) -> None:
        if self._closed:
            raise RuntimeError("contained app is closed")
        self._cancel.set()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._provider.close()

    def __enter__(self) -> ContainedAppController:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def image_bridge_message(
    source: Path,
    output_dir: Path,
    *,
    threshold_u8: int = 0,
    feather_radius_px: int = 0,
) -> dict[str, object]:
    image = describe_image(source)
    key = stable_output_key(Path(source.name))
    request = make_request(
        image,
        output_dir=output_dir,
        output_key=key,
        output_kinds=["mask", "cutout-png"],
    )
    request["edge"] = {
        "threshold_u8": threshold_u8,
        "feather_radius_px": feather_radius_px,
        "matting_mode": "alpha",
        "preserve_source_alpha": True,
    }
    return {
        "schema": BRIDGE_REQUEST_SCHEMA_V2,
        "operation": "run-image",
        "request": request,
    }


def video_bridge_message(request: VideoRequest, *, run: bool) -> dict[str, object]:
    return {
        "schema": BRIDGE_REQUEST_SCHEMA_V2,
        "operation": "run-video" if run else "estimate-video",
        "request": video_request_wire(request),
    }


class _GraphicalApp:
    def __init__(self, controller: ContainedAppController) -> None:
        import tkinter as tk
        from tkinter import filedialog, ttk

        self._tk = tk
        self._filedialog = filedialog
        self._controller = controller
        self._root = tk.Tk()
        self._root.title("Kilix Background Remover")
        self._root.minsize(720, 480)
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._active = False
        self._video_confirmation: str | None = None

        outer = ttk.Frame(self._root, padding=12)
        outer.pack(fill="both", expand=True)
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="image")
        ttk.Label(controls, text="Media").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.mode,
            values=("image", "video"),
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="ew")

        self.source = tk.StringVar()
        self.destination = tk.StringVar()
        ttk.Label(controls, text="Input").grid(row=1, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.source).grid(row=1, column=1, sticky="ew")
        ttk.Button(controls, text="Browse", command=self._browse_source).grid(row=1, column=2)
        ttk.Label(controls, text="Output").grid(row=2, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.destination).grid(row=2, column=1, sticky="ew")
        ttk.Button(controls, text="Browse", command=self._browse_destination).grid(row=2, column=2)

        self.output_kind = tk.StringVar(value=VideoOutputKind.MATTE.value)
        ttk.Label(controls, text="Video output").grid(row=3, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.output_kind,
            values=tuple(kind.value for kind in VideoOutputKind),
            state="readonly",
        ).grid(row=3, column=1, sticky="ew")
        self.background = tk.StringVar()
        ttk.Label(controls, text="Video background").grid(row=4, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.background).grid(row=4, column=1, sticky="ew")
        ttk.Button(controls, text="Browse", command=self._browse_background).grid(row=4, column=2)
        self.threshold = tk.IntVar(value=0)
        self.feather = tk.IntVar(value=0)
        edge = ttk.Frame(controls)
        edge.grid(row=5, column=1, sticky="w")
        ttk.Label(controls, text="Mask edge").grid(row=5, column=0, sticky="w")
        ttk.Label(edge, text="threshold").pack(side="left")
        ttk.Spinbox(edge, from_=0, to=255, textvariable=self.threshold, width=5).pack(side="left")
        ttk.Label(edge, text=" feather").pack(side="left")
        ttk.Spinbox(edge, from_=0, to=4096, textvariable=self.feather, width=5).pack(side="left")
        self.no_audio = tk.BooleanVar(value=False)
        self.raw_frames = tk.BooleanVar(value=False)
        options = ttk.Frame(controls)
        options.grid(row=6, column=1, sticky="w")
        ttk.Label(controls, text="Video policy").grid(row=6, column=0, sticky="w")
        ttk.Checkbutton(options, text="No audio", variable=self.no_audio).pack(side="left")
        ttk.Checkbutton(options, text="Raw frames", variable=self.raw_frames).pack(side="left")
        controls.columnconfigure(1, weight=1)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(10, 6))
        self.run_button = ttk.Button(actions, text="Run / estimate", command=self._run)
        self.run_button.pack(side="left")
        ttk.Button(actions, text="Cancel", command=self._cancel).pack(side="left", padx=6)
        self.status = tk.StringVar(value="Ready — input remains local")
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=12)

        self.preview = tk.Canvas(outer, background="#6f737a", highlightthickness=0)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", lambda _event: self._checkerboard())
        self._root.protocol("WM_DELETE_WINDOW", self._close)
        self._root.after(50, self._poll)

    def _checkerboard(self) -> None:
        canvas = self.preview
        canvas.delete("checker")
        size = 16
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        for y in range(0, height, size):
            for x in range(0, width, size):
                color = "#f0f0f0" if (x // size + y // size) % 2 == 0 else "#c9c9c9"
                canvas.create_rectangle(
                    x,
                    y,
                    x + size,
                    y + size,
                    fill=color,
                    outline=color,
                    tags="checker",
                )
        canvas.tag_lower("checker")

    def _browse_source(self) -> None:
        selected = self._filedialog.askopenfilename()
        if selected:
            self.source.set(selected)

    def _browse_destination(self) -> None:
        if self.mode.get() == "image":
            selected = self._filedialog.askdirectory()
        else:
            selected = self._filedialog.asksaveasfilename()
        if selected:
            self.destination.set(selected)

    def _browse_background(self) -> None:
        selected = self._filedialog.askopenfilename()
        if selected:
            self.background.set(selected)

    def _request(self) -> dict[str, object]:
        source = Path(self.source.get()).absolute()
        destination = Path(self.destination.get()).absolute()
        if self.mode.get() == "image":
            return image_bridge_message(
                source,
                destination,
                threshold_u8=self.threshold.get(),
                feather_radius_px=self.feather.get(),
            )
        kind = VideoOutputKind(self.output_kind.get())
        background = Path(self.background.get()).absolute() if self.background.get() else None
        request = VideoRequest(
            source=source,
            destination=destination,
            output_kind=kind,
            confirmation_sha256=self._video_confirmation,
            no_audio=self.no_audio.get(),
            raw_frames=self.raw_frames.get(),
            background_image=background if kind is VideoOutputKind.COMPOSITE_IMAGE else None,
            background_video=background if kind is VideoOutputKind.COMPOSITE_VIDEO else None,
        )
        return video_bridge_message(request, run=self._video_confirmation is not None)

    def _run(self) -> None:
        if self._active:
            return
        try:
            message = self._request()
        except Exception:
            self.status.set("The selected local paths or settings are invalid.")
            return
        self._active = True
        self.run_button.state(["disabled"])  # type: ignore[no-untyped-call]
        self.status.set("Working… q/Esc or Cancel stops the job")

        def work() -> None:
            try:
                self._events.put(("response", self._controller.dispatch(message)))
            except Exception as exc:
                self._events.put(("failure", type(exc).__name__))

        threading.Thread(target=work, name="kilix-f108-contained-job", daemon=True).start()

    def _cancel(self) -> None:
        if self._active:
            self._controller.request_cancel()
        self.status.set("Cancellation requested…")

    def _poll(self) -> None:
        try:
            kind, value = self._events.get_nowait()
        except queue.Empty:
            self._root.after(50, self._poll)
            return
        self._active = False
        self.run_button.state(["!disabled"])  # type: ignore[no-untyped-call]
        if kind == "response" and isinstance(value, Mapping):
            if value.get("error") is None:
                result = value.get("result")
                if isinstance(result, Mapping) and result.get("schema") == (
                    "kilix.background-removal.video-estimate/v1"
                ):
                    confirmation = result.get("confirmation_sha256")
                    self._video_confirmation = (
                        confirmation if isinstance(confirmation, str) else None
                    )
                    disclosure = (
                        " GIF transparency is hard-edged and palette-limited."
                        if result.get("gif_hard_edge_disclosure") is True
                        else ""
                    )
                    self.status.set("Estimate ready; press Run again to confirm." + disclosure)
                else:
                    self._video_confirmation = None
                    self.status.set("Complete — output verified and atomically committed")
            else:
                self._video_confirmation = None
                self.status.set("The local provider refused or failed the job safely.")
        else:
            self.status.set("The contained job failed safely.")
        self._root.after(50, self._poll)

    def _close(self) -> None:
        self._controller.close()
        self._root.destroy()

    def run(self) -> None:
        self._root.bind("<Escape>", lambda _event: self._cancel())
        self._root.mainloop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kilix-background-remover-app")
    parser.add_argument("--message", type=Path, help="run one bounded bridge message headlessly")
    parser.add_argument("--reference-profile", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with ContainedAppController(allow_reference_profile=args.reference_profile) as controller:
        if args.message is not None:
            if (
                args.message.is_symlink()
                or not args.message.is_file()
                or args.message.stat().st_size > MAX_SURFACE_JSON_BYTES
            ):
                raise SystemExit("contained app message must be a regular file")
            message = decode_surface_json(args.message.read_bytes())
            response = controller.dispatch(message)
            print(json.dumps(response, indent=2, sort_keys=True))
            return 0 if response.get("error") is None else 3
        _GraphicalApp(controller).run()
    return 0
