"""CLI controls for the two-step offline-video confirmation gate."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest
from PIL import Image

import kilix_background_remover.cli as cli


@pytest.fixture(scope="module")
def one_frame_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("cli-video")
    source = root / "source.mkv"
    completed = subprocess.run(
        [
            "/usr/bin/ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=8x8:rate=1:duration=1",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgra",
            str(source),
        ],
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "AV_LOG_FORCE_NOCOLOR": "1",
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return source


def _arguments(source: Path, destination: Path) -> list[str]:
    return [
        "video",
        str(source),
        str(destination),
        "--output-kind",
        "matte",
    ]


def test_video_cli_estimates_without_creating_output(
    one_frame_video: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "estimate-only.mkv"

    assert cli.main(_arguments(one_frame_video, destination)) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["status"] == "confirmation-required"
    assert len(document["estimate"]["confirmation_sha256"]) == 64
    assert document["estimate"]["frame_count"] == 1
    assert not destination.exists()


def test_video_cli_rejects_wrong_confirmation_before_provider_or_output(
    one_frame_video: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "wrong-confirmation.mkv"

    def provider_must_not_start(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider started before confirmation")

    monkeypatch.setattr(cli, "ReferenceFrameMasker", provider_must_not_start)
    arguments = [
        *_arguments(one_frame_video, destination),
        "--confirm-estimate",
        "0" * 64,
        "--reference-profile",
    ]

    assert cli.main(arguments) == 2
    document = json.loads(capsys.readouterr().out)

    assert document["error"]["phase"] == "confirm-estimate"
    assert document["error"]["job"]["code"] == "background.invalid-request"
    assert not destination.exists()


class _FastMasker:
    def __init__(self, _workspace: Path) -> None:
        pass

    def __enter__(self) -> _FastMasker:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def __call__(
        self,
        image: Image.Image,
        _index: int,
        _cancel: threading.Event | None,
    ) -> Image.Image:
        return image.getchannel("R")


def test_video_cli_commits_only_the_exact_confirmed_estimate(
    one_frame_video: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "confirmed.mkv"
    arguments = _arguments(one_frame_video, destination)
    assert cli.main(arguments) == 0
    estimate_document = json.loads(capsys.readouterr().out)
    confirmation = estimate_document["estimate"]["confirmation_sha256"]
    monkeypatch.setattr(cli, "ReferenceFrameMasker", _FastMasker)

    assert (
        cli.main(
            [
                *arguments,
                "--confirm-estimate",
                confirmation,
                "--reference-profile",
            ]
        )
        == 0
    )
    result_document = json.loads(capsys.readouterr().out)

    assert result_document["status"] == "committed"
    assert result_document["result"]["destination"] == str(destination)
    assert result_document["result"]["kind"] == "matte"
    assert result_document["result"]["frame_count"] == 1
    assert destination.is_file()
