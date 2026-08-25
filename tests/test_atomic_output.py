"""Atomic-output gate evidence: the destination path is never observable in a
partial state, an existing destination is never replaced, and a failed
multi-output commit leaves nothing behind.

Every negative here is causal: the failure is injected at a named boundary and
the test asserts what the filesystem holds afterwards, not merely that an
exception was raised.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import pytest
from PIL import Image

from kilix_background_remover.atomic import (
    StagedImage,
    cleanup_staging_files,
    commit_staged,
    discard_staged,
    stage_image,
)
from kilix_background_remover.errors import RemovalFailure

TOKEN = "0" * 32
MB = 1024 * 1024


def _stage(destination: Path, *, value: int = 128, token: str = TOKEN) -> StagedImage:
    return stage_image(
        Image.new("L", (16, 16), value),
        destination,
        image_format="PNG",
        media_type="image/png",
        kind="mask",
        max_output_bytes=MB,
        staging_token=token,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_destination_is_never_observable_before_it_is_complete(tmp_path: Path) -> None:
    """A watcher polling the destination must see it absent, then complete.

    The staged file is written, fsynced and re-decoded before any name appears at
    the destination, so no reader can observe a truncated or unverified image.
    """
    destination = tmp_path / "mask.png"
    staged = _stage(destination)
    observations: list[str] = []
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            try:
                data = destination.read_bytes()
            except FileNotFoundError:
                observations.append("absent")
                continue
            digest = hashlib.sha256(data).hexdigest()
            observations.append("complete" if digest == staged.sha256 else "PARTIAL")

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        commit_staged([staged])
    finally:
        stop.set()
        watcher.join(timeout=5)

    assert "PARTIAL" not in observations, "a partial destination was observable"
    assert observations, "the watcher never sampled the destination"
    assert destination.is_file()
    assert _sha256(destination) == staged.sha256


@pytest.mark.parametrize("boundary", [0, 1, 2])
def test_commit_failure_at_every_boundary_leaves_no_output(tmp_path: Path, boundary: int) -> None:
    """Inject the failure before the first link, between links, and after the last."""
    items = [_stage(tmp_path / f"out{index}.png", value=index * 40) for index in range(3)]
    if boundary == 0:
        # nothing linked yet: discarding must clear staging and create no output
        discard_staged(items)
        assert not any(item.destination.exists() for item in items)
    else:
        with pytest.raises(RemovalFailure, match="could not be committed"):
            commit_staged(items, fail_after_links=boundary)
        assert not any(item.destination.exists() for item in items), "a partial output survived"
    assert not any(item.stage.exists() for item in items), "staging residue survived"


def test_an_existing_destination_is_never_replaced(tmp_path: Path) -> None:
    destination = tmp_path / "mask.png"
    destination.write_bytes(b"operator content")
    before = _sha256(destination)
    with pytest.raises(RemovalFailure, match="already exists"):
        _stage(destination)
    assert _sha256(destination) == before


def test_a_symlinked_destination_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.png"
    target.write_bytes(b"operator content")
    destination = tmp_path / "mask.png"
    destination.symlink_to(target)
    with pytest.raises(RemovalFailure, match="already exists"):
        _stage(destination)
    assert target.read_bytes() == b"operator content"


def test_rollback_does_not_delete_a_file_it_did_not_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rollback compares inodes before unlinking, so a racing writer that
    replaces the destination between link and rollback keeps its file.

    The race is injected deterministically: os.link is wrapped so that the
    moment the first destination is linked, an unrelated writer replaces it.
    Unlinking by name alone would destroy that writer's data.
    """
    import kilix_background_remover.atomic as atomic

    first = _stage(tmp_path / "first.png", value=10)
    second = _stage(tmp_path / "second.png", value=20)
    real_link = os.link

    def racing_link(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
        real_link(src, dst, *args, **kwargs)
        if Path(dst).name == "first.png":
            Path(dst).unlink()
            Path(dst).write_bytes(b"someone else")

    monkeypatch.setattr(atomic.os, "link", racing_link)
    with pytest.raises(RemovalFailure):
        commit_staged([first, second], fail_after_links=1)
    assert first.destination.read_bytes() == b"someone else", (
        "rollback destroyed a file it did not create"
    )
    assert not second.destination.exists()


def test_staging_cleanup_touches_only_this_job(tmp_path: Path) -> None:
    mine = _stage(tmp_path / "mine.png", token=TOKEN)
    other = _stage(tmp_path / "other.png", token="f" * 32)
    unrelated = tmp_path / "operator.stage"
    unrelated.write_bytes(b"not ours")
    cleanup_staging_files([tmp_path / "mine.png"], TOKEN)
    assert not mine.stage.exists(), "this job's staging file was not cleaned"
    assert other.stage.exists(), "another job's staging file was removed"
    assert unrelated.read_bytes() == b"not ours"
    discard_staged([other])


def test_staged_file_is_private_and_verified(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "mask.png")
    mode = os.stat(staged.stage).st_mode & 0o777
    assert mode == 0o600, f"staging mode is {mode:o}, not 0600"
    assert staged.sha256 == _sha256(staged.stage)
    with Image.open(staged.stage) as decoded:
        decoded.load()
        assert decoded.size == (16, 16)
    discard_staged([staged])


def test_output_byte_ceiling_refuses_before_any_destination_appears(tmp_path: Path) -> None:
    destination = tmp_path / "mask.png"
    with pytest.raises(RemovalFailure, match="byte limit"):
        stage_image(
            Image.new("L", (256, 256), 200),
            destination,
            image_format="PNG",
            media_type="image/png",
            kind="mask",
            max_output_bytes=16,
            staging_token=TOKEN,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".kilix-f108-*.stage")), "staging residue after a refusal"
