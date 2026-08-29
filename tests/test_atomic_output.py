"""Atomic-output gate evidence: the destination path is never observable in a
partial state, an existing destination is never replaced, and a failed
multi-output commit leaves nothing behind.

Every negative here is causal: the failure is injected at a named boundary and
the test asserts what the filesystem holds afterwards, not merely that an
exception was raised.

Write-exhaustion mutation F-1 removes exception-path staging cleanup. In an
isolated export whose imported production path is asserted before scoring,
F-1 makes 2/2 write-exhaustion controls fail on named staging residue.

Metadata mutations M-1 and M-2 independently remove the encoder stripping
arguments and post-encode inspection. In two isolated exports whose imported
production paths are asserted before scoring, both 2/2 mutations are killed:
M-1 retains the PNG ICC profile and M-2 admits the hostile encoder's profile.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
from PIL import Image

from kilix_background_remover.atomic import (
    StagedImage,
    allocate_staging_path,
    cleanup_staging_files,
    commit_staged,
    discard_staged,
    finalize_staged_file,
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
    destination = tmp_path / "cutout.png"
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


@pytest.mark.parametrize(
    ("image_format", "suffix", "media_type"),
    [("PNG", ".png", "image/png"), ("WEBP", ".webp", "image/webp")],
)
def test_sensitive_metadata_is_stripped_without_mutating_the_source(
    tmp_path: Path, image_format: str, suffix: str, media_type: str
) -> None:
    image = Image.new("RGBA", (16, 16), (11, 22, 33, 44))
    image.info.update(
        {
            "icc_profile": b"kilix-sensitive-icc-profile",
            "exif": b"Exif\x00\x00kilix-sensitive-exif",
            "xmp": b"kilix-sensitive-xmp",
            "Comment": "kilix-sensitive-comment",
        }
    )
    source_info = image.info.copy()

    staged = stage_image(
        image,
        tmp_path / f"cutout{suffix}",
        image_format=image_format,
        media_type=media_type,
        kind="cutout",
        max_output_bytes=MB,
        staging_token=TOKEN,
    )
    with Image.open(staged.stage) as decoded:
        decoded.load()
        assert not {"icc_profile", "exif", "xmp", "Comment"} & decoded.info.keys()
    assert image.info == source_info, "metadata stripping mutated the caller's image"
    discard_staged([staged])


def test_staging_refuses_if_the_encoder_reintroduces_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-encode verification is the backstop if encoder defaults regress."""
    real_save = Image.Image.save

    def leaky_save(self, stream, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["icc_profile"] = b"kilix-injected-sensitive-profile"
        return real_save(self, stream, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", leaky_save)
    destination = tmp_path / "cutout.png"
    with pytest.raises(RemovalFailure, match="staged output could not be verified") as raised:
        stage_image(
            Image.new("RGBA", (16, 16), (11, 22, 33, 44)),
            destination,
            image_format="PNG",
            media_type="image/png",
            kind="cutout",
            max_output_bytes=MB,
            staging_token=TOKEN,
        )

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "encoded output retained metadata: icc_profile"
    assert not destination.exists()
    assert not list(tmp_path.glob(".kilix-f108-*.stage")), (
        "staging residue survived metadata verification"
    )


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


def test_enospc_during_encode_leaves_no_destination_or_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial staging write followed by ENOSPC is fully rolled back."""
    destination = tmp_path / "cutout.png"

    def fail_after_partial_write(self, stream, *args, **kwargs):  # type: ignore[no-untyped-def]
        stream.write(b"partial encoded output")
        stream.flush()
        raise OSError(errno.ENOSPC, "injected full-disk refusal")

    monkeypatch.setattr(Image.Image, "save", fail_after_partial_write)
    with pytest.raises(RemovalFailure, match="staged output could not be verified") as raised:
        stage_image(
            Image.new("RGBA", (256, 256), (200, 100, 50, 255)),
            destination,
            image_format="PNG",
            media_type="image/png",
            kind="cutout-png",
            max_output_bytes=MB,
            staging_token=TOKEN,
        )

    assert isinstance(raised.value.__cause__, OSError)
    assert raised.value.__cause__.errno == errno.ENOSPC
    assert not destination.exists(), "a destination appeared after ENOSPC"
    assert not list(tmp_path.glob(".kilix-f108-*.stage")), "staging residue survived ENOSPC"


def test_kernel_write_limit_during_real_png_encode_cleans_staging(tmp_path: Path) -> None:
    """A real kernel EFBIG during Pillow encoding exercises the same cleanup.

    RLIMIT_FSIZE is process-wide, so the control runs in a child. The child
    requires EFBIG as the causal exception; an output-byte-limit refusal or an
    unrelated encode failure cannot satisfy it.
    """
    destination = tmp_path / "mask.png"
    child = textwrap.dedent(
        """
        import errno
        import random
        import resource
        import signal
        import sys
        from pathlib import Path

        from PIL import Image

        from kilix_background_remover.atomic import stage_image
        from kilix_background_remover.errors import RemovalFailure

        directory = Path(sys.argv[1])
        destination = directory / "mask.png"
        before = resource.getrlimit(resource.RLIMIT_FSIZE)
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (4096, before[1]))
        try:
            image = Image.frombytes(
                "L", (1024, 1024), random.Random(108).randbytes(1024 * 1024)
            )
            try:
                stage_image(
                    image,
                    destination,
                    image_format="PNG",
                    media_type="image/png",
                    kind="mask",
                    max_output_bytes=2 * 1024 * 1024,
                    staging_token="0" * 32,
                )
            except RemovalFailure as failure:
                cause = failure.__cause__
                if not (
                    failure.code == "background.output-failed"
                    and failure.phase == "verify-output"
                    and isinstance(cause, OSError)
                    and cause.errno == errno.EFBIG
                ):
                    print(repr(failure), repr(cause), file=sys.stderr)
                    raise SystemExit(12)
            else:
                raise SystemExit(13)
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, before)
        """
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert not destination.exists(), "a destination appeared after the kernel write refusal"
    assert not list(tmp_path.glob(".kilix-f108-*.stage")), (
        "staging residue survived the kernel write refusal"
    )


def test_external_encoder_cannot_replace_the_reserved_inode(tmp_path: Path) -> None:
    destination = tmp_path / "output.mkv"
    reserved = allocate_staging_path(destination, staging_token=TOKEN)
    reserved.stage.unlink()
    reserved.stage.write_bytes(b"replacement inode")

    with pytest.raises(RemovalFailure, match="could not be verified") as raised:
        finalize_staged_file(
            reserved,
            media_type="video/x-matroska",
            kind="matte",
            max_output_bytes=MB,
            verify=lambda _path: None,
        )

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "the encoder replaced its reserved staging inode"
    assert not destination.exists()
    assert not reserved.stage.exists()


def test_external_encoder_byte_limit_is_checked_before_verification(tmp_path: Path) -> None:
    destination = tmp_path / "output.mkv"
    reserved = allocate_staging_path(destination, staging_token=TOKEN)
    reserved.stage.write_bytes(b"x" * 17)
    verified = False

    def verify(_path: Path) -> None:
        nonlocal verified
        verified = True

    with pytest.raises(RemovalFailure, match="byte limit"):
        finalize_staged_file(
            reserved,
            media_type="video/x-matroska",
            kind="matte",
            max_output_bytes=16,
            verify=verify,
        )

    assert not verified
    assert not destination.exists()
    assert not reserved.stage.exists()


def test_external_encoder_output_is_private_verified_and_atomically_committed(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output.mkv"
    payload = b"qualified encoded bytes"
    reserved = allocate_staging_path(destination, staging_token=TOKEN)
    reserved.stage.write_bytes(payload)
    staged = finalize_staged_file(
        reserved,
        media_type="video/x-matroska",
        kind="matte",
        max_output_bytes=MB,
        verify=lambda path: path.read_bytes() == payload or pytest.fail("wrong payload"),
    )

    assert stat.S_IMODE(reserved.stage.stat().st_mode) == 0o600
    assert staged.bytes == len(payload)
    assert staged.sha256 == hashlib.sha256(payload).hexdigest()
    commit_staged([staged])

    assert destination.read_bytes() == payload
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert not reserved.stage.exists()


def test_external_encoder_output_cannot_change_during_verification(tmp_path: Path) -> None:
    destination = tmp_path / "output.mkv"
    reserved = allocate_staging_path(destination, staging_token=TOKEN)
    reserved.stage.write_bytes(b"before")

    def mutate(path: Path) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.write(descriptor, b"AFTER!")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(path, 0o600)

    with pytest.raises(RemovalFailure, match="could not be verified") as raised:
        finalize_staged_file(
            reserved,
            media_type="video/x-matroska",
            kind="matte",
            max_output_bytes=MB,
            verify=mutate,
        )

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) in {
        "the staged output changed during verification",
        "the staged output content changed during verification",
    }
    assert not destination.exists()
    assert not reserved.stage.exists()
