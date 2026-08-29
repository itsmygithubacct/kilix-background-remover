from __future__ import annotations

import copy
import os
import threading
from pathlib import Path

from conftest import assert_valid_message
from jsonschema import Draft202012Validator
from PIL import Image

from kilix_background_remover.cancellation_v2 import DurableCancellationGate
from kilix_background_remover.contract_v2 import decode_gray8_png, strict_decode
from kilix_background_remover.frontend import describe_image
from kilix_background_remover.worker import WorkerSupervisor


def _validate_outcome(validators: dict[str, Draft202012Validator], outcome: object) -> None:
    progress = outcome.progress  # type: ignore[attr-defined]
    terminal = outcome.result or outcome.error  # type: ignore[attr-defined]
    assert terminal is not None
    for message in [*progress, terminal]:
        assert_valid_message(validators, message)
    sequences = [message["job"]["sequence"] for message in [*progress, terminal]]
    assert sequences == list(range(len(sequences)))


def test_real_onnx_worker_commits_all_image_outputs(
    request_factory: object,
    corpus: Path,
    validators: dict[str, Draft202012Validator],
    tmp_path: Path,
) -> None:
    background = {
        "mode": "image",
        "image": describe_image(corpus / "clutter.png"),
    }
    request = request_factory(  # type: ignore[operator]
        tmp_path,
        output_kinds=["mask", "cutout-png", "cutout-webp", "composite"],
        background=background,
    )
    with WorkerSupervisor() as supervisor:
        outcome = supervisor.run(request)
        first_pid = supervisor.pid
        second_dir = tmp_path / "second"
        second_dir.mkdir()
        second_request = request_factory(  # type: ignore[operator]
            second_dir, output_key="second", output_kinds=["mask"]
        )
        second = supervisor.run(second_request)
        assert supervisor.pid == first_pid
    assert outcome.ok
    assert second.ok
    _validate_outcome(validators, outcome)
    _validate_outcome(validators, second)
    assert outcome.result is not None
    mask = outcome.result["mask"]
    assert isinstance(mask, dict)
    paths = [Path(mask["path"])]
    outputs = outcome.result["outputs"]
    assert isinstance(outputs, list)
    paths.extend(Path(item["path"]) for item in outputs)
    assert len(paths) == 4
    for path in paths:
        assert path.is_file()
        assert os.stat(path).st_mode & 0o777 == 0o600
        with Image.open(path) as image:
            assert image.size == (128, 96)
            assert not image.info.get("exif")
    assert mask["encoding"] == "gray8"
    assert mask["semantics"] == "foreground-alpha"
    assert decode_gray8_png(Path(mask["path"]).read_bytes())[:2] == (128, 96)


def test_worker_fails_closed_for_profile_and_existing_output(
    request_factory: object, tmp_path: Path
) -> None:
    request = request_factory(tmp_path)  # type: ignore[operator]
    profile_request = copy.deepcopy(request)
    profile_request["model"]["artifact_sha256"] = "f" * 64
    with WorkerSupervisor() as supervisor:
        missing = supervisor.run(profile_request)
        assert missing.error is not None
        assert missing.error["job"]["code"] == "background.profile-unavailable"

        request = request_factory(tmp_path)  # type: ignore[operator]
        mask_path = Path(request["destinations"]["mask"])
        mask_path.write_bytes(b"do-not-overwrite")
        existing = supervisor.run(request)
    assert existing.error is not None
    assert existing.error["job"]["code"] == "background.output-failed"
    assert mask_path.read_bytes() == b"do-not-overwrite"
    cutout = Path(request["destinations"]["cutout_png"])
    assert not cutout.exists()
    assert not list(tmp_path.glob(".kilix-f108-*.stage"))


def test_cancellation_and_deadline_leave_no_partial_outputs_and_worker_recovers(
    request_factory: object, tmp_path: Path
) -> None:
    cancelled_dir = tmp_path / "cancelled"
    cancelled_dir.mkdir()
    cancelled_request = request_factory(cancelled_dir)  # type: ignore[operator]
    cancellation = threading.Event()
    cancellation.set()
    deadline_dir = tmp_path / "deadline"
    deadline_dir.mkdir()
    deadline_request = request_factory(  # type: ignore[operator]
        deadline_dir, deadline_ms=1
    )
    recovery_dir = tmp_path / "recovery"
    recovery_dir.mkdir()
    recovery_request = request_factory(recovery_dir)  # type: ignore[operator]

    cancellation_database = tmp_path / "cancellation.sqlite3"
    with WorkerSupervisor(cancellation_database=cancellation_database) as supervisor:
        cancelled = supervisor.run(cancelled_request, cancel=cancellation)
        deadline = supervisor.run(deadline_request)
        recovered = supervisor.run(recovery_request)
    assert cancelled.error is not None
    assert cancelled.error["job"]["code"] == "job.cancelled"
    assert deadline.error is not None
    assert deadline.error["job"]["code"] == "background.deadline"
    assert recovered.ok
    cancelled_job = cancelled_request["job"]
    deadline_job = deadline_request["job"]
    assert isinstance(cancelled_job, dict)
    assert isinstance(deadline_job, dict)
    gate = DurableCancellationGate(cancellation_database)
    recorded = gate.recorded_outcome(cancelled_job["request_id"])
    assert recorded is not None
    cancel_outcome = strict_decode(recorded)
    assert isinstance(cancel_outcome, dict)
    assert cancel_outcome["outcome"] == "accepted"
    assert cancelled.error["job"]["sequence"] > cancel_outcome["linearization_sequence"]
    assert gate.recorded_outcome(deadline_job["request_id"]) is None
    for directory in (cancelled_dir, deadline_dir):
        assert list(directory.iterdir()) == []
