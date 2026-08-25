from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest
from conftest import ROOT

from kilix_background_remover.app_bridge import (
    BRIDGE_REQUEST_SCHEMA,
    BRIDGE_RESPONSE_SCHEMA,
    run_bridge_message,
)
from kilix_background_remover.jobs import BatchEntry, BatchRunner
from kilix_background_remover.tui import render_progress
from kilix_background_remover.worker import WorkerSupervisor


def test_batch_is_ordered_mixed_failure_resumable_and_nonrolling(
    request_factory: object, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    state = tmp_path / "state"
    output.mkdir()
    state.mkdir()
    first = request_factory(output, output_key="first")  # type: ignore[operator]
    second = request_factory(output, output_key="second")  # type: ignore[operator]
    second["model"]["artifact_sha256"] = "e" * 64
    entries = [BatchEntry("first", first), BatchEntry("second", second)]
    with WorkerSupervisor() as supervisor:
        runner = BatchRunner(supervisor)
        outcomes = runner.run(entries, state_dir=state)
        resumed = runner.run(entries, state_dir=state)
    assert [item.key for item in outcomes] == ["first", "second"]
    assert [item.index for item in outcomes] == [0, 1]
    assert outcomes[0].outcome.ok
    assert not outcomes[1].outcome.ok
    assert Path(first["destinations"]["mask"]).is_file()
    assert Path(first["destinations"]["cutout_png"]).is_file()
    assert resumed[0].disposition == "resumed"
    assert resumed[0].outcome.ok
    assert resumed[1].disposition == "executed"
    assert (state / "first.json").is_file()
    assert (state / "second.json").is_file()


def test_batch_rejects_duplicate_keys(request_factory: object, tmp_path: Path) -> None:
    output = tmp_path / "output"
    state = tmp_path / "state"
    output.mkdir()
    state.mkdir()
    first = request_factory(output, output_key="first")  # type: ignore[operator]
    second = request_factory(output, output_key="second")  # type: ignore[operator]
    with WorkerSupervisor() as supervisor, pytest.raises(ValueError, match="unique"):
        BatchRunner(supervisor).run(
            [BatchEntry("same", first), BatchEntry("same", second)],
            state_dir=state,
        )


def test_bridge_accepts_only_fixed_fields_and_shares_worker(
    request_factory: object, tmp_path: Path
) -> None:
    request = request_factory(tmp_path)  # type: ignore[operator]
    message = {"schema": BRIDGE_REQUEST_SCHEMA, "operation": "run", "request": request}
    unavailable = run_bridge_message(message, allow_reference_profile=False)
    assert unavailable["error"]["job"]["code"] == "background.profile-unavailable"

    for forbidden in ("command", "environment", "model_url", "python_import"):
        mutated = dict(message)
        mutated[forbidden] = "forbidden"
        with pytest.raises(ValueError, match="unknown"):
            run_bridge_message(mutated, allow_reference_profile=False)

    request_with_command = copy.deepcopy(request)
    request_with_command["command"] = ["sh", "-c", "false"]
    with pytest.raises(Exception, match="missing or unknown"):
        run_bridge_message(
            {**message, "request": request_with_command}, allow_reference_profile=False
        )

    with WorkerSupervisor() as supervisor:
        response = run_bridge_message(
            message,
            allow_reference_profile=True,
            supervisor=supervisor,
        )
    assert response["schema"] == BRIDGE_RESPONSE_SCHEMA
    assert response["result"] is not None
    assert response["error"] is None


def test_tui_narrow_render_is_deterministic() -> None:
    progress = {
        "schema": "kilix.background-removal.progress/v1",
        "phase": "postprocess",
        "job": {"state": "running", "progress": 0.725},
    }
    assert render_progress(progress, 80) == "running    72.5%  postprocess"
    assert render_progress(progress, 12) == "running    7"
    assert len(render_progress(progress, 1)) == 1


def test_product_source_has_no_network_or_process_execution_imports() -> None:
    forbidden_roots = {"aiohttp", "http", "requests", "socket", "subprocess", "urllib"}
    for path in sorted((ROOT / "src" / "kilix_background_remover").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        assert not imports & forbidden_roots, (path.name, imports & forbidden_roots)
