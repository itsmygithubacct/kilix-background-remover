"""Contained lifecycle teardown for the app bridge.

Release gate 6 requires the TUI and app to share one worker and pass contained
lifecycle tests. The shared-worker half is covered by tests/test_batch_ui_bridge.py;
this is the teardown half - condition I2 of the acceptance ledger.

Mutation table, written before the controls:

=====  ==============================================  =========================
ID     Control                                         Mutation
=====  ==============================================  =========================
I2-a   a caller-supplied supervisor is NOT closed      always close, ignoring
       by the bridge                                   ownership
I2-b   a bridge-owned supervisor is torn down even     move close() out of the
       when the run raises                             finally block
I2-c   a bridge-owned supervisor is torn down on the   (null control for I2-b:
       ordinary path                                   proves teardown at all)
=====  ==============================================  =========================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kilix_background_remover.app_bridge import BRIDGE_REQUEST_SCHEMA, run_bridge_message
from kilix_background_remover.worker import WorkerSupervisor


def _message(request: dict[str, object]) -> dict[str, object]:
    return {"schema": BRIDGE_REQUEST_SCHEMA, "operation": "run", "request": request}


def _alive(pid: int | None) -> bool:
    return pid is not None and Path(f"/proc/{pid}").exists()


def test_i2c_a_bridge_owned_supervisor_is_torn_down(request_factory: Any, tmp_path: Path) -> None:
    """Null control for I2-b: without it, I2-b would pass against a bridge that
    never starts a worker at all."""
    out = tmp_path / "out"
    out.mkdir()
    seen: dict[str, int | None] = {}
    real_close = WorkerSupervisor.close

    def record_then_close(self: WorkerSupervisor) -> None:
        seen["pid"] = self.pid
        real_close(self)

    original = WorkerSupervisor.close
    WorkerSupervisor.close = record_then_close  # type: ignore[method-assign]
    try:
        response = run_bridge_message(_message(request_factory(out)), allow_reference_profile=True)
    finally:
        WorkerSupervisor.close = original  # type: ignore[method-assign]

    assert response["error"] is None, response["error"]
    assert "pid" in seen, "the bridge never closed the supervisor it owned"
    assert not _alive(seen["pid"]), "the bridge-owned worker survived teardown"


def test_i2b_a_bridge_owned_supervisor_is_torn_down_when_the_run_raises(
    request_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure path is where a leak hides: an exception must not skip
    teardown and strand a worker process."""
    out = tmp_path / "out"
    out.mkdir()
    seen: dict[str, int | None] = {}
    real_close = WorkerSupervisor.close

    def explode(self: WorkerSupervisor, *_args: object, **_kwargs: object) -> object:
        seen["pid"] = self.pid
        raise RuntimeError("injected mid-run failure")

    def record_then_close(self: WorkerSupervisor) -> None:
        seen.setdefault("pid", self.pid)
        real_close(self)

    monkeypatch.setattr(WorkerSupervisor, "run", explode)
    monkeypatch.setattr(WorkerSupervisor, "close", record_then_close)
    with pytest.raises(RuntimeError, match="injected mid-run failure"):
        run_bridge_message(_message(request_factory(out)), allow_reference_profile=True)

    assert "pid" in seen
    assert not _alive(seen["pid"]), "a worker leaked when the run raised"


def test_i2a_a_caller_supplied_supervisor_is_not_closed_by_the_bridge(
    request_factory: Any, tmp_path: Path
) -> None:
    """The shared-worker contract: the TUI and app hand the bridge a supervisor
    they own.  Closing it would tear down a worker still in use elsewhere."""
    out = tmp_path / "out"
    out.mkdir()
    with WorkerSupervisor() as supervisor:
        before = supervisor.pid
        response = run_bridge_message(
            _message(request_factory(out)),
            allow_reference_profile=True,
            supervisor=supervisor,
        )
        assert response["error"] is None, response["error"]
        assert supervisor.pid == before, "the bridge restarted a supervisor it does not own"
        assert _alive(before), "the bridge closed a supervisor it does not own"
