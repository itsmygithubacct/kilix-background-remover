"""Model-artifact integrity and mask-validity guards.

**Population, enumerated from source.** Every ``RemovalFailure`` code raised by
``worker.py`` and ``postprocess.py`` was listed and cross-referenced against the
existing suite. Twelve codes exist; nine are asserted somewhere. **Three had
zero coverage** and are the population here:

    background.artifact-invalid       the on-disk artifact digest recheck
    background.inference-failed       non-finite, constant and mis-shaped masks
    background.backend-unavailable    ONNX Runtime session construction failure

**Why this is outside the frozen contract ground.** The G5a/G5b contract audit
froze five sites, one of which (M01) is mask and edge semantics. M01's undefined
list is threshold comparison, model-value domain, operation order, feather
kernel and source-alpha combination. It does **not** cover what happens to a
non-finite, constant or mis-shaped model output - and the G5b remediation text
states "Non-finite model output is an inference error, never a mask sample",
which confirms the current behaviour rather than changing it. These guards are
therefore safe to bind now and will not be invalidated by G5b.

Mutation table, written before the controls:

====  ==============================================  =========================
ID    Control                                         Mutation
====  ==============================================  =========================
M-1   on-disk artifact digest is rechecked at load    drop the sha256_file compare
M-2   a non-finite model output is refused            drop the isfinite check
M-3   a constant nonzero model output is refused      drop the high == low branch
M-4   a mis-shaped model output is refused            drop the length check
M-5   an all-zero output warns, and does not fail     (null control for M-3)
====  ==============================================  =========================
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kilix_background_remover.errors import RemovalFailure
from kilix_background_remover.postprocess import prediction_to_mask
from kilix_background_remover.worker import _load_session, _profile

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "kilix_background_remover"


class _Model:
    def __init__(self, profile_id: str, artifact_sha256: str) -> None:
        self.profile_id = profile_id
        self.artifact_sha256 = artifact_sha256


class _Request:
    def __init__(self, model: _Model) -> None:
        self.model = model


def test_m1_a_tampered_artifact_is_refused_before_the_session_loads(
    tmp_path: Path,
) -> None:
    """The declared digest and the profile still agree; only the bytes on disk
    differ.  Nothing but the recheck stands between a swapped artifact and
    ``ort.InferenceSession`` loading it."""
    profile, _ = _profile()
    genuine = (PACKAGE / "reference_luma.onnx").read_bytes()
    assert hashlib.sha256(genuine).hexdigest() == profile["artifact_sha256"]

    # Size-preserving, so the failure cannot come from a length check elsewhere.
    tampered = bytearray(genuine)
    tampered[-1] ^= 0xFF
    swapped = tmp_path / "reference_luma.onnx"
    swapped.write_bytes(bytes(tampered))
    assert swapped.stat().st_size == len(genuine)

    request = _Request(_Model(profile["profile_id"], profile["artifact_sha256"]))
    with pytest.raises(RemovalFailure) as caught:
        _load_session(request, swapped, {})
    assert caught.value.code == "background.artifact-invalid"
    assert caught.value.phase == "load-model"


def test_m1_an_unknown_profile_is_refused_before_the_digest_is_read(
    tmp_path: Path,
) -> None:
    """Null-adjacent control: the profile check must fire first, so a caller
    cannot reach the artifact recheck with an unqualified profile."""
    profile, _ = _profile()
    request = _Request(_Model("not-a-profile", profile["artifact_sha256"]))
    absent = tmp_path / "missing.onnx"
    with pytest.raises(RemovalFailure) as caught:
        _load_session(request, absent, {})
    assert caught.value.code == "background.profile-unavailable"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_m2_a_non_finite_model_output_is_refused(bad: float) -> None:
    values = [0.5] * 16
    values[7] = bad
    with pytest.raises(RemovalFailure) as caught:
        prediction_to_mask(values, 4, 4)
    assert caught.value.code == "background.inference-failed"
    assert caught.value.safe_message == "The model returned a non-finite mask."


def test_m3_a_constant_nonzero_model_output_is_refused() -> None:
    """A model that returns the same nonzero value everywhere has produced no
    segmentation at all; normalising it would divide by zero."""
    with pytest.raises(RemovalFailure) as caught:
        prediction_to_mask([0.7] * 16, 4, 4)
    assert caught.value.code == "background.inference-failed"
    assert caught.value.safe_message == "The model returned a constant nonzero mask."


def test_m4_a_mis_shaped_model_output_is_refused() -> None:
    with pytest.raises(RemovalFailure) as caught:
        prediction_to_mask([0.5] * 15, 4, 4)
    assert caught.value.code == "background.inference-failed"
    assert caught.value.safe_message == "The model returned an invalid mask geometry."


def test_m5_an_all_zero_output_warns_rather_than_failing() -> None:
    """Null control for M-3.  Constant-zero is a legitimate 'no foreground'
    answer, so the constant guard must not swallow it.  Without this, M-3 would
    pass equally well against an implementation that rejected every constant."""
    mask, warnings = prediction_to_mask([0.0] * 16, 4, 4)
    assert mask.size == (4, 4)
    assert mask.getextrema() == (0, 0)
    assert [w["code"] for w in warnings] == ["background.no-salient-object"]


def test_m6_backend_unavailable_is_environment_reachable_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RECORDED REACHABILITY, not an input control.

    Guard order in ``_load_session`` is: profile identity, then the on-disk
    digest recheck, then session construction. To reach construction the file
    must *be* the genuine artifact, which is valid ONNX - so **no input can make
    the session constructor fail**. The code is reachable only by a broken
    environment.

    Induced here by breaking the environment rather than the input, so the guard
    is covered and its trigger is stated accurately instead of being implied to
    be input-driven.
    """
    import onnxruntime as ort

    profile, _ = _profile()

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("no backend")

    monkeypatch.setattr(ort, "InferenceSession", refuse)
    request = _Request(_Model(profile["profile_id"], profile["artifact_sha256"]))
    with pytest.raises(RemovalFailure) as caught:
        _load_session(request, PACKAGE / "reference_luma.onnx", {})
    assert caught.value.code == "background.backend-unavailable"
    assert caught.value.phase == "load-model"
