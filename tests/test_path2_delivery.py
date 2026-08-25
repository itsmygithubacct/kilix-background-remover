"""S0 path 2 is enforced here, not merely true by accident.

The binding decision (0.2.1-S0-PATH2-DECISION.md, obligations 4.2 and 4.4) is
that **no cutout weight is mirrored**: every profile is acquired by the user
from the author's distribution point and converted locally.

Today the package satisfies that by carrying no trained weight at all and
implementing no delivery path of any kind. That is compliance by absence, and
absence is not a control: a future change could add a mirrored artifact and
nothing would fail. These tests make the property enforced.

Mutation table, written before the controls:

===  =================================================  ========================
ID   Control                                            Mutation that must break it
===  =================================================  ========================
P2-1 no trained weight ships in the package             add a plausible weight file
P2-2 the only model artifact is the untrained graph     enlarge the reference graph
P2-3 the reference profile declares itself unqualified  flip release_qualified true
P2-4 no code path fetches or mirrors an artifact        add a urllib download helper
===  =================================================  ========================
"""

from __future__ import annotations

import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "kilix_background_remover"

# Extensions that carry trained model parameters.  A file with one of these is a
# weight unless it is the declared untrained reference graph.
WEIGHT_SUFFIXES = {".onnx", ".pth", ".pt", ".bin", ".safetensors", ".h5", ".pb", ".tflite"}

# The untrained architecture-only reference graph, and the ceiling that keeps it
# untrainable: a single ReduceMean node.  Any real cutout checkpoint is orders of
# magnitude larger - the smallest candidate in the tier table is 4.7 MB.
REFERENCE = "reference_luma.onnx"
REFERENCE_CEILING_BYTES = 4096

# Anything that could fetch bytes from elsewhere.
NETWORK_NAMES = ("urllib", "requests", "httpx", "socket", "ftplib", "http.client", "boto3")


def _model_artifacts() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*") if p.is_file() and p.suffix in WEIGHT_SUFFIXES)


def test_p2_1_no_trained_weight_ships_in_the_package() -> None:
    unexpected = [p.name for p in _model_artifacts() if p.name != REFERENCE]
    assert not unexpected, (
        f"path 2 forbids mirroring a cutout weight; found {unexpected}. "
        "If a profile is being delivered, it must be user-supplied acquisition."
    )


def test_p2_2_the_reference_graph_stays_too_small_to_be_a_checkpoint() -> None:
    reference = PACKAGE / REFERENCE
    assert reference.is_file(), "the untrained reference graph is missing"
    size = reference.stat().st_size
    assert size <= REFERENCE_CEILING_BYTES, (
        f"{REFERENCE} is {size} bytes, over the {REFERENCE_CEILING_BYTES}-byte ceiling. "
        "An artifact this large is no longer an architecture-only graph."
    )


def test_p2_3_the_reference_profile_declares_itself_unqualified() -> None:
    profile = json.loads((PACKAGE / "reference_profile.json").read_text(encoding="utf-8"))
    assert profile["release_qualified"] is False
    assert profile["weight_license"] == "not-applicable-no-trained-weights"
    assert profile["training_data"] == "none"


def test_p2_4_no_source_file_can_fetch_an_artifact() -> None:
    """Delivery is F100's user-supplied acquisition path.  This package must not
    grow its own fetcher, which is how mirroring reappears by increment."""
    offenders: dict[str, list[str]] = {}
    for source in PACKAGE.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        hits = [name for name in NETWORK_NAMES if f"import {name}" in text]
        if hits:
            offenders[source.name] = hits
    assert not offenders, f"a fetch capability appeared in {offenders}"


def test_p2_4_no_delivery_path_exists_and_the_absence_is_refused_loudly() -> None:
    """With no qualified profile installed, the product must refuse rather than
    silently fall back to the development reference."""
    surfaces = ["cli.py", "tui.py", "app_bridge.py"]
    for name in surfaces:
        text = (PACKAGE / name).read_text(encoding="utf-8")
        assert "No release-qualified model profile is installed." in text, (
            f"{name} lost its refusal for the absent-profile case"
        )
