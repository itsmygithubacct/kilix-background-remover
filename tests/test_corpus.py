from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

from conftest import ROOT
from PIL import Image

REQUIRED_CATEGORIES = {
    "clutter",
    "corrupt",
    "existing-alpha",
    "fur",
    "hair",
    "holes",
    "illustration",
    "large",
    "low-contrast",
    "negative",
    "plant",
    "portrait",
    "product",
    "reflective",
    "thin-structure",
    "translucent",
    "vehicle",
}


def test_corpus_manifest_has_owned_ground_truth(corpus: Path) -> None:
    manifest = json.loads((corpus / "corpus-manifest.json").read_text(encoding="utf-8"))
    assert manifest["quality_disposition"] == (
        "safety-and-contract-fixtures-only; no model quality claim"
    )
    categories: set[str] = set()
    assert len(manifest["items"]) == 15
    for item in manifest["items"]:
        assert item["license"] == "Apache-2.0"
        assert item["source"] == "deterministic-procedural-generator"
        categories.update(item["categories"])
        input_record = item["input"]
        input_path = corpus / input_record["path"]
        assert hashlib.sha256(input_path.read_bytes()).hexdigest() == input_record["sha256"]
        mask_record = item.get("ground_truth_mask")
        if mask_record is not None:
            mask_path = corpus / mask_record["path"]
            assert hashlib.sha256(mask_path.read_bytes()).hexdigest() == mask_record["sha256"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(mask_path) as mask:
                    assert mask.mode == "L"
                    assert mask.size == (mask_record["width"], mask_record["height"])
    assert categories >= REQUIRED_CATEGORIES


def test_corpus_regeneration_is_byte_identical(tmp_path: Path, corpus: Path) -> None:
    generated = tmp_path / "corpus"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "generate_corpus.py"),
            "--root",
            str(generated),
        ],
        check=True,
        timeout=30,
    )
    assert (generated / "SHA256SUMS").read_bytes() == (corpus / "SHA256SUMS").read_bytes()
    for line in (corpus / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        _digest, name = line.split("  ", 1)
        assert (generated / name).read_bytes() == (corpus / name).read_bytes()


def test_the_corpus_is_closed_in_both_directions(corpus: Path) -> None:
    """Every file present is listed, and every file listed is present.

    Discharges the enforceable half of acceptance condition B9. Release gate 2
    requires every corpus artifact to carry exact provenance and hashes; a
    manifest that verifies what it lists but accepts what it does not cannot
    make that claim, because an artifact with no provenance record could sit in
    the corpus indefinitely.

    Measured before this control existed: planting an unlisted PNG left both
    corpus tests passing. That is the same shape as the F100 R5 finding, where
    the wheel audit enforced closure over two name prefixes and discarded the
    member set it computed.

    Mutation that must break it: list a file in the manifest without shipping
    it, or ship a file without listing it. Both directions are asserted, so a
    one-sided check cannot satisfy this.
    """
    manifest = json.loads((corpus / "corpus-manifest.json").read_text(encoding="utf-8"))
    listed: set[str] = set()
    for item in manifest["items"]:
        listed.add(item["input"]["path"])
        mask = item.get("ground_truth_mask")
        if mask:
            listed.add(mask["path"])

    on_disk = {entry.name for entry in corpus.iterdir() if entry.is_file()}
    metadata = {"corpus-manifest.json", "SHA256SUMS"}
    payload = on_disk - metadata

    unlisted = sorted(payload - listed)
    assert not unlisted, f"corpus artifacts with no provenance record: {unlisted}"
    missing = sorted(listed - payload)
    assert not missing, f"manifest lists artifacts that are not present: {missing}"

    # SHA256SUMS must cover the same closed set plus the manifest itself.
    summed = {
        line.split("  ", 1)[1]
        for line in (corpus / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert summed == payload | {"corpus-manifest.json"}, (
        f"SHA256SUMS does not cover exactly the corpus: "
        f"extra={sorted(summed - payload - {'corpus-manifest.json'})} "
        f"missing={sorted((payload | {'corpus-manifest.json'}) - summed)}"
    )
