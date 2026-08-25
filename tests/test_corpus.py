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
