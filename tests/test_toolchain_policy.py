"""Release-toolchain identity must not drift between executable gates and prose.

Mutation U-1 restores the stale 0.12.3 metadata pin; mutation U-2 removes the
toolchain prerequisite from ``make check``. Their isolated roots are asserted
before scoring, and both 2/2 mutations are killed by the controls below.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = "0.12.5"
UV_VERSION_OUTPUT = f"uv {UV_VERSION} (x86_64-unknown-linux-gnu)"
UV_SHA256 = "b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46"


def test_release_uv_identity_matches_in_metadata_makefile_and_readme() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert project["tool"]["uv"]["required-version"] == f"=={UV_VERSION}"
    assert f"UV_RELEASE_VERSION := {UV_VERSION_OUTPUT}" in makefile
    assert f"UV_RELEASE_SHA256 := {UV_SHA256}" in makefile
    assert f"release-frozen `uv` {UV_VERSION} executable" in readme
    assert UV_SHA256 in readme


def test_primary_make_targets_enforce_the_toolchain_gate() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert re.search(r"^setup: toolchain-check$", makefile, flags=re.MULTILINE)
    assert re.search(r"^check: toolchain-check$", makefile, flags=re.MULTILINE)
    assert 'make setup UV="$UV"' in readme
    assert 'make check UV="$UV"' in readme
