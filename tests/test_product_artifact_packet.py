from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_installed_product_packet_closes_and_verifies() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "verify_product_artifacts.py"),
            str(root / "evidence" / "0.2.1-product-return-r1"),
        ],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "files=31/31" in completed.stdout
    assert "video=6/6" in completed.stdout
    assert "acceptance=0/1" in completed.stdout
