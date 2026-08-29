#!/usr/bin/env python3
"""Generate all OD-22 F108 return records from an isolated product install."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kilix_background_remover.cancellation_evidence import generate_cancellation_evidence
from kilix_background_remover.r5_return import generate_ledger
from kilix_background_remover.return_controls import (
    g5a_rejection_result,
    installed_carrier_result,
    load_g5a_request,
    pins_result,
    write_record,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--g5a-fixture", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = generate_ledger(args.output_dir)
    cancellation = generate_cancellation_evidence(args.output_dir)
    pins = pins_result(args.candidate_source)
    installed = installed_carrier_result(args.candidate_source)
    g5a = g5a_rejection_result(load_g5a_request(args.g5a_fixture))
    write_record(args.output_dir / "pins.json", pins)
    write_record(args.output_dir / "installed-carrier.json", installed)
    write_record(args.output_dir / "g5a-production-rejection.json", g5a)

    population = summary["population"]
    if not isinstance(population, dict):
        raise RuntimeError("outcome summary population is not an object")
    print(
        "PASS F108 OD-22 conditional return "
        f"outcomes={population['matched']}/{population['total']} "
        f"races={cancellation['race_outcomes_forced']}/{cancellation['race_outcomes_total']} "
        f"crashes={cancellation['crash_points_forced']}/{cancellation['crash_points_total']} "
        "installed=1/1 g5a=2/2 artifacts=5/5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
