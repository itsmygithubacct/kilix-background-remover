#!/usr/bin/env python3
"""Generate the untrained architecture-only ONNX reference profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> int:
    package = Path(__file__).resolve().parents[1] / "src" / "kilix_background_remover"
    model_path = package / "reference_luma.onnx"
    profile_path = package / "reference_profile.json"
    input_info = helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, [1, 3, "height", "width"]
    )
    output_info = helper.make_tensor_value_info(
        "mask", TensorProto.FLOAT, [1, 1, "height", "width"]
    )
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["image"], ["mask"], axes=[1], keepdims=1)],
        "kilix-f108-reference-luma-v1",
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph,
        producer_name="kilix-background-remover",
        producer_version="0.2.1",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    payload = model.SerializeToString(deterministic=True)
    model_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    profile = {
        "schema": "kilix.background-removal.dev-profile/v1",
        "profile_id": "f108-reference-luma-v1",
        "artifact_sha256": digest,
        "artifact_bytes": len(payload),
        "backend": "onnxruntime-cpu",
        "code_license": "Apache-2.0",
        "weight_license": "not-applicable-no-trained-weights",
        "training_data": "none",
        "release_qualified": False,
        "purpose": "supervised-worker-and-contract-feasibility-only",
    }
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{digest}  {model_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
