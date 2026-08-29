from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from kilix_background_remover.contract_v2 import ContractRuntime
from kilix_background_remover.frontend import describe_image, make_request

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "fixtures" / "corpus"


@pytest.fixture
def corpus() -> Path:
    return CORPUS


@pytest.fixture
def request_factory() -> Callable[..., dict[str, object]]:
    def build(
        output_dir: Path,
        *,
        source: Path | None = None,
        output_key: str = "case",
        output_kinds: list[str] | None = None,
        background: dict[str, object] | None = None,
        deadline_ms: int = 120_000,
    ) -> dict[str, object]:
        source = source or (CORPUS / "portrait-hair.png")
        return make_request(
            describe_image(source),
            output_dir=output_dir,
            output_key=output_key,
            output_kinds=output_kinds or ["mask", "cutout-png"],
            background=background,
            deadline_ms=deadline_ms,
        )

    return build


@pytest.fixture(scope="session")
def validators() -> dict[str, Draft202012Validator]:
    return ContractRuntime.load().validators


def assert_valid_message(
    validators: dict[str, Draft202012Validator], message: dict[str, object]
) -> None:
    identity = message["schema"]
    assert isinstance(identity, str)
    errors = list(validators[identity].iter_errors(message))
    assert not errors, [error.message for error in errors]
