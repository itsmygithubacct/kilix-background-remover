from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from kilix_background_remover.frontend import describe_image, make_request

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "fixtures" / "corpus"
SCHEMAS = ROOT / "contracts" / "schemas"


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
    resources: list[tuple[str, Resource[object]]] = []
    message_schemas: dict[str, object] = {}
    for path in sorted(SCHEMAS.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        identity = schema["$id"]
        resources.append((identity, Resource.from_contents(schema)))
        wire = schema.get("properties", {}).get("schema", {}).get("const")
        if isinstance(wire, str):
            message_schemas[wire] = schema
    registry = Registry().with_resources(resources)
    return {
        identity: Draft202012Validator(
            schema,
            registry=registry,
            format_checker=FormatChecker(),
        )
        for identity, schema in message_schemas.items()
    }


def assert_valid_message(
    validators: dict[str, Draft202012Validator], message: dict[str, object]
) -> None:
    identity = message["schema"]
    assert isinstance(identity, str)
    errors = list(validators[identity].iter_errors(message))
    assert not errors, [error.message for error in errors]
