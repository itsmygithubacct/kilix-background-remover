"""Stable, path-free provider failures."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(slots=True)
class RemovalFailure(Exception):
    code: str
    safe_message: str
    category: str
    phase: str
    retryable: bool = False

    def __post_init__(self) -> None:
        Exception.__init__(self, self.safe_message)


def diagnostic_reference(request_id: str, code: str) -> str:
    value = hashlib.sha256(f"{request_id}\0{code}".encode()).hexdigest()[:32]
    return f"diag-{value}"


def invalid_request(
    message: str = "The request does not match the conditional R5 v2 contract.",
) -> RemovalFailure:
    return RemovalFailure(
        "background.invalid-request", message, "input", "accepted", retryable=False
    )
