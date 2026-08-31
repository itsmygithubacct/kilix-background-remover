"""F108 acceptance authority for the conditionally authorized R5 carrier.

OD-22 permits this product return before G5b freezes.  Every load is therefore
bound to the candidate lock shipped in this package and must be rerun if any
candidate byte changes.  Only the installed public carrier is consumed; Track
G's private evidence directory is neither imported nor searched.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import struct
import zlib
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NoReturn, cast

import rfc8785
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

MAX_DOCUMENT_BYTES = 2_097_152
WIRE_TO_SCHEMA = {
    "kilix.media-job.request/v2": "urn:kilix:schema:kilix.media-job.request:v2",
    "kilix.media-job.cancel-request/v2": "urn:kilix:schema:kilix.media-job.cancel-request:v2",
    "kilix.media-job.cancel-outcome/v2": "urn:kilix:schema:kilix.media-job.cancel-outcome:v2",
    "kilix.media-job.progress/v2": "urn:kilix:schema:kilix.media-job.progress:v2",
    "kilix.media-job.result/v2": "urn:kilix:schema:kilix.media-job.result:v2",
    "kilix.media-job.error/v2": "urn:kilix:schema:kilix.media-job.error:v2",
    "kilix.background-removal.request/v2": ("urn:kilix:schema:kilix.background-removal.request:v2"),
    "kilix.background-removal.progress/v2": (
        "urn:kilix:schema:kilix.background-removal.progress:v2"
    ),
    "kilix.background-removal.result/v2": ("urn:kilix:schema:kilix.background-removal.result:v2"),
    "kilix.background-removal.error/v2": ("urn:kilix:schema:kilix.background-removal.error:v2"),
}
DESTINATION_KEYS = {
    "mask": "mask",
    "composite": "composite",
    "cutout-png": "cutout_png",
    "cutout-webp": "cutout_webp",
}
PROGRESS_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"queued"},
    "queued": {"queued", "loading", "running"},
    "loading": {"loading", "running"},
    "running": {"running", "encoding"},
    "encoding": {"encoding"},
}


@dataclass(frozen=True, slots=True)
class CandidateLock:
    distribution: str
    version: str
    manifest_sha256: str
    manifest_bytes: int
    registry_sha256: str
    wheel_sha256: str
    sdist_sha256: str


class ContractRefusal(ValueError):
    def __init__(self, stage: str, rule_id: str, detail: str = "") -> None:
        self.stage = stage
        self.rule_id = rule_id
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{stage}:{rule_id}{suffix}")


@dataclass(slots=True)
class ContractRuntime:
    root: Path
    lock: CandidateLock
    documents: dict[str, dict[str, Any]]
    registry: Registry[Any]
    validators: dict[str, Draft202012Validator]
    manifest_entries: dict[str, str]

    @classmethod
    def load(cls) -> ContractRuntime:
        lock = load_candidate_lock()
        # The candidate carrier is an optional distribution: it is not published to
        # any index, so the product must import and run without it and refuse in a
        # typed way when a contract operation actually needs it.
        try:
            from kilix_f108_f115_contracts import contract_root
        except ModuleNotFoundError:
            _refuse("carrier", "C-CARRIER-INSTALLED")
        try:
            installed_version = version(lock.distribution)
        except PackageNotFoundError:
            _refuse("carrier", "C-CARRIER-INSTALLED")
        if installed_version != lock.version:
            _refuse("carrier", "C-CARRIER-VERSION")
        root = Path(str(contract_root()))
        if not root.is_dir() or root.is_symlink():
            _refuse("carrier", "C-CARRIER-INSTALLED")
        manifest = root / "SHA256SUMS"
        manifest_bytes = manifest.read_bytes()
        if (
            len(manifest_bytes) != lock.manifest_bytes
            or hashlib.sha256(manifest_bytes).hexdigest() != lock.manifest_sha256
        ):
            _refuse("carrier", "C-CARRIER-MANIFEST")
        entries = verify_manifest_tree(root, manifest)
        registry_path = root / "registry-v2.json"
        if hashlib.sha256(registry_path.read_bytes()).hexdigest() != lock.registry_sha256:
            _refuse("carrier", "C-CARRIER-REGISTRY")
        documents, registry, validators = load_closed_registry(root, FormatChecker())
        return cls(root, lock, documents, registry, validators, entries)

    def decode(self, data: bytes) -> Any:
        return strict_decode(data)

    def validate_message(self, message: Mapping[str, Any]) -> None:
        wire = message.get("schema")
        if not isinstance(wire, str) or wire not in self.validators:
            _refuse("schema", "C-WIRE-IDENTITY")
        errors = sorted(
            self.validators[wire].iter_errors(message),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            _refuse("schema", "C-SHAPE")

    def accept_wire(self, data: bytes) -> dict[str, Any]:
        value = self.decode(data)
        if not isinstance(value, dict):
            _refuse("schema", "C-WIRE-IDENTITY")
        document = cast(dict[str, Any], value)
        self.validate_message(document)
        return document

    def validate_transcript(self, messages: Sequence[Mapping[str, Any]]) -> None:
        validate_transcript(messages, self)


def load_candidate_lock() -> CandidateLock:
    raw = (
        resources.files("kilix_background_remover").joinpath("candidate_r5_lock.json").read_bytes()
    )
    value = strict_decode(raw)
    if not isinstance(value, dict) or value.get("conditional") is not True:
        _refuse("carrier", "C-CARRIER-LOCK")
    carrier = value.get("carrier")
    if not isinstance(carrier, dict):
        _refuse("carrier", "C-CARRIER-LOCK")
    try:
        return CandidateLock(
            distribution=carrier["distribution"],
            version=carrier["version"],
            manifest_sha256=value["contract_manifest_sha256"],
            manifest_bytes=value["contract_manifest_bytes"],
            registry_sha256=value["registry_sha256"],
            wheel_sha256=carrier["wheel_sha256"],
            sdist_sha256=carrier["sdist_sha256"],
        )
    except (KeyError, TypeError) as exc:
        raise ContractRefusal("carrier", "C-CARRIER-LOCK") from exc


def canonical_bytes(value: Any) -> bytes:
    try:
        return rfc8785.dumps(value) + b"\n"
    except (TypeError, ValueError) as exc:
        raise ContractRefusal("canonical", "C-IJSON-DOMAIN") from exc


def strict_decode(data: bytes) -> Any:
    if len(data) > MAX_DOCUMENT_BYTES:
        _refuse("decode", "C-SIZE")
    if data.startswith(b"\xef\xbb\xbf"):
        _refuse("decode", "C-BOM")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractRefusal("decode", "C-UTF8") from exc

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in values:
            if key in decoded:
                _refuse("decode", "C-DUPLICATE-NAME")
            decoded[key] = value
        return decoded

    def nonfinite(_value: str) -> NoReturn:
        _refuse("decode", "C-NONFINITE")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    except ContractRefusal:
        raise
    except json.JSONDecodeError as exc:
        raise ContractRefusal("decode", "C-RFC8259") from exc
    if data != canonical_bytes(value):
        _refuse("canonical", "C-JCS-LF")
    return value


def parse_manifest(data: bytes) -> dict[str, str]:
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ContractRefusal("manifest", "C-MANIFEST-SHAPE") from exc
    entries: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00-\x1f\\]+)", line)
        if match is None:
            _refuse("manifest", "C-MANIFEST-SHAPE")
        digest, relative = match.groups()
        parts = relative.split("/")
        if relative.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            _refuse("manifest", "C-MANIFEST-PATH")
        if relative in entries:
            _refuse("manifest", "C-MANIFEST-DUPLICATE")
        entries[relative] = digest
    return entries


def verify_manifest_tree(root: Path, manifest: Path) -> dict[str, str]:
    entries = parse_manifest(manifest.read_bytes())
    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            _refuse("manifest", "C-MANIFEST-SYMLINK")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            _refuse("manifest", "C-MANIFEST-SPECIAL")
        if path == manifest:
            continue
        relative = path.relative_to(root).as_posix()
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if entries != actual:
        _refuse("manifest", "C-MANIFEST-EXACT")
    return entries


def _walk_refs(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "$ref" and isinstance(child, str):
                yield child
            yield from _walk_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_refs(child)


def ensure_closed_graph(documents: Mapping[str, Mapping[str, Any]]) -> None:
    graph: dict[str, set[str]] = {identity: set() for identity in documents}
    for identity, document in documents.items():
        for reference in _walk_refs(document):
            target = reference.split("#", 1)[0]
            if target and target not in documents:
                _refuse("schema", "C-REF-CLOSED")
            if target:
                graph[identity].add(target)
    active: set[str] = set()
    complete: set[str] = set()

    def visit(identity: str) -> None:
        if identity in active:
            _refuse("schema", "C-REGISTRY-CYCLE")
        if identity in complete:
            return
        active.add(identity)
        for target in graph[identity]:
            visit(target)
        active.remove(identity)
        complete.add(identity)

    for identity in graph:
        visit(identity)


def load_closed_registry(
    root: Path, format_checker: FormatChecker | None
) -> tuple[
    dict[str, dict[str, Any]],
    Registry[Any],
    dict[str, Draft202012Validator],
]:
    if format_checker is None:
        _refuse("schema", "C-FORMAT-REQUIRED")
    record = strict_decode((root / "registry-v2.json").read_bytes())
    if (
        not isinstance(record, dict)
        or record.get("registry_schema") != "kilix.contract-registry/v2"
    ):
        _refuse("schema", "C-REGISTRY-SHAPE")
    entries = record.get("resources")
    if not isinstance(entries, dict) or len(entries) != 12:
        _refuse("schema", "C-REGISTRY-COUNT")
    documents: dict[str, dict[str, Any]] = {}
    resources_to_register: list[tuple[str, Resource[Any]]] = []
    paths: set[str] = set()
    for identity, untyped_entry in entries.items():
        if not isinstance(identity, str) or not isinstance(untyped_entry, dict):
            _refuse("schema", "C-REGISTRY-ENTRY")
        entry = cast(dict[str, Any], untyped_entry)
        if set(entry) != {"bytes", "path", "sha256"} or not identity.endswith(":v2"):
            _refuse("schema", "C-REGISTRY-ENTRY")
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or relative in paths
            or not relative.startswith("schemas/")
        ):
            _refuse("schema", "C-REGISTRY-PATH")
        paths.add(relative)
        path = root / relative
        if path.is_symlink() or not path.is_file():
            _refuse("schema", "C-REGISTRY-FILE")
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            _refuse("schema", "C-REGISTRY-DIGEST")
        decoded = strict_decode(raw)
        if not isinstance(decoded, dict) or decoded.get("$id") != identity:
            _refuse("schema", "C-REGISTRY-ID")
        document = cast(dict[str, Any], decoded)
        try:
            Draft202012Validator.check_schema(document)
            resource = Resource.from_contents(document)
        except Exception as exc:
            raise ContractRefusal("schema", "C-SCHEMA-COMPILE") from exc
        documents[identity] = document
        resources_to_register.append((identity, resource))
    ensure_closed_graph(documents)
    registry: Registry[Any] = Registry().with_resources(resources_to_register)
    validators = {
        wire: Draft202012Validator(
            documents[identity], registry=registry, format_checker=format_checker
        )
        for wire, identity in WIRE_TO_SCHEMA.items()
    }
    return documents, registry, validators


def safe_path(value: str) -> bool:
    return (
        value.startswith("/")
        and value != "/"
        and "\x00" not in value
        and all(segment not in {"", ".", ".."} for segment in value.split("/")[1:])
    )


def _equal(left: Any, right: Any) -> bool:
    return canonical_bytes(left) == canonical_bytes(right)


def _background_projection(background: Mapping[str, Any]) -> dict[str, Any]:
    mode = background["mode"]
    if mode == "transparent":
        return {"mode": mode}
    if mode == "color":
        return {"mode": mode, "rgba": background["rgba"]}
    image = background["image"]
    return {
        "mode": "image",
        "image": {
            "sha256": image["sha256"],
            "width": image["width"],
            "height": image["height"],
        },
    }


def validate_request_semantics(request: Mapping[str, Any]) -> None:
    source = request["input"]
    limits = request["job"]["limits"]
    if not safe_path(source["path"]):
        _refuse("message-semantics", "BR-PATH-SAFE")
    if source["bytes"] > limits["max_input_bytes"]:
        _refuse("message-semantics", "BR-INPUT-BYTES")
    if source["width"] * source["height"] > limits["max_decoded_pixels"]:
        _refuse("message-semantics", "BR-INPUT-PIXELS")
    input_paths = {source["path"]}
    background = request["background"]
    if background["mode"] == "image":
        image = background["image"]
        if not safe_path(image["path"]):
            _refuse("message-semantics", "BR-PATH-SAFE")
        input_paths.add(image["path"])
        if (image["width"], image["height"]) != (source["width"], source["height"]):
            _refuse("message-semantics", "BR-BACKGROUND-GEOMETRY")
        if source["bytes"] + image["bytes"] > limits["max_input_bytes"]:
            _refuse("message-semantics", "BR-INPUT-BYTES")
        pixels = source["width"] * source["height"] + image["width"] * image["height"]
        if pixels > limits["max_decoded_pixels"]:
            _refuse("message-semantics", "BR-INPUT-PIXELS")
    expected_keys = {DESTINATION_KEYS[kind] for kind in request["output_kinds"]}
    if set(request["destinations"]) != expected_keys:
        _refuse("message-semantics", "BR-DESTINATION-KEYS")
    destinations = list(request["destinations"].values())
    if any(not safe_path(path) for path in destinations):
        _refuse("message-semantics", "BR-PATH-SAFE")
    if len(set(destinations)) != len(destinations):
        _refuse("message-semantics", "BR-DESTINATION-DISTINCT")
    if set(destinations) & input_paths:
        _refuse("message-semantics", "BR-DESTINATION-INPUT")


def validate_result_semantics(request: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    source = request["input"]
    expected_source = {
        "sha256": source["sha256"],
        "width": source["width"],
        "height": source["height"],
    }
    expected_settings = {
        "background": _background_projection(request["background"]),
        "edge": request["edge"],
    }
    if result["request_schema"] != "kilix.background-removal.request/v2":
        _refuse("message-semantics", "BR-REQUEST-SCHEMA")
    if not _equal(result["source"], expected_source):
        _refuse("message-semantics", "BR-SOURCE-PROJECTION")
    if not _equal(result["model"], request["model"]):
        _refuse("message-semantics", "BR-MODEL-PROJECTION")
    if not _equal(result["settings"], expected_settings):
        _refuse("message-semantics", "BR-SETTINGS-PROJECTION")
    mask = result["mask"]
    if (mask["width"], mask["height"]) != (source["width"], source["height"]):
        _refuse("message-semantics", "BR-MASK-GEOMETRY")
    if mask["path"] != request["destinations"]["mask"]:
        _refuse("message-semantics", "BR-MASK-DESTINATION")
    expected_kinds = set(request["output_kinds"]) - {"mask"}
    actual_kinds = [output["kind"] for output in result["outputs"]]
    if len(actual_kinds) != len(set(actual_kinds)) or set(actual_kinds) != expected_kinds:
        _refuse("message-semantics", "BR-OUTPUT-KINDS")
    for output in result["outputs"]:
        if (output["width"], output["height"]) != (source["width"], source["height"]):
            _refuse("message-semantics", "BR-OUTPUT-GEOMETRY")
        destination = request["destinations"][DESTINATION_KEYS[output["kind"]]]
        if output["path"] != destination:
            _refuse("message-semantics", "BR-OUTPUT-DESTINATION")
    total_bytes = mask["bytes"] + sum(output["bytes"] for output in result["outputs"])
    if total_bytes > request["job"]["limits"]["max_output_bytes"]:
        _refuse("message-semantics", "BR-OUTPUT-BYTES")

    def has_forbidden_path(value: Any, location: tuple[str, ...] = ()) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                parent_allowed = location == ("mask",) or (
                    len(location) == 2 and location[0] == "outputs"
                )
                if key == "path" and not parent_allowed:
                    return True
                if has_forbidden_path(child, (*location, key)):
                    return True
        elif isinstance(value, list):
            return any(
                has_forbidden_path(child, (*location, str(index)))
                for index, child in enumerate(value)
            )
        return False

    if has_forbidden_path(result):
        _refuse("message-semantics", "BR-RESULT-PATH")


def validate_transcript(messages: Sequence[Mapping[str, Any]], runtime: ContractRuntime) -> None:
    if not messages or messages[0].get("schema") != "kilix.background-removal.request/v2":
        _refuse("lifecycle", "LC-FIRST-REQUEST")
    for message in messages:
        runtime.validate_message(message)
    request = messages[0]
    validate_request_semantics(request)
    request_id = request["job"]["request_id"]
    previous_sequence = -1
    previous_state: str | None = None
    previous_progress = -1.0
    terminal: tuple[str, int] | None = None
    cancel_requests: dict[str, bytes] = {}
    cancel_outcomes: dict[str, Mapping[str, Any]] = {}
    accepted: Mapping[str, Any] | None = None
    for message in messages[1:]:
        if _message_request_id(message) != request_id:
            _refuse("lifecycle", "LC-REQUEST-ID")
        schema = message["schema"]
        if schema == "kilix.media-job.cancel-outcome/v2":
            cancellation_id = message["cancellation_id"]
            if cancellation_id in cancel_outcomes:
                if not _equal(cancel_outcomes[cancellation_id], message):
                    _refuse("lifecycle", "LC-CHANGED-OUTCOME-REPLAY")
                continue
        if schema == "kilix.background-removal.result/v2" and accepted is not None:
            _refuse("lifecycle", "LC-TERMINAL-EXCLUSIVE")
        event = _message_event(message)
        if event is not None:
            sequence, _state = event
            if sequence <= previous_sequence:
                _refuse("lifecycle", "LC-SEQUENCE")
            previous_sequence = sequence
        if schema == "kilix.background-removal.progress/v2":
            if terminal is not None:
                _refuse("lifecycle", "LC-PROGRESS-AFTER-TERMINAL")
            state = message["job"]["state"]
            if state not in PROGRESS_TRANSITIONS.get(previous_state, set()):
                _refuse("lifecycle", "LC-STATE-TRANSITION")
            progress = message["job"]["progress"]
            if progress < previous_progress:
                _refuse("lifecycle", "LC-PROGRESS-REGRESSION")
            previous_state = state
            previous_progress = progress
        elif schema == "kilix.background-removal.result/v2":
            if terminal is not None or accepted is not None:
                _refuse("lifecycle", "LC-TERMINAL-EXCLUSIVE")
            validate_result_semantics(request, message)
            terminal = ("committed", message["job"]["sequence"])
        elif schema == "kilix.background-removal.error/v2":
            if terminal is not None:
                _refuse("lifecycle", "LC-TERMINAL-EXCLUSIVE")
            job = message["job"]
            state = job["state"]
            if accepted is not None:
                if state != "cancelled" or job["code"] != "job.cancelled":
                    _refuse("lifecycle", "LC-ACCEPTED-CANCEL-TERMINAL")
                if job["sequence"] <= accepted["linearization_sequence"]:
                    _refuse("lifecycle", "LC-ACCEPTED-CANCEL-SEQUENCE")
            terminal = (state, job["sequence"])
        elif schema == "kilix.media-job.cancel-request/v2":
            cancellation_id = message["cancellation_id"]
            encoded = canonical_bytes(message)
            if cancel_requests and cancellation_id not in cancel_requests:
                _refuse("lifecycle", "LC-SECOND-CANCELLATION-ID")
            if cancellation_id in cancel_requests and cancel_requests[cancellation_id] != encoded:
                _refuse("lifecycle", "LC-CHANGED-CANCEL-REPLAY")
            cancel_requests[cancellation_id] = encoded
        elif schema == "kilix.media-job.cancel-outcome/v2":
            cancellation_id = message["cancellation_id"]
            if cancellation_id not in cancel_requests:
                _refuse("lifecycle", "LC-OUTCOME-WITHOUT-REQUEST")
            cancel_outcomes[cancellation_id] = message
            if message["outcome"] == "accepted":
                if terminal is not None:
                    _refuse("lifecycle", "LC-ACCEPTED-AFTER-TERMINAL")
                accepted = message
            else:
                if terminal is None:
                    _refuse("lifecycle", "LC-TERMINAL-WON-MISSING")
                if (
                    message["terminal_state"] != terminal[0]
                    or message["terminal_sequence"] != terminal[1]
                ):
                    _refuse("lifecycle", "LC-TERMINAL-WON-JOIN")
                if message["terminal_sequence"] >= message["linearization_sequence"]:
                    _refuse("lifecycle", "LC-TERMINAL-WON-ORDER")
    if cancel_requests.keys() != cancel_outcomes.keys():
        _refuse("lifecycle", "LC-MISSING-CANCEL-OUTCOME")
    if terminal is None:
        _refuse("lifecycle", "LC-MISSING-TERMINAL")
    if accepted is not None and terminal[0] != "cancelled":
        _refuse("lifecycle", "LC-ACCEPTED-CANCEL-TERMINAL")


def _message_request_id(message: Mapping[str, Any]) -> str:
    if "job" in message:
        return cast(str, message["job"]["request_id"])
    return cast(str, message["request_id"])


def _message_event(message: Mapping[str, Any]) -> tuple[int, str] | None:
    schema = message["schema"]
    if schema == "kilix.background-removal.progress/v2":
        return message["job"]["sequence"], message["job"]["state"]
    if schema == "kilix.background-removal.result/v2":
        return message["job"]["sequence"], "committed"
    if schema == "kilix.background-removal.error/v2":
        return message["job"]["sequence"], message["job"]["state"]
    if schema == "kilix.media-job.cancel-outcome/v2":
        return message["linearization_sequence"], "cancel-outcome"
    return None


def process_foreground_plane(case: Mapping[str, Any]) -> list[int]:
    width = cast(int, case["width"])
    height = cast(int, case["height"])
    plane = case["model_plane"]
    if not isinstance(plane, list) or len(plane) != width * height:
        _refuse("pixel", "PX-GEOMETRY")
    prepared: list[int] = []
    for sample in plane:
        if not isinstance(sample, int | float) or isinstance(sample, bool):
            _refuse("pixel", "PX-NONFINITE")
        numeric = float(sample)
        if not math.isfinite(numeric):
            _refuse("pixel", "PX-NONFINITE")
        a8 = math.floor(max(0.0, min(1.0, numeric)) * 255 + 0.5)
        if a8 < case["threshold_u8"]:
            prepared.append(0)
        elif case["matting_mode"] == "none":
            prepared.append(255)
        else:
            prepared.append(a8)
    radius = cast(int, case["feather_radius_px"])
    if radius:
        source = prepared
        prepared = []
        sample_count = (radius * 2 + 1) ** 2
        if width == 1 and height == 1:
            total = source[0] * sample_count
            prepared.append((total + sample_count // 2) // sample_count)
        else:
            for y in range(height):
                for x in range(width):
                    total = 0
                    for offset_y in range(-radius, radius + 1):
                        yy = min(height - 1, max(0, y + offset_y))
                        for offset_x in range(-radius, radius + 1):
                            xx = min(width - 1, max(0, x + offset_x))
                            total += source[yy * width + xx]
                    prepared.append((total + sample_count // 2) // sample_count)
    if case["preserve_source_alpha"]:
        alpha = case["source_alpha"]
        prepared = [
            min(value, source_alpha) for value, source_alpha in zip(prepared, alpha, strict=True)
        ]
    return prepared


def png_profile_accepted(case: Mapping[str, Any]) -> bool:
    chunks = case["chunks"]
    return bool(
        case["color_type"] == 0
        and case["bit_depth"] == 8
        and case["interlace"] == 0
        and chunks
        and chunks[0] == "IHDR"
        and chunks[-1] == "IEND"
        and chunks.count("IHDR") == 1
        and chunks.count("IEND") == 1
        and chunks.count("IDAT") >= 1
        and set(chunks) <= {"IHDR", "IDAT", "IEND"}
    )


def decode_gray8_png(data: bytes) -> tuple[int, int, list[int]]:
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        _refuse("pixel", "PX-PNG-PROFILE")
    offset = len(signature)
    chunks: list[tuple[str, bytes]] = []
    while offset < len(data):
        if offset + 12 > len(data):
            _refuse("pixel", "PX-PNG-DECODE")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        end = offset + length + 12
        if end > len(data):
            _refuse("pixel", "PX-PNG-DECODE")
        kind_bytes = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        if zlib.crc32(kind_bytes + payload) & 0xFFFFFFFF != expected_crc:
            _refuse("pixel", "PX-PNG-CRC")
        chunks.append((kind_bytes.decode("ascii", errors="replace"), payload))
        offset = end
    kinds = [kind for kind, _payload in chunks]
    if not (
        chunks
        and kinds[0] == "IHDR"
        and kinds[-1] == "IEND"
        and kinds.count("IHDR") == 1
        and kinds.count("IEND") == 1
        and kinds.count("IDAT") >= 1
        and set(kinds) <= {"IHDR", "IDAT", "IEND"}
        and len(chunks[0][1]) == 13
        and chunks[-1][1] == b""
    ):
        _refuse("pixel", "PX-PNG-PROFILE")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", chunks[0][1]
    )
    if (
        width < 1
        or height < 1
        or depth != 8
        or color != 0
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        _refuse("pixel", "PX-PNG-PROFILE")
    compressed = b"".join(payload for kind, payload in chunks if kind == "IDAT")
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise ContractRefusal("pixel", "PX-PNG-DECODE") from exc
    if len(raw) != height * (width + 1):
        _refuse("pixel", "PX-PNG-DECODE")
    pixels: list[int] = []
    for row in range(height):
        start = row * (width + 1)
        if raw[start] != 0:
            _refuse("pixel", "PX-PNG-DECODE")
        pixels.extend(raw[start + 1 : start + 1 + width])
    return width, height, pixels


def mutable_documents(runtime: ContractRuntime) -> dict[str, dict[str, Any]]:
    return deepcopy(runtime.documents)


def _refuse(stage: str, rule_id: str, detail: str = "") -> NoReturn:
    raise ContractRefusal(stage, rule_id, detail)
