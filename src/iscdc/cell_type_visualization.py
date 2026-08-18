"""Strict sidecar snapshots for offline cell-type visualizations.

The catalogue process keeps only the immutable generation's small manifest in memory.
Point data and annotation output remain in files under the sidecar root.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gzip
import hashlib
import json
import logging
import math
import os
import re
import shutil
import struct
import tempfile
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from itertools import chain
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping

STATUS_VERSION = 1
MANIFEST_VERSION = 1
REPORT_VERSION = 1
REPORT_SCHEMA_VERSION = REPORT_VERSION
FAILURE_REPORT_VERSION = 1
POINT_FORMAT_VERSION = 1
POINT_MAGIC = b"ISCDCCT\0"
POINT_MEDIA_TYPE = "application/vnd.iscdc.cell-type-points"
POINT_FLAG_CONFIDENCE = 0x0001
POINT_SUPPORTED_FLAGS = POINT_FLAG_CONFIDENCE
POINT_HEADER = struct.Struct("<8sHHIQQ")
POINT_HEADER_SIZE = POINT_HEADER.size

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
_SAFE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COLOR = re.compile(r"#[0-9A-Fa-f]{6}\Z")
_CL_ID = re.compile(r"CL:[0-9]{7}\Z")
_ENCODINGS = frozenset(("identity", "gzip", "br"))
_ENCODING_SUFFIXES = {"identity": ".bin", "gzip": ".bin.gz", "br": ".bin.br"}
_LOG = logging.getLogger(__name__)


class CellTypeVisualizationError(ValueError):
    """Raised when a sidecar artifact is unsafe, stale, or inconsistent."""


@dataclass(frozen=True)
class PointHeader:
    version: int
    flags: int
    count: int
    payload_offset: int
    total_size: int

    @property
    def has_confidence(self) -> bool:
        return bool(self.flags & POINT_FLAG_CONFIDENCE)


@dataclass(frozen=True)
class PointData:
    x: tuple[float, ...]
    y: tuple[float, ...]
    type_ids: tuple[int, ...]
    confidence: tuple[float, ...] | None

    @property
    def point_count(self) -> int:
        return len(self.x)


@dataclass(frozen=True)
class CellTypeCategory:
    type_id: int
    label: str
    color: str
    count: int
    cell_ontology_id: str | None = None
    state: str = "biological"


# Kept as a descriptive public spelling for producers.
CellTypeDefinition = CellTypeCategory


@dataclass(frozen=True)
class PointRepresentation:
    path: Path
    encoding: str
    size: int
    sha256: str
    content_size: int
    content_sha256: str

    @property
    def name(self) -> str:
        return self.path.name


CellTypePointFile = PointRepresentation


@dataclass(frozen=True)
class CellTypeSample:
    key: str
    id: str
    count: int
    bounds: tuple[float, float, float, float]
    category_counts: Mapping[int, int]
    representations: Mapping[str, PointRepresentation]

    @property
    def label(self) -> str:
        return self.id

    @property
    def file(self) -> PointRepresentation:
        return self.resolve("identity")

    def resolve(self, encoding: str = "identity") -> PointRepresentation:
        """Resolve a controlled encoding name; never accepts a filesystem path."""

        try:
            return self.representations[encoding]
        except (KeyError, TypeError) as exc:
            raise CellTypeVisualizationError(
                f"Sample {self.key!r} has no {encoding!r} point representation."
            ) from exc


@dataclass(frozen=True)
class CellTypeVisualization:
    dataset_id: str
    generation_id: str
    generated_at: str
    annotation_kind: str
    method: str
    coordinate_system: str
    coordinate_unit: str
    y_axis: str
    samples: Mapping[str, CellTypeSample]
    categories: tuple[CellTypeCategory, ...]
    manifest_path: Path
    report_path: Path
    inference_path: Path | None
    report: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @property
    def generation_dir(self) -> Path:
        return self.manifest_path.parent

    @property
    def cell_types(self) -> tuple[CellTypeCategory, ...]:
        return self.categories

    @property
    def has_confidence(self) -> bool:
        return self.annotation_kind == "inferred"


def _error(message: str) -> None:
    raise CellTypeVisualizationError(message)


def _strict_object(
    value: object,
    field: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _error(f"{field} must be an object.")
    keys = set(value)
    missing = required - keys
    unknown = keys - required - (optional or set())
    if missing:
        _error(f"{field} is missing fields: {', '.join(sorted(missing))}.")
    if unknown:
        _error(f"{field} contains unknown fields: {', '.join(sorted(unknown))}.")
    return value


def _string(value: object, field: str, *, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _error(f"{field} must be a non-blank, whitespace-trimmed string.")
    if len(value) > maximum:
        _error(f"{field} is too long.")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _error(f"{field} must be an integer greater than or equal to {minimum}.")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{field} must be numeric.")
    try:
        result = float(value)
    except OverflowError as exc:
        raise CellTypeVisualizationError(f"{field} is outside Float32 range.") from exc
    if not math.isfinite(result):
        _error(f"{field} must be finite.")
    if abs(result) > 3.4028234663852886e38:
        _error(f"{field} is outside Float32 range.")
    return result


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _error(f"{field} must be a lowercase SHA-256 digest.")
    return value


def _safe_name(value: object, field: str, *, key: bool = False) -> str:
    result = _string(value, field, maximum=200)
    pattern = _SAFE_KEY if key else _SAFE_NAME
    if pattern.fullmatch(result) is None or result in {".", ".."}:
        _error(f"{field} is not a safe name.")
    return result


def _timestamp(value: object, field: str) -> str:
    result = _string(value, field, maximum=64)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError:
        _error(f"{field} must be an ISO-8601 timestamp.")
    if parsed.tzinfo is None:
        _error(f"{field} must include a timezone.")
    return result


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise CellTypeVisualizationError("Document is not valid canonical JSON.") from exc


def _float_sequence(values: Iterable[object], field: str) -> list[float]:
    try:
        result = [_finite(value, f"{field}[{index}]") for index, value in enumerate(values)]
    except TypeError as exc:
        raise CellTypeVisualizationError(f"{field} must be iterable.") from exc
    return result


def encode_points(
    x: Iterable[object],
    y: Iterable[object],
    type_ids: Iterable[object],
    confidence: Iterable[object] | None = None,
) -> bytes:
    """Encode deterministic little-endian, structure-of-arrays point binary v1."""

    xs = _float_sequence(x, "x")
    ys = _float_sequence(y, "y")
    try:
        types = list(type_ids)
    except TypeError as exc:
        raise CellTypeVisualizationError("type_ids must be iterable.") from exc
    count = len(xs)
    if len(ys) != count or len(types) != count:
        _error("x, y, and type_ids must contain the same number of values.")
    if count > 0xFFFFFFFF:
        _error("Point count exceeds binary format v1 capacity.")
    clean_types: list[int] = []
    for index, value in enumerate(types):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFF:
            _error(f"type_ids[{index}] must be an integer between 0 and 65535.")
        clean_types.append(value)
    confidences: list[float] | None = None
    flags = 0
    if confidence is not None:
        confidences = _float_sequence(confidence, "confidence")
        if len(confidences) != count:
            _error("confidence must contain the same number of values as x.")
        if any(value < 0.0 or value > 1.0 for value in confidences):
            _error("confidence values must be between 0 and 1.")
        flags = POINT_FLAG_CONFIDENCE
    stride = 14 if confidences is not None else 10
    total_size = POINT_HEADER_SIZE + count * stride
    payload = bytearray(
        POINT_HEADER.pack(
            POINT_MAGIC,
            POINT_FORMAT_VERSION,
            flags,
            count,
            POINT_HEADER_SIZE,
            total_size,
        )
    )
    for values, code in ((xs, "f"), (ys, "f")):
        if values:
            payload.extend(struct.pack(f"<{count}{code}", *values))
    if confidences:
        payload.extend(struct.pack(f"<{count}f", *confidences))
    if clean_types:
        payload.extend(struct.pack(f"<{count}H", *clean_types))
    return bytes(payload)


def validate_point_header(
    payload: bytes | bytearray | memoryview,
    *,
    expected_count: int | None = None,
    expected_confidence: bool | None = None,
) -> PointHeader:
    """Validate binary v1's header and exact payload length."""

    view = memoryview(payload)
    if len(view) < POINT_HEADER_SIZE:
        _error("Point payload is shorter than its 32-byte header.")
    magic, version, flags, count, offset, total_size = POINT_HEADER.unpack_from(view)
    if magic != POINT_MAGIC:
        _error("Point payload has invalid magic.")
    if version != POINT_FORMAT_VERSION:
        _error(f"Unsupported point format version {version}.")
    if flags & ~POINT_SUPPORTED_FLAGS:
        _error("Point payload contains unsupported flags.")
    if offset != POINT_HEADER_SIZE:
        _error("Point payload has an invalid payload offset.")
    has_confidence = bool(flags & POINT_FLAG_CONFIDENCE)
    expected_size = POINT_HEADER_SIZE + count * (14 if has_confidence else 10)
    if total_size != expected_size or total_size != len(view):
        _error("Point payload total size is inconsistent.")
    if expected_count is not None and count != expected_count:
        _error("Point payload count does not match the manifest.")
    if expected_confidence is not None and has_confidence != expected_confidence:
        _error("Point payload confidence flag does not match the annotation kind.")
    return PointHeader(version, flags, count, offset, total_size)


def decode_points(payload: bytes | bytearray | memoryview) -> PointData:
    """Strictly decode point binary v1 into immutable Python tuples."""

    view = memoryview(payload)
    header = validate_point_header(view)
    count = header.count
    offset = header.payload_offset

    def unpack(code: str, width: int) -> tuple[Any, ...]:
        nonlocal offset
        if count == 0:
            return ()
        result = struct.unpack_from(f"<{count}{code}", view, offset)
        offset += count * width
        return result

    xs = unpack("f", 4)
    ys = unpack("f", 4)
    confidences = unpack("f", 4) if header.has_confidence else None
    type_ids = unpack("H", 2)
    if not all(math.isfinite(value) for value in chain(xs, ys)):
        _error("Point coordinates must be finite.")
    if confidences is not None and not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in confidences
    ):
        _error("Point confidence values must be finite and between 0 and 1.")
    return PointData(xs, ys, type_ids, confidences)


def _read_json(path: Path, field: str, *, maximum: int = 2_000_000) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _error(f"{field} is missing or unsafe.")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CellTypeVisualizationError(f"Unable to read {field}.") from exc
    if len(raw) > maximum:
        _error(f"{field} is too large.")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CellTypeVisualizationError(f"{field} is not valid JSON.") from exc
    if not isinstance(value, dict):
        _error(f"{field} must be an object.")
    return value


def _safe_file(generation_dir: Path, value: object, field: str) -> Path:
    name = _string(value, field, maximum=300)
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(
            part in {"", ".", ".."} or _SAFE_NAME.fullmatch(part) is None for part in relative.parts
        )
    ):
        _error(f"{field} is not a safe relative path.")
    path = generation_dir.joinpath(*relative.parts)
    if (
        any(
            generation_dir.joinpath(*relative.parts[:index]).is_symlink()
            for index in range(1, len(relative.parts) + 1)
        )
        or not path.is_file()
    ):
        _error(f"{field} does not identify a regular file.")
    try:
        resolved = path.resolve(strict=True)
        root = generation_dir.resolve(strict=True)
    except OSError as exc:
        raise CellTypeVisualizationError(f"Unable to resolve {field}.") from exc
    if resolved.parent != root and root not in resolved.parents:
        _error(f"{field} escapes its generation directory.")
    return resolved


def _file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise CellTypeVisualizationError(f"Unable to hash {path.name!r}.") from exc
    return size, digest.hexdigest()


@lru_cache(maxsize=1)
def _brotli_decoder() -> Any:
    library_name = ctypes.util.find_library("brotlidec")
    if not library_name:
        _error("Brotli validation is unavailable on this system.")
    try:
        library = ctypes.CDLL(library_name)
        function = library.BrotliDecoderDecompress
    except (OSError, AttributeError) as exc:
        raise CellTypeVisualizationError(
            "Brotli validation is unavailable on this system."
        ) from exc
    function.argtypes = (
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    )
    function.restype = ctypes.c_int
    return function


def _decompress_brotli(payload: bytes, expected_size: int) -> bytes:
    output = ctypes.create_string_buffer(max(1, expected_size))
    output_size = ctypes.c_size_t(expected_size)
    source = ctypes.create_string_buffer(payload)
    result = _brotli_decoder()(
        len(payload),
        ctypes.cast(source, ctypes.c_void_p),
        ctypes.byref(output_size),
        ctypes.cast(output, ctypes.c_void_p),
    )
    if result != 1 or output_size.value != expected_size:
        _error("Brotli point representation is invalid or has the wrong content size.")
    return output.raw[:expected_size]


@lru_cache(maxsize=1)
def _brotli_encoder() -> tuple[Any, Any]:
    library_name = ctypes.util.find_library("brotlienc")
    if not library_name:
        _error("Brotli encoding is unavailable on this system.")
    try:
        library = ctypes.CDLL(library_name)
        maximum_size = library.BrotliEncoderMaxCompressedSize
        compress = library.BrotliEncoderCompress
    except (OSError, AttributeError) as exc:
        raise CellTypeVisualizationError("Brotli encoding is unavailable on this system.") from exc
    maximum_size.argtypes = (ctypes.c_size_t,)
    maximum_size.restype = ctypes.c_size_t
    compress.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    )
    compress.restype = ctypes.c_int
    return maximum_size, compress


def _compress_brotli(payload: bytes) -> bytes:
    maximum_size, compress = _brotli_encoder()
    capacity = maximum_size(len(payload))
    output = ctypes.create_string_buffer(max(1, capacity))
    output_size = ctypes.c_size_t(capacity)
    source = ctypes.create_string_buffer(payload)
    # Quality 9 and lgwin 22 are fixed parts of the representation contract.
    result = compress(
        9,
        22,
        0,
        len(payload),
        ctypes.cast(source, ctypes.c_void_p),
        ctypes.byref(output_size),
        ctypes.cast(output, ctypes.c_void_p),
    )
    if result != 1:
        _error("Unable to encode the Brotli point representation.")
    return output.raw[: output_size.value]


def build_point_representations(payload: bytes) -> dict[str, bytes]:
    """Return canonical identity, deterministic gzip, and Brotli representations."""

    if not isinstance(payload, bytes):
        _error("Point payload must be bytes.")
    validate_point_header(payload)
    return {
        "identity": payload,
        "gzip": gzip.compress(payload, compresslevel=9, mtime=0),
        "br": _compress_brotli(payload),
    }


def _identity_payload(encoding: str, payload: bytes, expected_size: int) -> bytes:
    if encoding == "identity":
        result = payload
    elif encoding == "gzip":
        try:
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            result = decompressor.decompress(payload, expected_size + 1)
            if len(result) <= expected_size and not decompressor.unconsumed_tail:
                result += decompressor.flush(expected_size + 1 - len(result))
            if (
                len(result) != expected_size
                or decompressor.unconsumed_tail
                or not decompressor.eof
                or decompressor.unused_data
            ):
                _error("Gzip point representation has invalid framing or content size.")
        except zlib.error as exc:
            raise CellTypeVisualizationError("Gzip point representation is invalid.") from exc
    elif encoding == "br":
        result = _decompress_brotli(payload, expected_size)
    else:  # guarded by manifest validation
        _error(f"Unsupported encoding {encoding!r}.")
    if len(result) != expected_size:
        _error("Point representation content size does not match its manifest.")
    return result


def _dataset_value(dataset: object, *names: str, default: object = None) -> object:
    if isinstance(dataset, Mapping):
        for name in names:
            if name in dataset:
                return dataset[name]
        return default
    for name in names:
        if hasattr(dataset, name):
            return getattr(dataset, name)
    return default


@dataclass(frozen=True)
class _DatasetExpectation:
    dataset_id: str
    source_sha256: str
    obs_order_sha256: str | None
    observation_count: int
    coordinate_dimensions: int
    sample_ids: tuple[str, ...]
    dataset_type: str


def _dataset_expectation(
    dataset: object, dataset_id_hint: str | None = None
) -> _DatasetExpectation:
    dataset_id = _safe_name(
        _dataset_value(dataset, "dataset_id", "id", default=dataset_id_hint),
        "dataset.dataset_id",
    )
    source_sha = _sha256(_dataset_value(dataset, "sha256", "source_sha256"), "dataset.sha256")
    raw_obs_order_sha = _dataset_value(dataset, "obs_order_sha256")
    obs_order_sha = (
        _sha256(raw_obs_order_sha, "dataset.obs_order_sha256")
        if raw_obs_order_sha is not None
        else None
    )
    observation_count = _integer(
        _dataset_value(dataset, "n_obs", "observation_count"), "dataset.n_obs"
    )
    coordinate_dimensions = _integer(
        _dataset_value(dataset, "coordinate_dimensions"),
        "dataset.coordinate_dimensions",
    )
    raw_sample_ids = _dataset_value(dataset, "sample_ids")
    if not isinstance(raw_sample_ids, (list, tuple)) or not raw_sample_ids:
        _error("dataset.sample_ids must be a non-empty array.")
    sample_ids = tuple(
        _string(value, f"dataset.sample_ids[{index}]", maximum=200)
        for index, value in enumerate(raw_sample_ids)
    )
    if len(set(sample_ids)) != len(sample_ids):
        _error("dataset.sample_ids must be unique.")
    dataset_type = _string(
        _dataset_value(dataset, "dataset_type", default="full"), "dataset.dataset_type"
    )
    return _DatasetExpectation(
        dataset_id,
        source_sha,
        obs_order_sha,
        observation_count,
        coordinate_dimensions,
        sample_ids,
        dataset_type,
    )


def _validate_file_record(
    generation_dir: Path,
    value: object,
    field: str,
    *,
    required_path: str | None = None,
) -> tuple[Path, int, str]:
    record = _strict_object(
        value,
        field,
        required={"path", "size", "sha256"},
    )
    path = _safe_file(generation_dir, record["path"], f"{field}.path")
    if required_path is not None and record["path"] != required_path:
        _error(f"{field}.path must be {required_path!r}.")
    size = _integer(record["size"], f"{field}.size")
    digest = _sha256(record["sha256"], f"{field}.sha256")
    actual_size, actual_digest = _file_digest(path)
    if (actual_size, actual_digest) != (size, digest):
        _error(f"{field} size or SHA-256 does not match its file.")
    return path, size, digest


def _validate_representation(
    generation_dir: Path,
    value: object,
    field: str,
    *,
    encoding: str,
    expected_count: int,
    expected_confidence: bool,
    expected_bounds: tuple[float, float, float, float],
    category_ids: set[int],
    expected_category_counts: Mapping[int, int],
) -> tuple[PointRepresentation, bytes]:
    record = _strict_object(
        value,
        field,
        required={"path", "encoding", "size", "sha256", "content_size", "content_sha256"},
    )
    if record["encoding"] != encoding or encoding not in _ENCODINGS:
        _error(f"{field}.encoding is invalid.")
    path = _safe_file(generation_dir, record["path"], f"{field}.path")
    if not str(record["path"]).endswith(_ENCODING_SUFFIXES[encoding]):
        _error(f"{field}.path has the wrong suffix for {encoding}.")
    size = _integer(record["size"], f"{field}.size")
    digest = _sha256(record["sha256"], f"{field}.sha256")
    content_size = _integer(record["content_size"], f"{field}.content_size")
    expected_content_size = POINT_HEADER_SIZE + expected_count * (14 if expected_confidence else 10)
    if content_size != expected_content_size:
        _error(f"{field}.content_size is inconsistent with its point count.")
    content_digest = _sha256(record["content_sha256"], f"{field}.content_sha256")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CellTypeVisualizationError(f"Unable to read {field}.") from exc
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        _error(f"{field} size or SHA-256 does not match its file.")
    identity = _identity_payload(encoding, payload, content_size)
    if hashlib.sha256(identity).hexdigest() != content_digest:
        _error(f"{field} uncompressed SHA-256 does not match its manifest.")
    points = decode_points(identity)
    validate_point_header(
        identity,
        expected_count=expected_count,
        expected_confidence=expected_confidence,
    )
    observed = {category_id: 0 for category_id in category_ids}
    for type_id in points.type_ids:
        if type_id not in category_ids:
            _error(f"{field} refers to an unknown category ID.")
        observed[type_id] += 1
    if observed != dict(expected_category_counts):
        _error(f"{field} category counts do not match the manifest.")
    if expected_count:
        observed_bounds = (
            min(points.x),
            min(points.y),
            max(points.x),
            max(points.y),
        )
        if not all(
            math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
            for actual, expected in zip(observed_bounds, expected_bounds, strict=True)
        ):
            _error(f"{field} coordinate bounds do not match the manifest.")
    return (
        PointRepresentation(path, encoding, size, digest, content_size, content_digest),
        identity,
    )


def _load_manifest(
    generation_dir: Path,
    expectation: _DatasetExpectation,
) -> CellTypeVisualization:
    manifest_path = generation_dir / "manifest.json"
    manifest = _strict_object(
        _read_json(manifest_path, "manifest.json"),
        "manifest",
        required={
            "manifest_version",
            "dataset_id",
            "generation_id",
            "generated_at",
            "source",
            "annotation",
            "coordinates",
            "categories",
            "samples",
            "report",
            "provenance",
        },
        optional={"inference"},
    )
    if manifest["manifest_version"] != MANIFEST_VERSION:
        _error("Unsupported cell-type visualization manifest version.")
    if manifest["dataset_id"] != expectation.dataset_id:
        _error("Manifest dataset ID does not match the catalogue.")
    if manifest["generation_id"] != generation_dir.name:
        _error("Manifest generation ID does not match its directory.")
    _safe_name(manifest["generation_id"], "manifest.generation_id")
    generated_at = _timestamp(manifest["generated_at"], "manifest.generated_at")

    source = _strict_object(
        manifest["source"],
        "manifest.source",
        required={
            "sha256",
            "obs_order_sha256",
            "observation_count",
            "coordinate_dimensions",
            "sample_ids",
        },
    )
    if _sha256(source["sha256"], "manifest.source.sha256") != expectation.source_sha256:
        _error("Visualization source SHA-256 is stale.")
    obs_order_sha = _sha256(source["obs_order_sha256"], "manifest.source.obs_order_sha256")
    if expectation.obs_order_sha256 is not None and obs_order_sha != expectation.obs_order_sha256:
        _error("Visualization observation order is stale.")
    if (
        _integer(source["observation_count"], "manifest.source.observation_count")
        != expectation.observation_count
    ):
        _error("Visualization observation count is stale.")
    if (
        _integer(
            source["coordinate_dimensions"],
            "manifest.source.coordinate_dimensions",
        )
        != 2
        or expectation.coordinate_dimensions != 2
    ):
        _error("Cell-type visualizations require exactly two coordinate dimensions.")
    if not isinstance(source["sample_ids"], list):
        _error("manifest.source.sample_ids must be an array.")
    source_sample_ids = tuple(
        _string(value, f"manifest.source.sample_ids[{index}]", maximum=200)
        for index, value in enumerate(source["sample_ids"])
    )
    if source_sample_ids != expectation.sample_ids:
        _error("Visualization sample IDs are stale or reordered.")

    annotation = _strict_object(
        manifest["annotation"],
        "manifest.annotation",
        required={"kind", "method"},
    )
    annotation_kind = annotation["kind"]
    if annotation_kind not in {"source", "inferred"}:
        _error("manifest.annotation.kind must be 'source' or 'inferred'.")
    method = _string(annotation["method"], "manifest.annotation.method", maximum=200)

    coordinates = _strict_object(
        manifest["coordinates"],
        "manifest.coordinates",
        required={"system", "unit", "y_axis"},
    )
    coordinate_system = _string(coordinates["system"], "manifest.coordinates.system")
    coordinate_unit = _string(coordinates["unit"], "manifest.coordinates.unit")
    y_axis = coordinates["y_axis"]
    if y_axis not in {"up", "down"}:
        _error("manifest.coordinates.y_axis must be 'up' or 'down'.")

    if not isinstance(manifest["categories"], list) or not manifest["categories"]:
        _error("manifest.categories must be a non-empty array.")
    categories: list[CellTypeCategory] = []
    category_ids: set[int] = set()
    labels: set[str] = set()
    for index, raw_category in enumerate(manifest["categories"]):
        field = f"manifest.categories[{index}]"
        category = _strict_object(
            raw_category,
            field,
            required={"type_id", "label", "color", "count", "state"},
            optional={"cell_ontology_id"},
        )
        type_id = _integer(category["type_id"], f"{field}.type_id")
        if type_id > 0xFFFF or type_id != index:
            _error(f"{field}.type_id must be its zero-based category index.")
        category_ids.add(type_id)
        label = _string(category["label"], f"{field}.label", maximum=200)
        if label in labels:
            _error("Category labels must be unique.")
        labels.add(label)
        color = _string(category["color"], f"{field}.color", maximum=7)
        if _COLOR.fullmatch(color) is None:
            _error(f"{field}.color must be a six-digit hex color.")
        count = _integer(category["count"], f"{field}.count")
        state = category["state"]
        if state not in {"biological", "mixed", "uncertain"}:
            _error(f"{field}.state is invalid.")
        ontology_id = category.get("cell_ontology_id")
        if ontology_id is not None and (
            not isinstance(ontology_id, str) or _CL_ID.fullmatch(ontology_id) is None
        ):
            _error(f"{field}.cell_ontology_id must be a stable CL identifier.")
        if state == "mixed" and label != "Mixed":
            _error("The mixed prediction state must be labelled 'Mixed'.")
        if state == "uncertain" and label != "Uncertain":
            _error("The uncertain prediction state must be labelled 'Uncertain'.")
        if state != "biological" and ontology_id is not None:
            _error("Mixed and Uncertain states must not have Cell Ontology IDs.")
        if annotation_kind == "source" and state != "biological":
            _error("Source categories may not use inferred prediction states.")
        if annotation_kind == "inferred" and label in {"Mixed", "Uncertain"}:
            if state != label.lower():
                _error("Reserved inferred labels must use their matching prediction state.")
        if annotation_kind == "inferred" and state == "biological" and ontology_id is None:
            _error("Inferred biological categories require stable Cell Ontology IDs.")
        categories.append(
            CellTypeCategory(type_id, label, color.upper(), count, ontology_id, state)
        )
    if sum(category.count for category in categories) != expectation.observation_count:
        _error("Category counts do not equal the source observation count.")

    if not isinstance(manifest["samples"], list) or not manifest["samples"]:
        _error("manifest.samples must be a non-empty array.")
    samples: dict[str, CellTypeSample] = {}
    sample_ids: list[str] = []
    aggregate_counts = {category_id: 0 for category_id in category_ids}
    for index, raw_sample in enumerate(manifest["samples"]):
        field = f"manifest.samples[{index}]"
        sample = _strict_object(
            raw_sample,
            field,
            required={"key", "id", "count", "bounds", "category_counts", "representations"},
        )
        key = _safe_name(sample["key"], f"{field}.key", key=True)
        if key in samples:
            _error("Sample keys must be unique.")
        sample_id = _string(sample["id"], f"{field}.id", maximum=200)
        sample_ids.append(sample_id)
        count = _integer(sample["count"], f"{field}.count")
        bounds_raw = sample["bounds"]
        if not isinstance(bounds_raw, list) or len(bounds_raw) != 4:
            _error(f"{field}.bounds must contain [min_x, min_y, max_x, max_y].")
        bounds = tuple(
            _finite(value, f"{field}.bounds[{position}]")
            for position, value in enumerate(bounds_raw)
        )
        if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
            _error(f"{field}.bounds are inverted.")
        if not isinstance(sample["category_counts"], dict):
            _error(f"{field}.category_counts must be an object.")
        category_counts: dict[int, int] = {}
        for raw_id, raw_count in sample["category_counts"].items():
            if not isinstance(raw_id, str) or not raw_id.isascii() or not raw_id.isdecimal():
                _error(f"{field}.category_counts keys must be decimal category IDs.")
            category_id = int(raw_id)
            if category_id not in category_ids or category_id in category_counts:
                _error(f"{field}.category_counts refers to an unknown category ID.")
            category_counts[category_id] = _integer(raw_count, f"{field}.category_counts.{raw_id}")
        if set(category_counts) != category_ids or sum(category_counts.values()) != count:
            _error(f"{field}.category_counts must cover all categories and sum to count.")
        for category_id, category_count in category_counts.items():
            aggregate_counts[category_id] += category_count

        raw_representations = sample["representations"]
        if not isinstance(raw_representations, dict) or set(raw_representations) != _ENCODINGS:
            _error(f"{field}.representations must contain exactly identity, gzip, and br.")
        representations: dict[str, PointRepresentation] = {}
        identity_digest: str | None = None
        identity_size: int | None = None
        for encoding, raw_representation in raw_representations.items():
            representation, identity = _validate_representation(
                generation_dir,
                raw_representation,
                f"{field}.representations.{encoding}",
                encoding=encoding,
                expected_count=count,
                expected_confidence=annotation_kind == "inferred",
                expected_bounds=bounds,  # type: ignore[arg-type]
                category_ids=category_ids,
                expected_category_counts=category_counts,
            )
            digest = hashlib.sha256(identity).hexdigest()
            if identity_digest is None:
                identity_digest, identity_size = digest, len(identity)
            elif (digest, len(identity)) != (identity_digest, identity_size):
                _error(f"{field} representations do not encode identical content.")
            representations[encoding] = representation
        samples[key] = CellTypeSample(
            key,
            sample_id,
            count,
            bounds,  # type: ignore[arg-type]
            MappingProxyType(category_counts),
            MappingProxyType(representations),
        )
    if tuple(sample_ids) != expectation.sample_ids:
        _error("Manifest samples do not match source sample IDs in order.")
    if sum(sample.count for sample in samples.values()) != expectation.observation_count:
        _error("Sample counts do not equal the source observation count.")
    expected_global = {category.type_id: category.count for category in categories}
    if aggregate_counts != expected_global:
        _error("Per-sample category counts do not match global category counts.")

    report_path, _, _ = _validate_file_record(
        generation_dir, manifest["report"], "manifest.report", required_path="report.json"
    )
    report = _strict_object(
        _read_json(report_path, "report.json"),
        "report",
        required={
            "report_version",
            "dataset_id",
            "generation_id",
            "source_sha256",
            "status",
            "quality_control",
            "thresholds",
            "warnings",
        },
    )
    if report["report_version"] != REPORT_VERSION:
        _error("Unsupported cell-type visualization report version.")
    if report["dataset_id"] != expectation.dataset_id:
        _error("Report dataset ID does not match the manifest.")
    if report["generation_id"] != generation_dir.name:
        _error("Report generation ID does not match the manifest.")
    if _sha256(report["source_sha256"], "report.source_sha256") != expectation.source_sha256:
        _error("Report source SHA-256 does not match the manifest.")
    if report["status"] != "passed":
        _error("A successful generation requires a passed report.")
    for field_name in ("quality_control", "thresholds"):
        if not isinstance(report[field_name], dict):
            _error(f"report.{field_name} must be an object.")
        _canonical_json(report[field_name])
    if not isinstance(report["warnings"], list):
        _error("report.warnings must be an array.")
    for index, warning in enumerate(report["warnings"]):
        _string(warning, f"report.warnings[{index}]", maximum=2000)
    inference_path: Path | None = None
    if annotation_kind == "inferred":
        if "inference" not in manifest:
            _error("Inferred visualizations require inference.h5.")
        inference_path, _, _ = _validate_file_record(
            generation_dir,
            manifest["inference"],
            "manifest.inference",
            required_path="inference.h5",
        )
        try:
            with inference_path.open("rb") as stream:
                signature = stream.read(8)
        except OSError as exc:
            raise CellTypeVisualizationError("Unable to read inference.h5.") from exc
        if signature != b"\x89HDF\r\n\x1a\n":
            _error("inference.h5 does not have an HDF5 signature.")
    elif "inference" in manifest:
        _error("Source visualizations must not declare inference output.")
    provenance = _strict_object(
        manifest["provenance"],
        "manifest.provenance",
        required={"environment_lock_sha256", "references", "parameters"},
    )
    _sha256(
        provenance["environment_lock_sha256"],
        "manifest.provenance.environment_lock_sha256",
    )
    if not isinstance(provenance["references"], list):
        _error("manifest.provenance.references must be an array.")
    for index, raw_reference in enumerate(provenance["references"]):
        field = f"manifest.provenance.references[{index}]"
        reference = _strict_object(
            raw_reference,
            field,
            required={"id", "version", "sha256"},
        )
        _string(reference["id"], f"{field}.id", maximum=200)
        _string(reference["version"], f"{field}.version", maximum=200)
        _sha256(reference["sha256"], f"{field}.sha256")
    if not isinstance(provenance["parameters"], dict):
        _error("manifest.provenance.parameters must be an object.")
    # Prove provenance is bounded, finite JSON without retaining mutable input.
    if len(_canonical_json(provenance)) > 1_000_000:
        _error("manifest.provenance is too large.")
    return CellTypeVisualization(
        expectation.dataset_id,
        generation_dir.name,
        generated_at,
        annotation_kind,
        method,
        coordinate_system,
        coordinate_unit,
        y_axis,
        MappingProxyType(samples),
        tuple(categories),
        manifest_path,
        report_path,
        inference_path,
        MappingProxyType(report),
        MappingProxyType(dict(provenance)),
    )


def load_cell_type_visualization(root: Path, dataset: object) -> CellTypeVisualization:
    """Load one authoritative successful generation, raising on invalid data."""

    expectation = _dataset_expectation(dataset)
    if expectation.dataset_type != "full":
        _error("Cell-type visualizations are available only for full datasets.")
    dataset_dir = Path(root) / expectation.dataset_id
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        _error("Visualization dataset directory is missing or unsafe.")
    status_path = dataset_dir / "status.json"
    status = _strict_object(
        _read_json(status_path, "status.json", maximum=64_000),
        "status",
        required={"status_version", "state", "dataset_id", "updated_at"},
        optional={
            "generation_id",
            "manifest_sha256",
            "failure_id",
            "failure_report_sha256",
        },
    )
    if status["status_version"] != STATUS_VERSION:
        _error("Unsupported visualization status version.")
    if status["dataset_id"] != expectation.dataset_id:
        _error("Visualization status dataset ID does not match the catalogue.")
    _timestamp(status["updated_at"], "status.updated_at")
    state = status["state"]
    if state == "failure":
        if set(status) != {
            "status_version",
            "state",
            "dataset_id",
            "updated_at",
            "failure_id",
            "failure_report_sha256",
        }:
            _error("Failure status must not point to a generation.")
        failure_id = _safe_name(status["failure_id"], "status.failure_id")
        expected_report_sha = _sha256(
            status["failure_report_sha256"], "status.failure_report_sha256"
        )
        failure_dir = dataset_dir / "failures" / failure_id
        if failure_dir.is_symlink() or not failure_dir.is_dir():
            _error("The status failure directory is missing or unsafe.")
        failure_report_path = failure_dir / "report.json"
        _, actual_report_sha = _file_digest(failure_report_path)
        if actual_report_sha != expected_report_sha:
            _error("Status failure report SHA-256 binding does not match its file.")
        failure_report = _strict_object(
            _read_json(failure_report_path, "failure report"),
            "failure report",
            required={
                "failure_report_version",
                "dataset_id",
                "failure_id",
                "failed_at",
                "status",
                "stage",
                "category",
                "error",
                "details",
            },
        )
        if failure_report["failure_report_version"] != FAILURE_REPORT_VERSION:
            _error("Unsupported visualization failure report version.")
        if (
            failure_report["dataset_id"] != expectation.dataset_id
            or failure_report["failure_id"] != failure_id
        ):
            _error("Failure report does not match its status.")
        _timestamp(failure_report["failed_at"], "failure report.failed_at")
        if failure_report["status"] != "failed":
            _error("Failure report status must be 'failed'.")
        _string(failure_report["stage"], "failure report.stage", maximum=200)
        _string(failure_report["category"], "failure report.category", maximum=200)
        _string(failure_report["error"], "failure report.error", maximum=2000)
        if not isinstance(failure_report["details"], dict):
            _error("failure report.details must be an object.")
        _canonical_json(failure_report["details"])
        _error("The latest cell-type visualization generation failed.")
    if state != "success":
        _error("Visualization status state must be 'success' or 'failure'.")
    if set(status) != {
        "status_version",
        "state",
        "dataset_id",
        "updated_at",
        "generation_id",
        "manifest_sha256",
    }:
        _error("Success status must point to exactly one manifest-bound generation.")
    generation_id = _safe_name(status["generation_id"], "status.generation_id")
    expected_manifest_sha = _sha256(status["manifest_sha256"], "status.manifest_sha256")
    generations_dir = dataset_dir / "generations"
    if generations_dir.is_symlink() or not generations_dir.is_dir():
        _error("Visualization generations directory is missing or unsafe.")
    generation_dir = generations_dir / generation_id
    if generation_dir.is_symlink() or not generation_dir.is_dir():
        _error("The status generation directory is missing or unsafe.")
    manifest_path = generation_dir / "manifest.json"
    _, actual_manifest_sha = _file_digest(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        _error("Status manifest SHA-256 binding does not match the generation.")
    return _load_manifest(generation_dir, expectation)


def load_cell_type_visualizations(
    root: Path,
    datasets: Iterable[object] | Mapping[str, object],
) -> dict[str, CellTypeVisualization]:
    """Build a fail-open startup index; one bad dataset cannot affect another."""

    if isinstance(datasets, Mapping):
        items: Iterable[tuple[str | None, object]] = datasets.items()
    else:
        items = ((None, dataset) for dataset in datasets)
    result: dict[str, CellTypeVisualization] = {}
    for hint, dataset in items:
        try:
            if hint is not None and _dataset_value(dataset, "dataset_id", "id") is None:
                if not isinstance(dataset, Mapping):
                    _error("A dataset mapping value without an ID must be an object mapping.")
                dataset = dict(dataset)
                dataset["dataset_id"] = hint
            expectation = _dataset_expectation(dataset, hint)
            if expectation.dataset_type != "full" or expectation.coordinate_dimensions != 2:
                continue
            result[expectation.dataset_id] = load_cell_type_visualization(root, dataset)
        except Exception as exc:
            identifier = hint or _dataset_value(dataset, "dataset_id", "id", default="unknown")
            _LOG.warning("Cell-type visualization unavailable for %s: %s", identifier, exc)
    return result


# Short aliases for callers that use the snapshot terminology.
load_visualization_snapshot = load_cell_type_visualization
load_visualization_index = load_cell_type_visualizations


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    payload = _canonical_json(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def publish_failure(
    root: Path,
    dataset_id: str,
    error: str,
    *,
    stage: str = "generation",
    category: str = "quality_gate",
    details: Mapping[str, object] | None = None,
    failure_id: str | None = None,
    updated_at: datetime | None = None,
) -> None:
    """Publish an immutable failure report, then make failure authoritative."""

    safe_dataset_id = _safe_name(dataset_id, "dataset_id")
    message = _string(error, "error", maximum=2000)
    stage = _string(stage, "stage", maximum=200)
    category = _string(category, "category", maximum=200)
    detail_values = dict(details or {})
    _canonical_json(detail_values)
    instant = updated_at or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        _error("updated_at must include a timezone.")
    timestamp = instant.isoformat()
    if failure_id is None:
        unique = uuid.uuid4().hex[:10]
        failure_id = f"{instant.strftime('%Y%m%dT%H%M%S%fZ')}-{unique}"
    safe_failure_id = _safe_name(failure_id, "failure_id")
    report = {
        "failure_report_version": FAILURE_REPORT_VERSION,
        "dataset_id": safe_dataset_id,
        "failure_id": safe_failure_id,
        "failed_at": timestamp,
        "status": "failed",
        "stage": stage,
        "category": category,
        "error": message,
        "details": detail_values,
    }
    report_payload = _canonical_json(report) + b"\n"
    dataset_dir = Path(root) / safe_dataset_id
    if dataset_dir.is_symlink():
        _error("Visualization dataset directory is unsafe.")
    failures_dir = dataset_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    if failures_dir.is_symlink():
        _error("Visualization failures directory is unsafe.")
    final_dir = failures_dir / safe_failure_id
    if final_dir.exists() or final_dir.is_symlink():
        _error("Visualization failure reports are immutable and may not be replaced.")
    staging = Path(tempfile.mkdtemp(prefix=f".{safe_failure_id}.", dir=failures_dir))
    try:
        _write_file = staging / "report.json"
        with _write_file.open("xb") as stream:
            stream.write(report_payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, final_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    _atomic_json(
        dataset_dir / "status.json",
        {
            "status_version": STATUS_VERSION,
            "state": "failure",
            "dataset_id": safe_dataset_id,
            "updated_at": timestamp,
            "failure_id": safe_failure_id,
            "failure_report_sha256": hashlib.sha256(report_payload).hexdigest(),
        },
    )


def publish_generation(
    root: Path,
    manifest: Mapping[str, object],
    files: Mapping[str, bytes],
) -> CellTypeVisualization:
    """Publish an immutable generation, then atomically point status at it.

    ``files`` contains every generation file other than ``manifest.json``. The
    generation is validated before its success status becomes visible.
    """

    document = json.loads(_canonical_json(manifest))
    if not isinstance(document, dict):
        _error("manifest must be an object.")
    dataset_id = _safe_name(document.get("dataset_id"), "manifest.dataset_id")
    generation_id = _safe_name(document.get("generation_id"), "manifest.generation_id")
    safe_files: dict[PurePosixPath, bytes] = {}
    for raw_name, payload in files.items():
        if not isinstance(raw_name, str) or not isinstance(payload, bytes):
            _error("Generation files must map relative string paths to bytes.")
        relative = PurePosixPath(raw_name)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(
                part in {"", ".", ".."} or _SAFE_NAME.fullmatch(part) is None
                for part in relative.parts
            )
        ):
            _error("Generation file name is unsafe.")
        if relative.as_posix() == "manifest.json" or relative in safe_files:
            _error("Generation files contain a duplicate or reserved path.")
        safe_files[relative] = payload
    root_path = Path(root)
    dataset_dir = root_path / dataset_id
    if dataset_dir.is_symlink():
        _error("Visualization dataset directory is unsafe.")
    generations_dir = dataset_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    if generations_dir.is_symlink():
        _error("Visualization generations directory is unsafe.")
    final_dir = generations_dir / generation_id
    if final_dir.exists() or final_dir.is_symlink():
        _error("Visualization generations are immutable and may not be replaced.")
    staging_root = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations_dir))
    staging = staging_root / generation_id
    staging.mkdir()
    manifest_bytes = _canonical_json(document) + b"\n"
    try:
        for relative, payload in safe_files.items():
            destination = staging.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        expectation = _DatasetExpectation(
            dataset_id,
            str(document.get("source", {}).get("sha256", "")),
            str(document.get("source", {}).get("obs_order_sha256", "")),
            int(document.get("source", {}).get("observation_count", -1)),
            int(document.get("source", {}).get("coordinate_dimensions", -1)),
            tuple(document.get("source", {}).get("sample_ids", ())),
            "full",
        )
        snapshot = _load_manifest(staging, expectation)
        os.replace(staging, final_dir)
        staging_root.rmdir()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        _atomic_json(
            dataset_dir / "status.json",
            {
                "status_version": STATUS_VERSION,
                "state": "success",
                "dataset_id": dataset_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "generation_id": generation_id,
                "manifest_sha256": manifest_sha,
            },
        )
        return snapshot.__class__(
            **{
                **snapshot.__dict__,
                "manifest_path": final_dir / "manifest.json",
                "report_path": final_dir / "report.json",
                "inference_path": (
                    final_dir / "inference.h5" if snapshot.inference_path is not None else None
                ),
                "samples": MappingProxyType(
                    {
                        key: CellTypeSample(
                            sample.key,
                            sample.id,
                            sample.count,
                            sample.bounds,
                            sample.category_counts,
                            MappingProxyType(
                                {
                                    encoding: PointRepresentation(
                                        final_dir / representation.path.relative_to(staging),
                                        representation.encoding,
                                        representation.size,
                                        representation.sha256,
                                        representation.content_size,
                                        representation.content_sha256,
                                    )
                                    for encoding, representation in sample.representations.items()
                                }
                            ),
                        )
                        for key, sample in snapshot.samples.items()
                    }
                ),
            }
        )
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
