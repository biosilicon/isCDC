"""Offline cell-type annotation orchestration.

This module deliberately has no import-time dependency on MuData, SciPy, R, SingleR,
or spacexr.  The public web application can therefore import :mod:`iscdc` in its small
runtime environment; annotation dependencies are loaded only by an offline command.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_ROOT = PROJECT_ROOT / "annotation"
ASSET_ROOT = PROJECT_ROOT / "assets" / "cell_type_annotation"
DEFAULT_PLAN_PATH = ASSET_ROOT / "configs" / "catalogue.yaml"
DEFAULT_VOCABULARY_PATH = ASSET_ROOT / "vocabulary.yaml"
REFERENCE_SCHEMA_VERSION = "1.0"
EXCHANGE_SCHEMA_VERSION = "1.0"
RESULT_SCHEMA_VERSION = "1.0"
ALLOWED_METHODS = frozenset({"source", "singler", "rctd"})
ALLOWED_STATUSES = frozenset({"Source", "Predicted", "Mixed", "Uncertain"})
TERMINAL_PILOT_STATES = frozenset({"success", "scientific_failure"})
MAX_CONCURRENT_JOBS = 20
MAX_RCTD_TASK_CORES = 12
MAX_SINGLER_TASK_CORES = 30
MAX_TOTAL_CORES = 40


class CellTypeAnnotationError(RuntimeError):
    """A safe, actionable annotation workflow failure."""


class ScientificAnnotationFailure(CellTypeAnnotationError):
    """A completed analysis that failed a predeclared scientific quality gate."""


@dataclass(frozen=True)
class QualityGates:
    min_shared_genes: int | None = None
    min_balanced_accuracy: float | None = None
    min_macro_f1: float | None = None
    max_ece: float | None = None
    max_uncertain_fraction: float | None = None
    max_mixed_fraction: float | None = None
    min_marker_agreement: float | None = None
    require_calibration: bool = True


@dataclass(frozen=True)
class DatasetPlan:
    dataset_id: str
    method: str
    reference_id: str | None
    pilot: bool
    complete: bool
    parameters: Mapping[str, Any] = field(default_factory=dict)
    qc: QualityGates = field(default_factory=QualityGates)
    source_evidence: str | None = None
    coordinate_system: Mapping[str, str] = field(
        default_factory=lambda: {"x_axis": "right", "y_axis": "up"}
    )


@dataclass(frozen=True)
class ReferenceMetadata:
    reference_id: str
    species: str
    tissue: str
    version: str
    files: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class AnnotationPrediction:
    observation_ids: tuple[str, ...]
    labels: tuple[str, ...]
    ontology_ids: tuple[str | None, ...]
    statuses: tuple[str, ...]
    sample_ids: tuple[str, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    confidence: tuple[float, ...] | None
    method: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return len(self.observation_ids)


@dataclass(frozen=True)
class AnnotationOutcome:
    dataset_id: str
    status: str
    method: str
    report_path: Path | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "method": self.method,
            "report_path": None if self.report_path is None else str(self.report_path),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CatalogueRecord:
    dataset_id: str
    h5mu_path: Path
    sha256: str
    n_obs: int
    coordinate_dimensions: int
    sample_ids: tuple[str, ...]
    coordinate_unit: str
    spatial_unit: str


def _load_yaml(path: Path) -> Any:
    try:
        yaml = importlib.import_module("yaml")
    except ImportError as exc:  # pragma: no cover - depends on the invoking environment
        raise CellTypeAnnotationError("PyYAML is required in iscdc-cell-annotation") from exc
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CellTypeAnnotationError(
            f"Unable to read annotation configuration {path}: {exc}"
        ) from exc


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise CellTypeAnnotationError(f"Unknown {context} fields: {', '.join(unknown)}")


def _optional_fraction(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CellTypeAnnotationError(f"{name} must be a number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise CellTypeAnnotationError(f"{name} must be a finite number in [0, 1]")
    return result


def _parse_quality_gates(value: Any, dataset_id: str) -> QualityGates:
    if not isinstance(value, dict):
        raise CellTypeAnnotationError(f"{dataset_id}: qc must be a mapping")
    allowed = {
        "min_shared_genes",
        "min_balanced_accuracy",
        "min_macro_f1",
        "max_ece",
        "max_uncertain_fraction",
        "max_mixed_fraction",
        "min_marker_agreement",
        "require_calibration",
    }
    _reject_unknown(value, allowed, f"{dataset_id} qc")
    shared = value.get("min_shared_genes")
    if shared is not None and (
        isinstance(shared, bool) or not isinstance(shared, int) or shared < 1
    ):
        raise CellTypeAnnotationError(f"{dataset_id}: min_shared_genes must be a positive integer")
    require_calibration = value.get("require_calibration", True)
    if not isinstance(require_calibration, bool):
        raise CellTypeAnnotationError(f"{dataset_id}: require_calibration must be boolean")
    return QualityGates(
        min_shared_genes=shared,
        min_balanced_accuracy=_optional_fraction(
            value.get("min_balanced_accuracy"), f"{dataset_id}.min_balanced_accuracy"
        ),
        min_macro_f1=_optional_fraction(value.get("min_macro_f1"), f"{dataset_id}.min_macro_f1"),
        max_ece=_optional_fraction(value.get("max_ece"), f"{dataset_id}.max_ece"),
        max_uncertain_fraction=_optional_fraction(
            value.get("max_uncertain_fraction"), f"{dataset_id}.max_uncertain_fraction"
        ),
        max_mixed_fraction=_optional_fraction(
            value.get("max_mixed_fraction"), f"{dataset_id}.max_mixed_fraction"
        ),
        min_marker_agreement=_optional_fraction(
            value.get("min_marker_agreement"), f"{dataset_id}.min_marker_agreement"
        ),
        require_calibration=require_calibration,
    )


def load_catalogue_plan(path: Path = DEFAULT_PLAN_PATH) -> dict[str, DatasetPlan]:
    """Load and strictly validate the versioned catalogue annotation plan."""
    raw = _load_yaml(Path(path))
    if not isinstance(raw, dict):
        raise CellTypeAnnotationError("Annotation plan must be a mapping")
    _reject_unknown(raw, {"schema_version", "pilot_ids", "datasets"}, "plan")
    if raw.get("schema_version") != "1.0":
        raise CellTypeAnnotationError("Unsupported annotation plan schema_version")
    pilot_ids = raw.get("pilot_ids")
    datasets = raw.get("datasets")
    if not isinstance(pilot_ids, list) or not all(isinstance(item, str) for item in pilot_ids):
        raise CellTypeAnnotationError("pilot_ids must be a list of dataset IDs")
    if len(pilot_ids) != len(set(pilot_ids)):
        raise CellTypeAnnotationError("pilot_ids contains duplicates")
    if not isinstance(datasets, list):
        raise CellTypeAnnotationError("datasets must be a list")

    result: dict[str, DatasetPlan] = {}
    allowed = {
        "dataset_id",
        "method",
        "reference_id",
        "complete",
        "parameters",
        "qc",
        "source_evidence",
        "coordinate_system",
    }
    for entry in datasets:
        if not isinstance(entry, dict):
            raise CellTypeAnnotationError("Every datasets entry must be a mapping")
        dataset_id = entry.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id or dataset_id != dataset_id.strip():
            raise CellTypeAnnotationError("Every annotation dataset_id must be a non-blank string")
        _reject_unknown(entry, allowed, f"{dataset_id} plan")
        if dataset_id in result:
            raise CellTypeAnnotationError(f"Duplicate annotation plan for {dataset_id}")
        method = entry.get("method")
        if method not in ALLOWED_METHODS:
            raise CellTypeAnnotationError(f"{dataset_id}: method must be source, singler, or rctd")
        complete = entry.get("complete")
        if not isinstance(complete, bool):
            raise CellTypeAnnotationError(f"{dataset_id}: complete must be boolean")
        reference_id = entry.get("reference_id")
        if method != "source" and complete:
            if (
                not isinstance(reference_id, str)
                or not reference_id
                or reference_id.startswith("PENDING")
            ):
                raise CellTypeAnnotationError(
                    f"{dataset_id}: complete inferred plan requires reference_id"
                )
        elif reference_id is not None and not isinstance(reference_id, str):
            raise CellTypeAnnotationError(f"{dataset_id}: reference_id must be a string or null")
        parameters = entry.get("parameters", {})
        if not isinstance(parameters, dict):
            raise CellTypeAnnotationError(f"{dataset_id}: parameters must be a mapping")
        cores = parameters.get("cores")
        exclusive = parameters.get("exclusive", False)
        if not isinstance(exclusive, bool):
            raise CellTypeAnnotationError(
                f"{dataset_id}: parameters.exclusive must be boolean"
            )
        maximum_cores = (
            MAX_SINGLER_TASK_CORES if method == "singler" else MAX_RCTD_TASK_CORES
        )
        if method != "source" and (
            isinstance(cores, bool) or not isinstance(cores, int) or not 1 <= cores <= maximum_cores
        ):
            raise CellTypeAnnotationError(
                f"{dataset_id}: {method} parameters.cores must be in [1, {maximum_cores}]"
            )
        if method == "rctd" and parameters.get("rctd_mode") != "full":
            raise CellTypeAnnotationError(f"{dataset_id}: RCTD must use rctd_mode: full")
        coordinates = entry.get("coordinate_system", {"x_axis": "right", "y_axis": "up"})
        if not isinstance(coordinates, dict):
            raise CellTypeAnnotationError(f"{dataset_id}: coordinate_system must be a mapping")
        _reject_unknown(coordinates, {"x_axis", "y_axis"}, f"{dataset_id} coordinate_system")
        if coordinates.get("x_axis", "right") not in {"left", "right"}:
            raise CellTypeAnnotationError(f"{dataset_id}: x_axis must be left or right")
        if coordinates.get("y_axis", "up") not in {"up", "down"}:
            raise CellTypeAnnotationError(f"{dataset_id}: y_axis must be up or down")
        result[dataset_id] = DatasetPlan(
            dataset_id=dataset_id,
            method=method,
            reference_id=reference_id,
            pilot=dataset_id in pilot_ids,
            complete=complete,
            parameters=dict(parameters),
            qc=_parse_quality_gates(entry.get("qc", {}), dataset_id),
            source_evidence=entry.get("source_evidence"),
            coordinate_system=dict(coordinates),
        )
    missing_pilots = sorted(set(pilot_ids).difference(result))
    if missing_pilots:
        raise CellTypeAnnotationError(f"Unknown pilot IDs: {', '.join(missing_pilots)}")
    return result


def load_vocabulary(path: Path = DEFAULT_VOCABULARY_PATH) -> dict[str, str]:
    raw = _load_yaml(Path(path))
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0":
        raise CellTypeAnnotationError("Unsupported cell-type vocabulary")
    types = raw.get("cell_types")
    statuses = raw.get("prediction_statuses")
    if not isinstance(types, list) or not isinstance(statuses, list):
        raise CellTypeAnnotationError(
            "Vocabulary requires cell_types and prediction_statuses lists"
        )
    if {item.get("name") for item in statuses if isinstance(item, dict)} != {"Mixed", "Uncertain"}:
        raise CellTypeAnnotationError(
            "Vocabulary must define Mixed and Uncertain prediction statuses"
        )
    result: dict[str, str] = {}
    for entry in types:
        if not isinstance(entry, dict) or set(entry) != {"name", "ontology_id"}:
            raise CellTypeAnnotationError("Invalid cell_types vocabulary entry")
        name, ontology_id = entry["name"], entry["ontology_id"]
        if not isinstance(name, str) or not isinstance(ontology_id, str):
            raise CellTypeAnnotationError("Vocabulary names and ontology IDs must be strings")
        if (
            not ontology_id.startswith("CL:")
            or len(ontology_id) != 10
            or not ontology_id[3:].isdigit()
        ):
            raise CellTypeAnnotationError(f"Invalid stable Cell Ontology ID: {ontology_id}")
        if name in result:
            raise CellTypeAnnotationError(f"Duplicate vocabulary name: {name}")
        result[name] = ontology_id
    return result


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _atomic_directory(destination: Path, writer: Callable[[Path], None], force: bool) -> None:
    destination = destination.resolve()
    if destination.exists() and not force:
        raise CellTypeAnnotationError(f"Output already exists: {destination}; use --force")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    backup: Path | None = None
    try:
        writer(staging)
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.backup-{os.getpid()}")
            os.replace(destination, backup)
        os.replace(staging, destination)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise


def _require_annotation_environment() -> None:
    active = os.environ.get("CONDA_DEFAULT_ENV")
    if active != "iscdc-cell-annotation":
        raise CellTypeAnnotationError(
            "This mutating command requires `conda activate iscdc-cell-annotation`"
        )


def _validated_sparse_matrix(matrix: Any) -> Any:
    """Return a sparse matrix after strict count validation, never calling toarray()."""
    sparse = importlib.import_module("scipy.sparse")
    numpy = importlib.import_module("numpy")
    if len(getattr(matrix, "shape", ())) != 2:
        raise CellTypeAnnotationError("RNA X must be a two-dimensional count matrix")
    if sparse.issparse(matrix):
        result = matrix.tocoo(copy=False)
        values = result.data
        if (
            not numpy.isfinite(values).all()
            or (values < 0).any()
            or (values != numpy.floor(values)).any()
        ):
            raise CellTypeAnnotationError("RNA X must contain finite non-negative integer counts")
        if values.size and values.max() > numpy.iinfo(numpy.int64).max:
            raise CellTypeAnnotationError("RNA X counts exceed the signed 64-bit exchange range")
        return result.astype(numpy.int64, copy=False)
    # Dense and dense-backed inputs are inspected and sparsified by row chunks. This
    # bounds memory and, importantly, never converts a sparse input to dense.
    chunks = []
    row_count = int(matrix.shape[0])
    for start in range(0, row_count, 4096):
        raw_block = matrix[start : min(start + 4096, row_count), :]
        if sparse.issparse(raw_block):
            values = raw_block.data
            if (
                not numpy.isfinite(values).all()
                or (values < 0).any()
                or (values != numpy.floor(values)).any()
            ):
                raise CellTypeAnnotationError(
                    "RNA X must contain finite non-negative integer counts"
                )
            if values.size and values.max() > numpy.iinfo(numpy.int64).max:
                raise CellTypeAnnotationError(
                    "RNA X counts exceed the signed 64-bit exchange range"
                )
            chunks.append(raw_block.astype(numpy.int64, copy=False).tocsr())
            continue
        block = numpy.asarray(raw_block)
        if (
            not numpy.isfinite(block).all()
            or (block < 0).any()
            or (block != numpy.floor(block)).any()
        ):
            raise CellTypeAnnotationError("RNA X must contain finite non-negative integer counts")
        if block.size and block.max() > numpy.iinfo(numpy.int64).max:
            raise CellTypeAnnotationError("RNA X counts exceed the signed 64-bit exchange range")
        chunks.append(sparse.csr_matrix(block.astype(numpy.int64, copy=False)))
    return sparse.vstack(chunks, format="coo") if chunks else sparse.coo_matrix(matrix.shape)


def export_sparse_rna_exchange(h5mu_path: Path, destination: Path, *, force: bool = False) -> dict:
    """Export RNA as sparse Matrix Market plus aligned metadata without densifying.

    The matrix is observations by genes. R adapters transpose it explicitly for the
    conventions of SingleR/spacexr. The source file is opened read-only and its digest
    is checked again after export.
    """
    h5mu_path = Path(h5mu_path).resolve()
    before = sha256_file(h5mu_path)

    def write(staging: Path) -> None:
        try:
            mudata = importlib.import_module("mudata")
            scipy_io = importlib.import_module("scipy.io")
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise CellTypeAnnotationError(
                "MuData and SciPy are required in iscdc-cell-annotation"
            ) from exc
        mdata = None
        try:
            mdata = mudata.read_h5mu(h5mu_path, backed="r")
            if "rna" not in mdata.mod:
                raise CellTypeAnnotationError("The .h5mu file has no rna modality")
            rna = mdata.mod["rna"]
            if rna.X is None:
                raise CellTypeAnnotationError("RNA X is missing")
            exchange_matrix = _validated_sparse_matrix(rna.X)
            scipy_io.mmwrite(staging / "matrix.mtx", exchange_matrix, symmetry="general")

            observation_ids = tuple(map(str, rna.obs_names))
            gene_ids = tuple(map(str, rna.var_names))
            with (staging / "observations.tsv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(("observation_id", "sample_id"))
                samples = _rna_aligned_samples(mdata, observation_ids)
                writer.writerows(zip(observation_ids, samples, strict=True))
            with (staging / "genes.tsv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(("gene_id",))
                writer.writerows((gene_id,) for gene_id in gene_ids)
            coordinates = _rna_aligned_coordinates(mdata, observation_ids)
            with (staging / "spatial.tsv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(("observation_id", "x", "y"))
                writer.writerows(
                    (obs_id, _format_float(x), _format_float(y))
                    for obs_id, (x, y) in zip(observation_ids, coordinates, strict=True)
                )
            _write_json(
                staging / "exchange.json",
                {
                    "schema_version": EXCHANGE_SCHEMA_VERSION,
                    "source_h5mu": str(h5mu_path),
                    "source_sha256": before,
                    "n_observations": len(observation_ids),
                    "n_genes": len(gene_ids),
                    "matrix_orientation": "observations_by_genes",
                    "matrix_format": "MatrixMarket coordinate",
                    "files": {
                        name: {
                            "size": (staging / name).stat().st_size,
                            "sha256": sha256_file(staging / name),
                        }
                        for name in ("matrix.mtx", "observations.tsv", "genes.tsv", "spatial.tsv")
                    },
                },
            )
        finally:
            if mdata is not None:
                file_manager = getattr(mdata, "file", None)
                close = getattr(file_manager, "close", None)
                if callable(close):
                    close()

    _atomic_directory(Path(destination), write, force)
    after = sha256_file(h5mu_path)
    if before != after:
        raise CellTypeAnnotationError("Source .h5mu changed during sparse exchange export")
    return json.loads((Path(destination) / "exchange.json").read_text(encoding="utf-8"))


def _format_float(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise CellTypeAnnotationError("Spatial coordinates must be finite")
    return format(number, ".9g")


def _rna_aligned_samples(mdata: Any, observation_ids: Sequence[str]) -> tuple[str, ...]:
    if "sample_id" not in mdata.obs.columns:
        raise CellTypeAnnotationError("Top-level obs['sample_id'] is required")
    try:
        values = mdata.obs.loc[list(observation_ids), "sample_id"]
    except KeyError as exc:
        raise CellTypeAnnotationError(
            "RNA observations are not aligned to top-level observations"
        ) from exc
    result = tuple(map(str, values))
    if any(not value or value != value.strip() for value in result):
        raise CellTypeAnnotationError("sample_id contains blank or noncanonical values")
    return result


def _rna_aligned_coordinates(
    mdata: Any, observation_ids: Sequence[str]
) -> tuple[tuple[float, float], ...]:
    if "spatial" not in mdata.obsm:
        raise CellTypeAnnotationError("Top-level obsm['spatial'] is required")
    positions = {str(name): index for index, name in enumerate(mdata.obs_names)}
    try:
        indexes = [positions[name] for name in observation_ids]
    except KeyError as exc:
        raise CellTypeAnnotationError(
            "RNA observations are not aligned to spatial coordinates"
        ) from exc
    spatial = mdata.obsm["spatial"]
    if getattr(spatial, "shape", (0, 0))[1] != 2:
        raise CellTypeAnnotationError("Spatial coordinates must have exactly two dimensions")
    return tuple((float(spatial[index, 0]), float(spatial[index, 1])) for index in indexes)


def generate_source_labels(h5mu_path: Path, plan: DatasetPlan) -> AnnotationPrediction:
    """Build a complete source-label prediction without modifying the .h5mu file."""
    if plan.method != "source":
        raise CellTypeAnnotationError("generate_source_labels requires method: source")
    before = sha256_file(Path(h5mu_path))
    try:
        mudata = importlib.import_module("mudata")
    except ImportError as exc:  # pragma: no cover
        raise CellTypeAnnotationError("MuData is required in iscdc-cell-annotation") from exc
    mdata = None
    try:
        mdata = mudata.read_h5mu(h5mu_path, backed="r")
        if "cell_type" not in mdata.obs.columns:
            raise CellTypeAnnotationError("Verified source obs['cell_type'] is absent")
        labels_series = mdata.obs["cell_type"]
        pandas = importlib.import_module("pandas")
        if not isinstance(labels_series.dtype, pandas.CategoricalDtype):
            raise CellTypeAnnotationError("Verified source cell_type must be categorical")
        if labels_series.cat.ordered:
            raise CellTypeAnnotationError("Verified source cell_type must be unordered")
        if labels_series.isna().any():
            raise CellTypeAnnotationError("Source cell_type does not cover every observation")
        observation_ids = tuple(map(str, mdata.obs_names))
        labels = tuple(map(str, labels_series.astype(object)))
        if any(not label or label != label.strip() for label in labels):
            raise CellTypeAnnotationError("Source cell_type labels are blank or noncanonical")
        if set(labels_series.cat.categories) != set(labels):
            raise CellTypeAnnotationError("Source cell_type contains unused categories")
        coordinates = _rna_aligned_coordinates(mdata, observation_ids)
        samples = _rna_aligned_samples(mdata, observation_ids)
        ontology_map = plan.parameters.get("source_ontology_map", {})
        if not isinstance(ontology_map, dict):
            raise CellTypeAnnotationError("source_ontology_map must be a mapping")
        ontology_ids = tuple(ontology_map.get(label) for label in labels)
        prediction = AnnotationPrediction(
            observation_ids=observation_ids,
            labels=labels,
            ontology_ids=ontology_ids,
            statuses=("Source",) * len(labels),
            sample_ids=samples,
            x=tuple(point[0] for point in coordinates),
            y=tuple(point[1] for point in coordinates),
            confidence=None,
            method="source",
            diagnostics={"source_evidence": plan.source_evidence, "confidence_definition": None},
        )
    finally:
        if mdata is not None:
            close = getattr(getattr(mdata, "file", None), "close", None)
            if callable(close):
                close()
    if sha256_file(Path(h5mu_path)) != before:
        raise CellTypeAnnotationError("Source .h5mu changed while reading source labels")
    validate_prediction(prediction, plan, expected_observation_ids=prediction.observation_ids)
    return prediction


def validate_reference_pack(
    reference_dir: Path, reference_id: str | None = None
) -> ReferenceMetadata:
    """Validate immutable reference metadata, sizes, and SHA-256 checksums."""
    reference_dir = Path(reference_dir)
    metadata_path = reference_dir / "reference.json"
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CellTypeAnnotationError(f"Invalid reference metadata {metadata_path}: {exc}") from exc
    required = {"schema_version", "reference_id", "species", "tissue", "version", "files"}
    if not isinstance(raw, dict) or not required.issubset(raw):
        raise CellTypeAnnotationError("Reference metadata is incomplete")
    if raw["schema_version"] != REFERENCE_SCHEMA_VERSION:
        raise CellTypeAnnotationError("Unsupported reference schema_version")
    if reference_id is not None and raw["reference_id"] != reference_id:
        raise CellTypeAnnotationError("Reference ID does not match requested reference")
    files = raw["files"]
    if not isinstance(files, list) or not files:
        raise CellTypeAnnotationError("Reference files must be a non-empty list")
    validated: list[Mapping[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"name", "size", "sha256"}:
            raise CellTypeAnnotationError("Invalid reference file record")
        name = entry["name"]
        if not isinstance(name, str) or Path(name).name != name:
            raise CellTypeAnnotationError("Reference file names must be plain basenames")
        path = reference_dir / name
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise CellTypeAnnotationError(f"Reference file size mismatch: {name}")
        if sha256_file(path) != entry["sha256"]:
            raise CellTypeAnnotationError(f"Reference file checksum mismatch: {name}")
        validated.append(dict(entry))
    return ReferenceMetadata(
        reference_id=str(raw["reference_id"]),
        species=str(raw["species"]),
        tissue=str(raw["tissue"]),
        version=str(raw["version"]),
        files=tuple(validated),
        metadata=raw,
    )


def _run_rscript(
    script: Path,
    arguments: Sequence[str],
    log_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    command = ["Rscript", "--vanilla", str(script), *map(str, arguments)]
    process_environment = os.environ.copy()
    process_environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "GOTO_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "RCPP_PARALLEL_NUM_THREADS": "1",
        }
    )
    try:
        result = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=None,
            env=process_environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CellTypeAnnotationError(f"Unable to execute Rscript: {exc}") from exc
    log_path.write_text(
        f"command: {' '.join(command)}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        raise CellTypeAnnotationError(
            f"R annotation failed with exit code {result.returncode}: "
            f"{detail or 'Rscript produced no diagnostic output'}"
        )


def run_r_annotation(
    method: str,
    exchange_dir: Path,
    reference_dir: Path,
    result_dir: Path,
    config_path: Path,
    *,
    workers: int = 1,
    parameters: Mapping[str, Any] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Invoke the pinned SingleR or full-mode RCTD adapter."""
    scripts = {
        "singler": ANNOTATION_ROOT / "run_single_r.R",
        "rctd": ANNOTATION_ROOT / "run_rctd.R",
    }
    if method not in scripts:
        raise CellTypeAnnotationError("R orchestration method must be singler or rctd")
    maximum_workers = (
        MAX_SINGLER_TASK_CORES if method == "singler" else MAX_RCTD_TASK_CORES
    )
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= maximum_workers
    ):
        raise CellTypeAnnotationError(
            f"{method} workers must be between 1 and {maximum_workers}"
        )
    result_dir.mkdir(parents=True, exist_ok=False)
    runtime_config = result_dir / "runtime_config.json"
    _write_json(
        runtime_config,
        {
            "schema_version": "1.0",
            "method": method,
            "parameters": dict(parameters or {}),
            "catalogue_config": str(config_path),
        },
    )
    _run_rscript(
        scripts[method],
        (
            "--exchange",
            str(exchange_dir),
            "--reference",
            str(reference_dir),
            "--output",
            str(result_dir),
            "--config",
            str(runtime_config),
            "--workers",
            str(workers),
        ),
        result_dir / "rscript.log",
        runner=runner,
    )


def read_r_prediction(result_dir: Path, exchange_dir: Path, method: str) -> AnnotationPrediction:
    """Read the deliberately simple R/Python exchange output."""
    table_path = Path(result_dir) / "predictions.tsv"
    try:
        with table_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except OSError as exc:
        raise CellTypeAnnotationError(f"Unable to read R predictions: {exc}") from exc
    required = {"observation_id", "label", "ontology_id", "status", "confidence"}
    if not rows or not required.issubset(rows[0]):
        raise CellTypeAnnotationError("R predictions are empty or missing required columns")
    with (Path(exchange_dir) / "observations.tsv").open(encoding="utf-8", newline="") as stream:
        observations = list(csv.DictReader(stream, delimiter="\t"))
    with (Path(exchange_dir) / "spatial.tsv").open(encoding="utf-8", newline="") as stream:
        coordinates = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != len(observations) or len(rows) != len(coordinates):
        raise CellTypeAnnotationError("R predictions are not observation-complete")
    confidence_values: list[float] = []
    for row in rows:
        try:
            confidence_values.append(float(row["confidence"]))
        except ValueError as exc:
            raise CellTypeAnnotationError("R predictions contain invalid confidence") from exc
    return AnnotationPrediction(
        observation_ids=tuple(row["observation_id"] for row in rows),
        labels=tuple(row["label"] for row in rows),
        ontology_ids=tuple(row["ontology_id"] or None for row in rows),
        statuses=tuple(row["status"] for row in rows),
        sample_ids=tuple(row["sample_id"] for row in observations),
        x=tuple(float(row["x"]) for row in coordinates),
        y=tuple(float(row["y"]) for row in coordinates),
        confidence=tuple(confidence_values),
        method=method,
        diagnostics=_load_result_diagnostics(Path(result_dir)),
    )


def _load_result_diagnostics(result_dir: Path) -> Mapping[str, Any]:
    try:
        value = json.loads((result_dir / "diagnostics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CellTypeAnnotationError(f"Missing or invalid R diagnostics: {exc}") from exc
    if not isinstance(value, dict):
        raise CellTypeAnnotationError("R diagnostics must be a JSON object")
    return value


def validate_prediction(
    prediction: AnnotationPrediction,
    plan: DatasetPlan,
    *,
    expected_observation_ids: Sequence[str],
) -> dict[str, Any]:
    """Apply alignment, vocabulary, calibration, and dataset-specific QC gates."""
    lengths = {
        len(prediction.observation_ids),
        len(prediction.labels),
        len(prediction.ontology_ids),
        len(prediction.statuses),
        len(prediction.sample_ids),
        len(prediction.x),
        len(prediction.y),
    }
    if prediction.confidence is not None:
        lengths.add(len(prediction.confidence))
    if lengths != {prediction.n_obs} or prediction.n_obs == 0:
        raise CellTypeAnnotationError("Prediction fields must be non-empty and observation-aligned")
    if tuple(expected_observation_ids) != prediction.observation_ids:
        raise CellTypeAnnotationError("Prediction observations differ from source order")
    if len(set(prediction.observation_ids)) != prediction.n_obs:
        raise CellTypeAnnotationError("Prediction observation IDs are not unique")
    if any(not label or label != label.strip() for label in prediction.labels):
        raise CellTypeAnnotationError("Prediction labels must be non-blank canonical strings")
    if any(status not in ALLOWED_STATUSES for status in prediction.statuses):
        raise CellTypeAnnotationError("Prediction contains an unknown status")
    if any(not math.isfinite(value) for value in (*prediction.x, *prediction.y)):
        raise CellTypeAnnotationError("Prediction coordinates must be finite")

    source = plan.method == "source"
    if source:
        if prediction.confidence is not None or set(prediction.statuses) != {"Source"}:
            raise CellTypeAnnotationError(
                "Source annotations must omit confidence and use Source status"
            )
    else:
        if prediction.confidence is None:
            raise CellTypeAnnotationError("Inferred annotations require calibrated confidence")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in prediction.confidence):
            raise CellTypeAnnotationError("Calibrated confidence must be finite and in [0, 1]")
        if "Source" in prediction.statuses:
            raise CellTypeAnnotationError("Inferred annotations cannot use Source status")
        for label, ontology_id, status in zip(
            prediction.labels, prediction.ontology_ids, prediction.statuses, strict=True
        ):
            if status in {"Predicted"} and (
                not isinstance(ontology_id, str)
                or len(ontology_id) != 10
                or not ontology_id.startswith("CL:")
                or not ontology_id[3:].isdigit()
            ):
                raise CellTypeAnnotationError(
                    f"Inferred biological label {label!r} lacks a stable Cell Ontology ID"
                )
            if status in {"Mixed", "Uncertain"} and ontology_id is not None:
                raise CellTypeAnnotationError(
                    f"Prediction status {status} must not carry a Cell Ontology ID"
                )

    metrics = prediction.diagnostics.get("qc", prediction.diagnostics)
    if not isinstance(metrics, Mapping):
        raise CellTypeAnnotationError("Prediction diagnostics must contain QC metrics")
    calibration = prediction.diagnostics.get("calibration")
    if not source and plan.qc.require_calibration:
        if not isinstance(calibration, Mapping) or calibration.get("completed") is not True:
            raise CellTypeAnnotationError("Dataset-specific confidence calibration is incomplete")
        if calibration.get("confidence_definition") in {None, "", "raw_score", "max_weight"}:
            raise CellTypeAnnotationError(
                "Confidence definition is missing or is an uncalibrated raw value"
            )

    failures: list[str] = []
    gates = plan.qc
    comparisons = (
        ("shared_genes", gates.min_shared_genes, lambda actual, threshold: actual >= threshold),
        (
            "balanced_accuracy",
            gates.min_balanced_accuracy,
            lambda actual, threshold: actual >= threshold,
        ),
        ("macro_f1", gates.min_macro_f1, lambda actual, threshold: actual >= threshold),
        ("ece", gates.max_ece, lambda actual, threshold: actual <= threshold),
        (
            "marker_agreement",
            gates.min_marker_agreement,
            lambda actual, threshold: actual >= threshold,
        ),
    )
    for name, threshold, comparison in comparisons:
        if threshold is None:
            continue
        actual = metrics.get(name)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
        ):
            failures.append(f"{name} is missing or non-finite")
        elif not comparison(float(actual), threshold):
            failures.append(f"{name}={actual} fails threshold {threshold}")
    uncertain_fraction = prediction.statuses.count("Uncertain") / prediction.n_obs
    mixed_fraction = prediction.statuses.count("Mixed") / prediction.n_obs
    if (
        gates.max_uncertain_fraction is not None
        and uncertain_fraction > gates.max_uncertain_fraction
    ):
        failures.append(
            f"uncertain_fraction={uncertain_fraction:.6f} exceeds {gates.max_uncertain_fraction}"
        )
    if gates.max_mixed_fraction is not None and mixed_fraction > gates.max_mixed_fraction:
        failures.append(f"mixed_fraction={mixed_fraction:.6f} exceeds {gates.max_mixed_fraction}")
    if failures:
        raise ScientificAnnotationFailure(
            "Annotation QC rejected publication: " + "; ".join(failures)
        )
    return {
        "n_observations": prediction.n_obs,
        "uncertain_fraction": uncertain_fraction,
        "mixed_fraction": mixed_fraction,
        "calibrated": not source,
    }


def _resolve_settings(settings: Any | None) -> Any:
    if settings is not None:
        return settings
    config = importlib.import_module("iscdc.config")
    return config.Settings.from_environment()


def _catalogue_record(dataset_id: str, settings: Any) -> CatalogueRecord:
    # sqlite3 keeps annotation tooling independent from SQLAlchemy at import time.
    import sqlite3

    connection = sqlite3.connect(settings.database_path)
    try:
        row = connection.execute(
            """SELECT storage_dir, sha256, dataset_type, n_obs,
                      coordinate_dimensions, sample_ids, coordinate_unit, spatial_unit
               FROM datasets WHERE dataset_id = ?""",
            (dataset_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise CellTypeAnnotationError(f"Unknown catalogue dataset: {dataset_id}")
    if row[2] != "full":
        raise CellTypeAnnotationError(
            f"Cell-type visualization requires a full dataset: {dataset_id}"
        )
    sample_ids = json.loads(row[5])
    if not isinstance(sample_ids, list) or not sample_ids:
        raise CellTypeAnnotationError(f"Catalogue sample IDs are invalid for {dataset_id}")
    return CatalogueRecord(
        dataset_id=dataset_id,
        h5mu_path=Path(settings.data_root) / row[0] / "dataset.h5mu",
        sha256=str(row[1]),
        n_obs=int(row[3]),
        coordinate_dimensions=int(row[4]),
        sample_ids=tuple(map(str, sample_ids)),
        coordinate_unit=str(row[6]),
        spatial_unit=str(row[7]),
    )


def _visualization_root(settings: Any) -> Path:
    root = getattr(settings, "cell_type_visualization_root", None)
    if root is None:
        root = Path(settings.data_root).parent / "cell_type_visualizations"
    return Path(root)


def _reference_root(settings: Any) -> Path:
    return _visualization_root(settings) / "references"


def _work_root(settings: Any) -> Path:
    return _visualization_root(settings) / "work"


def build_cell_type_reference(
    reference_id: str,
    settings: Any | None = None,
    *,
    force: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict:
    """Build (when configured externally) or strictly verify one reference pack."""
    _require_annotation_environment()
    resolved = _resolve_settings(settings)
    destination = _reference_root(resolved) / reference_id
    if destination.exists() and not force:
        metadata = validate_reference_pack(destination, reference_id)
        return dict(metadata.metadata)
    recipe = ASSET_ROOT / "configs" / "references" / f"{reference_id}.yaml"
    if not recipe.is_file():
        raise CellTypeAnnotationError(
            f"Reference recipe is unavailable for {reference_id}; scientific curation is incomplete"
        )

    def writer(staging: Path) -> None:
        _run_rscript(
            ANNOTATION_ROOT / "build_reference.R",
            ("--config", str(recipe), "--output", str(staging)),
            staging / "rscript.log",
            runner=runner,
        )
        validate_reference_pack(staging, reference_id)

    _atomic_directory(destination, writer, force)
    return dict(validate_reference_pack(destination, reference_id).metadata)


def _cell_type_color(label: str) -> str:
    """Return a deterministic high-contrast color independent of category order."""
    value = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    hue = (value / 2**64 + 0.6180339887498949) % 1.0
    saturation = 0.58 + ((value >> 8) & 0xFF) / 2550
    lightness = 0.45 + ((value >> 16) & 0xFF) / 5100
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, min(saturation, 0.78))
    return f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"


def _obs_order_sha256(observation_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in observation_ids:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _file_record(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _inference_h5_bytes(result_dir: Path, prediction: AnnotationPrediction) -> bytes:
    """Preserve complete inferred diagnostics and sparse scores/weights in HDF5."""
    try:
        h5py = importlib.import_module("h5py")
        numpy = importlib.import_module("numpy")
        scipy_io = importlib.import_module("scipy.io")
        sparse = importlib.import_module("scipy.sparse")
    except ImportError as exc:  # pragma: no cover - annotation environment dependent
        raise CellTypeAnnotationError("h5py and SciPy are required to write inference.h5") from exc
    handle, temporary_name = tempfile.mkstemp(suffix=".h5")
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        rows: list[dict[str, str]]
        with (result_dir / "predictions.tsv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(temporary, "w") as output:
            output.attrs["inference_version"] = 1
            output.attrs["method"] = prediction.method
            output.attrs["diagnostics_json"] = json.dumps(
                prediction.diagnostics, ensure_ascii=False, allow_nan=False, sort_keys=True
            )
            calibration = prediction.diagnostics.get("calibration", {})
            if not isinstance(calibration, Mapping):
                raise CellTypeAnnotationError("Inference calibration diagnostics must be a mapping")
            calibration_group = output.create_group("calibration")
            for key, value in calibration.items():
                calibration_group.attrs[str(key)] = (
                    value
                    if isinstance(value, (str, int, float, bool)) and value is not None
                    else json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
                )
            output.create_dataset(
                "observation_id",
                data=numpy.asarray(prediction.observation_ids, dtype=object),
                dtype=string_dtype,
            )
            output.create_dataset(
                "label", data=numpy.asarray(prediction.labels, dtype=object), dtype=string_dtype
            )
            output.create_dataset(
                "status", data=numpy.asarray(prediction.statuses, dtype=object), dtype=string_dtype
            )
            for name in (
                "confidence",
                "best_score",
                "second_score",
                "delta",
                "top_weight",
                "second_weight",
                "entropy",
                "effective_types",
            ):
                if name == "confidence":
                    values = prediction.confidence
                elif rows and name in rows[0]:
                    values = tuple(float(row[name]) if row[name] else math.nan for row in rows)
                else:
                    values = None
                if values is not None:
                    output.create_dataset(name, data=numpy.asarray(values, dtype="float64"))
            if rows and "converged" in rows[0]:
                output.create_dataset(
                    "converged",
                    data=numpy.asarray(
                        [row["converged"].lower() in {"true", "t", "1"} for row in rows],
                        dtype="bool",
                    ),
                )
            for matrix_name, label_name, group_name in (
                ("candidate_scores.mtx", "candidate_score_labels.tsv", "candidate_scores"),
                ("cell_type_weights.mtx", "weight_labels.tsv", "cell_type_weights"),
            ):
                matrix_path = result_dir / matrix_name
                if not matrix_path.is_file():
                    continue
                matrix = sparse.csr_matrix(scipy_io.mmread(matrix_path))
                group = output.create_group(group_name)
                group.create_dataset("data", data=matrix.data)
                group.create_dataset("indices", data=matrix.indices)
                group.create_dataset("indptr", data=matrix.indptr)
                group.attrs["shape"] = matrix.shape
                with (result_dir / label_name).open(encoding="utf-8", newline="") as stream:
                    label_rows = list(csv.DictReader(stream, delimiter="\t"))
                group.create_dataset(
                    "labels",
                    data=numpy.asarray([row["label"] for row in label_rows], dtype=object),
                    dtype=string_dtype,
                )
        return temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)


def _sidecar_publish_generation(
    settings: Any,
    record: CatalogueRecord,
    plan: DatasetPlan,
    prediction: AnnotationPrediction,
    validation: Mapping[str, Any],
    reference: ReferenceMetadata | None,
    result_dir: Path | None,
) -> Any:
    """Build the strict sidecar manifest/files and publish them atomically."""
    try:
        sidecar = importlib.import_module("iscdc.cell_type_visualization")
    except ImportError as exc:
        raise CellTypeAnnotationError("Cell-type visualization sidecar is not installed") from exc
    publish = getattr(sidecar, "publish_generation", None)
    encode = getattr(sidecar, "encode_points", None)
    representations_builder = getattr(sidecar, "build_point_representations", None)
    if not callable(publish) or not callable(encode) or not callable(representations_builder):
        raise CellTypeAnnotationError("Cell-type visualization sidecar API is incomplete")
    display_labels = tuple(
        status if status in {"Mixed", "Uncertain"} else label
        for label, status in zip(prediction.labels, prediction.statuses, strict=True)
    )
    category_labels = list(dict.fromkeys(display_labels))
    type_by_label = {label: index for index, label in enumerate(category_labels)}
    type_ids = tuple(type_by_label[label] for label in display_labels)
    categories: list[dict[str, Any]] = []
    for label in category_labels:
        indexes = [index for index, value in enumerate(display_labels) if value == label]
        state = label.lower() if label in {"Mixed", "Uncertain"} else "biological"
        category: dict[str, Any] = {
            "type_id": type_by_label[label],
            "label": label,
            "color": _cell_type_color(label),
            "count": len(indexes),
            "state": state,
        }
        if state == "biological":
            ontology_ids = {prediction.ontology_ids[index] for index in indexes}
            ontology_ids.discard(None)
            if len(ontology_ids) == 1:
                category["cell_ontology_id"] = ontology_ids.pop()
        categories.append(category)

    files: dict[str, bytes] = {}
    sample_documents: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(record.sample_ids):
        indexes = [index for index, value in enumerate(prediction.sample_ids) if value == sample_id]
        if not indexes:
            raise CellTypeAnnotationError(f"Sample {sample_id} has no annotation observations")
        sample_key = f"sample_{sample_index}"
        confidence = (
            None
            if prediction.confidence is None
            else tuple(prediction.confidence[index] for index in indexes)
        )
        identity = encode(
            (prediction.x[index] for index in indexes),
            (prediction.y[index] for index in indexes),
            (type_ids[index] for index in indexes),
            confidence,
        )
        encoded_representations = representations_builder(identity)
        representation_records: dict[str, Any] = {}
        suffixes = {"identity": ".bin", "gzip": ".bin.gz", "br": ".bin.br"}
        content_sha = hashlib.sha256(identity).hexdigest()
        for encoding, content in encoded_representations.items():
            name = f"points/{sample_key}{suffixes[encoding]}"
            files[name] = content
            representation_records[encoding] = {
                **_file_record(name, content),
                "encoding": encoding,
                "content_size": len(identity),
                "content_sha256": content_sha,
            }
        xs = [prediction.x[index] for index in indexes]
        ys = [prediction.y[index] for index in indexes]
        counts = {str(type_id): 0 for type_id in range(len(category_labels))}
        for index in indexes:
            counts[str(type_ids[index])] += 1
        sample_documents.append(
            {
                "key": sample_key,
                "id": sample_id,
                "count": len(indexes),
                "bounds": [min(xs), min(ys), max(xs), max(ys)],
                "category_counts": counts,
                "representations": representation_records,
            }
        )

    generated_at = datetime.now(timezone.utc)
    generation_id = f"{generated_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:10]}"
    report = {
        "report_version": 1,
        "dataset_id": record.dataset_id,
        "generation_id": generation_id,
        "source_sha256": record.sha256,
        "status": "passed",
        "quality_control": {**dict(validation), **dict(prediction.diagnostics.get("qc", {}))},
        "thresholds": _quality_dict(plan.qc),
        "warnings": list(prediction.diagnostics.get("warnings", [])),
    }
    report_bytes = _json_bytes(report)
    files["report.json"] = report_bytes
    inference_record = None
    if prediction.method != "source":
        if result_dir is None:
            raise CellTypeAnnotationError(
                "Inferred publication requires complete R result diagnostics"
            )
        inference_bytes = _inference_h5_bytes(result_dir, prediction)
        files["inference.h5"] = inference_bytes
        inference_record = _file_record("inference.h5", inference_bytes)
    environment_sha = sha256_file(ANNOTATION_ROOT / "renv.lock")
    references: list[dict[str, str]] = []
    if reference is not None:
        references.append(
            {
                "id": reference.reference_id,
                "version": reference.version,
                "sha256": sha256_file(
                    _reference_root(settings) / reference.reference_id / "reference.json"
                ),
            }
        )
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "dataset_id": record.dataset_id,
        "generation_id": generation_id,
        "generated_at": generated_at.isoformat(),
        "source": {
            "sha256": record.sha256,
            "obs_order_sha256": _obs_order_sha256(prediction.observation_ids),
            "observation_count": record.n_obs,
            "coordinate_dimensions": record.coordinate_dimensions,
            "sample_ids": list(record.sample_ids),
        },
        "annotation": {
            "kind": "source" if prediction.method == "source" else "inferred",
            "method": {"source": "Source annotation", "singler": "SingleR", "rctd": "RCTD full"}[
                prediction.method
            ],
        },
        "coordinates": {
            "system": "cartesian",
            "unit": record.coordinate_unit,
            "y_axis": plan.coordinate_system.get("y_axis", "up"),
        },
        "categories": categories,
        "samples": sample_documents,
        "report": _file_record("report.json", report_bytes),
        "provenance": {
            "environment_lock_sha256": environment_sha,
            "references": references,
            "parameters": dict(plan.parameters),
        },
    }
    if inference_record is not None:
        manifest["inference"] = inference_record
    return publish(_visualization_root(settings), manifest, files)


def _publish_failure(
    settings: Any,
    dataset_id: str,
    report: Mapping[str, Any],
    *,
    category: str = "quality_gate",
) -> None:
    try:
        sidecar = importlib.import_module("iscdc.cell_type_visualization")
    except ImportError:
        return
    publish = getattr(sidecar, "publish_failure", None)
    if not callable(publish):
        return
    detail = str(report.get("detail", "Cell-type annotation failed"))
    publish(
        _visualization_root(settings),
        dataset_id,
        detail[:2000],
        stage="generation",
        category=category,
        details=dict(report),
    )


def generate_cell_type_visualization(
    dataset_id: str,
    settings: Any | None = None,
    *,
    force: bool = False,
    plan_path: Path = DEFAULT_PLAN_PATH,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> AnnotationOutcome:
    """Generate, validate, and atomically publish one configured full dataset."""
    _require_annotation_environment()
    resolved = _resolve_settings(settings)
    plans = load_catalogue_plan(plan_path)
    if dataset_id not in plans:
        raise CellTypeAnnotationError(f"Dataset is absent from annotation plan: {dataset_id}")
    plan = plans[dataset_id]
    record = _catalogue_record(dataset_id, resolved)
    actual_sha256 = sha256_file(record.h5mu_path)
    if actual_sha256 != record.sha256:
        raise CellTypeAnnotationError("Catalogue and .h5mu SHA-256 differ")
    status_path = _visualization_root(resolved) / dataset_id / "status.json"
    if status_path.exists() and not force:
        raise CellTypeAnnotationError(
            f"A visualization status already exists for {dataset_id}; use --force to supersede it"
        )
    if not plan.complete:
        detail = "Scientific configuration is explicitly incomplete"
        report = _failure_report(dataset_id, plan, record.sha256, "scientific_failure", detail)
        _publish_failure(resolved, dataset_id, report, category="scientific_failure")
        return AnnotationOutcome(dataset_id, "scientific_failure", plan.method, detail=detail)

    work_parent = _work_root(resolved)
    work_parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{dataset_id}-", dir=work_parent))
    try:
        if plan.method == "source":
            prediction = generate_source_labels(record.h5mu_path, plan)
            reference = None
            result: Path | None = None
        else:
            reference_dir = _reference_root(resolved) / str(plan.reference_id)
            reference = validate_reference_pack(reference_dir, plan.reference_id)
            exchange = work / "exchange"
            export_sparse_rna_exchange(record.h5mu_path, exchange)
            result = work / "r-result"
            run_r_annotation(
                plan.method,
                exchange,
                reference_dir,
                result,
                plan_path,
                workers=int(plan.parameters.get("cores", 1)),
                parameters=plan.parameters,
                runner=runner,
            )
            prediction = read_r_prediction(result, exchange, plan.method)
        if prediction.n_obs != record.n_obs:
            raise CellTypeAnnotationError("Prediction observation count differs from catalogue")
        expected_ids = (
            _exchange_observation_ids(exchange)
            if plan.method != "source"
            else prediction.observation_ids
        )
        validation = validate_prediction(prediction, plan, expected_observation_ids=expected_ids)
        if sha256_file(record.h5mu_path) != record.sha256:
            raise CellTypeAnnotationError("Source .h5mu changed before atomic publication")
        _sidecar_publish_generation(
            resolved,
            record,
            plan,
            prediction,
            validation,
            reference,
            result,
        )
        return AnnotationOutcome(dataset_id, "success", plan.method)
    except ScientificAnnotationFailure as exc:
        diagnostic_summary = _prediction_diagnostic_summary(
            prediction if "prediction" in locals() else None,
            result if "result" in locals() else None,
        )
        report = _failure_report(
            dataset_id,
            plan,
            record.sha256,
            "scientific_failure",
            str(exc),
            diagnostic_summary=diagnostic_summary,
        )
        _write_json(work / "failure.json", report)
        _publish_failure(resolved, dataset_id, report, category="scientific_failure")
        return AnnotationOutcome(
            dataset_id, "scientific_failure", plan.method, detail=str(exc)
        )
    except Exception as exc:
        report = _failure_report(dataset_id, plan, record.sha256, "failure", str(exc))
        _write_json(work / "failure.json", report)
        _publish_failure(resolved, dataset_id, report)
        if isinstance(exc, CellTypeAnnotationError):
            raise
        raise CellTypeAnnotationError(str(exc)) from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _exchange_observation_ids(exchange_dir: Path) -> tuple[str, ...]:
    with (exchange_dir / "observations.tsv").open(encoding="utf-8", newline="") as stream:
        return tuple(row["observation_id"] for row in csv.DictReader(stream, delimiter="\t"))


def _quality_dict(gates: QualityGates) -> dict[str, Any]:
    return {
        name: getattr(gates, name)
        for name in (
            "min_shared_genes",
            "min_balanced_accuracy",
            "min_macro_f1",
            "max_ece",
            "max_uncertain_fraction",
            "max_mixed_fraction",
            "min_marker_agreement",
            "require_calibration",
        )
    }


def _failure_report(
    dataset_id: str,
    plan: DatasetPlan,
    source_sha256: str,
    status: str,
    detail: str,
    *,
    diagnostic_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_sha256": source_sha256,
        "method": plan.method,
        "reference_id": plan.reference_id,
        "parameters": dict(plan.parameters),
        "qc_gates": _quality_dict(plan.qc),
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
    }
    if diagnostic_summary is not None:
        report["diagnostic_summary"] = dict(diagnostic_summary)
    return report


def _prediction_diagnostic_summary(
    prediction: AnnotationPrediction | None, result_dir: Path | None
) -> dict[str, Any]:
    """Retain bounded target-distribution evidence for iterative scientific repair."""
    if prediction is None:
        return {}
    summary: dict[str, Any] = {
        "n_observations": prediction.n_obs,
        "status_counts": {
            status: prediction.statuses.count(status) for status in sorted(set(prediction.statuses))
        },
        "qc": prediction.diagnostics.get("qc", {}),
        "calibration": prediction.diagnostics.get("calibration", {}),
        "status_reasons": prediction.diagnostics.get("status_reasons", {}),
    }
    if prediction.confidence:
        ordered = sorted(prediction.confidence)

        def quantile(fraction: float) -> float:
            return float(ordered[round(fraction * (len(ordered) - 1))])

        summary["confidence_quantiles"] = {
            "q01": quantile(0.01),
            "q05": quantile(0.05),
            "q25": quantile(0.25),
            "q50": quantile(0.50),
            "q75": quantile(0.75),
            "q95": quantile(0.95),
            "q99": quantile(0.99),
        }
    if result_dir is not None:
        table_path = Path(result_dir) / "predictions.tsv"
        try:
            with table_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
        except OSError:
            rows = []
        for name in (
            "best_score",
            "delta",
            "top_weight",
            "second_weight",
            "entropy",
            "effective_types",
        ):
            values = sorted(
                float(row[name])
                for row in rows
                if row.get(name) not in {None, ""} and math.isfinite(float(row[name]))
            )
            if values:
                summary[f"{name}_quantiles"] = {
                    "q01": values[round(0.01 * (len(values) - 1))],
                    "q05": values[round(0.05 * (len(values) - 1))],
                    "q25": values[round(0.25 * (len(values) - 1))],
                    "q50": values[round(0.50 * (len(values) - 1))],
                    "q75": values[round(0.75 * (len(values) - 1))],
                    "q95": values[round(0.95 * (len(values) - 1))],
                    "q99": values[round(0.99 * (len(values) - 1))],
                }
    return summary


def _resource_batches(
    dataset_ids: Sequence[str],
    plans: Mapping[str, DatasetPlan],
    *,
    max_jobs: int,
    max_cores: int = MAX_TOTAL_CORES,
) -> tuple[tuple[str, ...], ...]:
    """Pack jobs deterministically without exceeding job or declared-core limits."""
    batches: list[tuple[str, ...]] = []
    concurrent_batch: list[str] = []
    concurrent_cores = 0
    for dataset_id in dataset_ids:
        declared_cores = max(1, int(plans[dataset_id].parameters.get("cores", 1)))
        plan = plans[dataset_id]
        max_task_cores = (
            MAX_SINGLER_TASK_CORES
            if plan.method == "singler"
            else MAX_RCTD_TASK_CORES
        )
        if declared_cores > max_task_cores:
            raise CellTypeAnnotationError(
                f"{dataset_id}: declared cores {declared_cores} exceed per-task limit "
                f"{max_task_cores}"
            )
        if concurrent_batch and (
            len(concurrent_batch) >= max_jobs or concurrent_cores + declared_cores > max_cores
        ):
            batches.append(tuple(concurrent_batch))
            concurrent_batch = []
            concurrent_cores = 0
        concurrent_batch.append(dataset_id)
        concurrent_cores += declared_cores
    if concurrent_batch:
        batches.append(tuple(concurrent_batch))
    return tuple(batches)


def audit_cell_type_visualizations(
    dataset_ids: Sequence[str] | None = None,
    settings: Any | None = None,
    *,
    all_datasets: bool = False,
    jobs: int = 1,
    plan_path: Path = DEFAULT_PLAN_PATH,
) -> dict[str, Any]:
    """Generate missing/stale work under the pilot gate and return an audit report."""
    _require_annotation_environment()
    if (
        isinstance(jobs, bool)
        or not isinstance(jobs, int)
        or not 1 <= jobs <= MAX_CONCURRENT_JOBS
    ):
        raise CellTypeAnnotationError(
            f"jobs must be between 1 and {MAX_CONCURRENT_JOBS}"
        )
    plans = load_catalogue_plan(plan_path)
    resolved = _resolve_settings(settings)
    selected = list(dataset_ids or ())
    if all_datasets and selected:
        raise CellTypeAnnotationError("Choose --all or explicit dataset IDs, not both")
    pilot_ids = [dataset_id for dataset_id, plan in plans.items() if plan.pilot]
    if not all_datasets and not selected:
        selected = pilot_ids
    unknown = sorted(set(selected).difference(plans))
    if unknown:
        raise CellTypeAnnotationError("Unknown annotation datasets: " + ", ".join(unknown))

    outcomes: dict[str, AnnotationOutcome] = {}
    errors: dict[str, str] = {}

    def run_batch(batch: Sequence[str]) -> None:
        states = _published_states(resolved, batch, plans)
        pending = [
            dataset_id
            for dataset_id in batch
            if states.get(dataset_id) not in TERMINAL_PILOT_STATES
        ]
        if not pending:
            return
        exclusive = [
            dataset_id
            for dataset_id in pending
            if plans[dataset_id].parameters.get("exclusive") is True
        ]
        parallel = [dataset_id for dataset_id in pending if dataset_id not in exclusive]
        def run_one(dataset_id: str) -> None:
            try:
                outcomes[dataset_id] = generate_cell_type_visualization(
                    dataset_id,
                    resolved,
                    force=states.get(dataset_id) != "missing",
                    plan_path=plan_path,
                )
            except Exception as exc:  # every failure is captured in its immutable report
                errors[dataset_id] = str(exc)

        # Pack jobs deterministically: at most twenty jobs and no more than
        # 40 declared logical cores in aggregate. Per-round plans record the
        # equal allocation actually passed to every concurrently run task.
        for concurrent_batch in _resource_batches(parallel, plans, max_jobs=jobs):
            with ThreadPoolExecutor(
                max_workers=len(concurrent_batch), thread_name_prefix="cell-type-audit"
            ) as pool:
                futures = {
                    pool.submit(run_one, dataset_id): dataset_id
                    for dataset_id in concurrent_batch
                }
                for future in as_completed(futures):
                    future.result()
        if exclusive:
            prerequisite_ids = [
                dataset_id
                for dataset_id, plan in plans.items()
                if plan.parameters.get("exclusive") is not True
            ]
            prerequisite_states = _published_states(resolved, prerequisite_ids, plans)
            prerequisites_ready = all(
                prerequisite_states.get(dataset_id) == "success"
                for dataset_id in prerequisite_ids
            )
            if prerequisites_ready:
                for dataset_id in exclusive:
                    run_one(dataset_id)
            else:
                for dataset_id in exclusive:
                    errors[dataset_id] = (
                        "exclusive annotation deferred until every other dataset is successful"
                    )

    if all_datasets:
        run_batch(pilot_ids)
        pilot_states = _published_states(resolved, pilot_ids, plans)
        if all(state in TERMINAL_PILOT_STATES for state in pilot_states.values()):
            selected = list(plans)
            run_batch([dataset_id for dataset_id in selected if dataset_id not in pilot_ids])
        else:
            selected = pilot_ids
    else:
        run_batch(selected)

    states = _published_states(resolved, selected, plans)
    records: list[dict[str, Any]] = []
    for dataset_id in selected:
        plan = plans[dataset_id]
        record_errors: list[str] = []
        try:
            record = _catalogue_record(dataset_id, resolved)
            expected_sha256 = record.sha256
            if sha256_file(record.h5mu_path) != expected_sha256:
                record_errors.append("source_checksum_mismatch")
        except CellTypeAnnotationError as exc:
            expected_sha256 = None
            record_errors.append(str(exc))
        if not plan.complete:
            record_errors.append("scientific_configuration_incomplete")
        if plan.method != "source" and plan.complete:
            try:
                validate_reference_pack(
                    _reference_root(resolved) / str(plan.reference_id), plan.reference_id
                )
            except CellTypeAnnotationError as exc:
                record_errors.append(str(exc))
        if dataset_id in errors:
            record_errors.append(errors[dataset_id])
        records.append(
            {
                "dataset_id": dataset_id,
                "method": plan.method,
                "pilot": plan.pilot,
                "source_sha256": expected_sha256,
                "published_state": states.get(dataset_id, "missing"),
                "errors": record_errors,
                "ready": states.get(dataset_id) in TERMINAL_PILOT_STATES,
            }
        )
    success_count = sum(record["published_state"] == "success" for record in records)
    scientific_failure_count = sum(
        record["published_state"] == "scientific_failure" for record in records
    )
    failure_count = len(records) - success_count - scientific_failure_count
    return {
        "schema_version": "1.0",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "all" if all_datasets else "selected",
        "jobs": jobs,
        "datasets": records,
        "success_count": success_count,
        "scientific_failure_count": scientific_failure_count,
        "failure_count": failure_count,
        "ok": all(record["ready"] for record in records),
    }


def _published_states(
    settings: Any,
    dataset_ids: Iterable[str],
    plans: Mapping[str, DatasetPlan] | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    root = _visualization_root(settings)
    for dataset_id in dataset_ids:
        try:
            status = json.loads((root / dataset_id / "status.json").read_text(encoding="utf-8"))
            if (
                not isinstance(status, dict)
                or status.get("status_version") != 1
                or status.get("dataset_id") != dataset_id
            ):
                raise ValueError
            if status.get("state") == "success":
                generation_id = status.get("generation_id")
                manifest_path = root / dataset_id / "generations" / generation_id / "manifest.json"
                manifest_bytes = manifest_path.read_bytes()
                if hashlib.sha256(manifest_bytes).hexdigest() != status.get("manifest_sha256"):
                    raise ValueError
                manifest = json.loads(manifest_bytes)
                record = _catalogue_record(dataset_id, settings)
                current = manifest.get("source", {}).get("sha256") == record.sha256
                plan = plans.get(dataset_id) if plans is not None else None
                if plan is not None:
                    provenance = manifest.get("provenance", {})
                    expected_annotation = {
                        "kind": "source" if plan.method == "source" else "inferred",
                        "method": {
                            "source": "Source annotation",
                            "singler": "SingleR",
                            "rctd": "RCTD full",
                        }[plan.method],
                    }
                    current = current and manifest.get("annotation") == expected_annotation
                    current = current and provenance.get("parameters") == dict(plan.parameters)
                    current = current and provenance.get("environment_lock_sha256") == sha256_file(
                        ANNOTATION_ROOT / "renv.lock"
                    )
                    report_record = manifest.get("report", {})
                    report_bytes = (
                        manifest_path.parent / str(report_record.get("path"))
                    ).read_bytes()
                    report = json.loads(report_bytes)
                    current = current and report_record.get("sha256") == hashlib.sha256(
                        report_bytes
                    ).hexdigest()
                    current = current and report_record.get("size") == len(report_bytes)
                    current = current and report.get("dataset_id") == dataset_id
                    current = current and report.get("generation_id") == generation_id
                    current = current and report.get("source_sha256") == record.sha256
                    current = current and report.get("status") == "passed"
                    current = current and report.get("thresholds") == _quality_dict(plan.qc)
                    if plan.method == "source":
                        current = current and not provenance.get("references")
                    elif plan.complete:
                        reference_dir = _reference_root(settings) / str(plan.reference_id)
                        reference = validate_reference_pack(reference_dir, plan.reference_id)
                        expected_reference = {
                            "id": reference.reference_id,
                            "version": reference.version,
                            "sha256": sha256_file(reference_dir / "reference.json"),
                        }
                        current = current and provenance.get("references") == [expected_reference]
                    else:
                        current = False
                result[dataset_id] = "success" if current else "stale"
            elif status.get("state") == "failure":
                failure_id = status.get("failure_id")
                report_path = root / dataset_id / "failures" / failure_id / "report.json"
                report_bytes = report_path.read_bytes()
                if hashlib.sha256(report_bytes).hexdigest() != status.get("failure_report_sha256"):
                    raise ValueError
                report = json.loads(report_bytes)
                plan = plans.get(dataset_id) if plans is not None else None
                record = _catalogue_record(dataset_id, settings)
                scientific_current = (
                    report.get("failure_report_version") == 1
                    and report.get("dataset_id") == dataset_id
                    and report.get("failure_id") == failure_id
                    and report.get("category") == "scientific_failure"
                    and report.get("details", {}).get("status") == "scientific_failure"
                    and report.get("details", {}).get("source_sha256") == record.sha256
                    and (
                        plan is None
                        or (
                            report.get("details", {}).get("method") == plan.method
                            and report.get("details", {}).get("reference_id") == plan.reference_id
                            and report.get("details", {}).get("parameters") == dict(plan.parameters)
                            and report.get("details", {}).get("qc_gates") == _quality_dict(plan.qc)
                        )
                    )
                )
                result[dataset_id] = "scientific_failure" if scientific_current else "failure"
            else:
                result[dataset_id] = "missing"
        except (OSError, ValueError, TypeError, json.JSONDecodeError, CellTypeAnnotationError):
            result[dataset_id] = "missing"
    return result


def annotate_dataset(*args: Any, **kwargs: Any) -> AnnotationOutcome:
    """Backward-compatible descriptive alias for offline integrations."""
    return generate_cell_type_visualization(*args, **kwargs)


def annotate_all(
    settings: Any | None = None, *, force: bool = False, plan_path: Path = DEFAULT_PLAN_PATH
) -> list[AnnotationOutcome]:
    plans = load_catalogue_plan(plan_path)
    return [
        generate_cell_type_visualization(
            dataset_id, settings=settings, force=force, plan_path=plan_path
        )
        for dataset_id in plans
    ]


def prepare_references(
    reference_ids: Sequence[str], settings: Any | None = None, *, force: bool = False
) -> list[dict]:
    return [
        build_cell_type_reference(item, settings=settings, force=force) for item in reference_ids
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline cell-type annotation tooling")
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("build-cell-type-reference")
    reference.add_argument("reference_id")
    reference.add_argument("--force", action="store_true")
    generate = commands.add_parser("generate-cell-type-visualization")
    generate.add_argument("dataset_id")
    generate.add_argument("--force", action="store_true")
    audit = commands.add_parser("audit-cell-type-visualizations")
    audit.add_argument("--all", action="store_true", dest="all_datasets")
    audit.add_argument("dataset_ids", nargs="*")
    audit.add_argument("--jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-cell-type-reference":
        result = build_cell_type_reference(args.reference_id, force=args.force)
    elif args.command == "generate-cell-type-visualization":
        result = generate_cell_type_visualization(args.dataset_id, force=args.force).as_dict()
    else:
        result = audit_cell_type_visualizations(
            getattr(args, "dataset_ids", None),
            all_datasets=args.all_datasets,
            jobs=args.jobs,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
