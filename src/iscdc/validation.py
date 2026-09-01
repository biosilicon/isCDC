from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import mudata
import numpy as np
import pandas as pd
from pydantic import ValidationError
from scipy import sparse

from .schemas import DatabaseMetadata, MetadataDocument, ScalarOrList
from .technology import technology_vocabulary_message, unsupported_technologies

REQUIRED_DATABASE_FIELDS = {
    "schema_version",
    "dataset_id",
    "entry_id",
    "dataset_type",
    "source",
    "organism",
    "tissue",
    "spatial_unit",
    "coordinate_unit",
    "pairing_type",
}
STANDARD_MODALITIES = {
    "rna",
    "protein",
    "atac",
    "histone",
    "methylation",
    "metabolite",
    "lipid",
    "translatome",
    "vdj",
    "bacterial_taxa",
    "fungal_taxa",
    "microbiome",
}
RECOMMENDED_SPATIAL_UNITS = {"cell", "nucleus", "spot", "bin", "region"}
RECOMMENDED_COORDINATE_UNITS = {"micrometer", "millimeter", "pixel", "array_index"}
RECOMMENDED_VALUE_TYPES = {
    "counts",
    "binary",
    "intensity",
    "normalized",
    "log_normalized",
    "background_corrected_intensity",
    "accessibility_score",
    "unknown",
}
FEATURE_MASK_KEY = "feature_measured_by_source"
FEATURE_HARMONIZATION_KEY = "feature_harmonization"
COORDINATE_HARMONIZATION_KEY = "coordinate_harmonization"
SOURCE_FEATURE_COLUMN_PREFIX = "source_feature_ids__"
CELL_TYPE_PROVENANCE_KEY = "cell_type_provenance"
CELL_TYPE_PROVENANCE_VERSION = "1.0"
UNANNOTATED_CELL_TYPE = "Unannotated"
CELL_TYPE_PROVENANCE_FIELDS = {
    "version",
    "unannotated_label",
    "sources",
}
CELL_TYPE_SOURCE_FIELDS = {
    "source_file",
    "source_url",
    "source_sha256",
    "observation_id_column",
    "label_column",
    "annotated_count",
    "unannotated_count",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str


@dataclass(frozen=True)
class ModalitySummary:
    name: str
    technology: ScalarOrList
    value_type: str
    n_obs: int
    n_vars: int


@dataclass
class ValidationOutcome:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    n_obs: int = 0
    coordinate_dimensions: int = 0
    sample_ids: list[str] = field(default_factory=list)
    modalities: dict[str, ModalitySummary] = field(default_factory=dict)
    dataset_type: str = ""
    split_id: str | None = None
    challenge_type: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def report(self, checked_at: str, filename: str) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked_at": checked_at,
            "file": filename,
            "summary": {
                "dataset_type": self.dataset_type,
                "split_id": self.split_id,
                "challenge_type": self.challenge_type,
                "n_obs": self.n_obs,
                "coordinate_dimensions": self.coordinate_dimensions,
                "sample_ids": self.sample_ids,
                "modalities": {
                    name: asdict(summary) for name, summary in sorted(self.modalities.items())
                },
            },
            "errors": [asdict(issue) for issue in self.errors],
            "warnings": [asdict(issue) for issue in self.warnings],
        }


def _normalize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_normalize(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _canonical(value: Any) -> Any:
    normalized = _normalize(value)
    if isinstance(normalized, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in normalized.items()
            if item is not None and not (key == "derivation" and not item)
        }
    if isinstance(normalized, list):
        return [_canonical(item) for item in normalized]
    return normalized


def _has_blank_ids(values) -> bool:  # noqa: ANN001
    return any(not str(value).strip() for value in values)


def _error(outcome: ValidationOutcome, code: str, message: str, path: str) -> None:
    outcome.errors.append(ValidationIssue(code, message, path))


def _warn(outcome: ValidationOutcome, code: str, message: str, path: str) -> None:
    outcome.warnings.append(ValidationIssue(code, message, path))


def _database_model(
    database: Any, outcome: ValidationOutcome, path_prefix: str = "/uns/database"
) -> DatabaseMetadata | None:
    if not isinstance(database, Mapping):
        _error(
            outcome, "missing_database_metadata", "uns['database'] must be a mapping.", path_prefix
        )
        return None
    normalized = _normalize(database)
    if normalized.get("dataset_type") == "full" and not normalized.get("derivation"):
        normalized.pop("derivation", None)
    try:
        return DatabaseMetadata.model_validate(normalized)
    except ValidationError as exc:
        for detail in exc.errors(include_url=False):
            location = "/".join(map(str, detail["loc"]))
            _error(
                outcome,
                "invalid_database_metadata",
                detail["msg"],
                f"{path_prefix}/{location}" if location else path_prefix,
            )
        return None


def _computed_pairing_type(mdata) -> str:  # noqa: ANN001
    memberships = [set(map(str, adata.obs_names)) for adata in mdata.mod.values()]
    if all(values == memberships[0] for values in memberships[1:]):
        return "same_unit"
    if any(
        left.intersection(right)
        for index, left in enumerate(memberships)
        for right in memberships[index + 1 :]
    ):
        return "partially_shared"
    return "unpaired"


def _valid_http_url(value: Any) -> bool:
    if not isinstance(value, str) or value != value.strip():
        return False
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _valid_source_filename(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
    )


def _validate_cell_type_provenance(
    mdata,
    database: DatabaseMetadata | None,
    values: pd.Series,
    outcome: ValidationOutcome,
) -> None:  # noqa: ANN001
    path = f"/uns/{CELL_TYPE_PROVENANCE_KEY}"
    raw = _normalize(mdata.uns.get(CELL_TYPE_PROVENANCE_KEY))
    if not isinstance(raw, Mapping):
        _error(
            outcome,
            "missing_cell_type_provenance",
            "The reserved 'Unannotated' cell type requires source provenance.",
            path,
        )
        return
    if set(raw) != CELL_TYPE_PROVENANCE_FIELDS:
        _error(
            outcome,
            "invalid_cell_type_provenance",
            "Cell-type provenance must contain exactly version, unannotated_label, and sources.",
            path,
        )
    if (
        raw.get("version") != CELL_TYPE_PROVENANCE_VERSION
        or raw.get("unannotated_label") != UNANNOTATED_CELL_TYPE
    ):
        _error(
            outcome,
            "invalid_cell_type_provenance",
            "Cell-type provenance version or reserved label is invalid.",
            path,
        )

    sources = raw.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        _error(
            outcome,
            "invalid_cell_type_provenance",
            "Cell-type provenance sources must be a non-empty source_dataset_id mapping.",
            f"{path}/sources",
        )
        return

    labels = values.astype(object).to_numpy()
    unannotated = labels == UNANNOTATED_CELL_TYPE
    if database is not None and database.dataset_type == "full":
        row_sources = np.full(mdata.n_obs, database.dataset_id, dtype=object)
        allowed_source_ids = {database.dataset_id}
    elif database is not None and database.derivation is not None:
        allowed_source_ids = set(database.derivation.source_dataset_ids)
        if "source_dataset_id" not in mdata.obs:
            _error(
                outcome,
                "invalid_cell_type_provenance",
                "Derived cell-type provenance requires obs['source_dataset_id'].",
                "/obs/source_dataset_id",
            )
            return
        row_sources = mdata.obs["source_dataset_id"].astype(str).to_numpy()
    else:
        return

    expected_source_ids = set(map(str, row_sources[unannotated]))
    observed_source_ids: list[str] = []
    for source_id, item in sources.items():
        item_path = f"{path}/sources/{source_id}"
        if not isinstance(item, Mapping) or set(item) != CELL_TYPE_SOURCE_FIELDS:
            _error(
                outcome,
                "invalid_cell_type_provenance",
                "Each cell-type provenance source must contain the complete source contract.",
                item_path,
            )
            continue
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id != source_id.strip()
            or "/" in source_id
            or "\\" in source_id
        ):
            _error(
                outcome,
                "invalid_cell_type_provenance",
                "Cell-type provenance source_dataset_id keys must be safe non-blank strings.",
                f"{path}/sources",
            )
            continue
        observed_source_ids.append(source_id)
        if source_id not in allowed_source_ids:
            _error(
                outcome,
                "invalid_cell_type_provenance",
                "Cell-type provenance references an undeclared source dataset.",
                item_path,
            )

        annotated_count = item.get("annotated_count")
        unannotated_count = item.get("unannotated_count")
        valid_counts = (
            isinstance(annotated_count, int)
            and not isinstance(annotated_count, bool)
            and annotated_count >= 0
            and isinstance(unannotated_count, int)
            and not isinstance(unannotated_count, bool)
            and unannotated_count > 0
        )
        if not valid_counts:
            _error(
                outcome,
                "invalid_cell_type_provenance",
                "Cell-type provenance counts must be non-negative integers with "
                "unannotated_count > 0.",
                item_path,
            )
        source_rows = row_sources == source_id
        actual_unannotated = int(np.count_nonzero(source_rows & unannotated))
        actual_annotated = int(np.count_nonzero(source_rows & ~unannotated))
        if valid_counts and (
            annotated_count != actual_annotated or unannotated_count != actual_unannotated
        ):
            _error(
                outcome,
                "cell_type_provenance_count_mismatch",
                "Cell-type provenance counts do not match the stored observation labels.",
                item_path,
            )

        required_strings = ("observation_id_column", "label_column")
        if any(
            not isinstance(item.get(field), str)
            or not item[field].strip()
            or item[field] != item[field].strip()
            for field in required_strings
        ):
            _error(
                outcome,
                "invalid_cell_type_provenance",
                "Cell-type provenance alignment columns must be non-blank trimmed strings.",
                item_path,
            )
        if not _valid_source_filename(item.get("source_file")):
            _error(
                outcome,
                "invalid_cell_type_provenance",
                "Cell-type provenance source_file must be a safe original filename.",
                f"{item_path}/source_file",
            )
        if not _valid_http_url(item.get("source_url")):
            _error(
                outcome,
                "invalid_cell_type_provenance",
                "Cell-type provenance source_url must be an absolute HTTP(S) URL.",
                f"{item_path}/source_url",
            )
        if not isinstance(item.get("source_sha256"), str) or not SHA256_PATTERN.fullmatch(
            item["source_sha256"]
        ):
            _error(
                outcome,
                "invalid_cell_type_provenance",
                "Cell-type provenance source_sha256 must be lowercase SHA-256.",
                f"{item_path}/source_sha256",
            )

    if set(observed_source_ids) != expected_source_ids:
        _error(
            outcome,
            "cell_type_provenance_source_mismatch",
            "Cell-type provenance must describe exactly the sources containing Unannotated labels.",
            f"{path}/sources",
        )


def _validate_cell_type(
    mdata, database: DatabaseMetadata | None, outcome: ValidationOutcome
) -> None:  # noqa: ANN001
    """Validate the optional, observation-aligned canonical cell-type annotation."""
    if "cell_type" not in mdata.obs.columns:
        if CELL_TYPE_PROVENANCE_KEY in mdata.uns:
            _error(
                outcome,
                "orphan_cell_type_provenance",
                "Cell-type provenance must not exist without obs['cell_type'].",
                f"/uns/{CELL_TYPE_PROVENANCE_KEY}",
            )
        return

    values = mdata.obs["cell_type"]
    path = "/obs/cell_type"
    if not isinstance(values.dtype, pd.CategoricalDtype):
        _error(
            outcome,
            "invalid_cell_type_dtype",
            "Optional obs['cell_type'] must use pandas categorical dtype.",
            path,
        )
        return
    if values.cat.ordered:
        _error(
            outcome,
            "ordered_cell_type",
            "Optional obs['cell_type'] must be an unordered categorical.",
            path,
        )
    if not values.notna().all():
        _error(
            outcome,
            "null_cell_type",
            "Optional obs['cell_type'] must cover every observation when present.",
            path,
        )

    categories = list(values.cat.categories)
    if any(not isinstance(label, str) for label in categories):
        _error(
            outcome,
            "invalid_cell_type_label",
            "Cell-type categories must be strings.",
            path,
        )
        return
    if any(not label or label != label.strip() for label in categories):
        _error(
            outcome,
            "noncanonical_cell_type_label",
            "Cell-type categories must be non-blank strings without surrounding whitespace.",
            path,
        )
    used = set(values.dropna().astype(object))
    unused = [label for label in categories if label not in used]
    if unused:
        _error(
            outcome,
            "unused_cell_type_category",
            "Cell-type categorical metadata must not contain unused categories.",
            path,
        )
    if UNANNOTATED_CELL_TYPE in used:
        if not categories or categories[-1] != UNANNOTATED_CELL_TYPE:
            _error(
                outcome,
                "nonterminal_unannotated_cell_type",
                "The reserved 'Unannotated' category must be the final category.",
                path,
            )
        _validate_cell_type_provenance(mdata, database, values, outcome)
    elif CELL_TYPE_PROVENANCE_KEY in mdata.uns:
        _error(
            outcome,
            "orphan_cell_type_provenance",
            "Cell-type provenance is only valid when the reserved 'Unannotated' label is used.",
            f"/uns/{CELL_TYPE_PROVENANCE_KEY}",
        )


def _validate_common(
    mdata,
    outcome: ValidationOutcome,
    metadata: MetadataDocument | None = None,  # noqa: ANN001
) -> DatabaseMetadata | None:
    outcome.n_obs = int(mdata.n_obs)
    global_names = mdata.obs_names
    global_names_unique = bool(global_names.is_unique)

    if len(mdata.mod) < 2:
        _error(outcome, "too_few_modalities", "At least two modalities are required.", "/mod")
    if mdata.n_obs <= 0:
        _error(outcome, "empty_observations", "Top-level observations must not be empty.", "/obs")
    if not global_names_unique:
        _error(
            outcome,
            "duplicate_obs_names",
            "Top-level observation IDs must be unique.",
            "/obs/index",
        )
    elif _has_blank_ids(global_names):
        _error(
            outcome, "blank_obs_name", "Top-level observation IDs must not be blank.", "/obs/index"
        )

    internal_database = mdata.uns.get("database")
    database = _database_model(internal_database, outcome)
    if isinstance(internal_database, Mapping):
        missing_fields = REQUIRED_DATABASE_FIELDS.difference(internal_database)
        for name in sorted(missing_fields):
            _error(
                outcome,
                "missing_database_field",
                f"Required database field '{name}' is missing.",
                f"/uns/database/{name}",
            )
    if database is not None:
        outcome.dataset_type = database.dataset_type
        outcome.split_id = database.derivation.split_id if database.derivation else None
        outcome.challenge_type = (
            database.derivation.challenge_type if database.derivation else None
        )
        if metadata is not None:
            internal_values = _canonical(internal_database)
            yaml_values = _canonical(metadata.database_values())
            for name, value in internal_values.items():
                if name not in yaml_values:
                    _error(
                        outcome,
                        "metadata_field_missing_from_yaml",
                        f"Internal database field '{name}' is not present in metadata YAML.",
                        f"/metadata/database/{name}",
                    )
                elif value != yaml_values[name]:
                    _error(
                        outcome,
                        "database_metadata_mismatch",
                        f"Database field '{name}' differs between the file and metadata YAML.",
                        f"/uns/database/{name}",
                    )
        if database.spatial_unit not in RECOMMENDED_SPATIAL_UNITS:
            _warn(
                outcome,
                "nonstandard_spatial_unit",
                "Spatial unit is outside the recommended vocabulary.",
                "/uns/database/spatial_unit",
            )
        if database.coordinate_unit not in RECOMMENDED_COORDINATE_UNITS:
            _warn(
                outcome,
                "nonstandard_coordinate_unit",
                "Coordinate unit is outside the recommended vocabulary.",
                "/uns/database/coordinate_unit",
            )

    if "sample_id" not in mdata.obs.columns:
        _error(outcome, "missing_sample_id", "obs['sample_id'] is required.", "/obs/sample_id")
    else:
        samples = mdata.obs["sample_id"]
        if not samples.notna().all():
            _error(
                outcome,
                "null_sample_id",
                "Sample IDs must not contain null values.",
                "/obs/sample_id",
            )
        internal_sample_ids = sorted({str(value).strip() for value in samples.dropna().unique()})
        if "" in internal_sample_ids:
            _error(outcome, "blank_sample_id", "Sample IDs must not be blank.", "/obs/sample_id")
        outcome.sample_ids = internal_sample_ids
        if metadata is not None and set(internal_sample_ids) != set(metadata.sample_ids):
            _error(
                outcome,
                "sample_ids_mismatch",
                "Sample IDs differ between the file and metadata YAML.",
                "/metadata/sample_ids",
            )

    _validate_cell_type(mdata, database, outcome)

    if "spatial" not in mdata.obsm:
        _error(outcome, "missing_spatial", "obsm['spatial'] is required.", "/obsm/spatial")
    else:
        spatial_values = mdata.obsm["spatial"]
        if len(spatial_values.shape) != 2:
            _error(
                outcome,
                "invalid_spatial_rank",
                "Spatial coordinates must be a matrix.",
                "/obsm/spatial",
            )
        else:
            if spatial_values.shape[0] != mdata.n_obs:
                _error(
                    outcome,
                    "spatial_row_mismatch",
                    "Spatial coordinate rows must match top-level observations.",
                    "/obsm/spatial",
                )
            if spatial_values.shape[1] not in (2, 3):
                _error(
                    outcome,
                    "invalid_spatial_dimensions",
                    "Spatial coordinates must have two or three columns.",
                    "/obsm/spatial",
                )
            else:
                outcome.coordinate_dimensions = int(spatial_values.shape[1])
        if not np.issubdtype(spatial_values.dtype, np.number):
            _error(
                outcome,
                "nonnumeric_spatial",
                "Spatial coordinates must be numeric.",
                "/obsm/spatial",
            )
        else:
            coordinates = np.asarray(spatial_values)
            if not np.isfinite(coordinates).all():
                _error(
                    outcome,
                    "nonfinite_spatial",
                    "Spatial coordinates must be finite.",
                    "/obsm/spatial",
                )
            if spatial_values.dtype != np.float32:
                _warn(
                    outcome,
                    "nonrecommended_spatial_dtype",
                    "float32 is recommended for spatial coordinates.",
                    "/obsm/spatial",
                )

    internal_modality_names = set(mdata.mod)
    if metadata is not None and internal_modality_names != set(metadata.modalities):
        _error(
            outcome,
            "modalities_mismatch",
            "Modality names differ between the file and metadata YAML.",
            "/metadata/modalities",
        )

    covered: set[str] = set()
    global_name_set = set(map(str, global_names)) if global_names_unique else set()
    for name, adata in mdata.mod.items():
        path_prefix = f"/mod/{name}"
        if name not in STANDARD_MODALITIES:
            _warn(
                outcome,
                "nonstandard_modality_name",
                f"'{name}' is outside the recommended modality vocabulary.",
                path_prefix,
            )
        if adata.X is None or getattr(adata.X, "shape", None) in (None, ()):
            _error(
                outcome, "missing_x", "The primary data matrix X is required.", f"{path_prefix}/X"
            )
        elif adata.X.shape != adata.shape:
            _error(
                outcome,
                "x_shape_mismatch",
                "X shape must match the AnnData observation and feature dimensions.",
                f"{path_prefix}/X",
            )
        if adata.n_obs <= 0:
            _error(
                outcome,
                "empty_modality_obs",
                "Modality observations must not be empty.",
                path_prefix,
            )
        if adata.n_vars <= 0:
            _error(
                outcome, "empty_modality_vars", "Modality features must not be empty.", path_prefix
            )
        if not adata.obs_names.is_unique:
            _error(
                outcome,
                "duplicate_modality_obs_names",
                "Modality observation IDs must be unique.",
                f"{path_prefix}/obs/index",
            )
        elif _has_blank_ids(adata.obs_names):
            _error(
                outcome,
                "blank_modality_obs_name",
                "Modality observation IDs must not be blank.",
                f"{path_prefix}/obs/index",
            )
        if not adata.var_names.is_unique:
            _error(
                outcome,
                "duplicate_var_names",
                "Modality feature IDs must be unique.",
                f"{path_prefix}/var/index",
            )
        elif _has_blank_ids(adata.var_names):
            _error(
                outcome,
                "blank_var_name",
                "Modality feature IDs must not be blank.",
                f"{path_prefix}/var/index",
            )
        modality_names = set(map(str, adata.obs_names))
        if global_names_unique and not modality_names.issubset(global_name_set):
            _error(
                outcome,
                "modality_obs_not_global",
                "Every modality observation ID must belong to top-level observations.",
                f"{path_prefix}/obs/index",
            )
        covered.update(modality_names)

        assay = adata.uns.get("assay")
        technology: ScalarOrList = ""
        value_type = ""
        if not isinstance(assay, Mapping):
            _error(
                outcome,
                "missing_assay",
                "uns['assay'] must be a mapping.",
                f"{path_prefix}/uns/assay",
            )
        else:
            technology = _normalize(assay.get("technology", ""))
            raw_value_type = _normalize(assay.get("value_type", ""))
            value_type = raw_value_type.strip() if isinstance(raw_value_type, str) else ""
            valid_technology = isinstance(technology, str) and bool(technology.strip())
            if isinstance(technology, list):
                valid_technology = (
                    bool(technology)
                    and all(isinstance(item, str) and bool(item.strip()) for item in technology)
                    and len(technology) == len(set(technology))
                )
            if not valid_technology:
                _error(
                    outcome,
                    "missing_assay_field",
                    "Assay field 'technology' must be a non-empty string or unique string list.",
                    f"{path_prefix}/uns/assay/technology",
                )
            elif unsupported := unsupported_technologies(technology):
                _error(
                    outcome,
                    "unsupported_technology",
                    technology_vocabulary_message(unsupported),
                    f"{path_prefix}/uns/assay/technology",
                )
            if not value_type:
                _error(
                    outcome,
                    "missing_assay_field",
                    "Assay field 'value_type' must be a non-empty string.",
                    f"{path_prefix}/uns/assay/value_type",
                )
            yaml_assay = metadata.modalities.get(name) if metadata is not None else None
            if yaml_assay is not None and (
                _canonical(technology) != _canonical(yaml_assay.technology)
                or value_type != yaml_assay.value_type
            ):
                _error(
                    outcome,
                    "assay_metadata_mismatch",
                    f"Assay metadata for '{name}' differs from metadata YAML.",
                    f"{path_prefix}/uns/assay",
                )
            if value_type not in RECOMMENDED_VALUE_TYPES:
                _warn(
                    outcome,
                    "nonstandard_value_type",
                    "Value type is outside the recommended vocabulary.",
                    f"{path_prefix}/uns/assay/value_type",
                )
            elif value_type == "unknown":
                _warn(
                    outcome,
                    "unknown_value_type",
                    "A specific value type is preferred when it can be determined.",
                    f"{path_prefix}/uns/assay/value_type",
                )
        outcome.modalities[name] = ModalitySummary(
            name, technology, value_type, int(adata.n_obs), int(adata.n_vars)
        )

    if global_names_unique and covered != global_name_set:
        _error(
            outcome,
            "unrepresented_global_obs",
            "Top-level observations must equal the union of modality observations.",
            "/obs/index",
        )
    if database is not None and len(mdata.mod) >= 2:
        computed_pairing = _computed_pairing_type(mdata)
        if len(mdata.mod) == 2 and computed_pairing != "same_unit":
            _error(
                outcome,
                "two_modality_pairing_required",
                "Two-modality datasets must contain the same observations in both modalities.",
                "/mod",
            )
        if computed_pairing == "unpaired":
            _error(
                outcome,
                "unpaired_modalities",
                "At least one modality pair must share observations.",
                "/mod",
            )
        if database.pairing_type != computed_pairing:
            _error(
                outcome,
                f"{database.pairing_type}_mismatch",
                f"pairing_type must be '{computed_pairing}' for the modality memberships.",
                "/uns/database/pairing_type",
            )
    return database


def _source_pairs(mdata) -> list[tuple[str, str]]:  # noqa: ANN001
    return list(
        zip(
            mdata.obs["source_dataset_id"].astype(str),
            mdata.obs["source_obs_id"].astype(str),
            strict=True,
        )
    )


def _matrix_block_is_zero(matrix: Any, rows: list[int], columns: list[int]) -> bool:
    if not rows or not columns:
        return True
    block = matrix[rows, :][:, columns]
    if sparse.issparse(block):
        return block.nnz == 0
    return not np.any(np.asarray(block) != 0)


def _validate_feature_mask(
    adata,
    modality: str,
    source_ids: list[str],
    source_features: Mapping[str, set[str]],
    row_source_ids: list[str],
    required: bool,
    outcome: ValidationOutcome,
) -> None:  # noqa: ANN001
    path_prefix = f"/mod/{modality}"
    has_mask = FEATURE_MASK_KEY in adata.varm
    if required and not has_mask:
        _error(
            outcome,
            "missing_feature_measurement_mask",
            "A feature measurement mask is required for this feature policy.",
            f"{path_prefix}/varm/{FEATURE_MASK_KEY}",
        )
        return
    if not has_mask:
        return
    mask = np.asarray(adata.varm[FEATURE_MASK_KEY])
    expected = np.column_stack(
        [
            np.asarray(
                [str(feature) in source_features[source_id] for feature in adata.var_names],
                dtype=bool,
            )
            for source_id in source_ids
        ]
    )
    if (
        mask.shape != expected.shape
        or not np.issubdtype(mask.dtype, np.bool_)
        or not np.array_equal(mask, expected)
    ):
        _error(
            outcome,
            "invalid_feature_measurement_mask",
            "The feature measurement mask must be boolean and exactly match "
            "source feature coverage.",
            f"{path_prefix}/varm/{FEATURE_MASK_KEY}",
        )
        return
    measurement = adata.uns.get("feature_measurement")
    if not isinstance(measurement, Mapping):
        _error(
            outcome,
            "missing_feature_measurement_metadata",
            "Feature measurement metadata is required when a mask is stored.",
            f"{path_prefix}/uns/feature_measurement",
        )
        return
    if (
        measurement.get("mask_key") != FEATURE_MASK_KEY
        or list(map(str, _normalize(measurement.get("source_dataset_ids", [])))) != source_ids
        or measurement.get("placeholder_value") != 0
        or not isinstance(measurement.get("description"), str)
        or not measurement["description"].strip()
    ):
        _error(
            outcome,
            "invalid_feature_measurement_metadata",
            "Feature measurement metadata does not describe the stored mask.",
            f"{path_prefix}/uns/feature_measurement",
        )
    for source_index, source_id in enumerate(source_ids):
        missing_columns = np.flatnonzero(~mask[:, source_index]).tolist()
        rows = [index for index, row_source in enumerate(row_source_ids) if row_source == source_id]
        if not _matrix_block_is_zero(adata.X, rows, missing_columns):
            _error(
                outcome,
                "nonzero_unmeasured_feature",
                "Unmeasured feature positions must contain the declared zero placeholder.",
                f"{path_prefix}/X",
            )
            break


def _mapping_digest(feature_order: Sequence[str], mapping: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for feature in feature_order:
        canonical = mapping.get(feature)
        if canonical is None:
            continue
        digest.update(feature.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_source_feature_column(
    adata, source_id: str, outcome: ValidationOutcome, modality: str
) -> dict[str, list[str]]:  # noqa: ANN001
    column = f"{SOURCE_FEATURE_COLUMN_PREFIX}{source_id}"
    if column not in adata.var:
        _error(
            outcome,
            "missing_harmonized_source_features",
            f"Harmonized modality '{modality}' lacks provenance for source '{source_id}'.",
            f"/mod/{modality}/var/{column}",
        )
        return {}
    result: dict[str, list[str]] = {}
    for canonical, raw_value in zip(map(str, adata.var_names), adata.var[column], strict=True):
        try:
            values = json.loads(str(raw_value))
        except (TypeError, ValueError, json.JSONDecodeError):
            values = None
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            _error(
                outcome,
                "invalid_harmonized_source_features",
                f"Feature provenance for source '{source_id}' must be a non-empty unique list.",
                f"/mod/{modality}/var/{column}",
            )
            continue
        result[canonical] = values
    return result


def _aggregate_matrix(
    matrix: Any,
    source_feature_count: int,
    target_count: int,
    source_positions: list[int],
    target_positions: list[int],
) -> Any:
    if sparse.issparse(matrix):
        projection = sparse.csr_matrix(
            (
                np.ones(len(source_positions), dtype=matrix.dtype),
                (source_positions, target_positions),
            ),
            shape=(source_feature_count, target_count),
        )
        return (matrix @ projection).tocsr()
    values = np.asarray(matrix)
    result = np.zeros((values.shape[0], target_count), dtype=values.dtype)
    for source_position, target_position in zip(
        source_positions, target_positions, strict=True
    ):
        result[:, target_position] += values[:, source_position]
    return result


def _matrix_values_equal(left: Any, right: Any) -> bool:
    if sparse.issparse(left) or sparse.issparse(right):
        left_sparse = left if sparse.issparse(left) else sparse.csr_matrix(left)
        right_sparse = right if sparse.issparse(right) else sparse.csr_matrix(right)
        difference = (left_sparse - right_sparse).tocsr()
        return difference.nnz == 0
    return np.array_equal(np.asarray(left), np.asarray(right))


def _validate_harmonized_modality(
    mdata,
    adata,
    modality: str,
    derivation,
    sources: Mapping[str, Any],
    top_provenance: Mapping[str, tuple[str, str]],
    outcome: ValidationOutcome,
) -> None:  # noqa: ANN001
    summary = derivation.feature_harmonization
    assert summary is not None
    metadata = adata.uns.get(FEATURE_HARMONIZATION_KEY)
    if not isinstance(metadata, Mapping):
        _error(
            outcome,
            "missing_feature_harmonization_metadata",
            f"Modality '{modality}' lacks feature harmonization metadata.",
            f"/mod/{modality}/uns/{FEATURE_HARMONIZATION_KEY}",
        )
        return
    source_ids = summary.source_dataset_ids
    if (
        metadata.get("version") != summary.version
        or metadata.get("scope") != summary.scope
        or metadata.get("aggregation") != summary.aggregation
        or metadata.get("namespace") != summary.modalities.get(modality)
        or list(map(str, _normalize(metadata.get("source_dataset_ids", [])))) != source_ids
        or metadata.get("source_feature_column_prefix") != SOURCE_FEATURE_COLUMN_PREFIX
        or not isinstance(metadata.get("sources"), Mapping)
    ):
        _error(
            outcome,
            "invalid_feature_harmonization_metadata",
            f"Modality '{modality}' harmonization metadata disagrees with derivation.",
            f"/mod/{modality}/uns/{FEATURE_HARMONIZATION_KEY}",
        )
        return
    source_metadata = metadata["sources"]
    output_features = list(map(str, adata.var_names))
    source_mappings: dict[str, dict[str, str]] = {}
    ordered_canonical: dict[str, list[str]] = {}
    provenance_by_source = {
        source_id: _parse_source_feature_column(
            adata, source_id, outcome, modality
        )
        for source_id in source_ids
    }
    for source_id in source_ids:
        if source_id not in sources or modality not in sources[source_id].mod:
            _error(
                outcome,
                "missing_harmonization_source_modality",
                f"Source '{source_id}' lacks harmonized modality '{modality}'.",
                f"/sources/{source_id}/mod/{modality}",
            )
            return
        details = source_metadata.get(source_id)
        if not isinstance(details, Mapping):
            _error(
                outcome,
                "missing_harmonization_source_metadata",
                f"Modality '{modality}' lacks mapping metadata for source '{source_id}'.",
                f"/mod/{modality}/uns/{FEATURE_HARMONIZATION_KEY}/sources/{source_id}",
            )
            return
        source_adata = sources[source_id].mod[modality]
        feature_order = list(map(str, source_adata.var_names))
        kind = details.get("kind")
        if kind == "identity":
            mapping = dict(zip(feature_order, feature_order, strict=True))
        elif kind == "var_column":
            column = details.get("column")
            if not isinstance(column, str) or column not in source_adata.var:
                _error(
                    outcome,
                    "invalid_harmonization_var_column",
                    f"Source '{source_id}' lacks the declared feature mapping column.",
                    f"/sources/{source_id}/mod/{modality}/var",
                )
                return
            values = source_adata.var[column]
            if not values.notna().all() or _has_blank_ids(values):
                _error(
                    outcome,
                    "invalid_harmonization_var_values",
                    f"Source '{source_id}' feature mapping column contains invalid values.",
                    f"/sources/{source_id}/mod/{modality}/var/{column}",
                )
                return
            mapping = dict(
                zip(feature_order, [str(value).strip() for value in values], strict=True)
            )
        elif kind == "mapping_file":
            grouped = provenance_by_source[source_id]
            mapping = {
                raw_feature: canonical
                for canonical, raw_features in grouped.items()
                for raw_feature in raw_features
            }
        else:
            _error(
                outcome,
                "invalid_harmonization_mapping_kind",
                f"Source '{source_id}' has an invalid feature mapping kind.",
                f"/mod/{modality}/uns/{FEATURE_HARMONIZATION_KEY}/sources/{source_id}/kind",
            )
            return
        if any(raw_feature not in set(feature_order) for raw_feature in mapping):
            _error(
                outcome,
                "unknown_harmonized_source_feature",
                f"Source '{source_id}' mapping references an unknown feature.",
                f"/mod/{modality}/var",
            )
            return
        if details.get("mapping_sha256") != _mapping_digest(feature_order, mapping):
            _error(
                outcome,
                "feature_harmonization_hash_mismatch",
                f"Source '{source_id}' feature mapping hash is invalid.",
                f"/mod/{modality}/uns/{FEATURE_HARMONIZATION_KEY}/sources/{source_id}",
            )
        source_mappings[source_id] = mapping
        ordered_canonical[source_id] = list(
            dict.fromkeys(mapping[feature] for feature in feature_order if feature in mapping)
        )
    common = set(ordered_canonical[source_ids[0]]).intersection(
        *(set(ordered_canonical[source_id]) for source_id in source_ids[1:])
    )
    expected_features = [
        feature for feature in ordered_canonical[source_ids[0]] if feature in common
    ]
    if output_features != expected_features:
        _error(
            outcome,
            "harmonized_feature_intersection_mismatch",
            f"Modality '{modality}' does not match the ordered all-source intersection.",
            f"/mod/{modality}/var/index",
        )
        return
    for source_id in source_ids:
        grouped = provenance_by_source[source_id]
        expected_grouped: dict[str, list[str]] = {}
        for raw_feature in map(str, sources[source_id].mod[modality].var_names):
            canonical = source_mappings[source_id].get(raw_feature)
            if canonical in common:
                expected_grouped.setdefault(canonical, []).append(raw_feature)
        if grouped != expected_grouped:
            _error(
                outcome,
                "harmonized_feature_provenance_mismatch",
                f"Modality '{modality}' source feature provenance is incorrect.",
                f"/mod/{modality}/var",
            )
            return
    output_obs = list(map(str, adata.obs_names))
    for source_id in derivation.source_dataset_ids:
        output_rows = [
            index
            for index, obs_name in enumerate(output_obs)
            if top_provenance[obs_name][0] == source_id
        ]
        source_obs_ids = [top_provenance[output_obs[index]][1] for index in output_rows]
        source_adata = sources[source_id].mod[modality]
        source_rows = source_adata.obs_names.get_indexer(source_obs_ids)
        if np.any(source_rows < 0):
            continue
        raw_lookup = {
            feature: index
            for index, feature in enumerate(map(str, source_adata.var_names))
        }
        source_positions: list[int] = []
        target_positions: list[int] = []
        grouped = provenance_by_source[source_id]
        for target_position, canonical in enumerate(output_features):
            for raw_feature in grouped.get(canonical, []):
                source_positions.append(raw_lookup[raw_feature])
                target_positions.append(target_position)
        expected = _aggregate_matrix(
            source_adata.X[source_rows, :],
            source_adata.n_vars,
            len(output_features),
            source_positions,
            target_positions,
        )
        actual = adata.X[output_rows, :]
        if not _matrix_values_equal(actual, expected):
            _error(
                outcome,
                "harmonized_matrix_mismatch",
                f"Modality '{modality}' values do not match source aggregation.",
                f"/mod/{modality}/X",
            )
            return


def _validate_harmonized_coordinates(
    mdata,
    database: DatabaseMetadata,
    sources: Mapping[str, Any],
    pairs: Sequence[tuple[str, str]],
    outcome: ValidationOutcome,
) -> None:  # noqa: ANN001
    summary = database.derivation.coordinate_harmonization
    if summary is None:
        return
    metadata = mdata.uns.get(COORDINATE_HARMONIZATION_KEY)
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("sources"), Mapping):
        _error(
            outcome,
            "missing_coordinate_harmonization_metadata",
            "Coordinate harmonization metadata is required.",
            f"/uns/{COORDINATE_HARMONIZATION_KEY}",
        )
        return
    if (
        metadata.get("version") != summary.version
        or metadata.get("spatial_unit") != summary.spatial_unit
        or metadata.get("coordinate_unit") != summary.coordinate_unit
        or list(map(str, _normalize(metadata.get("source_dataset_ids", []))))
        != summary.source_dataset_ids
        or database.spatial_unit != summary.spatial_unit
        or database.coordinate_unit != summary.coordinate_unit
    ):
        _error(
            outcome,
            "invalid_coordinate_harmonization_metadata",
            "Coordinate harmonization metadata disagrees with derivation.",
            f"/uns/{COORDINATE_HARMONIZATION_KEY}",
        )
        return
    output_coordinates = np.asarray(mdata.obsm["spatial"])
    for source_id in database.derivation.source_dataset_ids:
        details = metadata["sources"].get(source_id)
        if not isinstance(details, Mapping) or source_id not in sources:
            _error(
                outcome,
                "missing_coordinate_source_metadata",
                f"Coordinate metadata for source '{source_id}' is missing.",
                f"/uns/{COORDINATE_HARMONIZATION_KEY}/sources/{source_id}",
            )
            continue
        source = sources[source_id]
        if details.get("kind") == "obsm":
            key = details.get("key")
            if not isinstance(key, str) or key not in source.obsm:
                expected_all = None
            else:
                expected_all = np.asarray(source.obsm[key])
        elif details.get("kind") == "obs_columns":
            x, y = details.get("x"), details.get("y")
            if (
                not isinstance(x, str)
                or not isinstance(y, str)
                or x not in source.obs
                or y not in source.obs
            ):
                expected_all = None
            else:
                expected_all = np.column_stack([source.obs[x].to_numpy(), source.obs[y].to_numpy()])
        else:
            expected_all = None
        if expected_all is None:
            _error(
                outcome,
                "invalid_coordinate_source_rule",
                f"Coordinate rule for source '{source_id}' cannot be applied.",
                f"/uns/{COORDINATE_HARMONIZATION_KEY}/sources/{source_id}",
            )
            continue
        expected_all = expected_all.astype(np.float32, copy=False)
        source_lookup = {name: index for index, name in enumerate(map(str, source.obs_names))}
        output_rows = [index for index, pair in enumerate(pairs) if pair[0] == source_id]
        source_rows = [source_lookup[pairs[index][1]] for index in output_rows]
        if not np.array_equal(output_coordinates[output_rows], expected_all[source_rows]):
            _error(
                outcome,
                "harmonized_coordinate_mismatch",
                f"Coordinates for source '{source_id}' do not match the declared rule.",
                "/obsm/spatial",
            )


def _cell_type_source_entries(mdata) -> dict[str, Mapping[str, Any]]:  # noqa: ANN001
    raw = _normalize(mdata.uns.get(CELL_TYPE_PROVENANCE_KEY))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("sources"), Mapping):
        return {}
    return {
        str(source_id): item
        for source_id, item in raw["sources"].items()
        if isinstance(item, Mapping)
    }


def _validate_derived_cell_type(
    mdata,
    sources: Mapping[str, Any],
    pairs: Sequence[tuple[str, str]],
    outcome: ValidationOutcome,
) -> None:  # noqa: ANN001
    source_has_labels = {
        source_id: "cell_type" in source.obs.columns for source_id, source in sources.items()
    }
    output_has_labels = "cell_type" in mdata.obs.columns
    all_sources_have_labels = bool(source_has_labels) and all(source_has_labels.values())
    if all_sources_have_labels and not output_has_labels:
        _error(
            outcome,
            "missing_derived_cell_type",
            "Derived data must propagate cell_type when every direct full source provides it.",
            "/obs/cell_type",
        )
        return
    if output_has_labels and not all_sources_have_labels:
        _error(
            outcome,
            "unexpected_derived_cell_type",
            "Derived data must omit cell_type when any direct full source lacks it.",
            "/obs/cell_type",
        )
        return
    if not output_has_labels:
        return

    source_labels = {
        source_id: dict(
            zip(
                map(str, source.obs_names),
                source.obs["cell_type"].astype(object),
                strict=True,
            )
        )
        for source_id, source in sources.items()
    }
    if any(
        source_id not in source_labels or obs_id not in source_labels[source_id]
        for source_id, obs_id in pairs
    ):
        return
    expected = [source_labels[source_id][obs_id] for source_id, obs_id in pairs]
    actual = mdata.obs["cell_type"].astype(object).tolist()
    if expected != actual:
        _error(
            outcome,
            "derived_cell_type_mismatch",
            "Derived cell_type values must exactly match their direct full-source observations.",
            "/obs/cell_type",
        )

    if UNANNOTATED_CELL_TYPE not in set(actual):
        return
    output_entries = _cell_type_source_entries(mdata)
    immutable_fields = CELL_TYPE_SOURCE_FIELDS.difference(
        {"annotated_count", "unannotated_count"}
    )
    for source_id, output_entry in output_entries.items():
        source = sources.get(source_id)
        source_entry = (
            _cell_type_source_entries(source).get(source_id) if source is not None else None
        )
        if source_entry is None or any(
            _canonical(output_entry.get(field)) != _canonical(source_entry.get(field))
            for field in immutable_fields
        ):
            _error(
                outcome,
                "derived_cell_type_provenance_mismatch",
                "Derived cell-type provenance must preserve the direct source-file contract.",
                f"/uns/{CELL_TYPE_PROVENANCE_KEY}/sources/{source_id}",
            )


def _validate_derivation(
    mdata,
    database: DatabaseMetadata,
    outcome: ValidationOutcome,
    source_paths: Mapping[str, Path] | None,
) -> None:  # noqa: ANN001
    if database.dataset_type == "full":
        return
    for column in ("source_dataset_id", "source_obs_id"):
        if column not in mdata.obs:
            _error(
                outcome,
                f"missing_{column}",
                f"obs['{column}'] is required for derived datasets.",
                f"/obs/{column}",
            )
        elif not mdata.obs[column].notna().all() or _has_blank_ids(mdata.obs[column].dropna()):
            _error(
                outcome,
                f"invalid_{column}",
                f"obs['{column}'] must contain non-blank values.",
                f"/obs/{column}",
            )
    if not all(column in mdata.obs for column in ("source_dataset_id", "source_obs_id")):
        return
    derivation = database.derivation
    if derivation is None:
        return
    pairs = _source_pairs(mdata)
    if len(pairs) != len(set(pairs)):
        _error(
            outcome,
            "duplicate_source_observation",
            "Source dataset/observation pairs must be unique.",
            "/obs",
        )
    observed_source_ids = {source_id for source_id, _ in pairs}
    declared_source_ids = derivation.source_dataset_ids
    if not observed_source_ids.issubset(set(declared_source_ids)):
        _error(
            outcome,
            "undeclared_source_dataset",
            "Every observed source_dataset_id must be declared in derivation.",
            "/obs/source_dataset_id",
        )
    if observed_source_ids != set(declared_source_ids):
        _error(
            outcome,
            "unused_declared_source_dataset",
            "Every declared source dataset must contribute at least one observation.",
            "/uns/database/derivation/source_dataset_ids",
        )
    if derivation.construction_type == "subset" and len(observed_source_ids) != 1:
        _error(
            outcome,
            "invalid_subset_sources",
            "A subset must use exactly one observed source dataset.",
            "/obs/source_dataset_id",
        )
    if source_paths is None:
        _error(
            outcome,
            "missing_source_context",
            "Derived validation requires paths for all source full datasets.",
            "/uns/database/derivation/source_dataset_ids",
        )
        return
    context_source_ids = list(declared_source_ids)
    if derivation.feature_harmonization is not None:
        context_source_ids = list(derivation.feature_harmonization.source_dataset_ids)
    if derivation.coordinate_harmonization is not None:
        coordinate_source_ids = derivation.coordinate_harmonization.source_dataset_ids
        if set(coordinate_source_ids) != set(context_source_ids):
            _error(
                outcome,
                "harmonization_source_mismatch",
                "Feature and coordinate harmonization must use the same source set.",
                "/uns/database/derivation",
            )
        else:
            context_source_ids = list(coordinate_source_ids)
    missing_paths = [source_id for source_id in context_source_ids if source_id not in source_paths]
    if missing_paths:
        _error(
            outcome,
            "missing_source_dataset",
            f"Source full datasets are unavailable: {', '.join(missing_paths)}.",
            "/uns/database/derivation/source_dataset_ids",
        )
        return

    sources: dict[str, Any] = {}
    source_obs: dict[str, set[str]] = {}
    source_samples: dict[str, dict[str, str]] = {}
    source_modality_obs: dict[str, dict[str, set[str]]] = {}
    source_modality_features: dict[str, dict[str, list[str]]] = {}
    try:
        for source_id in context_source_ids:
            path = Path(source_paths[source_id])
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
                    source = mudata.read_h5mu(path, backed="r")
            except Exception as exc:
                _error(
                    outcome,
                    "source_read_failed",
                    f"Unable to read source dataset '{source_id}': {exc}",
                    f"/sources/{source_id}",
                )
                continue
            sources[source_id] = source
            source_outcome = ValidationOutcome()
            source_database = _validate_common(source, source_outcome)
            if source_outcome.errors:
                _error(
                    outcome,
                    "invalid_source_dataset",
                    f"Source dataset '{source_id}' does not satisfy schema 1.2.",
                    f"/sources/{source_id}",
                )
            if (
                source_database is None
                or source_database.dataset_type != "full"
                or source_database.dataset_id != source_id
            ):
                _error(
                    outcome,
                    "source_not_full",
                    f"Source '{source_id}' must identify a schema 1.2 full dataset "
                    "with the same dataset_id.",
                    f"/sources/{source_id}",
                )
            names = list(map(str, source.obs_names))
            source_obs[source_id] = set(names)
            source_samples[source_id] = dict(
                zip(names, source.obs["sample_id"].astype(str), strict=True)
            )
            source_modality_obs[source_id] = {
                name: set(map(str, adata.obs_names)) for name, adata in source.mod.items()
            }
            source_modality_features[source_id] = {
                name: list(map(str, adata.var_names)) for name, adata in source.mod.items()
            }
        if len(sources) != len(context_source_ids):
            return

        for source_id, obs_id in pairs:
            if source_id not in source_obs or obs_id not in source_obs[source_id]:
                _error(
                    outcome,
                    "source_observation_not_found",
                    f"Source observation ({source_id!r}, {obs_id!r}) does not exist.",
                    "/obs/source_obs_id",
                )

        _validate_derived_cell_type(mdata, sources, pairs, outcome)

        top_provenance = dict(zip(map(str, mdata.obs_names), pairs, strict=True))
        top_pair_set = set(pairs)
        sample_forward: dict[tuple[str, str], set[str]] = {}
        sample_reverse: dict[str, set[tuple[str, str]]] = {}
        if "sample_id" in mdata.obs:
            for pair, output_sample in zip(pairs, mdata.obs["sample_id"].astype(str), strict=True):
                source_id, obs_id = pair
                original_sample = source_samples.get(source_id, {}).get(obs_id)
                if original_sample is None:
                    continue
                source_sample = (source_id, original_sample)
                sample_forward.setdefault(source_sample, set()).add(output_sample)
                sample_reverse.setdefault(output_sample, set()).add(source_sample)
            if any(len(values) != 1 for values in sample_forward.values()) or any(
                len(values) != 1 for values in sample_reverse.values()
            ):
                _error(
                    outcome,
                    "ambiguous_derived_sample_id",
                    "Derived sample IDs must reversibly distinguish every "
                    "source dataset/sample pair.",
                    "/obs/sample_id",
                )
            if len(declared_source_ids) > 1 and any(
                output_samples != {f"{source_id}::{original_sample}"}
                for (source_id, original_sample), output_samples in sample_forward.items()
            ):
                _error(
                    outcome,
                    "noncanonical_composite_sample_id",
                    "Samples in a multi-source derived file must use "
                    "'<source_dataset_id>::<original_sample_id>'.",
                    "/obs/sample_id",
                )

        for modality, adata in mdata.mod.items():
            observed_pairs = {
                top_provenance[str(name)] for name in adata.obs_names if str(name) in top_provenance
            }
            expected_pairs = {
                pair
                for pair in top_pair_set
                if pair[1] in source_modality_obs.get(pair[0], {}).get(modality, set())
            }
            if observed_pairs != expected_pairs:
                _error(
                    outcome,
                    "invalid_modality_source_membership",
                    f"Modality '{modality}' does not preserve source observation membership.",
                    f"/mod/{modality}/obs/index",
                )

            if derivation.feature_harmonization is not None:
                _validate_harmonized_modality(
                    mdata,
                    adata,
                    modality,
                    derivation,
                    sources,
                    top_provenance,
                    outcome,
                )
                continue

            relevant_ids = [
                source_id
                for source_id in declared_source_ids
                if modality in source_modality_features[source_id]
            ]
            if not relevant_ids:
                _error(
                    outcome,
                    "modality_without_source",
                    f"Modality '{modality}' has no declared source dataset.",
                    f"/mod/{modality}",
                )
                continue
            feature_lists = [
                source_modality_features[source_id][modality] for source_id in relevant_ids
            ]
            output_features = list(map(str, adata.var_names))
            policy = derivation.feature_merge_policy
            feature_space_valid = True
            if policy == "preserve":
                feature_space_valid = (
                    all(values == feature_lists[0] for values in feature_lists[1:])
                    and output_features == feature_lists[0]
                )
            elif policy == "intersection":
                expected = set(feature_lists[0]).intersection(
                    *(set(values) for values in feature_lists[1:])
                )
                feature_space_valid = set(output_features) == expected and len(
                    output_features
                ) == len(expected)
            elif policy == "union":
                expected = set().union(*(set(values) for values in feature_lists))
                feature_space_valid = set(output_features) == expected and len(
                    output_features
                ) == len(expected)
            else:
                reference_id = derivation.reference_dataset_id
                feature_space_valid = (
                    bool(reference_id)
                    and modality in source_modality_features[reference_id]
                    and output_features == source_modality_features[reference_id][modality]
                )
            if not feature_space_valid:
                _error(
                    outcome,
                    "feature_merge_policy_mismatch",
                    f"Modality '{modality}' feature space does not match policy '{policy}'.",
                    f"/mod/{modality}/var/index",
                )

            feature_sets = {
                source_id: set(source_modality_features[source_id].get(modality, []))
                for source_id in declared_source_ids
            }
            missing_measurements = any(
                any(feature not in feature_sets[source_id] for feature in output_features)
                for source_id in declared_source_ids
            )
            mask_required = policy == "union" or (policy == "reference" and missing_measurements)
            row_sources = [top_provenance[str(name)][0] for name in adata.obs_names]
            _validate_feature_mask(
                adata,
                modality,
                declared_source_ids,
                feature_sets,
                row_sources,
                mask_required,
                outcome,
            )
        _validate_harmonized_coordinates(mdata, database, sources, pairs, outcome)
    finally:
        for source in sources.values():
            source.file.close()


def validate_mudata(mdata, metadata: MetadataDocument | None = None) -> ValidationOutcome:  # noqa: ANN001
    """Validate the schema 1.2 structure already loaded in memory."""
    outcome = ValidationOutcome()
    _validate_common(mdata, outcome, metadata)
    return outcome


def validate_h5mu(
    path: Path,
    metadata: MetadataDocument | None = None,
    *,
    source_paths: Mapping[str, Path] | None = None,
    peer_paths: Sequence[Path] = (),
) -> ValidationOutcome:
    outcome = ValidationOutcome()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
            mdata = mudata.read_h5mu(path, backed="r")
    except Exception as exc:
        _error(outcome, "h5mu_read_failed", f"Unable to read MuData file: {exc}", "/")
        return outcome
    try:
        database = _validate_common(mdata, outcome, metadata)
        if database is not None:
            _validate_derivation(mdata, database, outcome, source_paths)
            if database.dataset_type in {"train", "test"}:
                for peer_path in peer_paths:
                    if database.dataset_type == "train":
                        peer_outcome = validate_train_test_pair(path, Path(peer_path))
                    else:
                        peer_outcome = validate_train_test_pair(Path(peer_path), path)
                    outcome.errors.extend(peer_outcome.errors)
                    outcome.warnings.extend(peer_outcome.warnings)
    finally:
        mdata.file.close()
    return outcome


def validate_train_test_pair(train_path: Path, test_path: Path) -> ValidationOutcome:
    """Validate split identity and source-observation disjointness for two files."""
    outcome = ValidationOutcome()
    loaded: list[Any] = []
    try:
        for label, path in (("train", train_path), ("test", test_path)):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
                    loaded.append(mudata.read_h5mu(path, backed="r"))
            except Exception as exc:
                _error(
                    outcome,
                    "split_file_read_failed",
                    f"Unable to read {label} file: {exc}",
                    f"/{label}",
                )
                return outcome
        train, test = loaded
        train_database = _database_model(train.uns.get("database"), outcome, "/train/uns/database")
        test_database = _database_model(test.uns.get("database"), outcome, "/test/uns/database")
        if train_database is None or test_database is None:
            return outcome
        if train_database.dataset_type != "train" or test_database.dataset_type != "test":
            _error(
                outcome,
                "invalid_split_dataset_types",
                "The pair must contain one train and one test dataset.",
                "/",
            )
            return outcome
        if train_database.derivation is None or test_database.derivation is None:
            return outcome
        if train_database.derivation.split_id != test_database.derivation.split_id:
            _error(
                outcome,
                "split_id_mismatch",
                "Train and test datasets must use the same split_id.",
                "/uns/database/derivation/split_id",
            )
        if train_database.derivation.challenge_type != test_database.derivation.challenge_type:
            _error(
                outcome,
                "challenge_type_mismatch",
                "Train and test datasets must use the same challenge_type.",
                "/uns/database/derivation/challenge_type",
            )
        train_harmonization = train_database.derivation.feature_harmonization
        test_harmonization = test_database.derivation.feature_harmonization
        if (train_harmonization is None) != (test_harmonization is None):
            _error(
                outcome,
                "feature_harmonization_mismatch",
                "Train and test must either both use feature harmonization or neither use it.",
                "/uns/database/derivation/feature_harmonization",
            )
        elif train_harmonization is not None and test_harmonization is not None:
            if train_harmonization != test_harmonization:
                _error(
                    outcome,
                    "feature_harmonization_mismatch",
                    "Train and test feature harmonization summaries must match.",
                    "/uns/database/derivation/feature_harmonization",
                )
            if set(train.mod) != set(test.mod) or any(
                list(map(str, train.mod[name].var_names))
                != list(map(str, test.mod[name].var_names))
                for name in set(train.mod).intersection(test.mod)
            ):
                _error(
                    outcome,
                    "harmonized_feature_order_mismatch",
                    "Harmonized train and test modality feature orders must match exactly.",
                    "/mod",
                )
        if (
            train_database.derivation.coordinate_harmonization
            != test_database.derivation.coordinate_harmonization
        ):
            _error(
                outcome,
                "coordinate_harmonization_mismatch",
                "Train and test coordinate harmonization summaries must match.",
                "/uns/database/derivation/coordinate_harmonization",
            )
        for label, value in (("train", train), ("test", test)):
            if not all(column in value.obs for column in ("source_dataset_id", "source_obs_id")):
                _error(
                    outcome,
                    "missing_source_provenance",
                    f"The {label} dataset lacks source provenance columns.",
                    f"/{label}/obs",
                )
                return outcome
        train_pairs = set(_source_pairs(train))
        test_pairs = set(_source_pairs(test))
        if not train_pairs.isdisjoint(test_pairs):
            _error(
                outcome,
                "train_test_source_overlap",
                "Train and test datasets contain overlapping source observations.",
                "/obs",
            )
    finally:
        for value in loaded:
            value.file.close()
    return outcome
