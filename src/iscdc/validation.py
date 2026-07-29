from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mudata
import numpy as np
from pydantic import ValidationError
from scipy import sparse

from .schemas import DatabaseMetadata, MetadataDocument, ScalarOrList

REQUIRED_DATABASE_FIELDS = {
    "schema_version",
    "dataset_id",
    "dataset_type",
    "source",
    "organism",
    "tissue",
    "spatial_unit",
    "coordinate_unit",
    "pairing_type",
}
STANDARD_MODALITIES = {"rna", "protein", "atac", "methylation", "metabolite", "lipid"}
RECOMMENDED_SPATIAL_UNITS = {"cell", "nucleus", "spot", "bin", "region"}
RECOMMENDED_COORDINATE_UNITS = {"micrometer", "millimeter", "pixel", "array_index"}
RECOMMENDED_PAIRING_TYPES = {"same_unit", "partially_shared", "unpaired"}
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
        if (
            database.pairing_type in RECOMMENDED_PAIRING_TYPES
            and database.pairing_type != computed_pairing
        ):
            _error(
                outcome,
                f"{database.pairing_type}_mismatch",
                f"pairing_type must be '{computed_pairing}' for the modality memberships.",
                "/uns/database/pairing_type",
            )
        elif database.pairing_type not in RECOMMENDED_PAIRING_TYPES:
            _warn(
                outcome,
                "nonstandard_pairing_type",
                "Pairing type is outside the recommended vocabulary and was not checked.",
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
    missing_paths = [
        source_id for source_id in declared_source_ids if source_id not in source_paths
    ]
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
        for source_id in declared_source_ids:
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
                    f"Source dataset '{source_id}' does not satisfy schema 1.1.",
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
                    f"Source '{source_id}' must identify a schema 1.1 full dataset "
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
        if len(sources) != len(declared_source_ids):
            return

        for source_id, obs_id in pairs:
            if source_id not in source_obs or obs_id not in source_obs[source_id]:
                _error(
                    outcome,
                    "source_observation_not_found",
                    f"Source observation ({source_id!r}, {obs_id!r}) does not exist.",
                    "/obs/source_obs_id",
                )

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
    finally:
        for source in sources.values():
            source.file.close()


def validate_mudata(mdata, metadata: MetadataDocument | None = None) -> ValidationOutcome:  # noqa: ANN001
    """Validate the schema 1.1 structure already loaded in memory."""
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
