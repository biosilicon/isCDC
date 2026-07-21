from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import mudata
import numpy as np

from .schemas import MetadataDocument

REQUIRED_DATABASE_FIELDS = {
    "schema_version",
    "dataset_id",
    "source",
    "organism",
    "tissue",
    "spatial_unit",
    "coordinate_unit",
    "pairing_type",
}
STANDARD_MODALITIES = {"rna", "protein", "atac", "methylation", "metabolite", "lipid"}
RECOMMENDED_SPATIAL_UNITS = {"cell", "nucleus", "spot", "bin", "region"}
RECOMMENDED_COORDINATE_UNITS = {"micrometer", "millimeter", "pixel"}
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


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str


@dataclass(frozen=True)
class ModalitySummary:
    name: str
    technology: str
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

    @property
    def valid(self) -> bool:
        return not self.errors

    def report(self, checked_at: str, filename: str) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked_at": checked_at,
            "file": filename,
            "summary": {
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


def _has_blank_ids(index) -> bool:  # noqa: ANN001
    return any(not str(value).strip() for value in index)


def validate_h5mu(path: Path, metadata: MetadataDocument) -> ValidationOutcome:
    outcome = ValidationOutcome()

    def error(code: str, message: str, path_value: str) -> None:
        outcome.errors.append(ValidationIssue(code, message, path_value))

    def warn(code: str, message: str, path_value: str) -> None:
        outcome.warnings.append(ValidationIssue(code, message, path_value))

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
            mdata = mudata.read_h5mu(path, backed="r")
    except Exception as exc:  # MuData exposes several backend-specific read exceptions.
        error("h5mu_read_failed", f"Unable to read MuData file: {exc}", "/")
        return outcome

    try:
        outcome.n_obs = int(mdata.n_obs)
        global_names = mdata.obs_names
        global_names_unique = bool(global_names.is_unique)

        if len(mdata.mod) < 2:
            error("too_few_modalities", "At least two modalities are required.", "/mod")
        if mdata.n_obs <= 0:
            error("empty_observations", "Top-level observations must not be empty.", "/obs")
        if not global_names_unique:
            error("duplicate_obs_names", "Top-level observation IDs must be unique.", "/obs/index")
        elif _has_blank_ids(global_names):
            error("blank_obs_name", "Top-level observation IDs must not be blank.", "/obs/index")

        internal_database = mdata.uns.get("database")
        yaml_database = metadata.database_values()
        pairing_type = None
        if not isinstance(internal_database, Mapping):
            error(
                "missing_database_metadata", "uns['database'] must be a mapping.", "/uns/database"
            )
        else:
            missing_fields = REQUIRED_DATABASE_FIELDS.difference(internal_database)
            for name in sorted(missing_fields):
                error(
                    "missing_database_field",
                    f"Required database field '{name}' is missing.",
                    f"/uns/database/{name}",
                )
            for name, value in internal_database.items():
                if name not in yaml_database:
                    error(
                        "metadata_field_missing_from_yaml",
                        f"Internal database field '{name}' is not present in metadata YAML.",
                        f"/metadata/database/{name}",
                    )
                elif _normalize(value) != _normalize(yaml_database[name]):
                    error(
                        "database_metadata_mismatch",
                        f"Database field '{name}' differs between the file and metadata YAML.",
                        f"/uns/database/{name}",
                    )
            pairing_type = _normalize(internal_database.get("pairing_type"))
            if internal_database.get("schema_version") != "1.0":
                warn(
                    "unrecognized_schema_version",
                    "The validator was developed for schema version 1.0.",
                    "/uns/database/schema_version",
                )
            if internal_database.get("spatial_unit") not in RECOMMENDED_SPATIAL_UNITS:
                warn(
                    "nonstandard_spatial_unit",
                    "Spatial unit is outside the recommended vocabulary.",
                    "/uns/database/spatial_unit",
                )
            if internal_database.get("coordinate_unit") not in RECOMMENDED_COORDINATE_UNITS:
                warn(
                    "nonstandard_coordinate_unit",
                    "Coordinate unit is outside the recommended vocabulary.",
                    "/uns/database/coordinate_unit",
                )

        if "sample_id" not in mdata.obs.columns:
            error("missing_sample_id", "obs['sample_id'] is required.", "/obs/sample_id")
        else:
            sample_series = mdata.obs["sample_id"]
            if not sample_series.notna().all():
                error(
                    "null_sample_id", "Sample IDs must not contain null values.", "/obs/sample_id"
                )
            internal_sample_ids = sorted(
                {str(value).strip() for value in sample_series.dropna().unique()}
            )
            if "" in internal_sample_ids:
                error("blank_sample_id", "Sample IDs must not be blank.", "/obs/sample_id")
            outcome.sample_ids = internal_sample_ids
            if set(internal_sample_ids) != set(metadata.sample_ids):
                error(
                    "sample_ids_mismatch",
                    "Sample IDs differ between the file and metadata YAML.",
                    "/metadata/sample_ids",
                )

        if "spatial" not in mdata.obsm:
            error("missing_spatial", "obsm['spatial'] is required.", "/obsm/spatial")
        else:
            spatial = mdata.obsm["spatial"]
            if len(spatial.shape) != 2:
                error(
                    "invalid_spatial_rank", "Spatial coordinates must be a matrix.", "/obsm/spatial"
                )
            else:
                if spatial.shape[0] != mdata.n_obs:
                    error(
                        "spatial_row_mismatch",
                        "Spatial coordinate rows must match top-level observations.",
                        "/obsm/spatial",
                    )
                if spatial.shape[1] not in (2, 3):
                    error(
                        "invalid_spatial_dimensions",
                        "Spatial coordinates must have two or three columns.",
                        "/obsm/spatial",
                    )
                else:
                    outcome.coordinate_dimensions = int(spatial.shape[1])
            if not np.issubdtype(spatial.dtype, np.number):
                error("nonnumeric_spatial", "Spatial coordinates must be numeric.", "/obsm/spatial")
            elif spatial.dtype != np.float32:
                warn(
                    "nonrecommended_spatial_dtype",
                    "float32 is recommended for spatial coordinates.",
                    "/obsm/spatial",
                )

        internal_modality_names = set(mdata.mod)
        yaml_modality_names = set(metadata.modalities)
        if internal_modality_names != yaml_modality_names:
            error(
                "modalities_mismatch",
                "Modality names differ between the file and metadata YAML.",
                "/metadata/modalities",
            )

        masks: dict[str, np.ndarray] = {}
        covered = np.zeros(mdata.n_obs, dtype=bool) if global_names_unique else None
        for name, adata in mdata.mod.items():
            path_prefix = f"/mod/{name}"
            if name not in STANDARD_MODALITIES:
                warn(
                    "nonstandard_modality_name",
                    f"'{name}' is outside the recommended modality vocabulary.",
                    path_prefix,
                )
            if adata.X is None or getattr(adata.X, "shape", None) in (None, ()):
                error("missing_x", "The primary data matrix X is required.", f"{path_prefix}/X")
            elif adata.X.shape != adata.shape:
                error(
                    "x_shape_mismatch",
                    "X shape must match the AnnData observation and feature dimensions.",
                    f"{path_prefix}/X",
                )
            if adata.n_obs <= 0:
                error("empty_modality_obs", "Modality observations must not be empty.", path_prefix)
            if adata.n_vars <= 0:
                error("empty_modality_vars", "Modality features must not be empty.", path_prefix)
            if not adata.obs_names.is_unique:
                error(
                    "duplicate_modality_obs_names",
                    "Modality observation IDs must be unique.",
                    f"{path_prefix}/obs/index",
                )
            elif _has_blank_ids(adata.obs_names):
                error(
                    "blank_modality_obs_name",
                    "Modality observation IDs must not be blank.",
                    f"{path_prefix}/obs/index",
                )
            if not adata.var_names.is_unique:
                error(
                    "duplicate_var_names",
                    "Modality feature IDs must be unique.",
                    f"{path_prefix}/var/index",
                )
            elif _has_blank_ids(adata.var_names):
                error(
                    "blank_var_name",
                    "Modality feature IDs must not be blank.",
                    f"{path_prefix}/var/index",
                )

            if global_names_unique:
                positions = global_names.get_indexer(adata.obs_names)
                if np.any(positions < 0):
                    error(
                        "modality_obs_not_global",
                        "Every modality observation ID must belong to top-level observations.",
                        f"{path_prefix}/obs/index",
                    )
                mask = np.zeros(mdata.n_obs, dtype=bool)
                valid_positions = positions[positions >= 0]
                mask[valid_positions] = True
                masks[name] = mask
                if covered is not None:
                    covered[valid_positions] = True

            assay = adata.uns.get("assay")
            technology = ""
            value_type = ""
            if not isinstance(assay, Mapping):
                error(
                    "missing_assay", "uns['assay'] must be a mapping.", f"{path_prefix}/uns/assay"
                )
            else:
                for field_name in ("technology", "value_type"):
                    if not str(assay.get(field_name, "")).strip():
                        error(
                            "missing_assay_field",
                            f"Assay field '{field_name}' is required.",
                            f"{path_prefix}/uns/assay/{field_name}",
                        )
                technology = str(_normalize(assay.get("technology", "")))
                value_type = str(_normalize(assay.get("value_type", "")))
                yaml_assay = metadata.modalities.get(name)
                if yaml_assay is not None and (
                    technology != yaml_assay.technology or value_type != yaml_assay.value_type
                ):
                    error(
                        "assay_metadata_mismatch",
                        f"Assay metadata for '{name}' differs from metadata YAML.",
                        f"{path_prefix}/uns/assay",
                    )
                if value_type not in RECOMMENDED_VALUE_TYPES:
                    warn(
                        "nonstandard_value_type",
                        "Value type is outside the recommended vocabulary.",
                        f"{path_prefix}/uns/assay/value_type",
                    )
                elif value_type == "unknown":
                    warn(
                        "unknown_value_type",
                        "A specific value type is preferred when it can be determined.",
                        f"{path_prefix}/uns/assay/value_type",
                    )
            outcome.modalities[name] = ModalitySummary(
                name=name,
                technology=technology,
                value_type=value_type,
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
            )

        if covered is not None and not covered.all():
            error(
                "unrepresented_global_obs",
                "Top-level observations must equal the union of modality observations.",
                "/obs/index",
            )

        mask_values = list(masks.values())
        if pairing_type == "same_unit" and mask_values:
            if not all(mask.all() for mask in mask_values):
                error(
                    "same_unit_mismatch",
                    "same_unit requires every modality to contain all top-level observations.",
                    "/uns/database/pairing_type",
                )
        elif pairing_type == "partially_shared" and len(mask_values) >= 2:
            has_overlap = any(
                np.any(left & right)
                for index, left in enumerate(mask_values)
                for right in mask_values[index + 1 :]
            )
            all_identical = all(np.array_equal(mask_values[0], mask) for mask in mask_values[1:])
            if not has_overlap or all_identical:
                error(
                    "partially_shared_mismatch",
                    "partially_shared requires overlapping but non-identical "
                    "modality observations.",
                    "/uns/database/pairing_type",
                )
        elif pairing_type == "unpaired" and len(mask_values) >= 2:
            membership_count = np.sum(np.stack(mask_values), axis=0)
            if np.any(membership_count > 1):
                error(
                    "unpaired_mismatch",
                    "unpaired modalities must not share observation IDs.",
                    "/uns/database/pairing_type",
                )
        elif pairing_type not in RECOMMENDED_PAIRING_TYPES:
            warn(
                "nonstandard_pairing_type",
                "Pairing type is outside the recommended vocabulary and was not checked.",
                "/uns/database/pairing_type",
            )
    finally:
        mdata.file.close()

    return outcome
