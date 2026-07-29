from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from .validation import validate_h5mu, validate_mudata, validate_train_test_pair

SCHEMA_VERSION = "1.1"
CONFIG_POLICIES = {"preserve", "intersection", "union", "reference"}
CHALLENGE_TYPES = {"same_slice", "cross_slice_same_subject", "cross_subject"}
DATASET_TYPES = {"train", "test"}
FEATURE_MASK_KEY = "feature_measured_by_source"

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
REQUIRED_DERIVATION_FIELDS = {
    "construction_type",
    "source_dataset_ids",
    "split_id",
    "challenge_type",
    "selection_description",
    "feature_merge_policy",
    "processing_description",
}


class SplitterError(ValueError):
    """Raised when a split configuration or MuData file is invalid."""


@dataclass(frozen=True)
class Region:
    sample_id: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class SpatialConfig:
    path: Path
    split_id: str
    challenge_type: str
    source: Path
    output_dir: Path
    train_id: str
    test_id: str
    regions: tuple[Region, ...]


@dataclass(frozen=True)
class ComposeSide:
    dataset_id: str
    sources: tuple[Path, ...]
    reference_dataset_id: str | None


@dataclass(frozen=True)
class ComposeConfig:
    path: Path
    split_id: str
    challenge_type: str
    feature_merge_policy: str
    output_dir: Path
    train: ComposeSide
    test: ComposeSide


@dataclass
class SourceDataset:
    path: Path
    mdata: md.MuData
    database: dict[str, Any]
    dataset_id: str

    @property
    def obs_names(self) -> list[str]:
        return [str(value) for value in self.mdata.obs_names]

    def close(self) -> None:
        self.mdata.file.close()


@dataclass(frozen=True)
class ProductValidation:
    dataset_type: str
    split_id: str
    challenge_type: str
    source_pairs: frozenset[tuple[str, str]]
    modalities: dict[str, tuple[str, ...]]


def _normalise(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_normalise(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SplitterError(f"{context} must be a mapping")
    return value


def _check_keys(
    value: Mapping[str, Any], required: set[str], allowed: set[str], context: str
) -> None:
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(allowed))
    if missing:
        raise SplitterError(f"{context} is missing required field(s): {', '.join(missing)}")
    if unknown:
        raise SplitterError(f"{context} contains unknown field(s): {', '.join(unknown)}")


def _required_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SplitterError(f"{context} must be a non-empty string")
    return value.strip()


def _dataset_id(value: Any, context: str) -> str:
    result = _required_string(value, context)
    if Path(result).name != result or result in {".", ".."}:
        raise SplitterError(f"{context} must be a safe file-name component")
    return result


def _config_path(value: Any, base: Path, context: str) -> Path:
    text = _required_string(value, context)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_yaml(path: Path | str) -> tuple[Path, Mapping[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except OSError as exc:
        raise SplitterError(f"unable to read config '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise SplitterError(f"invalid YAML in '{config_path}': {exc}") from exc
    return config_path, _as_mapping(value, "config")


def _check_config_version(value: Any) -> None:
    if value != SCHEMA_VERSION:
        raise SplitterError(f"config schema_version must be '{SCHEMA_VERSION}'")


def _challenge_type(value: Any, context: str) -> str:
    result = _required_string(value, context)
    if result not in CHALLENGE_TYPES:
        raise SplitterError(
            f"{context} must be one of: " + ", ".join(sorted(CHALLENGE_TYPES))
        )
    return result


def load_spatial_config(path: Path | str) -> SpatialConfig:
    config_path, values = _load_yaml(path)
    required = {
        "schema_version",
        "split_id",
        "challenge_type",
        "feature_merge_policy",
        "source",
        "output_dir",
        "train",
        "test",
    }
    _check_keys(values, required, required, "spatial config")
    _check_config_version(values["schema_version"])
    if values["feature_merge_policy"] != "preserve":
        raise SplitterError("spatial feature_merge_policy must be 'preserve'")

    train = _as_mapping(values["train"], "spatial config train")
    test = _as_mapping(values["test"], "spatial config test")
    _check_keys(train, {"dataset_id"}, {"dataset_id"}, "spatial config train")
    _check_keys(test, {"dataset_id", "regions"}, {"dataset_id", "regions"}, "spatial config test")
    raw_regions = test["regions"]
    if not isinstance(raw_regions, list) or not raw_regions:
        raise SplitterError("spatial config test.regions must be a non-empty list")

    regions: list[Region] = []
    region_fields = {"sample_id", "x_min", "x_max", "y_min", "y_max"}
    for index, raw_region in enumerate(raw_regions):
        context = f"spatial config test.regions[{index}]"
        region = _as_mapping(raw_region, context)
        _check_keys(region, region_fields, region_fields, context)
        coordinates: dict[str, float] = {}
        for name in ("x_min", "x_max", "y_min", "y_max"):
            raw_value = region[name]
            if isinstance(raw_value, bool):
                raise SplitterError(f"{context}.{name} must be a finite number")
            try:
                coordinates[name] = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise SplitterError(f"{context}.{name} must be a finite number") from exc
            if not np.isfinite(coordinates[name]):
                raise SplitterError(f"{context}.{name} must be a finite number")
        if coordinates["x_min"] > coordinates["x_max"]:
            raise SplitterError(f"{context}.x_min must not exceed x_max")
        if coordinates["y_min"] > coordinates["y_max"]:
            raise SplitterError(f"{context}.y_min must not exceed y_max")
        regions.append(
            Region(
                sample_id=_required_string(region["sample_id"], f"{context}.sample_id"),
                **coordinates,
            )
        )

    base = config_path.parent
    train_id = _dataset_id(train["dataset_id"], "spatial config train.dataset_id")
    test_id = _dataset_id(test["dataset_id"], "spatial config test.dataset_id")
    if train_id == test_id:
        raise SplitterError("train and test dataset_id values must differ")
    return SpatialConfig(
        path=config_path,
        split_id=_required_string(values["split_id"], "spatial config split_id"),
        challenge_type=_challenge_type(
            values["challenge_type"], "spatial config challenge_type"
        ),
        source=_config_path(values["source"], base, "spatial config source"),
        output_dir=_config_path(values["output_dir"], base, "spatial config output_dir"),
        train_id=train_id,
        test_id=test_id,
        regions=tuple(regions),
    )


def _load_compose_side(value: Any, base: Path, name: str) -> ComposeSide:
    side = _as_mapping(value, f"compose config {name}")
    fields = {"dataset_id", "sources", "reference_dataset_id"}
    _check_keys(side, fields, fields, f"compose config {name}")
    raw_sources = side["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise SplitterError(f"compose config {name}.sources must be a non-empty list")
    sources = tuple(
        _config_path(source, base, f"compose config {name}.sources[{index}]")
        for index, source in enumerate(raw_sources)
    )
    if len(set(sources)) != len(sources):
        raise SplitterError(f"compose config {name}.sources contains a duplicate source")
    reference = side["reference_dataset_id"]
    if reference is not None:
        reference = _required_string(reference, f"compose config {name}.reference_dataset_id")
    return ComposeSide(
        dataset_id=_dataset_id(side["dataset_id"], f"compose config {name}.dataset_id"),
        sources=sources,
        reference_dataset_id=reference,
    )


def load_compose_config(path: Path | str) -> ComposeConfig:
    config_path, values = _load_yaml(path)
    fields = {
        "schema_version",
        "split_id",
        "challenge_type",
        "feature_merge_policy",
        "output_dir",
        "train",
        "test",
    }
    _check_keys(values, fields, fields, "compose config")
    _check_config_version(values["schema_version"])
    policy = values["feature_merge_policy"]
    if policy not in CONFIG_POLICIES:
        raise SplitterError(
            "compose feature_merge_policy must be one of: " + ", ".join(sorted(CONFIG_POLICIES))
        )
    base = config_path.parent
    train = _load_compose_side(values["train"], base, "train")
    test = _load_compose_side(values["test"], base, "test")
    if train.dataset_id == test.dataset_id:
        raise SplitterError("train and test dataset_id values must differ")
    overlap = set(train.sources).intersection(test.sources)
    if overlap:
        duplicate = next(iter(overlap))
        raise SplitterError(f"a source cannot be assigned to both train and test: {duplicate}")
    if policy == "reference":
        if train.reference_dataset_id is None or test.reference_dataset_id is None:
            raise SplitterError("reference policy requires a reference_dataset_id on both sides")
    elif train.reference_dataset_id is not None or test.reference_dataset_id is not None:
        raise SplitterError("reference_dataset_id must be null unless policy is 'reference'")
    return ComposeConfig(
        path=config_path,
        split_id=_required_string(values["split_id"], "compose config split_id"),
        challenge_type=_challenge_type(
            values["challenge_type"], "compose config challenge_type"
        ),
        feature_merge_policy=policy,
        output_dir=_config_path(values["output_dir"], base, "compose config output_dir"),
        train=train,
        test=test,
    )


def _blank_ids(values: Sequence[Any]) -> bool:
    return any(not str(value).strip() for value in values)


def _computed_pairing_type(modalities: Mapping[str, ad.AnnData]) -> str:
    observation_sets = [set(map(str, adata.obs_names)) for adata in modalities.values()]
    if all(values == observation_sets[0] for values in observation_sets[1:]):
        return "same_unit"
    has_overlap = any(
        bool(left.intersection(right))
        for index, left in enumerate(observation_sets)
        for right in observation_sets[index + 1 :]
    )
    return "partially_shared" if has_overlap else "unpaired"


def _validate_common_structure(mdata: md.MuData, context: str) -> dict[str, Any]:
    if mdata.n_obs <= 0:
        raise SplitterError(f"{context}: top-level observations must not be empty")
    if len(mdata.mod) < 2:
        raise SplitterError(f"{context}: at least two modalities are required")
    if not mdata.obs_names.is_unique or _blank_ids(mdata.obs_names):
        raise SplitterError(f"{context}: top-level observation IDs must be unique and non-empty")
    if "sample_id" not in mdata.obs:
        raise SplitterError(f"{context}: obs['sample_id'] is required")
    samples = mdata.obs["sample_id"]
    if not samples.notna().all() or _blank_ids(samples.tolist()):
        raise SplitterError(f"{context}: sample IDs must be non-null and non-empty")
    if "spatial" not in mdata.obsm:
        raise SplitterError(f"{context}: obsm['spatial'] is required")
    coordinates = np.asarray(mdata.obsm["spatial"])
    if coordinates.ndim != 2 or coordinates.shape != (mdata.n_obs, coordinates.shape[1]):
        raise SplitterError(f"{context}: spatial coordinates must have one row per observation")
    if coordinates.shape[1] not in (2, 3):
        raise SplitterError(f"{context}: spatial coordinates must have two or three columns")
    if not np.issubdtype(coordinates.dtype, np.number) or not np.isfinite(coordinates).all():
        raise SplitterError(f"{context}: spatial coordinates must be finite numeric values")

    database = _as_mapping(mdata.uns.get("database"), f"{context}: uns['database']")
    missing_database = sorted(REQUIRED_DATABASE_FIELDS.difference(database))
    if missing_database:
        raise SplitterError(
            f"{context}: database metadata is missing: {', '.join(missing_database)}"
        )
    if database["schema_version"] != SCHEMA_VERSION:
        raise SplitterError(f"{context}: database schema_version must be '{SCHEMA_VERSION}'")
    for field_name in ("dataset_id", "dataset_type", "spatial_unit", "coordinate_unit"):
        _required_string(database[field_name], f"{context}: database.{field_name}")
    for field_name in ("source", "organism", "tissue"):
        _metadata_values(database[field_name], f"{context}: database.{field_name}")

    global_names = set(map(str, mdata.obs_names))
    covered: set[str] = set()
    for modality, adata in mdata.mod.items():
        modality_context = f"{context}: modality '{modality}'"
        if adata.X is None or getattr(adata.X, "shape", None) != adata.shape:
            raise SplitterError(f"{modality_context}: X must match the AnnData shape")
        if adata.n_obs <= 0 or adata.n_vars <= 0:
            raise SplitterError(f"{modality_context}: observations and features must be non-empty")
        if not adata.obs_names.is_unique or _blank_ids(adata.obs_names):
            raise SplitterError(f"{modality_context}: observation IDs must be unique and non-empty")
        if not adata.var_names.is_unique or _blank_ids(adata.var_names):
            raise SplitterError(f"{modality_context}: feature IDs must be unique and non-empty")
        modality_names = set(map(str, adata.obs_names))
        if not modality_names.issubset(global_names):
            raise SplitterError(f"{modality_context}: observation IDs must belong to the top level")
        covered.update(modality_names)
        assay = _as_mapping(adata.uns.get("assay"), f"{modality_context}: uns['assay']")
        _metadata_values(assay.get("technology"), f"{modality_context}: assay.technology")
        _required_string(assay.get("value_type"), f"{modality_context}: assay.value_type")
    if covered != global_names:
        raise SplitterError(f"{context}: top-level observations must equal the modality union")
    computed_pairing = _computed_pairing_type(mdata.mod)
    if database["pairing_type"] != computed_pairing:
        raise SplitterError(
            f"{context}: pairing_type must be '{computed_pairing}' for the modality memberships"
        )
    shared_outcome = validate_mudata(mdata)
    if not shared_outcome.valid:
        issue = shared_outcome.errors[0]
        raise SplitterError(f"{context}: schema 1.1 validation failed: {issue.message}")
    return dict(_normalise(database))


def _read_source(path: Path, *, backed: bool = False) -> SourceDataset:
    if not path.is_file():
        raise SplitterError(f"source file does not exist: {path}")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
            mdata = md.read_h5mu(path, backed="r" if backed else None)
    except Exception as exc:
        raise SplitterError(f"unable to read source '{path}': {exc}") from exc
    try:
        database = _validate_common_structure(mdata, str(path))
        if database["dataset_type"] != "full":
            raise SplitterError(f"{path}: source dataset_type must be 'full'")
        if database.get("derivation"):
            raise SplitterError(f"{path}: a full dataset must not have derivation metadata")
        dataset_id = _dataset_id(database["dataset_id"], f"{path}: database.dataset_id")
    except Exception:
        mdata.file.close()
        raise
    return SourceDataset(path=path, mdata=mdata, database=database, dataset_id=dataset_id)


def _metadata_values(value: Any, context: str) -> list[str]:
    normalised = _normalise(value)
    raw_values = normalised if isinstance(normalised, list) else [normalised]
    if not raw_values:
        raise SplitterError(f"{context} must not be empty")
    result: list[str] = []
    for item in raw_values:
        if not isinstance(item, str) or not item.strip():
            raise SplitterError(f"{context} must contain non-empty strings")
        result.append(item.strip())
    return result


def _deduplicated_metadata(sources: Sequence[SourceDataset], field: str) -> str | list[str]:
    values: list[str] = []
    for source in sources:
        for value in _metadata_values(source.database[field], f"{source.path}: database.{field}"):
            if value not in values:
                values.append(value)
    return values[0] if len(values) == 1 else values


def _ensure_source_compatibility(sources: Sequence[SourceDataset]) -> None:
    if not sources:
        raise SplitterError("at least one source is required")
    dataset_ids = [source.dataset_id for source in sources]
    if len(set(dataset_ids)) != len(dataset_ids):
        raise SplitterError("source dataset_id values must be unique")
    for field in ("spatial_unit", "coordinate_unit"):
        values = {source.database[field] for source in sources}
        if len(values) != 1:
            raise SplitterError(f"all sources must have the same {field}")
    modalities = {name for source in sources for name in source.mdata.mod}
    for modality in modalities:
        value_types = {
            _normalise(source.mdata.mod[modality].uns["assay"]["value_type"])
            for source in sources
            if modality in source.mdata.mod
        }
        if len(value_types) != 1:
            raise SplitterError(f"all sources for modality '{modality}' must share value_type")


def coordinate_ranges(path: Path | str, sample_id: str | None = None) -> dict[str, Any]:
    source = _read_source(Path(path).expanduser().resolve(), backed=True)
    try:
        coordinates = np.asarray(source.mdata.obsm["spatial"])
        samples = source.mdata.obs["sample_id"].astype(str).to_numpy()
        available_samples = list(dict.fromkeys(samples.tolist()))
        if sample_id is not None and sample_id not in available_samples:
            raise SplitterError(f"sample_id '{sample_id}' does not exist in {source.path}")

        def summary(mask: np.ndarray) -> dict[str, int | float]:
            selected = coordinates[mask]
            return {
                "n_obs": int(selected.shape[0]),
                "x_min": float(selected[:, 0].min()),
                "x_max": float(selected[:, 0].max()),
                "y_min": float(selected[:, 1].min()),
                "y_max": float(selected[:, 1].max()),
            }

        selected_samples = [sample_id] if sample_id is not None else available_samples
        return {
            "dataset_id": source.dataset_id,
            "coordinate_unit": source.database["coordinate_unit"],
            "coordinate_dimensions": int(coordinates.shape[1]),
            "global": summary(np.ones(source.mdata.n_obs, dtype=bool)),
            "samples": {
                sample: summary(samples == sample)
                for sample in selected_samples
                if sample is not None
            },
        }
    finally:
        source.close()


coordinate_range = coordinate_ranges


def _copy_matrix(value: Any) -> Any:
    if sparse.issparse(value):
        return value.copy()
    return np.asarray(value).copy()


def _align_matrix(adata: ad.AnnData, target_features: Sequence[str]) -> Any:
    lookup = {str(feature): index for index, feature in enumerate(adata.var_names)}
    positions = [lookup.get(feature) for feature in target_features]
    if all(position is not None for position in positions):
        return _copy_matrix(adata.X[:, [int(position) for position in positions]])

    if sparse.issparse(adata.X):
        columns = [
            adata.X[:, int(position)]
            if position is not None
            else sparse.csr_matrix((adata.n_obs, 1), dtype=adata.X.dtype)
            for position in positions
        ]
        return sparse.hstack(columns, format="csr")
    matrix = np.zeros((adata.n_obs, len(target_features)), dtype=np.asarray(adata.X).dtype)
    target_positions = [index for index, position in enumerate(positions) if position is not None]
    source_positions = [int(position) for position in positions if position is not None]
    if target_positions:
        matrix[:, target_positions] = np.asarray(adata.X)[:, source_positions]
    return matrix


def _stack_matrices(matrices: Sequence[Any]) -> Any:
    if any(sparse.issparse(matrix) for matrix in matrices):
        return sparse.vstack(
            [
                matrix if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
                for matrix in matrices
            ],
            format="csr",
        )
    return np.vstack(matrices)


def _make_anndata(
    matrices: Sequence[Any],
    obs_names: Sequence[str],
    features: Sequence[str],
    technology: str | list[str],
    value_type: str,
) -> ad.AnnData:
    result = ad.AnnData(
        X=_stack_matrices(matrices),
        obs=pd.DataFrame(index=pd.Index(obs_names, dtype=str)),
        var=pd.DataFrame(index=pd.Index(features, dtype=str)),
    )
    result.uns["assay"] = {"technology": technology, "value_type": value_type}
    return result


def _ordered_technologies(sources: Sequence[SourceDataset], modality: str) -> str | list[str]:
    values: list[str] = []
    for source in sources:
        if modality not in source.mdata.mod:
            continue
        technology = source.mdata.mod[modality].uns["assay"]["technology"]
        for value in _metadata_values(technology, f"{source.path}: {modality} assay.technology"):
            if value not in values:
                values.append(value)
    return values[0] if len(values) == 1 else values


def _minimal_mudata(
    modalities: Mapping[str, ad.AnnData],
    top_names: Sequence[str],
    sample_ids: Sequence[str],
    source_dataset_ids: Sequence[str],
    source_obs_ids: Sequence[str],
    coordinates: np.ndarray,
    database: Mapping[str, Any],
) -> md.MuData:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
        warnings.filterwarnings("ignore", message="Cannot join columns with the same name.*")
        result = md.MuData(dict(modalities))
        result = result[list(top_names), :].copy()
    actual_names = list(map(str, result.obs_names))
    positions = pd.Index(top_names).get_indexer(actual_names)
    if np.any(positions < 0) or len(actual_names) != len(top_names):
        raise SplitterError("internal error: output observation membership changed")
    result.obs["sample_id"] = np.asarray(sample_ids, dtype=object)[positions].tolist()
    result.obs["source_dataset_id"] = np.asarray(source_dataset_ids, dtype=object)[
        positions
    ].tolist()
    result.obs["source_obs_id"] = np.asarray(source_obs_ids, dtype=object)[positions].tolist()
    result.obsm["spatial"] = np.asarray(coordinates)[positions].copy()
    result.uns["database"] = deepcopy(dict(database))
    return result


def _derivation_database(
    *,
    dataset_id: str,
    dataset_type: str,
    sources: Sequence[SourceDataset],
    split_id: str,
    challenge_type: str,
    construction_type: str,
    feature_policy: str,
    selection_description: str,
    processing_description: str,
    pairing_type: str,
    reference_dataset_id: str | None = None,
) -> dict[str, Any]:
    derivation: dict[str, Any] = {
        "construction_type": construction_type,
        "source_dataset_ids": [source.dataset_id for source in sources],
        "split_id": split_id,
        "challenge_type": challenge_type,
        "selection_description": selection_description,
        "feature_merge_policy": feature_policy,
        "processing_description": processing_description,
        "random_seed": None,
    }
    if feature_policy == "reference":
        derivation["reference_dataset_id"] = reference_dataset_id
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_type": dataset_type,
        "source": _deduplicated_metadata(sources, "source"),
        "organism": _deduplicated_metadata(sources, "organism"),
        "tissue": _deduplicated_metadata(sources, "tissue"),
        "spatial_unit": sources[0].database["spatial_unit"],
        "coordinate_unit": sources[0].database["coordinate_unit"],
        "pairing_type": pairing_type,
        "derivation": derivation,
    }


def _region_description(regions: Sequence[Region]) -> str:
    descriptions = [
        f"sample_id={region.sample_id!r}, x=[{region.x_min:g}, {region.x_max:g}], "
        f"y=[{region.y_min:g}, {region.y_max:g}]"
        for region in regions
    ]
    return "union of closed rectangles: " + "; ".join(descriptions)


def _build_spatial_product(
    source: SourceDataset,
    selected: np.ndarray,
    *,
    dataset_id: str,
    dataset_type: str,
    split_id: str,
    challenge_type: str,
    selection_description: str,
) -> md.MuData:
    global_names = np.asarray(source.obs_names, dtype=object)
    selected_names = global_names[selected].tolist()
    selected_set = set(selected_names)
    modalities: dict[str, ad.AnnData] = {}
    for modality, source_adata in source.mdata.mod.items():
        modality_names = [str(name) for name in source_adata.obs_names if str(name) in selected_set]
        if not modality_names:
            raise SplitterError(
                f"spatial {dataset_type} split has no observations for modality '{modality}'"
            )
        matrix = _copy_matrix(source_adata[modality_names, :].X)
        modalities[modality] = _make_anndata(
            [matrix],
            modality_names,
            list(map(str, source_adata.var_names)),
            _normalise(source_adata.uns["assay"]["technology"]),
            str(source_adata.uns["assay"]["value_type"]),
        )
    pairing_type = _computed_pairing_type(modalities)
    source_positions = pd.Index(source.obs_names).get_indexer(selected_names)
    database = _derivation_database(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        sources=[source],
        split_id=split_id,
        challenge_type=challenge_type,
        construction_type="subset",
        feature_policy="preserve",
        selection_description=selection_description,
        processing_description=(
            "No matrix values or spatial coordinates were transformed; source feature order "
            "and modality membership were preserved."
        ),
        pairing_type=pairing_type,
    )
    return _minimal_mudata(
        modalities,
        selected_names,
        source.mdata.obs.iloc[source_positions]["sample_id"].astype(str).tolist(),
        [source.dataset_id] * len(selected_names),
        selected_names,
        np.asarray(source.mdata.obsm["spatial"])[source_positions],
        database,
    )


def _side_modalities(sources: Sequence[SourceDataset]) -> set[str]:
    return {name for source in sources for name in source.mdata.mod}


def _determine_target_features(
    sources: Sequence[SourceDataset],
    modalities: set[str],
    policy: str,
    reference: SourceDataset | None,
) -> dict[str, list[str]]:
    if policy == "reference":
        if reference is None:
            raise SplitterError("a reference source is required for reference policy")
        if set(reference.mdata.mod) != modalities:
            raise SplitterError("reference datasets must contain exactly the final modality set")
        return {
            modality: list(map(str, reference.mdata.mod[modality].var_names))
            for modality in sorted(modalities)
        }

    targets = {}
    for modality in sorted(modalities):
        feature_lists = [
            list(map(str, source.mdata.mod[modality].var_names))
            for source in sources
            if modality in source.mdata.mod
        ]
        if policy == "preserve":
            if any(features != feature_lists[0] for features in feature_lists[1:]):
                raise SplitterError(
                    f"preserve requires identical feature order for modality '{modality}'"
                )
            targets[modality] = feature_lists[0]
        elif policy == "intersection":
            common = set(feature_lists[0]).intersection(
                *(set(values) for values in feature_lists[1:])
            )
            target = [feature for feature in feature_lists[0] if feature in common]
            if not target:
                raise SplitterError(f"feature intersection is empty for modality '{modality}'")
            targets[modality] = target
        else:
            target = list(dict.fromkeys(feature for values in feature_lists for feature in values))
            targets[modality] = target
    return targets


def _feature_mask(
    sources: Sequence[SourceDataset], modality: str, features: Sequence[str]
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for source in sources:
        measured = (
            set(map(str, source.mdata.mod[modality].var_names))
            if modality in source.mdata.mod
            else set()
        )
        columns.append(np.asarray([feature in measured for feature in features], dtype=bool))
    return np.column_stack(columns)


def _processing_description(
    policy: str, *, has_missing_features: bool, comparison_note: str
) -> str:
    if policy == "preserve":
        description = (
            "Identical source feature spaces were preserved; X values and coordinates are "
            "unchanged."
        )
    elif policy == "intersection":
        description = (
            "Each modality was aligned to the common feature intersection; retained X values and "
            "coordinates are unchanged."
        )
    elif policy == "union" or has_missing_features:
        description = (
            f"Each modality was aligned using the {policy} feature policy; zero is a storage "
            f"placeholder for unmeasured features identified by varm['{FEATURE_MASK_KEY}']; "
            "measured X values and coordinates are unchanged."
        )
    else:
        description = (
            "Each modality was aligned to the reference feature order; every source measured all "
            "target features, so no missing-feature mask was needed; X values and coordinates are "
            "unchanged."
        )
    return f"{description} {comparison_note}".strip()


def _build_composite_product(
    sources: Sequence[SourceDataset],
    *,
    dataset_id: str,
    dataset_type: str,
    split_id: str,
    challenge_type: str,
    policy: str,
    target_features: Mapping[str, Sequence[str]],
    reference_dataset_id: str | None,
    comparison_note: str = "",
) -> md.MuData:
    modalities: dict[str, ad.AnnData] = {}
    has_missing_features = False
    for modality, features in target_features.items():
        contributing = [source for source in sources if modality in source.mdata.mod]
        matrices: list[Any] = []
        modality_obs_names: list[str] = []
        for source in contributing:
            source_adata = source.mdata.mod[modality]
            matrices.append(_align_matrix(source_adata, features))
            modality_obs_names.extend(
                f"{source.dataset_id}::{obs_name}" for obs_name in map(str, source_adata.obs_names)
            )
        value_type = str(contributing[0].mdata.mod[modality].uns["assay"]["value_type"])
        result = _make_anndata(
            matrices,
            modality_obs_names,
            features,
            _ordered_technologies(sources, modality),
            value_type,
        )
        mask = _feature_mask(sources, modality, features)
        has_missing_features = has_missing_features or not mask.all()
        if policy == "union" or (policy == "reference" and not mask.all()):
            result.varm[FEATURE_MASK_KEY] = mask
            result.uns["feature_measurement"] = {
                "mask_key": FEATURE_MASK_KEY,
                "source_dataset_ids": [source.dataset_id for source in sources],
                "placeholder_value": 0,
                "description": (
                    "False means the feature was not measured by that source dataset; the stored "
                    "zero is not a true measured zero."
                ),
            }
        modalities[modality] = result

    top_names: list[str] = []
    sample_ids: list[str] = []
    source_dataset_ids: list[str] = []
    source_obs_ids: list[str] = []
    coordinates: list[np.ndarray] = []
    for source in sources:
        source_names = source.obs_names
        top_names.extend(f"{source.dataset_id}::{name}" for name in source_names)
        sample_ids.extend(
            f"{source.dataset_id}::{sample}"
            for sample in source.mdata.obs["sample_id"].astype(str).tolist()
        )
        source_dataset_ids.extend([source.dataset_id] * len(source_names))
        source_obs_ids.extend(source_names)
        coordinates.append(np.asarray(source.mdata.obsm["spatial"]))

    pairing_type = _computed_pairing_type(modalities)
    construction_type = "subset" if len(sources) == 1 else "composite"
    database = _derivation_database(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        sources=sources,
        split_id=split_id,
        challenge_type=challenge_type,
        construction_type=construction_type,
        feature_policy=policy,
        selection_description=(
            "All observations from every assigned full source dataset were retained."
        ),
        processing_description=_processing_description(
            policy,
            has_missing_features=has_missing_features,
            comparison_note=comparison_note,
        ),
        pairing_type=pairing_type,
        reference_dataset_id=reference_dataset_id,
    )
    return _minimal_mudata(
        modalities,
        top_names,
        sample_ids,
        source_dataset_ids,
        source_obs_ids,
        np.vstack(coordinates),
        database,
    )


def _source_pairs(mdata: md.MuData) -> frozenset[tuple[str, str]]:
    return frozenset(
        (str(dataset_id), str(obs_id))
        for dataset_id, obs_id in zip(
            mdata.obs["source_dataset_id"], mdata.obs["source_obs_id"], strict=True
        )
    )


def _validate_product(path: Path, sources: Mapping[str, SourceDataset]) -> ProductValidation:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
            mdata = md.read_h5mu(path)
    except Exception as exc:
        raise SplitterError(f"unable to reread output '{path}': {exc}") from exc
    try:
        database = _validate_common_structure(mdata, str(path))
        dataset_type = database["dataset_type"]
        if dataset_type not in DATASET_TYPES:
            raise SplitterError(f"{path}: output dataset_type must be train or test")
        for column in ("source_dataset_id", "source_obs_id"):
            if column not in mdata.obs or not mdata.obs[column].notna().all():
                raise SplitterError(f"{path}: obs['{column}'] must be present and non-null")
        pairs = _source_pairs(mdata)
        if len(pairs) != mdata.n_obs:
            raise SplitterError(f"{path}: source observation pairs must be unique")
        derivation = _as_mapping(database.get("derivation"), f"{path}: database.derivation")
        missing = sorted(REQUIRED_DERIVATION_FIELDS.difference(derivation))
        if missing:
            raise SplitterError(f"{path}: derivation is missing: {', '.join(missing)}")
        source_ids = [str(value) for value in _normalise(derivation["source_dataset_ids"])]
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise SplitterError(
                f"{path}: derivation source_dataset_ids must be non-empty and unique"
            )
        expected_construction = "subset" if len(source_ids) == 1 else "composite"
        if derivation["construction_type"] != expected_construction:
            raise SplitterError(
                f"{path}: construction_type must be '{expected_construction}' for its sources"
            )
        policy = derivation["feature_merge_policy"]
        if policy not in CONFIG_POLICIES:
            raise SplitterError(f"{path}: invalid feature_merge_policy")
        if policy == "reference" and derivation.get("reference_dataset_id") not in source_ids:
            raise SplitterError(f"{path}: reference_dataset_id must belong to this output side")
        source_observations = {
            source_id: set(source.obs_names) for source_id, source in sources.items()
        }
        for source_id, source_obs_id in pairs:
            if (
                source_id not in source_observations
                or source_obs_id not in source_observations[source_id]
            ):
                raise SplitterError(
                    f"{path}: source pair ({source_id!r}, {source_obs_id!r}) does not exist"
                )

        top_provenance = {
            str(obs_name): (str(source_id), str(source_obs_id))
            for obs_name, source_id, source_obs_id in zip(
                mdata.obs_names,
                mdata.obs["source_dataset_id"],
                mdata.obs["source_obs_id"],
                strict=True,
            )
        }
        for modality, adata in mdata.mod.items():
            observed_pairs = {top_provenance[str(name)] for name in adata.obs_names}
            expected_pairs = {
                (source_id, str(obs_name))
                for source_id in source_ids
                if modality in sources[source_id].mdata.mod
                for obs_name in sources[source_id].mdata.mod[modality].obs_names
                if (source_id, str(obs_name)) in pairs
            }
            if observed_pairs != expected_pairs:
                raise SplitterError(
                    f"{path}: modality '{modality}' source membership is incomplete"
                )
        return ProductValidation(
            dataset_type=dataset_type,
            split_id=str(derivation["split_id"]),
            challenge_type=str(derivation["challenge_type"]),
            source_pairs=pairs,
            modalities={
                name: tuple(map(str, adata.var_names)) for name, adata in mdata.mod.items()
            },
        )
    finally:
        mdata.file.close()


def _write_products(
    output_dir: Path,
    train_id: str,
    test_id: str,
    train: md.MuData,
    test: md.MuData,
    sources: Sequence[SourceDataset],
) -> tuple[Path, Path]:
    if output_dir.exists():
        raise SplitterError(f"output directory already exists: {output_dir}")
    source_by_id = {source.dataset_id: source for source in sources}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        train_path = temporary / f"{train_id}.h5mu"
        test_path = temporary / f"{test_id}.h5mu"
        train.write_h5mu(train_path)
        test.write_h5mu(test_path)
        train_validation = _validate_product(train_path, source_by_id)
        test_validation = _validate_product(test_path, source_by_id)
        source_paths = {source.dataset_id: source.path for source in sources}
        for product_path in (train_path, test_path):
            shared_outcome = validate_h5mu(product_path, source_paths=source_paths)
            if not shared_outcome.valid:
                issue = shared_outcome.errors[0]
                raise SplitterError(f"generated file failed schema 1.1 validation: {issue.message}")
        if train_validation.dataset_type != "train" or test_validation.dataset_type != "test":
            raise SplitterError("generated files have incorrect dataset_type values")
        if train_validation.split_id != test_validation.split_id:
            raise SplitterError("generated train and test files have different split_id values")
        if train_validation.challenge_type != test_validation.challenge_type:
            raise SplitterError(
                "generated train and test files have different challenge_type values"
            )
        if set(train_validation.modalities) != set(test_validation.modalities):
            raise SplitterError("generated train and test modality sets are not comparable")
        if not train_validation.source_pairs.isdisjoint(test_validation.source_pairs):
            raise SplitterError("generated train and test source observation pairs overlap")
        shared_pair_outcome = validate_train_test_pair(train_path, test_path)
        if not shared_pair_outcome.valid:
            raise SplitterError(shared_pair_outcome.errors[0].message)
        expected_pairs = {
            (source.dataset_id, obs_name) for source in sources for obs_name in source.obs_names
        }
        if train_validation.source_pairs | test_validation.source_pairs != expected_pairs:
            raise SplitterError(
                "generated train and test files do not completely cover all sources"
            )
        if output_dir.exists():
            raise SplitterError(f"output directory already exists: {output_dir}")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / f"{train_id}.h5mu", output_dir / f"{test_id}.h5mu"


def spatial_split(config_path: Path | str) -> tuple[Path, Path]:
    config = load_spatial_config(config_path)
    if config.output_dir.exists():
        raise SplitterError(f"output directory already exists: {config.output_dir}")
    source = _read_source(config.source)
    try:
        if config.train_id == source.dataset_id or config.test_id == source.dataset_id:
            raise SplitterError("output dataset_id values must differ from source dataset_id")
        samples = source.mdata.obs["sample_id"].astype(str).to_numpy()
        coordinates = np.asarray(source.mdata.obsm["spatial"])
        available_samples = set(samples.tolist())
        region_samples = {region.sample_id for region in config.regions}
        unknown_samples = sorted(region_samples - available_samples)
        if unknown_samples:
            raise SplitterError(
                "test regions reference unknown sample_id value(s): " + ", ".join(unknown_samples)
            )
        test_mask = np.zeros(source.mdata.n_obs, dtype=bool)
        for region in config.regions:
            test_mask |= (
                (samples == region.sample_id)
                & (coordinates[:, 0] >= region.x_min)
                & (coordinates[:, 0] <= region.x_max)
                & (coordinates[:, 1] >= region.y_min)
                & (coordinates[:, 1] <= region.y_max)
            )
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            raise SplitterError("spatial train and test partitions must both be non-empty")
        region_description = _region_description(config.regions)
        train = _build_spatial_product(
            source,
            train_mask,
            dataset_id=config.train_id,
            dataset_type="train",
            split_id=config.split_id,
            challenge_type=config.challenge_type,
            selection_description=f"Complement of the {region_description}.",
        )
        test = _build_spatial_product(
            source,
            test_mask,
            dataset_id=config.test_id,
            dataset_type="test",
            split_id=config.split_id,
            challenge_type=config.challenge_type,
            selection_description=f"The {region_description}.",
        )
        return _write_products(
            config.output_dir, config.train_id, config.test_id, train, test, [source]
        )
    finally:
        source.close()


def _find_reference(
    sources: Sequence[SourceDataset], dataset_id: str | None
) -> SourceDataset | None:
    if dataset_id is None:
        return None
    return next((source for source in sources if source.dataset_id == dataset_id), None)


def compose_split(config_path: Path | str) -> tuple[Path, Path]:
    config = load_compose_config(config_path)
    if config.output_dir.exists():
        raise SplitterError(f"output directory already exists: {config.output_dir}")
    all_paths = config.train.sources + config.test.sources
    sources: list[SourceDataset] = []
    try:
        for path in all_paths:
            sources.append(_read_source(path))
        _ensure_source_compatibility(sources)
        source_ids = {source.dataset_id for source in sources}
        if config.train.dataset_id in source_ids or config.test.dataset_id in source_ids:
            raise SplitterError(
                "output dataset_id values must differ from all source dataset_id values"
            )
        train_sources = sources[: len(config.train.sources)]
        test_sources = sources[len(config.train.sources) :]
        train_modalities = _side_modalities(train_sources)
        test_modalities = _side_modalities(test_sources)
        if train_modalities != test_modalities or len(train_modalities) < 2:
            raise SplitterError(
                "train and test must have the same final modality set with at least two modalities"
            )
        train_reference = _find_reference(train_sources, config.train.reference_dataset_id)
        test_reference = _find_reference(test_sources, config.test.reference_dataset_id)
        if config.feature_merge_policy == "reference" and (
            train_reference is None or test_reference is None
        ):
            raise SplitterError("each reference_dataset_id must identify a source on its own side")
        train_targets = _determine_target_features(
            train_sources,
            train_modalities,
            config.feature_merge_policy,
            train_reference,
        )
        test_targets = _determine_target_features(
            test_sources,
            test_modalities,
            config.feature_merge_policy,
            test_reference,
        )
        if config.feature_merge_policy == "reference" and train_targets != test_targets:
            raise SplitterError("reference datasets must use identical modality feature order")
        comparison_note = ""
        if train_targets != test_targets:
            comparison_note = (
                "Train and test feature spaces were computed independently from each side's "
                "declared full sources and differ."
            )
        train = _build_composite_product(
            train_sources,
            dataset_id=config.train.dataset_id,
            dataset_type="train",
            split_id=config.split_id,
            challenge_type=config.challenge_type,
            policy=config.feature_merge_policy,
            target_features=train_targets,
            reference_dataset_id=config.train.reference_dataset_id,
            comparison_note=comparison_note,
        )
        test = _build_composite_product(
            test_sources,
            dataset_id=config.test.dataset_id,
            dataset_type="test",
            split_id=config.split_id,
            challenge_type=config.challenge_type,
            policy=config.feature_merge_policy,
            target_features=test_targets,
            reference_dataset_id=config.test.reference_dataset_id,
            comparison_note=comparison_note,
        )
        return _write_products(
            config.output_dir,
            config.train.dataset_id,
            config.test.dataset_id,
            train,
            test,
            sources,
        )
    finally:
        for source in sources:
            source.close()


def _print_range(result: Mapping[str, Any]) -> None:
    print(f"dataset_id: {result['dataset_id']}")
    print(f"coordinate_unit: {result['coordinate_unit']}")
    print(f"coordinate_dimensions: {result['coordinate_dimensions']}")

    def print_summary(label: str, summary: Mapping[str, Any]) -> None:
        print(
            f"{label}: n_obs={summary['n_obs']} "
            f"x=[{summary['x_min']:g}, {summary['x_max']:g}] "
            f"y=[{summary['y_min']:g}, {summary['y_max']:g}]"
        )

    print_summary("global", result["global"])
    for sample, summary in result["samples"].items():
        print_summary(f"sample {sample}", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split schema 1.1 spatial MuData datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    range_parser = subparsers.add_parser("range", help="show x/y coordinate ranges")
    range_parser.add_argument("source", type=Path, metavar="FULL.h5mu")
    range_parser.add_argument("--sample-id")
    range_parser.add_argument("--json", action="store_true", dest="as_json")

    spatial_parser = subparsers.add_parser("spatial", help="split one full dataset by regions")
    spatial_parser.add_argument("config", type=Path, metavar="CONFIG.yaml")

    compose_parser = subparsers.add_parser("compose", help="assign full datasets to train/test")
    compose_parser.add_argument("config", type=Path, metavar="CONFIG.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "range":
            result = coordinate_ranges(args.source, args.sample_id)
            if args.as_json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                _print_range(result)
        elif args.command == "spatial":
            train_path, test_path = spatial_split(args.config)
            print(train_path)
            print(test_path)
        else:
            train_path, test_path = compose_split(args.config)
            print(train_path)
            print(test_path)
    except (SplitterError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
