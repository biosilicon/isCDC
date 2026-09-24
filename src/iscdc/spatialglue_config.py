"""SpatialGlue input and configuration contract; safe in the website environment."""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import yaml

from .config import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "assets/spatial_domain/spatialglue.yaml"
LOCK_PATH = PROJECT_ROOT / "annotation/spatial_domain/gpu/requirements.lock.txt"
MODALITIES = (
    "rna",
    "translatome",
    "protein",
    "atac",
    "histone",
    "methylation",
    "metabolite",
    "lipid",
    "tcr",
    "bcr",
    "vdj",
)
MICROBIAL = {"microbiome", "bacterial_taxa", "fungal_taxa", "bacterial_rna", "fungal_rna"}
CONTINUOUS = {"normalized", "log_normalized", "intensity", "background_corrected_intensity"}
VALUE_TYPES = {
    "rna": {"counts", "normalized", "log_normalized"},
    "translatome": {"counts", "normalized", "log_normalized"},
    "protein": {"counts", "binary", "unknown"} | CONTINUOUS,
    "atac": {"counts", "binary"},
    "histone": {"counts"},
    "methylation": {"normalized"},
    "metabolite": {"unknown"} | CONTINUOUS,
    "lipid": CONTINUOUS,
    "tcr": {"counts", "binary"},
    "bcr": {"counts", "binary"},
    "vdj": {"counts", "binary"},
}
DEFAULTS = {
    "seed": 42,
    "resolution": 1.0,
    "n_pcs": 20,
    "expression_neighbors": 50,
    "n_top_genes": 3000,
    "target_sum": 10000,
    "input_dims": 50,
    "threads": 8,
    "memory_budget_gb": 768,
    "gpu_memory_budget_gb": 80,
    "gpu_reserve_gb": 8,
    "device": "cuda:0",
    "y_axis": "up",
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "dim_output": 64,
    "feature_neighbors": 20,
    "epochs": None,
    "spatial_neighbors": None,
    "weight_factors": None,
    "input_modalities": None,
    "graph_workspace_mb": 128,
}
DATASET_ONLY = {
    "threads",
    "memory_budget_gb",
    "gpu_memory_budget_gb",
    "gpu_reserve_gb",
    "device",
    "y_axis",
    "input_modalities",
}


def modality_records(record):
    values = record.get("modalities", {}) if isinstance(record, dict) else record.modalities
    if isinstance(values, dict):
        return values
    return {m.name: {"value_type": m.value_type} for m in values}


def select_modalities(record, requested=None):
    available = modality_records(record)
    if set(available) & MICROBIAL:
        raise ValueError("Microbiome-related Databases are excluded from SpatialGlue")
    if requested is None:
        requested = list(available)
    if (
        not isinstance(requested, list)
        or len(requested) not in {2, 3}
        or any(not isinstance(m, str) for m in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("SpatialGlue requires two or three explicit modalities")
    for name in requested:
        if name not in VALUE_TYPES or name not in available:
            raise ValueError(f"Unsupported or missing SpatialGlue modality: {name}")
        value = available[name]
        value_type = value["value_type"] if isinstance(value, dict) else value.value_type
        if value_type not in VALUE_TYPES[name]:
            raise ValueError(f"Unverified SpatialGlue input: {name}:{value_type}")
        recipe(record, name, value_type)
    return [m for m in MODALITIES if m in requested]


def recipe(record, name, value_type):
    """Versioned recipes; unknown is only admitted for the documented scSpaMet source."""
    dataset_id = record.get("dataset_id", "") if isinstance(record, dict) else record.dataset_id
    if value_type == "unknown":
        if not dataset_id.startswith("zenodo_6784251_scspamet_tonsil_"):
            raise ValueError(f"Unverified unknown-scale source for {name}: {dataset_id}")
        return "source_continuous_v1"
    if name == "methylation":
        # S032 source acceptance records this slice's DNAm assay as signed
        # MethSCAn residuals. Its siblings contain actual [0, 1] fractions.
        if dataset_id == "GSE270498_spatial_dmt_me11_replicate_50um":
            return "methylation_residual_v1"
        return "methylation_fraction_v1"
    if name in {"tcr", "bcr", "vdj"}:
        return f"receptor_{value_type}_v1"
    if value_type in {"normalized", "log_normalized"}:
        return "source_continuous_v1"
    if name in {"atac", "histone"}:
        return "epigenome_lsi_v1"
    if name in {"rna", "translatome"}:
        return "transcript_counts_v1"
    if value_type == "binary":
        return "binary_markers_v1"
    if name == "protein":
        return "protein_clr_v1" if value_type == "counts" else "protein_asinh_v1"
    return "msi_intensity_v1"


def combination_id(modalities):
    if len(modalities) not in {2, 3} or len(set(modalities)) != len(modalities):
        raise ValueError("Invalid SpatialGlue combination")
    ordered = [m for m in MODALITIES if m in modalities]
    if len(ordered) != len(modalities):
        raise ValueError("Unknown combination modality")
    return "__".join(ordered)


def expand_combinations(record, params):
    available = modality_records(record)
    if set(available) & MICROBIAL:
        raise ValueError("Microbiome-related Databases are excluded from SpatialGlue")
    if set(available) - set(MODALITIES):
        raise ValueError("Unverified modality needs a preprocessing recipe")
    requested = params.get("input_modalities")
    if requested is not None:
        return [select_modalities(record, requested)]
    ordered = [m for m in MODALITIES if m in available]
    if len(available) == 4:
        return [select_modalities(record, list(group)) for group in combinations(ordered, 3)]
    return [select_modalities(record)]


def load_parameters(path, dataset_id, sample_id=None, *, modalities=None):
    params = dict(DEFAULTS)
    if path is not None:
        config = yaml.safe_load(Path(path).read_text())
        if not isinstance(config, dict) or set(config) - {"defaults", "datasets"}:
            raise ValueError("SpatialGlue config accepts defaults and datasets")
        datasets = config.get("datasets", {})
        if not isinstance(datasets, dict):
            raise ValueError("datasets must be a mapping")
        dataset = datasets.get(dataset_id, {})
        if not isinstance(dataset, dict) or set(dataset) - {"parameters", "samples"}:
            raise ValueError("Dataset config accepts parameters and samples")
        samples = dataset.get("samples", {})
        if not isinstance(samples, dict):
            raise ValueError("samples must be a mapping")
        # Validate all sample overrides even during the read-only audit.
        for values in samples.values():
            if not isinstance(values, dict) or set(values) & DATASET_ONLY:
                raise ValueError(
                    "Sample overrides cannot change modalities, resources or orientation"
                )
            if set(values) - set(DEFAULTS):
                raise ValueError("Unknown SpatialGlue sample parameter")
        for values in (
            config.get("defaults", {}),
            dataset.get("parameters", {}),
            samples.get(sample_id, {}) if sample_id else {},
        ):
            if not isinstance(values, dict) or set(values) - set(DEFAULTS):
                raise ValueError("Unknown SpatialGlue parameter")
            params.update(values)
    for key, value in params.items():
        if key == "input_modalities":
            if value is not None and (
                not isinstance(value, list) or any(not isinstance(m, str) for m in value)
            ):
                raise ValueError("input_modalities must be a list")
        elif key == "device":
            if (
                not isinstance(value, str)
                or not value.startswith("cuda:")
                or not value[5:].isdigit()
            ):
                raise ValueError("SpatialGlue requires an explicit CUDA device, e.g. cuda:0")
        elif key == "y_axis":
            if value not in {"up", "down"}:
                raise ValueError("y_axis must be up or down")
        elif key == "weight_factors":
            if value is not None and (
                not isinstance(value, list)
                or not value
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    or v <= 0
                    for v in value
                )
            ):
                raise ValueError("weight_factors must be positive finite numbers")
        elif value is None and key in {"epochs", "spatial_neighbors"}:
            continue
        else:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and key not in {"seed", "weight_decay"})
            ):
                raise ValueError(f"Invalid SpatialGlue parameter: {key}")
            if (
                isinstance(DEFAULTS[key], int) or key in {"epochs", "spatial_neighbors"}
            ) and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
    if params["seed"] >= 2**32 or any(
        params[k] < 2
        for k in ("n_top_genes", "input_dims", "n_pcs", "dim_output", "expression_neighbors")
    ):
        raise ValueError("Invalid seed or feature dimensions")
    if (
        params["threads"] > 80
        or params["memory_budget_gb"] > 768
        or params["gpu_memory_budget_gb"] > 80
        or params["gpu_reserve_gb"] < 8
    ):
        raise ValueError("SpatialGlue resource ceilings are 80 threads/768 GiB RAM/80 GiB GPU")
    if modalities is not None:
        triplet = len(modalities) == 3
        epigenome = (
            not triplet and "rna" in modalities and bool(set(modalities) & {"atac", "histone"})
        )
        params["input_modalities"] = list(modalities)
        if params["epochs"] is None:
            params["epochs"] = 1600 if epigenome else 600
        if params["spatial_neighbors"] is None:
            params["spatial_neighbors"] = 6 if epigenome else 3
        if params["weight_factors"] is None:
            classic = "rna" in modalities and bool(set(modalities) & {"protein", "atac", "histone"})
            params["weight_factors"] = (
                [1] * 9 if triplet else ([1, 5, 1, 1] if classic else [1] * 4)
            )
        if len(params["weight_factors"]) != (9 if triplet else 4):
            raise ValueError("Incorrect number of SpatialGlue loss weights")
    return params


def eligibility(record, params):
    if record.get("dataset_type", "full") != "full" or record["coordinate_dimensions"] != 2:
        raise ValueError("SpatialGlue requires a 2D full Database")
    if record["spatial_unit"] not in {"single_cell", "near_cellular", "spot_level"}:
        raise ValueError("An explicit biological spatial resolution is required")
    groups = expand_combinations(record, params)
    return (
        groups[0] if len(groups) == 1 else [m for m in MODALITIES if m in modality_records(record)]
    )
