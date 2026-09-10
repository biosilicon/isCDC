"""Biological resolution classes, ordered from finest to coarsest."""

from collections.abc import Iterable, Mapping
from typing import Any

SPATIAL_RESOLUTION_LABELS = {
    "single_cell": "Single-cell",
    "near_cellular": "Near-cellular",
    "spot_level": "Spot-level",
}
SPATIAL_RESOLUTION_DESCRIPTIONS = {
    "single_cell": "Each observation represents an individual cell or nucleus, including "
    "validated cell segmentation.",
    "near_cellular": "Measurements approach cell scale, but observations do not identify "
    "individual cells.",
    "spot_level": "Coarser spatial observations may contain multiple cells, including "
    "coarse bins.",
}
LEGACY_SPATIAL_UNITS = {"cell", "nucleus", "spot", "bin", "spot/bin", "region"}


def spatial_resolution_label(value: str) -> str:
    return SPATIAL_RESOLUTION_LABELS.get(value, f"{value} (legacy unit)")


def require_spatial_resolution(value: str) -> None:
    if value not in SPATIAL_RESOLUTION_LABELS:
        raise ValueError(
            "spatial_unit must be single_cell, near_cellular or spot_level; "
            "classify legacy observations explicitly before importing or splitting."
        )


def coarsest_resolution(values: Iterable[str]) -> str:
    values = list(values)
    if not values:
        raise ValueError("At least one source resolution is required.")
    for value in values:
        require_spatial_resolution(value)
    order = list(SPATIAL_RESOLUTION_LABELS)
    return max(values, key=order.index)


def original_spatial_unit(database: Mapping[str, Any]) -> str:
    value = database.get("original_spatial_unit")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("original_spatial_unit must record the original observation unit.")
    return value.strip()
