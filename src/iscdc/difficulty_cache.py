"""Verified offline reuse of domain-classifier results (never used by the website)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

FINGERPRINT_VERSION = "1.0"


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def valid_fingerprint(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("version") == FINGERPRINT_VERSION
        and isinstance(value.get("sha256"), str)
        and len(value["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in value["sha256"])
    )


def input_fingerprint(side: Any) -> dict[str, str]:
    """Hash the matrix and metadata consumed by difficulty._read_side.

    Fixed row blocks and canonical float64 CSR make storage layout, compression,
    and dense/sparse encoding irrelevant. Non-input modalities, spatial coordinates,
    entry IDs, descriptions and annotations do not enter the classifier or diagnostics.
    Bump this version when the fingerprint contract changes, and METHOD_VERSION when
    evaluation starts consuming other inputs.
    """
    digest = hashlib.sha256()
    metadata = {
        "version": FINGERPRINT_VERSION,
        "shape": list(side.adata.shape),
        "obs_names": side.modality_obs_names.tolist(),
        "feature_names": side.feature_names.tolist(),
        "allowed_features": side.allowed_features.tolist(),
        "value_type": side.value_type,
        "technology": side.technology,
        "hierarchy": side.hierarchy,
    }
    digest.update(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for start in range(0, side.adata.n_obs, 1024):
        block = sparse.csr_matrix(side.adata.X[start : start + 1024, :], dtype=np.float64)
        block.sum_duplicates()
        block.sort_indices()
        block.eliminate_zeros()
        # Fixed endian and index width also ignore storage dtype differences.
        for values, dtype in (
            (block.indptr, "<i8"), (block.indices, "<i8"), (block.data, "<f8")
        ):
            digest.update(np.asarray(values, dtype=dtype).tobytes())
    return {"version": FINGERPRINT_VERSION, "sha256": digest.hexdigest()}


def _number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (float, int))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _valid_metrics(row: dict[str, Any], parameters: dict[str, Any]) -> bool:
    """Validate reusable per-Challenge metrics; cohort statistics are regenerated."""
    repeats = row.get("repeat_results")
    if not isinstance(repeats, list) or len(repeats) != parameters["repeats"]:
        return False
    folds = []
    for repeat in repeats:
        if not isinstance(repeat, dict):
            return False
        values = repeat.get("fold_aurocs")
        if (
            not isinstance(values, list)
            or len(values) != parameters["folds"]
            or not all(_number(value) for value in values)
            or not _number(repeat.get("mean_auroc"))
            or not math.isclose(repeat["mean_auroc"], float(np.mean(values)), abs_tol=1e-12)
        ):
            return False
        folds.extend(values)
    for name, expected in (
        ("mean_auroc", float(np.mean(folds))),
        ("std_auroc", float(np.std(folds))),
        ("domain_shift_score", float(np.clip(2 * (np.mean(folds) - 0.5), 0, 1))),
    ):
        if not _number(row.get(name)) or not math.isclose(row[name], expected, abs_tol=1e-12):
            return False
    return isinstance(row.get("warnings"), list) and all(
        isinstance(warning, dict) and isinstance(warning.get("code"), str)
        for warning in row["warnings"]
    )


def reusable_rows(
    report: Any, *, parameters: dict[str, Any], software: dict[str, str], method_version: str
) -> dict[str, dict[str, Any]]:
    """Reject incompatible caches and retry failed or malformed individual results."""
    if (
        not isinstance(report, dict)
        or report.get("report_version") != "1.0"
        or report.get("method_version") != method_version
        or report.get("parameters") != parameters
        or report.get("software") != software
        or not isinstance(report.get("challenges"), list)
    ):
        return {}
    rows = report["challenges"]
    ids = [row.get("split_id") if isinstance(row, dict) else None for row in rows]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        return {}
    return {
        row["split_id"]: copy.deepcopy(row)
        for row in rows
        if row.get("status") == "success"
        and row.get("input_modality") == parameters["input_modality"]
        and _valid_metrics(row, parameters)
    }


def reuse_result(
    previous: dict[str, Any] | None, challenge: Any, fingerprints: dict[str, Any]
) -> dict[str, Any] | None:
    """Rebind current file identities only with exact-file or semantic-input proof."""
    if previous is None:
        return None
    for name in ("train", "test"):
        old = previous.get(name)
        current = getattr(challenge, name)
        if not isinstance(old, dict) or old.get("dataset_id") != current.dataset_id:
            return None
        if old.get("sha256") != current.sha256 and (
            not valid_fingerprint(old.get("input_fingerprint"))
            or old["input_fingerprint"] != fingerprints[name]
        ):
            return None
    result = copy.deepcopy(previous)
    # Challenge type only groups the ranking; it is not a classifier input.
    result["challenge_type"] = challenge.challenge_type
    for name in ("train", "test"):
        result[name]["sha256"] = getattr(challenge, name).sha256
        result[name]["input_fingerprint"] = fingerprints[name]
    return result
