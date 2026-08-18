#!/usr/bin/env python3
"""Fetch a deterministic sparse CELLxGENE Census reference subset.

This command is invoked only from the isolated annotation environment. It selects
whole-donor training and calibration partitions from a pinned Census release and
never materializes the expression matrix as a dense array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cellxgene_census
import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.io import mmwrite

REQUIRED_OBS_COLUMNS = (
    "soma_joinid",
    "dataset_id",
    "donor_id",
    "assay",
    "tissue",
    "tissue_general",
    "disease",
    "development_stage",
    "cell_type",
    "cell_type_ontology_term_id",
    "is_primary_data",
)
EXCLUDED_ONTOLOGY_IDS = frozenset({"unknown", "CL:0000000"})
TILEDB_CONFIG = {
    # Reference downloads run alongside annotation jobs.  Keep TileDB's hidden
    # thread pools within the same per-task worker budget as RCTD calibration.
    "sm.compute_concurrency_level": 3,
    "sm.io_concurrency_level": 3,
}


class ReferenceFetchError(RuntimeError):
    """A deterministic reference selection or download failure."""


def _load_recipe(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ReferenceFetchError("Reference recipe must be a mapping")
    required = {
        "reference_id",
        "species",
        "method",
        "source",
        "obs_filter",
        "holdout_donor",
        "gene_key",
        "target_panel",
        "sampling",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ReferenceFetchError("Reference recipe is missing: " + ", ".join(missing))
    source = raw["source"]
    if not isinstance(source, dict) or not {
        "census_release",
        "collection_id",
        "dataset_id",
        "dataset_version_id",
        "license",
        "citation",
    } <= set(source):
        raise ReferenceFetchError("Reference source metadata is incomplete")
    if raw["method"] not in {"singler", "rctd"}:
        raise ReferenceFetchError("Reference method must be singler or rctd")
    if raw["gene_key"] not in {"feature_id", "feature_name"}:
        raise ReferenceFetchError("gene_key must be feature_id or feature_name")
    target_panel = raw["target_panel"]
    if not isinstance(target_panel, dict) or not {
        "h5mu_path",
        "sha256",
        "modality",
        "min_shared_genes",
    } <= set(target_panel):
        raise ReferenceFetchError("target_panel metadata is incomplete")
    sampling = raw["sampling"]
    if not isinstance(sampling, dict):
        raise ReferenceFetchError("sampling must be a mapping")
    for key in (
        "seed",
        "min_train_cells_per_type",
        "min_holdout_cells_per_type",
        "max_train_cells_per_type",
        "max_holdout_cells_per_type",
    ):
        value = sampling.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ReferenceFetchError(f"sampling.{key} must be a positive integer")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_count_depth_metadata(matrix: h5py.Group | h5py.Dataset, sample_size: int) -> dict:
    """Summarize per-observation counts without materializing an observations-by-genes array."""
    if isinstance(matrix, h5py.Group):
        encoding = matrix.attrs.get("encoding-type")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        shape = tuple(map(int, matrix.attrs.get("shape", ())))
        if len(shape) != 2:
            raise ReferenceFetchError("Target RNA sparse matrix has no valid shape")
        data = np.asarray(matrix["data"][:], dtype=np.float64)
        if encoding == "csr_matrix":
            indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
            cumulative = np.empty(len(data) + 1, dtype=np.float64)
            cumulative[0] = 0
            np.cumsum(data, out=cumulative[1:])
            depths = cumulative[indptr[1:]] - cumulative[indptr[:-1]]
        elif encoding == "csc_matrix":
            indices = np.asarray(matrix["indices"][:], dtype=np.int64)
            depths = np.bincount(indices, weights=data, minlength=shape[0])
        else:
            raise ReferenceFetchError("Target RNA X must be CSR or CSC for depth calibration")
    else:
        depths = np.empty(matrix.shape[0], dtype=np.float64)
        chunk_rows = 8192
        for start in range(0, matrix.shape[0], chunk_rows):
            stop = min(start + chunk_rows, matrix.shape[0])
            depths[start:stop] = np.asarray(matrix[start:stop, :]).sum(axis=1)
    if (
        depths.ndim != 1
        or not len(depths)
        or not np.all(np.isfinite(depths))
        or np.any(depths < 0)
    ):
        raise ReferenceFetchError("Target RNA X has invalid count depths")
    positive = np.sort(depths[depths > 0])
    if not len(positive):
        raise ReferenceFetchError("Target RNA X has no nonzero observations")
    sample_count = min(sample_size, len(positive))
    indexes = np.rint(np.linspace(0, len(positive) - 1, sample_count)).astype(np.int64)
    probabilities = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    return {
        "observation_count": int(len(depths)),
        "nonzero_observation_count": int(len(positive)),
        "zero_observation_count": int(np.count_nonzero(depths == 0)),
        "mean": float(np.mean(depths)),
        "quantiles": {
            f"q{int(probability * 100):02d}": float(np.quantile(depths, probability))
            for probability in probabilities
        },
        "deterministic_positive_depth_sample": positive[indexes].astype(int).tolist(),
    }


def load_target_panel(recipe_path: Path, recipe: dict[str, Any]) -> tuple[set[str], dict]:
    """Read and checksum the target RNA panel and optional sparse depth summary."""
    config = recipe["target_panel"]
    h5mu_path = (recipe_path.parent / config["h5mu_path"]).resolve()
    if not h5mu_path.is_file():
        raise ReferenceFetchError(f"Target panel .h5mu is missing: {h5mu_path}")
    actual_sha256 = _sha256(h5mu_path)
    if actual_sha256 != config["sha256"]:
        raise ReferenceFetchError("Target panel .h5mu checksum changed")
    modality = str(config["modality"])
    with h5py.File(h5mu_path, "r") as handle:
        try:
            modality_group = handle["mod"][modality]
            var = modality_group["var"]
        except KeyError as exc:
            raise ReferenceFetchError(f"Target panel modality is missing: {modality}") from exc
        index_key = var.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        raw_values = var[index_key][:]
        calibration = recipe.get("calibration", {})
        count_depth = None
        if calibration.get("match_target_count_depth") is True:
            sample_size = calibration.get("target_depth_sample_size")
            if (
                isinstance(sample_size, bool)
                or not isinstance(sample_size, int)
                or sample_size < 20
            ):
                raise ReferenceFetchError(
                    "calibration.target_depth_sample_size must be an integer of at least 20"
                )
            try:
                count_depth = _target_count_depth_metadata(modality_group["X"], sample_size)
            except KeyError as exc:
                raise ReferenceFetchError("Target panel RNA X is missing") from exc
    values = {
        (value.decode() if isinstance(value, bytes) else str(value)).strip()
        for value in raw_values
    }
    values.discard("")
    if not values:
        raise ReferenceFetchError("Target panel has no gene identifiers")
    metadata = {
        "dataset_path": str(h5mu_path),
        "dataset_sha256": actual_sha256,
        "modality": modality,
        "target_gene_count": len(values),
    }
    if count_depth is not None:
        metadata["count_depth"] = count_depth
    return values, metadata


def _stable_priority(join_id: int, seed: int, partition: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{partition}\0{join_id}".encode()).digest()


def deterministic_group_sample(
    frame: pd.DataFrame,
    *,
    group_column: str,
    maximum: int,
    seed: int,
    partition: str,
) -> pd.DataFrame:
    """Take a stable bounded sample from every observed group."""
    selected: list[pd.DataFrame] = []
    for _, group in frame.groupby(group_column, observed=True, sort=True):
        ordered = group.copy()
        ordered["_priority"] = [
            _stable_priority(int(value), seed, partition) for value in ordered["soma_joinid"]
        ]
        ordered = ordered.sort_values("_priority", kind="mergesort").head(maximum)
        selected.append(ordered.drop(columns="_priority"))
    if not selected:
        return frame.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)


def select_partitions(obs: pd.DataFrame, recipe: dict[str, Any]) -> pd.DataFrame:
    """Select aligned train and whole-donor held-out calibration observations."""
    normalized = obs.copy()
    for column in ("donor_id", "cell_type", "cell_type_ontology_term_id"):
        normalized[column] = normalized[column].astype("string").str.strip()
    normalized = normalized[
        normalized["cell_type"].notna()
        & normalized["cell_type_ontology_term_id"].notna()
        & (normalized["cell_type"] != "")
        & ~normalized["cell_type"].str.casefold().eq("unknown")
        & ~normalized["cell_type_ontology_term_id"].isin(EXCLUDED_ONTOLOGY_IDS)
    ].copy()
    holdout_donor = str(recipe["holdout_donor"])
    if holdout_donor not in set(normalized["donor_id"]):
        raise ReferenceFetchError(f"Configured holdout donor is absent: {holdout_donor}")

    sampling = recipe["sampling"]
    train = normalized[normalized["donor_id"] != holdout_donor].copy()
    holdout = normalized[normalized["donor_id"] == holdout_donor].copy()
    train_counts = train["cell_type_ontology_term_id"].value_counts()
    retained_types = set(
        train_counts[train_counts >= sampling["min_train_cells_per_type"]].index
    )
    holdout_counts = holdout["cell_type_ontology_term_id"].value_counts()
    calibration_types = retained_types & set(
        holdout_counts[holdout_counts >= sampling["min_holdout_cells_per_type"]].index
    )
    if len(retained_types) < 2:
        raise ReferenceFetchError("Fewer than two cell types satisfy training thresholds")
    if len(calibration_types) < 2:
        raise ReferenceFetchError("Fewer than two cell types satisfy holdout thresholds")

    train = train[train["cell_type_ontology_term_id"].isin(retained_types)].copy()
    holdout = holdout[holdout["cell_type_ontology_term_id"].isin(calibration_types)].copy()
    seed = sampling["seed"]
    train = deterministic_group_sample(
        train,
        group_column="cell_type_ontology_term_id",
        maximum=sampling["max_train_cells_per_type"],
        seed=seed,
        partition="train",
    )
    holdout = deterministic_group_sample(
        holdout,
        group_column="cell_type_ontology_term_id",
        maximum=sampling["max_holdout_cells_per_type"],
        seed=seed,
        partition="holdout",
    )
    train["partition"] = "train"
    holdout["partition"] = "holdout"
    result = pd.concat([train, holdout], ignore_index=True)
    result = result.sort_values(
        ["partition", "cell_type_ontology_term_id", "soma_joinid"], kind="mergesort"
    ).reset_index(drop=True)
    if result["soma_joinid"].duplicated().any():
        raise ReferenceFetchError("Reference selection contains duplicate Census join IDs")
    return result


def _validate_dataset_metadata(census: Any, recipe: dict[str, Any]) -> dict[str, Any]:
    source = recipe["source"]
    datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    rows = datasets[datasets["dataset_id"] == source["dataset_id"]]
    if len(rows) != 1:
        raise ReferenceFetchError("Pinned Census dataset ID is absent or duplicated")
    row = rows.iloc[0]
    if str(row["dataset_version_id"]) != source["dataset_version_id"]:
        raise ReferenceFetchError("Pinned Census dataset version changed")
    if str(row["collection_id"]) != source["collection_id"]:
        raise ReferenceFetchError("Pinned Census collection changed")
    return {
        key: (int(row[key]) if key == "dataset_total_cell_count" else str(row[key]))
        for key in (
            "collection_id",
            "collection_name",
            "dataset_id",
            "dataset_version_id",
            "dataset_title",
            "dataset_h5ad_path",
            "dataset_total_cell_count",
            "citation",
        )
    }


def _gene_keys(var: pd.DataFrame, key: str) -> pd.Series:
    if key == "feature_id":
        values = var.index.astype(str) if key not in var else var[key].astype("string")
    else:
        if key not in var:
            raise ReferenceFetchError(f"Census var metadata has no {key}")
        values = var[key].astype("string")
    return pd.Series(values, index=var.index, dtype="string").str.strip()


def collapse_duplicate_genes(
    matrix: sparse.spmatrix, var: pd.DataFrame, gene_key: str
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Merge repeated normalized symbols without densifying the count matrix."""
    keys = _gene_keys(var, gene_key)
    valid = keys.notna() & (keys != "")
    matrix = sparse.csr_matrix(matrix)[:, valid.to_numpy()]
    selected_var = var.loc[valid].copy()
    selected_keys = keys.loc[valid].astype(str).to_numpy()
    unique_keys, inverse = np.unique(selected_keys, return_inverse=True)
    mapper = sparse.csr_matrix(
        (
            np.ones(len(inverse), dtype=np.int8),
            (np.arange(len(inverse), dtype=np.int64), inverse),
        ),
        shape=(len(inverse), len(unique_keys)),
    )
    matrix = sparse.csr_matrix(matrix @ mapper)
    genes = pd.DataFrame({"gene_id": unique_keys})
    if gene_key == "feature_id":
        names = selected_var.get("feature_name", pd.Series("", index=selected_var.index))
        annotations = pd.DataFrame(
            {"gene_id": selected_keys, "gene_symbol": names.astype(str).to_numpy()}
        )
        symbol_by_key = annotations.groupby("gene_id", sort=True)["gene_symbol"].first()
        genes["gene_symbol"] = [symbol_by_key[value] for value in unique_keys]
    else:
        feature_ids = selected_var.get(
            "feature_id", pd.Series(selected_var.index.astype(str), index=selected_var.index)
        )
        annotations = pd.DataFrame(
            {"gene_id": selected_keys, "source_feature_id": feature_ids.astype(str).to_numpy()}
        )
        ids_by_key = annotations.groupby("gene_id", sort=True)["source_feature_id"].agg(
            lambda values: ",".join(sorted(set(values)))
        )
        genes["source_feature_id"] = [ids_by_key[value] for value in unique_keys]
    return matrix, genes


def _validate_counts(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix)
    if not np.all(np.isfinite(result.data)) or np.any(result.data < 0):
        raise ReferenceFetchError("Census raw matrix has invalid counts")
    if not np.allclose(result.data, np.rint(result.data), rtol=0, atol=1e-6):
        raise ReferenceFetchError("Census raw matrix is not integer-valued")
    result.data = np.rint(result.data).astype(np.int64)
    result.eliminate_zeros()
    return result


def fetch_reference(recipe_path: Path, output: Path) -> dict[str, Any]:
    recipe = _load_recipe(recipe_path)
    source = recipe["source"]
    target_genes, target_metadata = load_target_panel(recipe_path, recipe)
    output.mkdir(parents=True, exist_ok=True)
    value_filter = (
        f"is_primary_data == True and dataset_id == '{source['dataset_id']}' and "
        f"({recipe['obs_filter']})"
    )
    with cellxgene_census.open_soma(
        census_version=source["census_release"], tiledb_config=TILEDB_CONFIG
    ) as census:
        dataset_metadata = _validate_dataset_metadata(census, recipe)
        obs = cellxgene_census.get_obs(
            census,
            recipe["species"],
            value_filter=value_filter,
            column_names=list(REQUIRED_OBS_COLUMNS),
        )
        selected = select_partitions(obs, recipe)
        join_ids = selected["soma_joinid"].astype(np.int64).to_numpy()
        census_var = cellxgene_census.get_var(
            census,
            recipe["species"],
            column_names=["soma_joinid", "feature_id", "feature_name"],
        )
        census_gene_keys = _gene_keys(census_var, recipe["gene_key"])
        shared = census_var[census_gene_keys.isin(target_genes)]
        minimum = int(recipe["target_panel"]["min_shared_genes"])
        if len(shared) < minimum:
            raise ReferenceFetchError(
                f"Target and Census share {len(shared)} genes; configured minimum is {minimum}"
            )
        var_join_ids = shared["soma_joinid"].astype(np.int64).to_numpy()
        adata = cellxgene_census.get_anndata(
            census,
            recipe["species"],
            X_name="raw",
            obs_coords=join_ids,
            var_coords=var_join_ids,
            obs_column_names=list(REQUIRED_OBS_COLUMNS),
            var_column_names=["feature_id", "feature_name"],
        )

    downloaded_ids = adata.obs["soma_joinid"].astype(np.int64).to_numpy()
    position = {int(value): index for index, value in enumerate(downloaded_ids)}
    if set(position) != set(map(int, join_ids)):
        raise ReferenceFetchError("Downloaded Census matrix does not match selected observations")
    order = np.fromiter((position[int(value)] for value in join_ids), dtype=np.int64)
    matrix = _validate_counts(adata.X[order, :])
    matrix, genes = collapse_duplicate_genes(matrix, adata.var, recipe["gene_key"])
    if matrix.shape != (len(selected), len(genes)) or matrix.nnz == 0:
        raise ReferenceFetchError("Downloaded sparse reference matrix is empty or misaligned")

    observation_columns = [
        "soma_joinid",
        "partition",
        "dataset_id",
        "donor_id",
        "assay",
        "tissue",
        "tissue_general",
        "disease",
        "development_stage",
        "cell_type",
        "cell_type_ontology_term_id",
    ]
    selected[observation_columns].to_csv(
        output / "census_observations.tsv", sep="\t", index=False
    )
    genes.to_csv(output / "census_genes.tsv", sep="\t", index=False)
    mmwrite(output / "census_matrix.mtx", matrix, symmetry="general")
    metadata = {
        "schema_version": "1.0",
        "reference_id": recipe["reference_id"],
        "census_release": source["census_release"],
        "dataset": dataset_metadata,
        "declared_license": source["license"],
        "declared_citation": source["citation"],
        "query": value_filter,
        "holdout_donor": str(recipe["holdout_donor"]),
        "gene_key": recipe["gene_key"],
        "source_observation_count": int(len(obs)),
        "selected_observation_count": int(len(selected)),
        "train_observation_count": int((selected["partition"] == "train").sum()),
        "holdout_observation_count": int((selected["partition"] == "holdout").sum()),
        "gene_count": int(len(genes)),
        "matrix_nnz": int(matrix.nnz),
        "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "target_panel": {**target_metadata, "shared_gene_count": int(len(genes))},
    }
    if not all(math.isfinite(float(value)) for value in (matrix.shape[0], matrix.shape[1])):
        raise ReferenceFetchError("Reference dimensions are invalid")
    (output / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fetch_reference(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
