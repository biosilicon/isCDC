"""Offline domain-classifier difficulty estimates for catalogue Challenges.

The score produced here is deliberately a distribution-separability proxy.  It
does not estimate downstream prediction performance or distinguish biological
shift from technical shift.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import mudata
import numpy as np
from scipy import sparse, stats

from .config import Settings
from .database import create_database_engine, create_session_factory, initialize_database
from .models import Dataset
from .repository import CatalogueFilters, CatalogueIntegrityError, Challenge, list_challenges

METHOD_VERSION = "1.0"
REPORT_VERSION = "1.0"
DEFAULT_INPUT_MODALITY = "rna"
HIERARCHY_COLUMNS = (
    "sample_id",
    "source_dataset_id",
    "donor_id",
    "donor",
    "subject_id",
    "subject",
    "slice_id",
    "slice",
    "tissue",
    "platform",
    "organism",
)


class DifficultyEvaluationError(RuntimeError):
    """Raised when a difficulty report cannot be evaluated or written."""


class ChallengeEvaluationError(DifficultyEvaluationError):
    """Raised when one Challenge cannot produce a valid estimate."""


@dataclass(frozen=True)
class DifficultyConfig:
    input_modality: str = DEFAULT_INPUT_MODALITY
    seed: int = 42
    repeats: int = 5
    folds: int = 5
    max_observations_per_domain: int = 5000
    max_features: int = 2000
    representation_dimensionality: int = 50
    count_target_sum: float = 10_000.0
    logistic_c: float = 1.0
    minimum_observations_per_domain: int = 100
    low_sample_warning_threshold: int = 500
    low_feature_overlap_threshold: float = 0.8
    unstable_auroc_std_threshold: float = 0.05
    unstable_percentile_std_threshold: float = 10.0
    small_category_threshold: int = 5

    def __post_init__(self) -> None:
        if not self.input_modality.strip():
            raise ValueError("input_modality must be non-empty")
        positive = {
            "repeats": self.repeats,
            "folds": self.folds,
            "max_observations_per_domain": self.max_observations_per_domain,
            "max_features": self.max_features,
            "representation_dimensionality": self.representation_dimensionality,
            "minimum_observations_per_domain": self.minimum_observations_per_domain,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if self.max_features < self.representation_dimensionality:
            raise ValueError("max_features must cover representation_dimensionality")


@dataclass
class _SideData:
    dataset: Dataset
    path: Path
    mdata: Any
    adata: Any
    feature_names: np.ndarray
    allowed_features: np.ndarray
    modality_obs_names: np.ndarray
    hierarchy: dict[str, list[str]]
    value_type: str
    technology: str | list[str]

    def close(self) -> None:
        file_manager = getattr(self.mdata, "file", None)
        if file_manager is not None:
            file_manager.close()


def _normal_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_normal_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _normal_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normal_value(item) for item in value]
    return value


def _read_side(dataset: Dataset, settings: Settings, modality: str) -> _SideData:
    path = settings.data_root / dataset.storage_dir / "dataset.h5mu"
    if not path.is_file():
        raise ChallengeEvaluationError(f"dataset file is missing: {path}")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
            warnings.filterwarnings(
                "ignore",
                message="Cannot join columns with the same name because var_names are intersecting",
                category=UserWarning,
                module="mudata",
            )
            mdata = mudata.read_h5mu(path, backed="r")
    except Exception as exc:
        raise ChallengeEvaluationError(f"unable to read {dataset.dataset_id}: {exc}") from exc
    try:
        if modality not in mdata.mod:
            raise ChallengeEvaluationError(
                f"dataset {dataset.dataset_id!r} does not contain modality {modality!r}"
            )
        adata = mdata.mod[modality]
        if adata.X is None or adata.n_obs == 0 or adata.n_vars == 0:
            raise ChallengeEvaluationError(
                f"dataset {dataset.dataset_id!r} has an empty {modality!r} matrix"
            )
        feature_names = np.asarray(adata.var_names.astype(str), dtype=object)
        if len(set(feature_names)) != len(feature_names):
            raise ChallengeEvaluationError(
                f"dataset {dataset.dataset_id!r} has duplicate {modality!r} feature IDs"
            )
        allowed = np.ones(adata.n_vars, dtype=bool)
        if "feature_measured_by_source" in adata.varm:
            mask = np.asarray(adata.varm["feature_measured_by_source"], dtype=bool)
            if mask.ndim != 2 or mask.shape[0] != adata.n_vars:
                raise ChallengeEvaluationError(
                    f"dataset {dataset.dataset_id!r} has an invalid feature measurement mask"
                )
            allowed &= mask.all(axis=1)

        database = mdata.uns.get("database")
        if not isinstance(database, Mapping):
            raise ChallengeEvaluationError(
                f"dataset {dataset.dataset_id!r} lacks database metadata"
            )
        database_id = str(database.get("dataset_id", ""))
        if database_id != dataset.dataset_id:
            raise ChallengeEvaluationError(
                f"catalogue ID {dataset.dataset_id!r} does not match file ID {database_id!r}"
            )
        assay = adata.uns.get("assay")
        if not isinstance(assay, Mapping):
            raise ChallengeEvaluationError(
                f"dataset {dataset.dataset_id!r} lacks assay metadata for {modality!r}"
            )
        value_type = str(assay.get("value_type", "")).strip().lower()
        if not value_type:
            raise ChallengeEvaluationError(
                f"dataset {dataset.dataset_id!r} lacks a value_type for {modality!r}"
            )

        top_names = np.asarray(mdata.obs_names.astype(str), dtype=object)
        positions = {name: index for index, name in enumerate(top_names)}
        modality_names = np.asarray(adata.obs_names.astype(str), dtype=object)
        try:
            top_indices = np.asarray([positions[name] for name in modality_names], dtype=np.int64)
        except KeyError as exc:
            raise ChallengeEvaluationError(
                f"dataset {dataset.dataset_id!r} has modality observations absent from MuData obs"
            ) from exc
        hierarchy: dict[str, list[str]] = {}
        for column in HIERARCHY_COLUMNS:
            if column in mdata.obs:
                values = mdata.obs.iloc[top_indices][column].astype(str)
                hierarchy[column] = sorted(set(values))
        return _SideData(
            dataset=dataset,
            path=path,
            mdata=mdata,
            adata=adata,
            feature_names=feature_names,
            allowed_features=allowed,
            modality_obs_names=modality_names,
            hierarchy=hierarchy,
            value_type=value_type,
            technology=_normal_value(assay.get("technology", "")),
        )
    except Exception:
        mdata.file.close()
        raise


def _common_features(train: _SideData, test: _SideData) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_lookup = {
        str(name): index
        for index, name in enumerate(train.feature_names)
        if train.allowed_features[index]
    }
    test_lookup = {
        str(name): index
        for index, name in enumerate(test.feature_names)
        if test.allowed_features[index]
    }
    names = sorted(train_lookup.keys() & test_lookup.keys())
    return (
        np.asarray([train_lookup[name] for name in names], dtype=np.int64),
        np.asarray([test_lookup[name] for name in names], dtype=np.int64),
        names,
    )


def _matrix_rows(adata: Any, row_indices: np.ndarray, feature_indices: np.ndarray) -> Any:
    # h5py-backed sparse datasets require monotonic indices. Sampling order is
    # restored after the read so the seed still defines the resulting row order.
    order = np.argsort(row_indices, kind="stable")
    sorted_rows = row_indices[order]
    matrix = adata.X[sorted_rows, :]
    matrix = matrix[:, feature_indices]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    matrix = matrix[inverse, :]
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _valid_rows(
    side: _SideData, features: np.ndarray, value_type: str
) -> tuple[np.ndarray, dict[str, int]]:
    valid_parts: list[np.ndarray] = []
    non_finite = 0
    negative = 0
    zero_total = 0
    chunk_size = 8192
    for start in range(0, side.adata.n_obs, chunk_size):
        stop = min(start + chunk_size, side.adata.n_obs)
        rows = np.arange(start, stop, dtype=np.int64)
        matrix = _matrix_rows(side.adata, rows, features)
        if sparse.issparse(matrix):
            finite_rows = np.asarray(np.isfinite(matrix.data))
            finite_by_row = np.ones(matrix.shape[0], dtype=bool)
            if not finite_rows.all():
                bad_entries = np.flatnonzero(~finite_rows)
                bad_rows = np.searchsorted(matrix.indptr, bad_entries, side="right") - 1
                finite_by_row[np.unique(bad_rows)] = False
            negative_by_row = np.asarray(matrix.min(axis=1).toarray()).ravel() < 0
            totals = np.asarray(matrix.sum(axis=1)).ravel()
        else:
            finite_by_row = np.isfinite(matrix).all(axis=1)
            negative_by_row = (matrix < 0).any(axis=1)
            totals = matrix.sum(axis=1)
        acceptable = finite_by_row.copy()
        non_finite += int((~finite_by_row).sum())
        if value_type == "counts":
            acceptable &= ~negative_by_row
            acceptable &= totals > 0
            negative += int((finite_by_row & negative_by_row).sum())
            zero_total += int((finite_by_row & ~negative_by_row & (totals <= 0)).sum())
        valid_parts.append(rows[acceptable])
    valid = np.concatenate(valid_parts) if valid_parts else np.empty(0, dtype=np.int64)
    return valid, {
        "non_finite_rows_excluded": non_finite,
        "negative_rows_excluded": negative,
        "zero_total_rows_excluded": zero_total,
    }


def _preprocess(matrix: Any, value_type: str, target_sum: float) -> Any:
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        if value_type == "counts":
            totals = np.asarray(matrix.sum(axis=1)).ravel()
            matrix = sparse.diags(target_sum / totals) @ matrix
            matrix.data = np.log1p(matrix.data)
        return matrix
    dense = np.asarray(matrix, dtype=np.float64).copy()
    if value_type == "counts":
        totals = dense.sum(axis=1)
        dense *= (target_sum / totals)[:, None]
        np.log1p(dense, out=dense)
    return dense


def _sample_hash(side: _SideData, rows: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in side.modality_obs_names[rows]:
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _fold_auroc(
    matrix: Any,
    labels: np.ndarray,
    training: np.ndarray,
    held_out: np.ndarray,
    config: DifficultyConfig,
    seed: int,
) -> tuple[float, int]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ModuleNotFoundError as exc:  # pragma: no cover - checked at CLI boundary
        raise DifficultyEvaluationError(
            "challenge difficulty evaluation requires scikit-learn"
        ) from exc

    train_matrix = matrix[training]
    if sparse.issparse(train_matrix):
        means = np.asarray(train_matrix.mean(axis=0)).ravel()
        squared_means = np.asarray(train_matrix.multiply(train_matrix).mean(axis=0)).ravel()
        variances = squared_means - means**2
    else:
        variances = np.var(train_matrix, axis=0)
    finite_variable = np.isfinite(variances) & (variances > 0)
    candidates = np.flatnonzero(finite_variable)
    if len(candidates) < config.representation_dimensionality:
        raise ChallengeEvaluationError(
            "fewer than 50 non-constant features remain in a classifier training fold"
        )
    # Stable descending variance order; original columns are already ordered by
    # feature ID and mergesort therefore gives deterministic tie-breaking.
    ordered = candidates[np.argsort(-variances[candidates], kind="stable")]
    selected = ordered[: config.max_features]
    n_components = config.representation_dimensionality
    if min(len(training), len(selected)) < n_components:
        raise ChallengeEvaluationError("a classifier fold cannot support 50 PCA components")

    selected_train = train_matrix[:, selected]
    selected_held_out = matrix[held_out][:, selected]
    if sparse.issparse(selected_train):
        selected_train = selected_train.toarray()
        selected_held_out = selected_held_out.toarray()
    pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=seed)
    train_representation = pca.fit_transform(selected_train)
    held_out_representation = pca.transform(selected_held_out)
    classifier = LogisticRegression(
        C=config.logistic_c,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    classifier.fit(train_representation, labels[training])
    probabilities = classifier.predict_proba(held_out_representation)[:, 1]
    return float(roc_auc_score(labels[held_out], probabilities)), int(len(selected))


def _hierarchy_diagnostics(train: _SideData, test: _SideData) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for column in HIERARCHY_COLUMNS:
        train_values = train.hierarchy.get(column)
        test_values = test.hierarchy.get(column)
        if train_values is None and test_values is None:
            continue
        train_set = set(train_values or [])
        test_set = set(test_values or [])
        diagnostics[column] = {
            "train_unique_count": len(train_set),
            "test_unique_count": len(test_set),
            "overlap_count": len(train_set & test_set),
            "domain_disjoint": bool(train_set and test_set and train_set.isdisjoint(test_set)),
        }
    return diagnostics


def _evaluate_challenge(
    challenge: Challenge, settings: Settings, config: DifficultyConfig
) -> dict[str, Any]:
    if challenge.status != "complete" or challenge.train is None or challenge.test is None:
        raise ChallengeEvaluationError(f"challenge is incomplete: {challenge.status}")
    train = _read_side(challenge.train, settings, config.input_modality)
    test: _SideData | None = None
    try:
        test = _read_side(challenge.test, settings, config.input_modality)
        if train.value_type != test.value_type:
            raise ChallengeEvaluationError(
                f"input value_type differs: train={train.value_type!r}, test={test.value_type!r}"
            )
        train_features, test_features, feature_names = _common_features(train, test)
        if len(feature_names) <= config.representation_dimensionality:
            raise ChallengeEvaluationError(
                f"only {len(feature_names)} fully measured common features; at least "
                f"{config.representation_dimensionality + 1} are required"
            )

        train_valid, train_exclusions = _valid_rows(train, train_features, train.value_type)
        test_valid, test_exclusions = _valid_rows(test, test_features, test.value_type)
        sample_count = min(
            len(train_valid), len(test_valid), config.max_observations_per_domain
        )
        if sample_count < config.minimum_observations_per_domain:
            raise ChallengeEvaluationError(
                f"only {sample_count} valid observations per domain; at least "
                f"{config.minimum_observations_per_domain} are required"
            )

        repeat_results: list[dict[str, Any]] = []
        fold_aurocs: list[float] = []
        selected_feature_counts: list[int] = []
        seeds = np.random.SeedSequence(config.seed).spawn(config.repeats)
        for repeat_index, seed_sequence in enumerate(seeds):
            repeat_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
            rng = np.random.default_rng(repeat_seed)
            train_rows = rng.choice(train_valid, size=sample_count, replace=False)
            test_rows = rng.choice(test_valid, size=sample_count, replace=False)
            train_matrix = _preprocess(
                _matrix_rows(train.adata, train_rows, train_features),
                train.value_type,
                config.count_target_sum,
            )
            test_matrix = _preprocess(
                _matrix_rows(test.adata, test_rows, test_features),
                test.value_type,
                config.count_target_sum,
            )
            if sparse.issparse(train_matrix) or sparse.issparse(test_matrix):
                matrix = sparse.vstack((train_matrix, test_matrix), format="csr")
            else:
                matrix = np.vstack((train_matrix, test_matrix))
            labels = np.concatenate(
                (np.zeros(sample_count, dtype=np.int8), np.ones(sample_count, dtype=np.int8))
            )

            from sklearn.model_selection import StratifiedKFold

            splitter = StratifiedKFold(
                n_splits=config.folds, shuffle=True, random_state=repeat_seed
            )
            repeat_aurocs: list[float] = []
            repeat_features: list[int] = []
            for fold_index, (training, held_out) in enumerate(splitter.split(matrix, labels)):
                fold_seed = (repeat_seed + fold_index + 1) % (2**32 - 1)
                auroc, selected_count = _fold_auroc(
                    matrix, labels, training, held_out, config, fold_seed
                )
                repeat_aurocs.append(auroc)
                repeat_features.append(selected_count)
            fold_aurocs.extend(repeat_aurocs)
            selected_feature_counts.extend(repeat_features)
            repeat_results.append(
                {
                    "repeat": repeat_index,
                    "seed": repeat_seed,
                    "mean_auroc": float(np.mean(repeat_aurocs)),
                    "fold_aurocs": repeat_aurocs,
                    "train_sample_hash": _sample_hash(train, train_rows),
                    "test_sample_hash": _sample_hash(test, test_rows),
                    "train_observations_used": sample_count,
                    "test_observations_used": sample_count,
                    "selected_feature_counts": repeat_features,
                }
            )

        mean_auroc = float(np.mean(fold_aurocs))
        std_auroc = float(np.std(fold_aurocs, ddof=0))
        train_feature_total = len(train.feature_names)
        test_feature_total = len(test.feature_names)
        overlap_denominator = min(train_feature_total, test_feature_total)
        overlap_fraction = len(feature_names) / overlap_denominator
        hierarchy = _hierarchy_diagnostics(train, test)
        result_warnings: list[dict[str, Any]] = []
        if sample_count < config.low_sample_warning_threshold:
            result_warnings.append(
                {"code": "low_sample_size", "observations_per_domain": sample_count}
            )
        if overlap_fraction < config.low_feature_overlap_threshold:
            result_warnings.append(
                {"code": "low_feature_overlap", "overlap_fraction": overlap_fraction}
            )
        if train.value_type != "counts":
            result_warnings.append(
                {
                    "code": "preprocessed_input",
                    "value_type": train.value_type,
                    "message": (
                        "input was not normalized again; cross-pipeline comparability is limited"
                    ),
                }
            )
        if std_auroc > config.unstable_auroc_std_threshold:
            result_warnings.append({"code": "unstable_auroc", "std_auroc": std_auroc})
        if mean_auroc < 0.5:
            result_warnings.append({"code": "below_chance_mean_auroc", "mean_auroc": mean_auroc})
        for column, diagnostic in hierarchy.items():
            if diagnostic["domain_disjoint"]:
                result_warnings.append(
                    {"code": "hierarchy_domain_aligned", "field": column}
                )

        return {
            "split_id": challenge.split_id,
            "challenge_type": challenge.challenge_type,
            "status": "success",
            "input_modality": config.input_modality,
            "train": {
                "dataset_id": train.dataset.dataset_id,
                "sha256": train.dataset.sha256,
                "observations_total": int(train.adata.n_obs),
                "observations_valid": int(len(train_valid)),
                "features_total": train_feature_total,
                "technology": train.technology,
                **train_exclusions,
            },
            "test": {
                "dataset_id": test.dataset.dataset_id,
                "sha256": test.dataset.sha256,
                "observations_total": int(test.adata.n_obs),
                "observations_valid": int(len(test_valid)),
                "features_total": test_feature_total,
                "technology": test.technology,
                **test_exclusions,
            },
            "value_type": train.value_type,
            "preprocessing": (
                f"library_size_{int(config.count_target_sum)}_log1p"
                if train.value_type == "counts"
                else "preserved_declared_preprocessed_values"
            ),
            "common_fully_measured_features": len(feature_names),
            "feature_overlap_fraction": overlap_fraction,
            "selected_feature_count_min": min(selected_feature_counts),
            "selected_feature_count_max": max(selected_feature_counts),
            "representation_dimensionality": config.representation_dimensionality,
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc,
            "domain_shift_score": float(np.clip(2 * (mean_auroc - 0.5), 0, 1)),
            "difficulty_rank": None,
            "difficulty_percentile": None,
            "same_category_percentile": None,
            "repeat_percentile_std": None,
            "repeat_results": repeat_results,
            "hierarchy_diagnostics": hierarchy,
            "warnings": result_warnings,
        }
    finally:
        train.close()
        if test is not None:
            test.close()


def _percentiles(values: Sequence[float]) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    average_ranks = stats.rankdata(values, method="average")
    return [float(100 * (rank - 1) / (len(values) - 1)) for rank in average_ranks]


def _apply_rankings(results: list[dict[str, Any]], config: DifficultyConfig) -> dict[str, Any]:
    successful = [item for item in results if item["status"] == "success"]
    aurocs = [item["mean_auroc"] for item in successful]
    percentiles = _percentiles(aurocs)
    for item, percentile in zip(successful, percentiles, strict=True):
        item["difficulty_rank"] = 1 + sum(value < item["mean_auroc"] for value in aurocs)
        item["difficulty_percentile"] = percentile

    categories: dict[str, list[dict[str, Any]]] = {}
    for item in successful:
        categories.setdefault(item["challenge_type"], []).append(item)
    for category_items in categories.values():
        category_percentiles = _percentiles(
            [item["mean_auroc"] for item in category_items]
        )
        for item, percentile in zip(category_items, category_percentiles, strict=True):
            item["same_category_percentile"] = None if math.isnan(percentile) else percentile
            if len(category_items) < config.small_category_threshold:
                item["warnings"].append(
                    {"code": "small_category_pool", "category_size": len(category_items)}
                )

    repeat_percentile_rows: list[list[float]] = []
    if len(successful) >= 2:
        for repeat_index in range(config.repeats):
            repeat_values = [
                item["repeat_results"][repeat_index]["mean_auroc"] for item in successful
            ]
            repeat_percentile_rows.append(_percentiles(repeat_values))
        percentile_matrix = np.asarray(repeat_percentile_rows, dtype=float)
        for item_index, item in enumerate(successful):
            percentile_std = float(np.nanstd(percentile_matrix[:, item_index], ddof=0))
            item["repeat_percentile_std"] = percentile_std
            if percentile_std > config.unstable_percentile_std_threshold:
                item["warnings"].append(
                    {
                        "code": "unstable_repeat_percentile",
                        "std_percentage_points": percentile_std,
                    }
                )

    correlations: list[float] = []
    if len(successful) >= 2:
        repeat_values = [
            [item["repeat_results"][repeat_index]["mean_auroc"] for item in successful]
            for repeat_index in range(config.repeats)
        ]
        for left in range(config.repeats):
            for right in range(left + 1, config.repeats):
                correlation = stats.spearmanr(repeat_values[left], repeat_values[right]).statistic
                if np.isfinite(correlation):
                    correlations.append(float(correlation))
    return {
        "pairwise_spearman_count": len(correlations),
        "mean_pairwise_spearman": float(np.mean(correlations)) if correlations else None,
        "minimum_pairwise_spearman": min(correlations) if correlations else None,
    }


def _failure_result(
    challenge: Challenge, exc: Exception, config: DifficultyConfig
) -> dict[str, Any]:
    return {
        "split_id": challenge.split_id,
        "challenge_type": challenge.challenge_type,
        "status": "failed",
        "input_modality": config.input_modality,
        "mean_auroc": None,
        "std_auroc": None,
        "domain_shift_score": None,
        "difficulty_rank": None,
        "difficulty_percentile": None,
        "same_category_percentile": None,
        "repeat_percentile_std": None,
        "error": str(exc),
        "warnings": [],
    }


def evaluate_catalogue(
    settings: Settings, config: DifficultyConfig | None = None
) -> dict[str, Any]:
    """Evaluate all Challenges and return a serializable report without writing it."""
    config = config or DifficultyConfig()
    try:
        import sklearn
    except ModuleNotFoundError as exc:
        raise DifficultyEvaluationError(
            "challenge difficulty evaluation requires scikit-learn; install "
            "requirements-difficulty.txt"
        ) from exc

    engine = create_database_engine(settings.database_path)
    try:
        initialize_database(engine)
        session_factory = create_session_factory(engine)
        with session_factory() as session:
            challenges, total = list_challenges(
                session, CatalogueFilters(), offset=0, limit=1_000_000
            )
            if len(challenges) != total:
                raise DifficultyEvaluationError("unable to load the complete Challenge catalogue")
            results: list[dict[str, Any]] = []
            for challenge in challenges:
                try:
                    results.append(_evaluate_challenge(challenge, settings, config))
                except (ChallengeEvaluationError, OSError, ValueError) as exc:
                    results.append(_failure_result(challenge, exc, config))
    except CatalogueIntegrityError as exc:
        raise DifficultyEvaluationError(str(exc)) from exc
    finally:
        engine.dispose()

    ranking_stability = _apply_rankings(results, config)
    results.sort(
        key=lambda item: (
            item["status"] != "success",
            item["mean_auroc"] if item["mean_auroc"] is not None else math.inf,
            item["split_id"],
        )
    )
    success_count = sum(item["status"] == "success" for item in results)
    return {
        "report_version": REPORT_VERSION,
        "method_version": METHOD_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "interpretation": (
            "Train-test separability proxy only; not absolute difficulty, biological shift, "
            "or expected downstream model performance."
        ),
        "parameters": asdict(config),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "challenge_count": len(results),
        "success_count": success_count,
        "failure_count": len(results) - success_count,
        "ranking_stability": ranking_stability,
        "challenges": results,
    }


def write_report_atomically(report: dict[str, Any], path: Path, *, force: bool = False) -> Path:
    """Write a report beside a temporary file and atomically activate it."""
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise DifficultyEvaluationError(
            f"output file already exists: {destination}; use --force to replace it"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if destination.exists() and not force:
            raise DifficultyEvaluationError(
                f"output file already exists: {destination}; use --force to replace it"
            )
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_and_write(
    settings: Settings,
    output: Path,
    *,
    config: DifficultyConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate the catalogue, then atomically publish the complete report."""
    if output.expanduser().resolve().exists() and not force:
        raise DifficultyEvaluationError(
            f"output file already exists: {output.expanduser().resolve()}; "
            "use --force to replace it"
        )
    report = evaluate_catalogue(settings, config)
    write_report_atomically(report, output, force=force)
    return report
