"""PyTorch datasets for schema 1.1 ``.h5mu`` files.

This module is intentionally optional. Importing the rest of :mod:`iscdc` does
not require PyTorch; users of this module should install
``requirements-pytorch.txt``.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypedDict

import h5py
import mudata
import numpy as np
import pandas as pd
from anndata.io import sparse_dataset
from scipy import sparse

try:
    import torch
    from torch.utils.data import Dataset, get_worker_info
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by installation boundaries
    raise ModuleNotFoundError(
        "iscdc.pytorch requires PyTorch; install requirements-pytorch.txt first"
    ) from exc


SCHEMA_VERSION = "1.1"
FEATURE_MASK_KEY = "feature_measured_by_source"
FIXED_OBS_COLUMNS = ("sample_id",)
DERIVED_OBS_COLUMNS = ("source_dataset_id", "source_obs_id")

__all__ = [
    "H5MuDataset",
    "H5MuDatasetError",
    "H5MuPredictionDataset",
    "H5MuSample",
]


class H5MuDatasetError(ValueError):
    """Raised when a file cannot satisfy the PyTorch dataset contract."""


class H5MuSample(TypedDict):
    """Default observation returned by :class:`H5MuDataset`."""

    modalities: dict[str, torch.Tensor]
    modality_masks: dict[str, torch.Tensor]
    feature_masks: dict[str, torch.Tensor]
    spatial: torch.Tensor
    obs: dict[str, str | bool | int | float]


def _normalize_metadata(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_normalize_metadata(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _normalize_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata(item) for item in value]
    return deepcopy(value)


def _as_string(value: Any, context: str) -> str:
    if not isinstance(value, (str, np.str_)) or not str(value).strip():
        raise H5MuDatasetError(f"{context} must be a non-empty string")
    return str(value)


def _normalize_obs_value(value: Any, column: str) -> str | bool | int | float:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (datetime, date)):
        value = value.isoformat()
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        raise H5MuDatasetError(f"obs column '{column}' contains a missing or non-finite value")
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise H5MuDatasetError(f"obs column '{column}' contains a missing value")
    if not isinstance(value, (str, bool, int, float)):
        raise H5MuDatasetError(
            f"obs column '{column}' contains unsupported value type "
            f"'{type(value).__name__}'; select scalar string, boolean, or numeric columns"
        )
    return value


def _normalize_index(index: Any, length: int) -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError("dataset indices must be integers")
    normalized = int(index)
    if normalized < 0:
        normalized += length
    if normalized < 0 or normalized >= length:
        raise IndexError("dataset index out of range")
    return normalized


class H5MuDataset(Dataset[Any]):
    """Map top-level MuData observations to multimodal PyTorch samples.

    Matrix values are read lazily from the file. Missing modalities and
    source-specific unmeasured features are represented by zero placeholders
    accompanied by explicit boolean masks.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        modalities: Sequence[str] | None = None,
        obs_columns: Sequence[str] = (),
        dtype: torch.dtype = torch.float32,
        transform: Callable[[H5MuSample], Any] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise H5MuDatasetError(f".h5mu file does not exist: {self.path}")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if transform is not None and not callable(transform):
            raise TypeError("transform must be callable")
        self.dtype = dtype
        self.transform = transform

        self._file: h5py.File | None = None
        self._matrices: dict[str, Any] = {}
        self._owner_pid: int | None = None

        mdata = self._read_metadata()
        try:
            self._initialize_from_mudata(mdata, modalities, obs_columns)
        finally:
            mdata.file.close()

    def _read_metadata(self):  # noqa: ANN202
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
                return mudata.read_h5mu(self.path, backed="r")
        except Exception as exc:
            raise H5MuDatasetError(f"unable to read .h5mu file '{self.path}': {exc}") from exc

    def _initialize_from_mudata(
        self,
        mdata,  # noqa: ANN001
        requested_modalities: Sequence[str] | None,
        requested_obs_columns: Sequence[str],
    ) -> None:
        database = mdata.uns.get("database")
        if not isinstance(database, Mapping):
            raise H5MuDatasetError("uns['database'] must be a mapping")
        database = _normalize_metadata(database)
        if database.get("schema_version") != SCHEMA_VERSION:
            raise H5MuDatasetError("only schema 1.1 .h5mu files are supported")
        self.dataset_id = _as_string(database.get("dataset_id"), "database.dataset_id")
        self.dataset_type = _as_string(database.get("dataset_type"), "database.dataset_type")
        if self.dataset_type not in {"full", "train", "test"}:
            raise H5MuDatasetError("database.dataset_type must be full, train, or test")
        self.database_metadata = database

        if mdata.n_obs <= 0 or not mdata.obs_names.is_unique:
            raise H5MuDatasetError("top-level observations must be non-empty and unique")
        self.n_obs = int(mdata.n_obs)
        self.obs_names = mdata.obs_names.astype(str).copy()

        if "sample_id" not in mdata.obs:
            raise H5MuDatasetError("obs['sample_id'] is required")
        if "spatial" not in mdata.obsm:
            raise H5MuDatasetError("obsm['spatial'] is required")
        spatial = np.asarray(mdata.obsm["spatial"])
        if (
            spatial.ndim != 2
            or spatial.shape[0] != self.n_obs
            or spatial.shape[1] not in (2, 3)
            or not np.issubdtype(spatial.dtype, np.number)
            or not np.isfinite(spatial).all()
        ):
            raise H5MuDatasetError("obsm['spatial'] must be a finite N x 2 or N x 3 matrix")
        self._spatial = np.asarray(spatial, dtype=np.float32).copy()

        available_modalities = tuple(map(str, mdata.mod.keys()))
        self.modalities = self._select_modalities(available_modalities, requested_modalities)
        self.feature_names: dict[str, pd.Index] = {}
        self.assay_metadata: dict[str, dict[str, Any]] = {}
        self._row_indices: dict[str, np.ndarray] = {}
        self._feature_masks: dict[str, np.ndarray | None] = {}
        self._feature_mask_sources: dict[str, np.ndarray | None] = {}

        global_names = pd.Index(self.obs_names)
        for modality in self.modalities:
            adata = mdata.mod[modality]
            if adata.X is None or adata.n_obs <= 0 or adata.n_vars <= 0:
                raise H5MuDatasetError(f"mod['{modality}'].X must be a non-empty matrix")
            if not adata.obs_names.is_unique or not adata.var_names.is_unique:
                raise H5MuDatasetError(
                    f"modality '{modality}' observation and feature IDs must be unique"
                )
            global_positions = global_names.get_indexer(adata.obs_names.astype(str))
            if np.any(global_positions < 0):
                raise H5MuDatasetError(
                    f"modality '{modality}' contains observations absent from top-level obs"
                )
            row_indices = np.full(self.n_obs, -1, dtype=np.int64)
            row_indices[global_positions] = np.arange(adata.n_obs, dtype=np.int64)
            self._row_indices[modality] = row_indices
            self.feature_names[modality] = adata.var_names.astype(str).copy()

            assay = adata.uns.get("assay")
            if not isinstance(assay, Mapping):
                raise H5MuDatasetError(f"mod['{modality}'].uns['assay'] must be a mapping")
            self.assay_metadata[modality] = _normalize_metadata(assay)
            self._load_feature_mask(modality, adata, mdata.obs)

        self._load_obs_values(mdata.obs, requested_obs_columns)

    @staticmethod
    def _select_modalities(
        available: tuple[str, ...], requested: Sequence[str] | None
    ) -> tuple[str, ...]:
        if requested is None:
            selected = available
        else:
            if isinstance(requested, (str, bytes)):
                raise TypeError("modalities must be a sequence of modality names")
            selected = tuple(requested)
            if not selected:
                raise H5MuDatasetError("at least one modality must be selected")
            if any(not isinstance(name, str) or not name for name in selected):
                raise TypeError("modality names must be non-empty strings")
            if len(set(selected)) != len(selected):
                raise H5MuDatasetError("modalities must not contain duplicates")
            missing = [name for name in selected if name not in available]
            if missing:
                raise H5MuDatasetError(f"unknown modalities: {', '.join(missing)}")
        if not selected:
            raise H5MuDatasetError("the .h5mu file contains no modalities")
        return selected

    def _load_feature_mask(self, modality: str, adata, obs: pd.DataFrame) -> None:  # noqa: ANN001
        if FEATURE_MASK_KEY not in adata.varm:
            self._feature_masks[modality] = None
            self._feature_mask_sources[modality] = None
            return
        mask = np.asarray(adata.varm[FEATURE_MASK_KEY])
        measurement = adata.uns.get("feature_measurement")
        if not isinstance(measurement, Mapping):
            raise H5MuDatasetError(
                f"modality '{modality}' has a feature mask without feature_measurement metadata"
            )
        source_ids = _normalize_metadata(measurement.get("source_dataset_ids"))
        if (
            measurement.get("mask_key") != FEATURE_MASK_KEY
            or not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(value, str) or not value for value in source_ids)
            or mask.shape != (adata.n_vars, len(source_ids))
            or not np.issubdtype(mask.dtype, np.bool_)
        ):
            raise H5MuDatasetError(f"modality '{modality}' has invalid feature mask metadata")
        if "source_dataset_id" not in obs:
            raise H5MuDatasetError(
                f"modality '{modality}' has a feature mask but obs['source_dataset_id'] is absent"
            )
        source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
        row_sources = np.empty(self.n_obs, dtype=np.int64)
        for index, value in enumerate(obs["source_dataset_id"].astype(str)):
            if value not in source_lookup:
                raise H5MuDatasetError(
                    f"source dataset '{value}' is absent from modality '{modality}' feature mask"
                )
            row_sources[index] = source_lookup[value]
        self._feature_masks[modality] = np.asarray(mask, dtype=bool).copy()
        self._feature_mask_sources[modality] = row_sources

    def _load_obs_values(
        self, obs: pd.DataFrame, requested_columns: Sequence[str]
    ) -> None:
        if isinstance(requested_columns, (str, bytes)):
            raise TypeError("obs_columns must be a sequence of column names")
        requested = tuple(requested_columns)
        if any(not isinstance(name, str) or not name for name in requested):
            raise TypeError("obs column names must be non-empty strings")
        if len(set(requested)) != len(requested):
            raise H5MuDatasetError("obs_columns must not contain duplicates")
        if "obs_id" in requested:
            raise H5MuDatasetError("'obs_id' is the observation index, not an obs column")
        missing = [name for name in requested if name not in obs]
        if missing:
            raise H5MuDatasetError(f"unknown obs columns: {', '.join(missing)}")

        columns = list(FIXED_OBS_COLUMNS)
        if self.dataset_type in {"train", "test"}:
            absent = [name for name in DERIVED_OBS_COLUMNS if name not in obs]
            if absent:
                raise H5MuDatasetError(
                    "derived datasets require obs columns: " + ", ".join(absent)
                )
            columns.extend(DERIVED_OBS_COLUMNS)
        columns.extend(name for name in requested if name not in columns)
        self.obs_columns = tuple(columns)
        self._obs_values: dict[str, tuple[str | bool | int | float, ...]] = {}
        for column in self.obs_columns:
            self._obs_values[column] = tuple(
                _normalize_obs_value(value, column) for value in obs[column].tolist()
            )

    def __len__(self) -> int:
        return self.n_obs

    def _ensure_open(self) -> None:
        current_pid = os.getpid()
        if (
            self._file is not None
            and self._owner_pid == current_pid
            and bool(self._file.id.valid)
        ):
            return
        self.close()
        try:
            handle = h5py.File(self.path, "r")
            matrices: dict[str, Any] = {}
            for modality in self.modalities:
                node = handle[f"mod/{modality}/X"]
                if isinstance(node, h5py.Group):
                    encoding = node.attrs.get("encoding-type")
                    if isinstance(encoding, bytes):
                        encoding = encoding.decode("utf-8")
                    if encoding not in {"csr_matrix", "csc_matrix"}:
                        raise H5MuDatasetError(
                            f"unsupported X encoding '{encoding}' for modality '{modality}'"
                        )
                    matrices[modality] = sparse_dataset(node)
                elif isinstance(node, h5py.Dataset):
                    matrices[modality] = node
                else:  # pragma: no cover - h5py exposes only groups and datasets here
                    raise H5MuDatasetError(f"unsupported X storage for modality '{modality}'")
        except Exception:
            if "handle" in locals():
                handle.close()
            raise
        self._file = handle
        self._matrices = matrices
        self._owner_pid = current_pid

    def _feature_masks_for_indices(
        self, modality: str, indices: np.ndarray, present: np.ndarray
    ) -> np.ndarray:
        n_vars = len(self.feature_names[modality])
        result = np.zeros((len(indices), n_vars), dtype=bool)
        if not np.any(present):
            return result
        stored_mask = self._feature_masks[modality]
        if stored_mask is None:
            result[present] = True
            return result
        source_positions = self._feature_mask_sources[modality]
        assert source_positions is not None
        result[present] = stored_mask[:, source_positions[indices[present]]].T
        return result

    def _all_features_measured(self, modality: str, indices: np.ndarray) -> bool:
        stored_mask = self._feature_masks[modality]
        if stored_mask is None:
            return True
        source_positions = self._feature_mask_sources[modality]
        assert source_positions is not None
        used_sources = np.unique(source_positions[indices])
        return bool(stored_mask[:, used_sources].all())

    def _read_modality_batch(
        self, modality: str, indices: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_rows = self._row_indices[modality][indices]
        present = local_rows >= 0
        values = torch.zeros((len(indices), len(self.feature_names[modality])), dtype=self.dtype)
        if np.any(present):
            unique_rows, inverse = np.unique(local_rows[present], return_inverse=True)
            block = self._matrices[modality][unique_rows.tolist(), :]
            if sparse.issparse(block):
                block = block.toarray()
            dense = np.asarray(block)
            if dense.ndim == 1:
                dense = dense.reshape(1, -1)
            restored = np.ascontiguousarray(dense[inverse])
            values[torch.from_numpy(np.flatnonzero(present))] = torch.as_tensor(
                restored, dtype=self.dtype
            )
        modality_mask = torch.from_numpy(present.copy())
        feature_mask = torch.from_numpy(
            self._feature_masks_for_indices(modality, indices, present)
        )
        return values, modality_mask, feature_mask

    def build_sample(
        self,
        *,
        index: int,
        modalities: dict[str, torch.Tensor],
        modality_masks: dict[str, torch.Tensor],
        feature_masks: dict[str, torch.Tensor],
        spatial: torch.Tensor,
        obs: dict[str, str | bool | int | float],
    ) -> H5MuSample:
        """Assemble one sample; subclasses may override this stable hook."""
        del index
        return {
            "modalities": modalities,
            "modality_masks": modality_masks,
            "feature_masks": feature_masks,
            "spatial": spatial,
            "obs": obs,
        }

    def __getitem__(self, index: int) -> Any:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[Any]:
        normalized = np.asarray(
            [_normalize_index(index, self.n_obs) for index in indices], dtype=np.int64
        )
        if normalized.size == 0:
            return []
        self._ensure_open()

        try:
            return self._getitems_from_open_file(normalized)
        finally:
            # A main-process handle could later be inherited by forked DataLoader
            # workers. Keep handles persistent only inside workers, which are
            # already isolated processes.
            if get_worker_info() is None:
                self.close()

    def _getitems_from_open_file(self, normalized: np.ndarray) -> list[Any]:

        values_by_modality: dict[str, torch.Tensor] = {}
        modality_masks: dict[str, torch.Tensor] = {}
        feature_masks: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            values, modality_mask, feature_mask = self._read_modality_batch(
                modality, normalized
            )
            values_by_modality[modality] = values
            modality_masks[modality] = modality_mask
            feature_masks[modality] = feature_mask
        spatial = torch.from_numpy(np.ascontiguousarray(self._spatial[normalized]))

        samples: list[Any] = []
        for batch_index, observation_index in enumerate(normalized.tolist()):
            obs = {"obs_id": str(self.obs_names[observation_index])}
            obs.update(
                {
                    column: self._obs_values[column][observation_index]
                    for column in self.obs_columns
                }
            )
            sample = self.build_sample(
                index=observation_index,
                modalities={
                    name: values_by_modality[name][batch_index] for name in self.modalities
                },
                modality_masks={
                    name: modality_masks[name][batch_index] for name in self.modalities
                },
                feature_masks={
                    name: feature_masks[name][batch_index] for name in self.modalities
                },
                spatial=spatial[batch_index],
                obs=obs,
            )
            if self.transform is not None:
                sample = self.transform(sample)
            samples.append(sample)
        return samples

    def close(self) -> None:
        """Close the HDF5 handle owned by the current process, if any."""
        handle = getattr(self, "_file", None)
        if handle is not None:
            try:
                handle.close()
            finally:
                self._file = None
                self._matrices = {}
                self._owner_pid = None

    def __enter__(self) -> H5MuDataset:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_matrices"] = {}
        state["_owner_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class H5MuPredictionDataset(Dataset[Any]):
    """Paired single-input, single-target view over a schema 1.1 file."""

    def __init__(
        self,
        path: str | Path,
        input_modality: str,
        target_modality: str,
        *,
        dtype: torch.dtype = torch.float32,
        input_transform: Callable[[torch.Tensor], Any] | None = None,
        target_transform: Callable[[torch.Tensor], Any] | None = None,
        transform: Callable[[tuple[Any, Any]], Any] | None = None,
    ) -> None:
        if input_modality == target_modality:
            raise H5MuDatasetError("input_modality and target_modality must be different")
        for name, callback in (
            ("input_transform", input_transform),
            ("target_transform", target_transform),
            ("transform", transform),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable")
        self.input_modality = input_modality
        self.target_modality = target_modality
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.transform = transform
        self.dataset = H5MuDataset(
            path, modalities=(input_modality, target_modality), dtype=dtype
        )

        paired = np.flatnonzero(
            (self.dataset._row_indices[input_modality] >= 0)
            & (self.dataset._row_indices[target_modality] >= 0)
        ).astype(np.int64)
        if paired.size == 0:
            self.dataset.close()
            raise H5MuDatasetError(
                f"modalities '{input_modality}' and '{target_modality}' have no paired observations"
            )
        for modality in (input_modality, target_modality):
            if not self.dataset._all_features_measured(modality, paired):
                self.dataset.close()
                raise H5MuDatasetError(
                    f"modality '{modality}' has unmeasured features in paired observations; "
                    "use H5MuDataset to retain feature masks"
                )
        paired.setflags(write=False)
        self.observation_indices = paired
        self.obs_names = self.dataset.obs_names[paired]
        self.path = self.dataset.path
        self.dataset_id = self.dataset.dataset_id
        self.dataset_type = self.dataset.dataset_type
        self.input_feature_names = self.dataset.feature_names[input_modality]
        self.target_feature_names = self.dataset.feature_names[target_modality]

    def __len__(self) -> int:
        return len(self.observation_indices)

    def __getitem__(self, index: int) -> Any:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[Any]:
        normalized = [_normalize_index(index, len(self)) for index in indices]
        if not normalized:
            return []
        global_indices = self.observation_indices[normalized].tolist()
        samples = self.dataset.__getitems__(global_indices)
        pairs: list[Any] = []
        for sample in samples:
            input_value = sample["modalities"][self.input_modality]
            target_value = sample["modalities"][self.target_modality]
            if self.input_transform is not None:
                input_value = self.input_transform(input_value)
            if self.target_transform is not None:
                target_value = self.target_transform(target_value)
            pair: Any = (input_value, target_value)
            if self.transform is not None:
                pair = self.transform(pair)
            pairs.append(pair)
        return pairs

    def close(self) -> None:
        self.dataset.close()

    def __enter__(self) -> H5MuPredictionDataset:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self.close()
