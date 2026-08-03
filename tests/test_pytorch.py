from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pytest
import torch
from scipy import sparse
from torch.utils.data import DataLoader

from iscdc.pytorch import H5MuDataset, H5MuDatasetError, H5MuPredictionDataset


def _write_dataset(
    tmp_path: Path,
    *,
    pairing_type: str = "same_unit",
    sparse_x: bool = False,
    name: str = "training.h5mu",
) -> Path:
    rna_obs = ["c1", "c2", "c3"]
    if pairing_type == "same_unit":
        protein_obs = ["c1", "c2", "c3"]
    elif pairing_type == "partially_shared":
        protein_obs = ["c2", "c3", "c4"]
    elif pairing_type == "unpaired":
        protein_obs = ["c4", "c5", "c6"]
    else:  # pragma: no cover - helper guard
        raise ValueError(pairing_type)

    rna_x = np.asarray([[1, 10], [2, 20], [3, 30]], dtype=np.uint32)
    protein_x = np.asarray([[101], [102], [103]], dtype=np.uint32)
    if sparse_x:
        rna_x = sparse.csr_matrix(rna_x)
        protein_x = sparse.csr_matrix(protein_x)
    rna = ad.AnnData(X=rna_x)
    rna.obs_names = rna_obs
    rna.var_names = ["g1", "g2"]
    rna.uns["assay"] = {"technology": "test", "value_type": "counts"}
    protein = ad.AnnData(X=protein_x)
    protein.obs_names = protein_obs
    protein.var_names = ["p1"]
    protein.uns["assay"] = {"technology": "test", "value_type": "intensity"}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
        mdata = md.MuData({"rna": rna, "protein": protein})
    mdata.obs["sample_id"] = [f"sample_{index % 2}" for index in range(mdata.n_obs)]
    mdata.obs["label"] = np.arange(mdata.n_obs, dtype=np.int64)
    mdata.obsm["spatial"] = np.arange(mdata.n_obs * 2, dtype=np.float32).reshape(-1, 2)
    mdata.uns["database"] = {
        "schema_version": "1.1",
        "dataset_id": "training_data",
        "dataset_type": "full",
        "source": "TEST",
        "organism": "Homo sapiens",
        "tissue": "kidney",
        "spatial_unit": "cell",
        "coordinate_unit": "micrometer",
        "pairing_type": pairing_type,
    }
    path = tmp_path / name
    mdata.write_h5mu(path)
    return path


def _write_feature_mask_dataset(tmp_path: Path) -> Path:
    obs_names = ["a1", "b1"]
    rna = ad.AnnData(X=np.asarray([[1, 2], [3, 0]], dtype=np.uint32))
    rna.obs_names = obs_names
    rna.var_names = ["g1", "g2"]
    rna.uns["assay"] = {"technology": "test", "value_type": "counts"}
    rna.varm["feature_measured_by_source"] = np.asarray(
        [[True, True], [True, False]], dtype=bool
    )
    rna.uns["feature_measurement"] = {
        "mask_key": "feature_measured_by_source",
        "source_dataset_ids": ["source_a", "source_b"],
        "placeholder_value": 0,
        "description": "False values are unmeasured features.",
    }
    protein = ad.AnnData(X=np.asarray([[10], [20]], dtype=np.uint32))
    protein.obs_names = obs_names
    protein.var_names = ["p1"]
    protein.uns["assay"] = {"technology": "test", "value_type": "intensity"}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
        mdata = md.MuData({"rna": rna, "protein": protein})
    mdata.obs["sample_id"] = ["sample_a", "sample_b"]
    mdata.obs["source_dataset_id"] = ["source_a", "source_b"]
    mdata.obs["source_obs_id"] = ["a1", "b1"]
    mdata.obsm["spatial"] = np.asarray([[0, 0], [1, 1]], dtype=np.float32)
    mdata.uns["database"] = {
        "schema_version": "1.1",
        "dataset_id": "masked_train",
        "dataset_type": "train",
        "source": ["A", "B"],
        "organism": "Homo sapiens",
        "tissue": "kidney",
        "spatial_unit": "cell",
        "coordinate_unit": "micrometer",
        "pairing_type": "same_unit",
        "derivation": {
            "construction_type": "composite",
            "source_dataset_ids": ["source_a", "source_b"],
            "split_id": "masked_split",
            "challenge_type": "cross_subject",
            "selection_description": "test",
            "feature_merge_policy": "union",
            "processing_description": "test",
        },
    }
    path = tmp_path / "masked.h5mu"
    mdata.write_h5mu(path)
    return path


@pytest.mark.parametrize("sparse_x", [False, True])
def test_dataset_reads_dense_and_sparse_batches_in_requested_order(tmp_path, sparse_x):
    dataset = H5MuDataset(_write_dataset(tmp_path, sparse_x=sparse_x))
    samples = dataset.__getitems__([2, 0, 2])

    assert [sample["obs"]["obs_id"] for sample in samples] == ["c3", "c1", "c3"]
    torch.testing.assert_close(samples[0]["modalities"]["rna"], torch.tensor([3.0, 30.0]))
    torch.testing.assert_close(samples[1]["modalities"]["rna"], torch.tensor([1.0, 10.0]))
    assert samples[0]["modalities"]["rna"].dtype == torch.float32
    dataset.close()


def test_general_dataset_preserves_missing_modalities_and_selected_obs(tmp_path):
    path = _write_dataset(tmp_path, pairing_type="partially_shared")
    dataset = H5MuDataset(path, modalities=("protein", "rna"), obs_columns=("label",))
    missing_index = dataset.obs_names.get_loc("c1")
    shared_index = dataset.obs_names.get_loc("c2")

    missing = dataset[missing_index]
    assert dataset.modalities == ("protein", "rna")
    assert dataset.dataset_id == "training_data"
    assert list(dataset.feature_names["rna"]) == ["g1", "g2"]
    assert missing["obs"] == {
        "obs_id": "c1",
        "sample_id": "sample_0",
        "label": 0,
    }
    assert not missing["modality_masks"]["protein"]
    assert not missing["feature_masks"]["protein"].any()
    torch.testing.assert_close(missing["modalities"]["protein"], torch.zeros(1))
    assert missing["modality_masks"]["rna"]
    assert missing["feature_masks"]["rna"].all()

    shared = dataset[shared_index]
    assert shared["modality_masks"]["protein"]
    torch.testing.assert_close(shared["modalities"]["protein"], torch.tensor([101.0]))
    torch.testing.assert_close(shared["spatial"], torch.tensor([2.0, 3.0]))


def test_source_specific_feature_masks_are_returned(tmp_path):
    dataset = H5MuDataset(_write_feature_mask_dataset(tmp_path))
    source_a = dataset[dataset.obs_names.get_loc("a1")]
    source_b = dataset[dataset.obs_names.get_loc("b1")]

    assert source_a["feature_masks"]["rna"].tolist() == [True, True]
    assert source_b["feature_masks"]["rna"].tolist() == [True, False]
    assert source_b["obs"]["source_dataset_id"] == "source_b"
    assert source_b["obs"]["source_obs_id"] == "b1"


def test_transform_and_subclass_hook_apply_to_batched_loading(tmp_path):
    path = _write_dataset(tmp_path)

    dataset = H5MuDataset(path, transform=lambda sample: sample["modalities"]["rna"] + 1)
    torch.testing.assert_close(dataset[0], torch.tensor([2.0, 11.0]))

    class RnaOnlyDataset(H5MuDataset):
        def build_sample(self, **values):  # noqa: ANN003, ANN202
            return values["modalities"]["rna"]

    subclassed = RnaOnlyDataset(path)
    batch = next(iter(DataLoader(subclassed, batch_size=2)))
    torch.testing.assert_close(batch, torch.tensor([[1.0, 10.0], [2.0, 20.0]]))


def test_default_collation_and_process_local_reopening(tmp_path):
    dataset = H5MuDataset(_write_dataset(tmp_path, sparse_x=True))
    dataset[0]
    assert dataset._file is None
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._file is None
    torch.testing.assert_close(restored[0]["modalities"]["rna"], torch.tensor([1.0, 10.0]))

    loader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)
    batches = list(loader)
    assert sum(len(batch["obs"]["obs_id"]) for batch in batches) == len(dataset)
    torch.testing.assert_close(batches[0]["modalities"]["rna"][0], torch.tensor([1.0, 10.0]))
    dataset.close()
    restored.close()


def test_prediction_dataset_filters_to_pairs_and_applies_transforms(tmp_path):
    path = _write_dataset(tmp_path, pairing_type="partially_shared")
    dataset = H5MuPredictionDataset(
        path,
        "rna",
        "protein",
        input_transform=lambda value: value + 1,
        target_transform=lambda value: value * 2,
        transform=lambda pair: (pair[0] * 3, pair[1] * 4),
    )

    assert list(dataset.obs_names) == ["c2", "c3"]
    assert dataset.observation_indices.tolist() == [1, 2]
    assert dataset.__getitems__([]) == []
    x, y = dataset[0]
    torch.testing.assert_close(x, torch.tensor([9.0, 63.0]))
    torch.testing.assert_close(y, torch.tensor([808.0]))

    batch_x, batch_y = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch_x.shape == (2, 2)
    assert batch_y.shape == (2, 1)


def test_prediction_dataset_rejects_unpaired_or_unmeasured_data(tmp_path):
    with pytest.raises(H5MuDatasetError, match="no paired observations"):
        H5MuPredictionDataset(
            _write_dataset(tmp_path, pairing_type="unpaired"), "rna", "protein"
        )
    with pytest.raises(H5MuDatasetError, match="unmeasured features"):
        H5MuPredictionDataset(_write_feature_mask_dataset(tmp_path), "rna", "protein")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"modalities": ("missing",)}, "unknown modalities"),
        ({"obs_columns": ("missing",)}, "unknown obs columns"),
        ({"obs_columns": ("obs_id",)}, "observation index"),
    ],
)
def test_invalid_selections_report_clear_errors(tmp_path, kwargs, message):
    with pytest.raises(H5MuDatasetError, match=message):
        H5MuDataset(_write_dataset(tmp_path), **kwargs)
