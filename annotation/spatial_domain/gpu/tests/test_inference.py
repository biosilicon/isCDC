"""Real CPU preprocessing and CUDA trainers in the locked SpatialGlue environment."""

from __future__ import annotations

import copy
import os
import random

import anndata as ad
import mudata as md
import numpy as np
import pytest
import torch
from scipy import sparse

from iscdc import spatialglue as glue
from iscdc import spatialglue_config as config


def example(modalities, n=36):
    rng = np.random.default_rng(42)
    ids = np.array([f"cell_{i:03d}" for i in range(n)])
    matrices, names = {}, {}
    for name in modalities:
        size = 8 if name == "protein" else 50
        matrix = rng.poisson(2, (n, size))
        matrix[: n // 2, : size // 2] += rng.poisson(10, (n // 2, size // 2))
        matrices[name] = sparse.csr_matrix(matrix, dtype=float)
        names[name] = np.array([f"{name}_{i}" for i in range(size)])
    coords = np.column_stack((np.arange(n) % 6, np.arange(n) // 6)).astype(float)
    params = config.load_parameters(None, "test", modalities=modalities)
    params.update(threads=2, epochs=3)
    return matrices, names, ids, coords, params


def test_partial_pairing_is_by_id_and_keeps_exclusion_reasons():
    matrices, names, ids, coords, params = example(["rna", "protein", "atac"])
    mods = {}
    for name, matrix in matrices.items():
        data = ad.AnnData(matrix.copy())
        data.obs_names, data.var_names = ids, names[name]
        mods[name] = data
    mods["rna"].X[2] = 0
    mods["protein"] = mods["protein"][list(range(35, 0, -1))].copy()
    mdata = md.MuData(mods)
    aligned, features, valid, reasons, coverage = glue.paired_sample(
        mdata, list(matrices), np.arange(len(ids)), ids
    )
    assert reasons[0] == "missing_modality" and reasons[2] == "zero_counts"
    assert valid.sum() == 34
    np.testing.assert_array_equal(
        aligned["protein"].toarray(), matrices["protein"][valid].toarray()
    )
    assert coverage["modalities"]["protein"]["missing"] == 1
    assert coverage["zero_counts"] == 1
    mods["atac"].X[1, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        glue.paired_sample(
            mdata,
            list(matrices),
            np.arange(len(ids)),
            ids,
            {"rna": "counts", "protein": "counts", "atac": "binary"},
        )


@pytest.mark.parametrize("bad", [-1, 0.5, float("nan"), float("inf")])
def test_invalid_values_rejected_even_in_excluded_rows(bad):
    matrices, names, ids, *_ = example(["rna", "protein"])
    mods = {}
    for name, matrix in matrices.items():
        data = ad.AnnData(matrix.copy())
        data.obs_names, data.var_names = ids, names[name]
        mods[name] = data
    mods["rna"].X[0, 0] = bad
    mods["protein"] = mods["protein"][1:].copy()
    with pytest.raises(ValueError, match="non-negative integers"):
        glue.paired_sample(md.MuData(mods), list(mods), np.arange(len(ids)), ids)


@pytest.mark.parametrize(
    "modalities",
    [
        ["rna", "protein"],
        ["rna", "atac"],
        ["rna", "histone"],
        ["rna", "protein", "histone"],
        ["rna", "atac", "histone"],
    ],
)
def test_preprocessing_preserves_source_and_returns_independent_finite_dimensions(modalities):
    matrices, names, ids, coords, params = example(modalities)
    before = {m: glue.matrix_sha256(x) for m, x in matrices.items()}
    prepared, selected, notes, stages = glue.prepare_modalities(
        matrices, names, ids, coords, params
    )
    dimensions = [d.obsm["feat"].shape[1] for d in prepared.values()]
    assert min(dimensions) >= 2
    if "protein" in modalities:
        assert dimensions[modalities.index("rna")] > dimensions[modalities.index("protein")]
    for name in modalities:
        assert np.isfinite(prepared[name].obsm["feat"]).all()
        assert glue.matrix_sha256(matrices[name]) == before[name]
        assert stages[f"{name}_input"] == before[name]
        assert len(selected[name]) >= prepared[name].obsm["feat"].shape[1]


def test_leiden_uses_igraph_and_stable_id_order(monkeypatch):
    import scanpy as sc

    original = sc.tl.leiden
    original_neighbors = sc.pp.neighbors
    previous_jobs = sc.settings.n_jobs
    calls = []
    neighbor_jobs = []

    def checked(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(sc.tl, "leiden", checked)

    def checked_neighbors(*args, **kwargs):
        neighbor_jobs.append(sc.settings.n_jobs)
        return original_neighbors(*args, **kwargs)

    monkeypatch.setattr(sc.pp, "neighbors", checked_neighbors)
    _, _, ids, _, params = example(["rna", "protein"])
    rng = np.random.default_rng(42)
    embedding = rng.normal(size=(len(ids), 8))
    embedding[:18] += 10
    first = glue.cluster(embedding, ids, params)[0]
    second = glue.cluster(embedding, ids, params)[0]
    np.testing.assert_array_equal(first, second)
    assert first[0] == 1
    assert neighbor_jobs == [params["threads"], params["threads"]]
    assert sc.settings.n_jobs == previous_jobs
    assert all(c["flavor"] == "igraph" and c["directed"] is False for c in calls)


@pytest.mark.parametrize("modalities", [["rna", "protein"], ["rna", "protein", "atac"]])
def test_official_cuda_trainers_honor_parameters_and_replay(modalities):
    # Explicit CUDA test: lack of the required hardware must fail, never skip or use CPU.
    assert torch.cuda.is_available(), "CUDA integration tests require the A800"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    matrices, names, ids, coords, params = example(modalities)
    prepared, *_ = glue.prepare_modalities(matrices, names, ids, coords, params)
    outputs = []
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for _ in range(2):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        embedding, stages = glue.train(copy.deepcopy(prepared), params)
        assert embedding.shape == (len(ids), 64) and np.isfinite(embedding).all()
        assert len(stages) == len(modalities) * 5
        outputs.append(embedding)
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-6, rtol=1e-5)


def test_generation_preserves_source_and_separates_samples(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from iscdc.spatial_domain_visualization import load_spatial_domain_visualization

    matrices, names, ids, coords, params = example(["rna", "protein"])
    mods = {}
    for name, matrix in matrices.items():
        data = ad.AnnData(matrix)
        data.obs_names, data.var_names = ids, names[name]
        mods[name] = data
    mods["rna"].X[1] = 0
    mods["protein"] = mods["protein"][1:].copy()
    data = md.MuData(mods)
    data.obs["sample_id"] = ["A"] * 18 + ["B"] * 18
    data.obsm["spatial"] = coords
    source = tmp_path / "datasets/example/dataset.h5mu"
    source.parent.mkdir(parents=True)
    data.write_h5mu(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    record = {
        "dataset_id": "example",
        "dataset_type": "full",
        "storage_dir": "example",
        "sha256": before,
        "n_obs": 36,
        "sample_ids": ["A", "B"],
        "spatial_unit": "spot_level",
        "coordinate_dimensions": 2,
        "coordinate_unit": "pixel",
        "modalities": {"rna": {"value_type": "counts"}, "protein": {"value_type": "counts"}},
    }
    seen = []

    def train(prepared, params):
        adata = prepared["rna"]
        seen.append(tuple(adata.obs_names))
        return np.random.default_rng(42).normal(size=(adata.n_obs, 64)).astype(np.float32), {}

    monkeypatch.setattr(glue, "train", train)
    path = tmp_path / "config.yaml"
    path.write_text("defaults: {threads: 2}\n")
    root = tmp_path / "sidecars"
    settings = SimpleNamespace(data_root=tmp_path / "datasets")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    result = glue.generate(record, settings, root, path)
    assert result["state"] == "success" and len(seen) == 2
    assert not set(seen[0]) & set(seen[1])
    snapshot = load_spatial_domain_visualization(
        root, record, method_family="spatialglue", combination_id="rna__protein"
    )
    assert snapshot.report["samples"]["A"]["not_analyzed"] == 2
    assert snapshot.report["samples"]["B"]["not_analyzed"] == 0
    assert sum(s.count for s in snapshot.samples.values()) == 36
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="Existing SpatialGlue"):
        glue.generate(record, settings, root, path)
