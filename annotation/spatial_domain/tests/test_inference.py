"""Real adapter checks in the isolated CPU environment, independent of network/data."""

import numpy as np
import pytest
from scipy import sparse

from iscdc.spatial_domain_annotation import (
    DEFAULTS,
    DomainInferenceError,
    prepare_rna,
    run_algorithm,
)


def example():
    rng = np.random.default_rng(42)
    counts = rng.poisson(2, (90, 40))
    counts[:45, :20] += rng.poisson(15, (45, 20))
    counts[45:, 20:] += rng.poisson(15, (45, 20))
    coords = np.column_stack((np.arange(90) % 9, np.arange(90) // 9)).astype(float)
    ids = [f"cell_{i:03d}" for i in range(90)]
    return counts, coords, ids


@pytest.mark.parametrize("sparse_input", [False, True])
def test_counts_preprocessing_keeps_zero_rows_explicit_and_does_not_mutate(sparse_input):
    counts, coords, ids = example()
    counts[0] = 0
    original = counts.copy()
    matrix = sparse.csr_matrix(counts) if sparse_input else counts
    data, valid, genes = prepare_rna(matrix, ids, coords, DEFAULTS)
    assert not valid[0] and valid[1:].all()
    assert data.n_obs == 89 and data.n_vars == 40
    assert len(genes) == 40
    np.testing.assert_array_equal(counts, original)
    np.testing.assert_array_equal(data.obsm["spatial"], coords[1:])


def test_invalid_counts_rejected():
    counts, coords, ids = example()
    counts = counts.astype(float)
    counts[0, 0] = -0.5
    with pytest.raises(DomainInferenceError, match="integers"):
        prepare_rna(counts, ids, coords, DEFAULTS)


@pytest.mark.parametrize("method", ["BANKSY", "GraphST"])
def test_real_adapter_is_deterministic_and_returns_aligned_domains(method):
    counts, coords, ids = example()
    params = {**DEFAULTS, "threads": 2, "graphst_threads": 2, "graphst_epochs": 10}
    outputs = []
    for _ in range(2):
        data, valid, _ = prepare_rna(counts, ids, coords, params)
        labels, dims = run_algorithm(data, method, params)
        assert valid.all() and dims == 20
        assert labels.shape == (90,) and labels.min() == 1
        assert len(set(labels)) >= 2
        outputs.append(labels)
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("null_metadata", [False, True])
def test_generation_preserves_partial_coverage_and_separates_samples(
    tmp_path, monkeypatch, null_metadata
):
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace

    import anndata as ad
    import mudata as md
    import pandas as pd

    import iscdc.spatial_domain_annotation as module
    from iscdc.spatial_domain_visualization import load_spatial_domain_visualization

    counts, coords, ids = example()
    counts[0] = 0
    rna = ad.AnnData(sparse.csr_matrix(counts), obs=pd.DataFrame(index=ids))
    protein = ad.AnnData(np.ones((91, 2)), obs=pd.DataFrame(index=ids + ["protein_only"]))
    data = md.MuData({"rna": rna, "protein": protein})
    data.obs["sample_id"] = ["A"] * 45 + ["B"] * 46
    data.obsm["spatial"] = np.vstack([coords, [10, 10]])
    source = tmp_path / "datasets" / "example" / "dataset.h5mu"
    source.parent.mkdir(parents=True)
    data.write_h5mu(source)
    if null_metadata:
        import h5py

        with h5py.File(source, "r+") as handle:
            database = handle["uns"].create_group("database")
            database.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
            node = database.create_dataset("derivation", data=h5py.Empty("f"))
            node.attrs.update({"encoding-type": "null", "encoding-version": "0.1.0"})
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    record = {
        "dataset_id": "example",
        "dataset_type": "full",
        "storage_dir": "example",
        "sha256": before,
        "n_obs": 91,
        "sample_ids": ["A", "B"],
        "spatial_unit": "single_cell",
        "coordinate_dimensions": 2,
        "coordinate_unit": "pixel",
        "modalities": {"rna": {"value_type": "counts"}},
    }
    seen = []

    def fake_algorithm(sample, method, params):
        seen.append(tuple(sample.obs_names))
        return (np.arange(sample.n_obs) % 2 + 1).astype(np.uint16), 20

    monkeypatch.setattr(module, "run_algorithm", fake_algorithm)
    config = tmp_path / "params.yaml"
    config.write_text("defaults: {threads: 2}\n")
    root = tmp_path / "domains"
    result = module.generate(record, SimpleNamespace(data_root=tmp_path / "datasets"), root, config)
    assert result["state"] == "success"
    assert len(seen) == 2 and not set(seen[0]) & set(seen[1])
    assert set(seen[0]) == set(ids[1:45])
    snapshot = load_spatial_domain_visualization(root, record)
    assert snapshot.report["samples"]["A"]["not_analyzed"] == 1
    assert snapshot.report["samples"]["B"]["not_analyzed"] == 1
    assert sum(s.count for s in snapshot.samples.values()) == 91
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(DomainInferenceError, match="Existing"):
        module.generate(record, SimpleNamespace(data_root=tmp_path / "datasets"), root, config)
    assert Path(source).exists()


def test_hvg_retries_only_numerical_failures_and_records_selected_span(monkeypatch):
    import scanpy as sc

    from iscdc.spatial_domain_annotation import select_variable_genes

    calls = []

    def fit(data, **kwargs):
        calls.append(kwargs)
        if kwargs["span"] < 0.75:
            raise ValueError(b"svddc failed in l2fit.")

    monkeypatch.setattr(sc.pp, "highly_variable_genes", fit)
    report = select_variable_genes(object(), 3000)
    assert [call["span"] for call in calls] == [0.3, 0.5, 0.75]
    assert all(call["flavor"] == "seurat_v3" and call["n_top_genes"] == 3000 for call in calls)
    assert report["selected_span"] == 0.75
    assert len(report["attempts"]) == 3 and "error" in report["attempts"][0]

    def unexpected(*args, **kwargs):
        raise ValueError("invalid input")

    monkeypatch.setattr(sc.pp, "highly_variable_genes", unexpected)
    with pytest.raises(ValueError, match="invalid input"):
        select_variable_genes(object(), 3000)


def test_hvg_exhaustion_is_explicit(monkeypatch):
    import scanpy as sc

    from iscdc.spatial_domain_annotation import select_variable_genes

    spans = []

    def fail(data, **kwargs):
        spans.append(kwargs["span"])
        raise ValueError(b"There are other near singularities as well.")

    monkeypatch.setattr(sc.pp, "highly_variable_genes", fail)
    with pytest.raises(DomainInferenceError, match="all LOESS spans"):
        select_variable_genes(object(), 3000)
    assert spans == [0.3, 0.5, 0.75, 1.0]


def test_null_reader_is_idempotent_and_does_not_accept_unknown_encodings(tmp_path):
    import h5py
    from anndata.io import read_elem

    from iscdc.spatial_domain_annotation import enable_h5mu_null_reader

    enable_h5mu_null_reader()
    enable_h5mu_null_reader()
    with h5py.File(tmp_path / "null.h5", "w") as handle:
        node = handle.create_dataset("empty", data=h5py.Empty("f"))
        node.attrs.update({"encoding-type": "null", "encoding-version": "0.1.0"})
        assert read_elem(node) is None
        node.attrs["encoding-version"] = "9.9.9"
        with pytest.raises(Exception, match="No read method registered"):
            read_elem(node)
        malformed = handle.create_dataset("malformed", data=[1])
        malformed.attrs.update({"encoding-type": "null", "encoding-version": "0.1.0"})
        with pytest.raises(DomainInferenceError, match="null dataspace"):
            read_elem(malformed)
