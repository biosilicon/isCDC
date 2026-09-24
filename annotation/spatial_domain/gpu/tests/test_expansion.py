"""Semantic preprocessing and bounded exact graph regression tests (locked GPU env)."""

import numpy as np
import pytest
from scipy import sparse

from iscdc import spatialglue as glue
from iscdc import spatialglue_config as config
from iscdc.spatialglue_graph import exact_neighbors, normalize_graph


@pytest.mark.parametrize("metric,dimensions", [("correlation", 7), ("euclidean", 2)])
def test_exact_graph_matches_all_reference_distances_with_ties(metric, dimensions):
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(42)
    features = rng.normal(size=(61, dimensions))
    features[4:8] = features[2]
    ids = np.array([f"cell_{60 - i:03d}" for i in range(61)])
    informative = np.ones(61, bool)
    informative[0] = False
    got = exact_neighbors(
        features, ids, 5, metric=metric, informative=informative, workspace_mb=0.01
    )
    assert got[0].nnz == 0 and got[:, 0].nnz == 0
    distances = cdist(features, features, metric=metric)
    np.fill_diagonal(distances, np.inf)
    distances[:, 0] = np.inf
    for i in range(1, len(features)):
        cutoff = np.sort(distances[i])[4]
        # Floating point equivalent distances are allowed only at the k boundary.
        selected = got[i].indices
        assert len(selected) == 5 and i not in selected
        assert (distances[i, selected] <= cutoff + 1e-12).all()
    replay = exact_neighbors(
        features, ids, 5, metric=metric, informative=informative, workspace_mb=0.01
    )
    assert (got != replay).nnz == 0


def test_sparse_normalization_equals_original_dense_math():
    rng = np.random.default_rng(4)
    a = (rng.uniform(size=(41, 41)) > 0.94).astype(float)
    np.fill_diagonal(a, 0)
    symmetric = np.minimum(a + a.T, 1) + np.eye(41)
    d = symmetric.sum(axis=1) ** -0.5
    expected = d[:, None] * symmetric * d[None, :]
    np.testing.assert_allclose(normalize_graph(sparse.csr_matrix(a)).toarray(), expected, rtol=1e-6)


@pytest.mark.parametrize(
    "name,value_type",
    [
        ("translatome", "counts"),
        ("protein", "intensity"),
        ("protein", "background_corrected_intensity"),
        ("protein", "binary"),
        ("protein", "normalized"),
        ("rna", "normalized"),
        ("rna", "log_normalized"),
        ("metabolite", "intensity"),
        ("lipid", "intensity"),
        ("metabolite", "normalized"),
        ("methylation", "normalized"),
        ("tcr", "counts"),
        ("bcr", "binary"),
        ("vdj", "counts"),
        ("metabolite", "unknown"),
        ("protein", "unknown"),
    ],
)
def test_recipe_handles_real_value_semantics_without_mutating_source(name, value_type):
    rng = np.random.default_rng(42)
    values = rng.uniform(size=(42, 2 if name == "protein" else 17))
    if value_type == "counts":
        values = np.floor(values * 10)
    elif value_type == "binary":
        values = (values > 0.7).astype(float)
    elif value_type == "background_corrected_intensity":
        values -= 0.5
    values[0] = 0
    original = sparse.csr_matrix(values)
    ids = np.array([f"cell_{i}" for i in range(42)])
    params = config.load_parameters(None, "test", modalities=["rna", "protein"])
    record = {"dataset_id": "zenodo_6784251_scspamet_tonsil_donore_1"}
    params.update(
        input_value_types={name: value_type},
        preprocessing_recipes={name: config.recipe(record, name, value_type)},
    )
    result, selected, notes, _ = glue.prepare_modalities(
        {name: original},
        {name: np.array([f"f_{i}" for i in range(values.shape[1])])},
        ids,
        rng.normal(size=(42, 2)),
        params,
    )
    np.testing.assert_array_equal(original.toarray(), values)
    assert np.isfinite(result[name].obsm["feat"]).all()
    assert not result[name].obs["feature_informative"].iloc[0]
    assert notes[name]["source_value_type"] == value_type
    assert len(selected[name]) == values.shape[1]


def test_receptor_zero_rows_are_retained_without_requiring_double_positive():
    import anndata as ad
    import mudata as md

    ids = np.array([f"cell_{i}" for i in range(12)])
    mods = {}
    for name in ("rna", "tcr", "bcr"):
        x = np.ones((12, 4)) if name == "rna" else np.zeros((12, 4))
        if name != "rna":
            x[0 if name == "tcr" else 1, 0] = 1
        mods[name] = ad.AnnData(sparse.csr_matrix(x))
        mods[name].obs_names = ids
    matrices, _, valid, reasons, coverage = glue.paired_sample(
        md.MuData(mods),
        list(mods),
        np.arange(12),
        ids,
        {"rna": "counts", "tcr": "binary", "bcr": "binary"},
    )
    assert valid.all() and not reasons.any()
    assert coverage["modalities"]["tcr"]["zero_counts"] == 11
    assert matrices["bcr"].shape[0] == 12


def test_signed_methylation_residual_preserves_source_and_fraction_guard():
    import anndata as ad
    import mudata as md

    ids = np.array([f"cell_{i}" for i in range(12)])
    original = np.random.default_rng(42).normal(size=(12, 6))
    original[0] = 0  # An observed zero residual is valid and must stay in the analysis.
    mods = {
        "rna": ad.AnnData(sparse.csr_matrix(np.ones((12, 4)))),
        "methylation": ad.AnnData(sparse.csr_matrix(original)),
    }
    for mod in mods.values():
        mod.obs_names = ids
    data = md.MuData(mods)
    record = {"dataset_id": "GSE270498_spatial_dmt_me11_replicate_50um"}
    types = {"rna": "counts", "methylation": "normalized"}
    args = (data, list(mods), np.arange(12), ids, types)
    matrices, names, valid, _, _ = glue.paired_sample(*args, record=record)
    assert valid.all()
    np.testing.assert_array_equal(matrices["methylation"].toarray(), original)
    params = config.load_parameters(None, record["dataset_id"], modalities=list(mods))
    params.update(
        input_value_types=types,
        preprocessing_recipes={"methylation": config.recipe(record, "methylation", "normalized")},
    )
    prepared, _, notes, _ = glue.prepare_modalities(
        {"methylation": matrices["methylation"]}, names, ids, np.zeros((12, 2)), params
    )
    assert np.isfinite(prepared["methylation"].obsm["feat"]).all()
    assert notes["methylation"]["recipe"] == "methylation_residual_v1"
    assert notes["methylation"]["source_value_semantics"] == "signed MethSCAn residual"
    np.testing.assert_array_equal(data.mod["methylation"].X.toarray(), original)
    with pytest.raises(ValueError, match="fraction outside"):
        glue.paired_sample(*args)
    with pytest.raises(ValueError, match="fraction outside"):
        glue.paired_sample(*args, record={"dataset_id": "GSE270498_spatial_dmt_me11_50um"})
    for nonfinite in (np.nan, np.inf, -np.inf):
        data.mod["methylation"].X.data[0] = nonfinite
        with pytest.raises(ValueError, match="Non-finite"):
            glue.paired_sample(*args, record=record)


def test_preprocessing_cache_revalidates_content_and_input(monkeypatch, tmp_path):
    import iscdc.spatialglue_preprocessing as prep

    monkeypatch.setenv("ISCDC_GLUE_CACHE_ROOT", str(tmp_path))
    original = prep._prepare_modalities
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(prep, "_prepare_modalities", counted)
    rng = np.random.default_rng(42)
    matrix = sparse.csr_matrix(rng.poisson(2, (20, 8)).astype(float))
    params = config.load_parameters(None, "cache", modalities=["rna", "protein"])
    ids = np.array([f"cell_{i}" for i in range(20)])
    args = (
        {"rna": matrix},
        {"rna": np.array([f"g{i}" for i in range(8)])},
        ids,
        rng.normal(size=(20, 2)),
        params,
    )
    first = prep.prepare_modalities(*args)
    second = prep.prepare_modalities(*args)
    assert len(calls) == 1
    np.testing.assert_array_equal(first[0]["rna"].obsm["feat"], second[0]["rna"].obsm["feat"])
    next(tmp_path.glob("*.npz")).write_bytes(b"corrupted cache")
    prep.prepare_modalities(*args)
    assert len(calls) == 2
    matrix[0, 0] += 1
    prep.prepare_modalities(*args)
    assert len(calls) == 3


def test_phase_profiles_cover_sparse_resources():
    from iscdc.spatialglue_batch import phase_estimates
    from iscdc.spatialglue_resources import estimates

    params = config.load_parameters(None, "test", modalities=["rna", "protein"])
    job = {
        "n_obs": 690322,
        "input_bytes": 500_000_000,
        "modalities": ["rna", "protein"],
        "feature_counts": {"rna": 477, "protein": 27},
    }
    job.update(estimates(job["n_obs"], job["feature_counts"], params, job["input_bytes"]))
    phases = phase_estimates(job)
    assert set(phases) == {"pairing", "preprocessing", "graph", "training", "clustering"}
    assert phases["graph"]["gpu"] < phases["training"]["gpu"]
    assert max(p["host"] for p in phases.values()) < 32 * 2**30


def test_normalized_continuous_recipe_is_scale_invariant():
    rng = np.random.default_rng(42)
    values = sparse.csr_matrix(rng.uniform(size=(30, 6)))
    ids = np.array([f"cell_{i}" for i in range(30)])
    params = config.load_parameters(None, "test", modalities=["rna", "protein"])
    params["input_value_types"] = {"metabolite": "normalized"}
    results = []
    for scale in (1, 1e-12):
        prepared, selected, _, _ = glue.prepare_modalities(
            {"metabolite": values * scale},
            {"metabolite": np.array([f"mz_{i}" for i in range(6)])},
            ids,
            rng.normal(size=(30, 2)),
            params,
        )
        assert len(selected["metabolite"]) == 6
        results.append(prepared["metabolite"].obsm["feat"])
    np.testing.assert_allclose(results[0], results[1], atol=1e-6, rtol=1e-6)
