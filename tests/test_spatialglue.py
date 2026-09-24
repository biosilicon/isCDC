"""Website-side SpatialGlue contracts; no algorithm/GPU dependencies."""

from __future__ import annotations

import copy
import gzip
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from test_spatial_domain_visualization import ModeButtons, generation

from iscdc import spatialglue_config as config
from iscdc.spatial_domain_visualization import (
    domain_directory,
    file_record,
    load_spatial_domain_visualization,
    load_spatial_domain_visualizations,
    obs_order_sha256,
    publish_generation,
    publish_method_failure,
)


def record(modalities=("rna", "protein")):
    return {
        "dataset_id": "example",
        "dataset_type": "full",
        "coordinate_dimensions": 2,
        "spatial_unit": "spot_level",
        "modalities": {m: {"value_type": "counts"} for m in modalities},
    }


def glue_generation(source=None):
    source, manifest, files = generation(source)
    source = copy.deepcopy(source)
    source.setdefault("modalities", record()["modalities"])
    manifest["manifest_version"] = 2
    manifest["method"] = "SpatialGlue"
    provenance = manifest["provenance"]
    provenance.pop("input_modality")
    provenance.update(
        input_modalities=["rna", "protein"],
        unused_modalities=[],
        input_value_types={"rna": "counts", "protein": "counts"},
        device={"device": "cuda:0", "uuid": "GPU-test"},
        adapter_sha256="c" * 64,
    )
    report = json.loads(files["report.json"])
    report["report_version"] = 2
    for sample in report["samples"].values():
        sample["features"] = {}
        for name in ("rna", "protein"):
            features = sample["selected_genes"] if name == "rna" else ["CD3", "CD4"]
            path = f"features/{name}.txt.gz"
            files[path] = gzip.compress(("\n".join(features) + "\n").encode(), mtime=0)
            sample["features"][name] = {
                "count": len(features),
                "ids_sha256": obs_order_sha256(features),
                "file": file_record(path, files[path]),
            }
        sample["coverage"] = {
            "missing_modality": 0,
            "zero_counts": 1,
            "modalities": {m: {"missing": 0, "zero_counts": 1} for m in ("rna", "protein")},
        }
        sample["parameters"]["input_modalities"] = ["rna", "protein"]
        sample["clustering"] = {
            "flavor": "igraph",
            "directed": False,
            "n_iterations": -1,
            "resolution": 1.0,
            "seed": 42,
        }
        files["embeddings/sample_0.npy"] = b"fixture-embedding"
        sample["joint_embedding"] = file_record("embeddings/sample_0.npy", b"fixture-embedding")
    files["report.json"] = json.dumps(report).encode()
    manifest["report"] = file_record("report.json", files["report.json"])
    return source, manifest, files


def test_explicit_subset_canonical_order_and_unsupported_input():
    full = record(("histone", "protein", "rna", "atac"))
    with pytest.raises(ValueError, match="two or three"):
        config.select_modalities(full)
    assert config.select_modalities(full, ["atac", "rna", "protein"]) == ["rna", "protein", "atac"]
    full["modalities"]["protein"]["value_type"] = "normalized"
    assert config.select_modalities(full, ["rna", "protein"]) == ["rna", "protein"]
    assert config.select_modalities(full, ["protein", "atac"]) == ["protein", "atac"]
    for requested in (["rna", "rna"], ["rna", "vdj"]):
        with pytest.raises(ValueError):
            config.select_modalities(full, requested)


def test_all_four_triplets_and_microbiome_exclusion():
    source = record(("rna", "protein", "atac", "histone"))
    groups = config.expand_combinations(source, config.DEFAULTS)
    assert groups == [
        ["rna", "protein", "atac"],
        ["rna", "protein", "histone"],
        ["rna", "atac", "histone"],
        ["protein", "atac", "histone"],
    ]
    assert len({config.combination_id(g) for g in groups}) == 4
    for name in config.MICROBIAL:
        with pytest.raises(ValueError, match="Microbiome"):
            config.expand_combinations(record(("rna", name)), config.DEFAULTS)
    source["modalities"]["protein"]["value_type"] = "unknown"
    with pytest.raises(ValueError, match="unknown-scale"):
        config.expand_combinations(source, config.DEFAULTS)


def combination_generation(source, modalities):
    source, manifest, files = glue_generation(source)
    combination = config.combination_id(modalities)
    manifest["manifest_version"] = 3
    manifest["method"] = "SpatialGlue_3M" if len(modalities) == 3 else "SpatialGlue"
    manifest["provenance"].update(
        input_modalities=modalities,
        unused_modalities=sorted(set(source["modalities"]) - set(modalities)),
        input_value_types={m: source["modalities"][m]["value_type"] for m in modalities},
        combination_id=combination,
        recipe_version=1,
        partial_run=False,
    )
    report = json.loads(files["report.json"])
    report["report_version"] = 3
    for sample in report["samples"].values():
        template = sample["features"]["protein"]
        sample["features"] = {m: sample["features"].get(m, template) for m in modalities}
        if "rna" not in modalities:
            sample["selected_genes"] = []
            sample["selected_genes_sha256"] = obs_order_sha256([])
        sample["parameters"]["input_modalities"] = modalities
        sample["coverage"]["modalities"] = {m: {"missing": 0, "zero_counts": 1} for m in modalities}
    files["report.json"] = json.dumps(report).encode()
    manifest["report"] = file_record("report.json", files["report.json"])
    return source, manifest, files


def test_four_combinations_coexist_and_failed_status_is_independent(tmp_path):
    from iscdc.spatial_domain_visualization import load_spatialglue_combinations

    source, _, _ = generation()
    source["modalities"] = record(("rna", "protein", "atac", "histone"))["modalities"]
    groups = config.expand_combinations(source, config.DEFAULTS)
    for modalities in groups:
        source, manifest, files = combination_generation(source, modalities)
        publish_generation(
            tmp_path,
            source,
            manifest,
            files,
            method_family="spatialglue",
            combination_id=config.combination_id(modalities),
        )
    snapshots = load_spatialglue_combinations(tmp_path, [source])[source["dataset_id"]]
    assert len(snapshots) == 4
    assert (
        next(iter(snapshots["protein__atac__histone"].report["samples"].values()))["selected_genes"]
        == []
    )
    publish_method_failure(
        tmp_path,
        source["dataset_id"],
        "failed",
        method_family="spatialglue",
        combination_id="rna__protein__atac",
    )
    assert len(load_spatialglue_combinations(tmp_path, [source])[source["dataset_id"]]) == 3
    with pytest.raises(ValueError, match="Combination identity"):
        publish_generation(
            tmp_path / "wrong",
            source,
            manifest,
            files,
            method_family="spatialglue",
            combination_id="rna__protein__atac",
        )


def test_resource_estimates_scale_linearly_after_sparse_adaptation():
    from iscdc.spatialglue_resources import estimates

    params = config.load_parameters(None, "test", modalities=["rna", "protein"])
    small = estimates(100000, {"rna": 477, "protein": 27}, params, 1000000)
    large = estimates(690322, {"rna": 477, "protein": 27}, params, 6903220)
    assert large["estimated_host_bytes"] < small["estimated_host_bytes"] * 7
    assert large["estimated_gpu_bytes"] < small["estimated_gpu_bytes"] * 7


def test_methylation_residual_recipe_is_limited_to_verified_source():
    assert (
        config.recipe(
            {"dataset_id": "GSE270498_spatial_dmt_me11_replicate_50um"}, "methylation", "normalized"
        )
        == "methylation_residual_v1"
    )
    for dataset in ("GSE270498_spatial_dmt_me11_50um", "unknown_residual_dataset"):
        assert (
            config.recipe({"dataset_id": dataset}, "methylation", "normalized")
            == "methylation_fraction_v1"
        )


def test_scheduler_backfills_a_fitting_job_and_refills_after_release():
    from iscdc.spatialglue_batch import GIB, next_pending

    def queued(n, host):
        return {"workload": {"n_obs": n}, "phases": {"pairing": {"host": host * GIB, "gpu": GIB}}}

    pending = [queued(690322, 20), queued(9000, 2)]
    active = [{"cpus": [0], "threads": 8, "lease_host": 10 * GIB, "lease_gpu": GIB}]
    assert next_pending(pending, active, list(range(16)), 25 * GIB, 8 * GIB, 0) == (1, 1)
    assert next_pending(pending, [], list(range(16)), 25 * GIB, 8 * GIB, 0) == (0, 0)
    assert next_pending(pending, active, [0], 25 * GIB, 8 * GIB, 0) is None
    assert next_pending(pending, active, list(range(16)), 25 * GIB, GIB, 0) is None


def test_scheduler_calibration_is_scale_specific_and_reduces_overestimation():
    from iscdc.spatialglue_batch import GIB, fit_resources, phase_estimates
    from iscdc.spatialglue_resources import estimates

    def workload(n):
        row = {
            "n_obs": n,
            "input_bytes": n * 100,
            "modalities": ["rna", "protein"],
            "feature_counts": {"rna": 477, "protein": 27},
        }
        params = config.load_parameters(None, "test", modalities=row["modalities"])
        row.update(estimates(n, row["feature_counts"], params, row["input_bytes"]))
        return row

    large, small = workload(690322), workload(9000)
    observed = {"rss_bytes": 3 * GIB, "gpu_bytes": 4 * GIB}
    model = fit_resources(
        [{"returncode": 0, "workload": large, "phase_peaks": {"training": observed}}]
    )
    fitted = phase_estimates(large, model)["training"]
    assert observed["gpu_bytes"] < fitted["gpu"] < observed["gpu_bytes"] * 1.1
    assert fitted["gpu"] < phase_estimates(large)["training"]["gpu"]
    assert phase_estimates(small, model) == phase_estimates(small)


@pytest.mark.anyio
async def test_four_combination_routes_and_selector(
    settings, write_h5mu, write_metadata, metadata_values
):
    import mudata

    from iscdc.app import create_app
    from iscdc.importer import import_dataset
    from iscdc.spatial_domain_annotation import catalogue_records

    path = write_h5mu()
    data = mudata.read_h5mu(path)
    data.mod["protein"].uns["assay"]["value_type"] = "counts"
    metadata_values["modalities"]["protein"]["value_type"] = "counts"
    for name in ("atac", "histone"):
        data.mod[name] = data.mod["protein"].copy()
        metadata_values["modalities"][name] = {"technology": "Xenium", "value_type": "counts"}
    data.update()
    data.write_h5mu(path)
    settings = replace(
        settings, spatial_domain_visualization_root=settings.data_root.parent / "domains"
    )
    import_dataset(path, write_metadata(), settings)
    source = catalogue_records(settings)[0]
    root = settings.spatial_domain_visualization_root
    for group in config.expand_combinations(source, config.DEFAULTS):
        source, manifest, files = combination_generation(source, group)
        publish_generation(
            root,
            source,
            manifest,
            files,
            method_family="spatialglue",
            combination_id=config.combination_id(group),
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(settings)), base_url="http://test"
    ) as client:
        response = await client.get(f"/databases/{source['dataset_id']}")
        assert response.status_code == 200
        assert "data-spatialglue-combination" in response.text
        assert 'value="spatialglue:protein__atac__histone"' in response.text
        assert ModeButtons(response.text).buttons["spatialglue"]["aria-pressed"] == "true"
        for group in config.expand_combinations(source, config.DEFAULTS):
            combination = config.combination_id(group)
            url = (
                f"/databases/{source['dataset_id']}/spatial-domain-visualization/"
                f"spatialglue/{combination}/run-1/sample_0"
            )
            assert (await client.get(url)).status_code == 200
            assert (await client.head(url)).status_code == 200
            assert (await client.get(url.replace(combination, "unverified"))).status_code == 404


@pytest.mark.parametrize(
    "mods,epochs,neighbors,weights",
    [
        (["rna", "protein"], 600, 3, [1, 5, 1, 1]),
        (["rna", "atac"], 1600, 6, [1, 5, 1, 1]),
        (["rna", "histone"], 1600, 6, [1, 5, 1, 1]),
        (["rna", "atac", "histone"], 600, 3, [1] * 9),
    ],
)
def test_resolved_recipes_and_sample_overrides(tmp_path, mods, epochs, neighbors, weights):
    params = config.load_parameters(None, "example", modalities=mods)
    assert (params["epochs"], params["spatial_neighbors"], params["weight_factors"]) == (
        epochs,
        neighbors,
        weights,
    )
    path = tmp_path / "config.yaml"
    path.write_text("datasets:\n  example:\n    samples:\n      A: {epochs: 17, resolution: 0.6}\n")
    params = config.load_parameters(path, "example", "A", modalities=mods)
    assert params["epochs"] == 17 and params["resolution"] == 0.6
    path.write_text("datasets: {example: {samples: {A: {input_modalities: [rna, protein]}}}}")
    with pytest.raises(ValueError, match="Sample overrides"):
        config.load_parameters(path, "example")


@pytest.mark.parametrize(
    "key,value",
    [
        ("device", "cpu"),
        ("threads", 81),
        ("gpu_memory_budget_gb", 81),
        ("epochs", 1.5),
        ("resolution", float("nan")),
    ],
)
def test_invalid_configuration(tmp_path, key, value):
    import yaml

    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({"defaults": {key: value}}))
    with pytest.raises(ValueError):
        config.load_parameters(path, "example")


def test_cuda_unavailable_and_preflight(monkeypatch):
    from iscdc.spatialglue_resources import GIB, check_resources, cuda_info

    params = config.load_parameters(None, "example", modalities=["rna", "protein"])
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=256 * GIB)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False, mem_get_info=lambda device: (8 * GIB, 80 * GIB)
            )
        ),
    )
    with pytest.raises(ValueError, match="never falls back"):
        cuda_info(params)
    with pytest.raises(ValueError, match="GPU resource preflight"):
        check_resources(10, {"rna": 10, "protein": 3}, params)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=8 * GIB)),
    )
    with pytest.raises(ValueError, match="host resource preflight"):
        check_resources(10, {"rna": 10, "protein": 3}, params)


def test_two_versions_coexist_and_fail_independently(tmp_path):
    source, old, old_files = generation()
    publish_generation(tmp_path, source, old, old_files)
    source, new, files = glue_generation(source)
    publish_generation(tmp_path, source, new, files, method_family="spatialglue")
    assert load_spatial_domain_visualization(tmp_path, source).manifest["method"] == "BANKSY"
    assert (
        load_spatial_domain_visualization(tmp_path, source, method_family="spatialglue").manifest[
            "method"
        ]
        == "SpatialGlue"
    )
    publish_method_failure(tmp_path, "example", "CUDA failure", method_family="spatialglue")
    assert load_spatial_domain_visualizations(tmp_path, [source], method_family="spatialglue") == {}
    assert "example" in load_spatial_domain_visualizations(tmp_path, [source])


@pytest.mark.parametrize("corruption", ["order", "value_type", "coverage", "backend", "features"])
def test_invalid_multimodal_manifest_rejected(tmp_path, corruption):
    source, manifest, files = glue_generation()
    report = json.loads(files["report.json"])
    sample = next(iter(report["samples"].values()))
    if corruption == "order":
        manifest["provenance"]["input_modalities"].reverse()
    elif corruption == "value_type":
        source["modalities"]["protein"]["value_type"] = "intensity"
    elif corruption == "coverage":
        sample["coverage"]["zero_counts"] = 0
    elif corruption == "backend":
        sample["clustering"]["flavor"] = "leidenalg"
    else:
        sample["features"]["protein"]["ids_sha256"] = "f" * 64
    files["report.json"] = json.dumps(report).encode()
    manifest["report"] = file_record("report.json", files["report.json"])
    with pytest.raises(ValueError):
        publish_generation(tmp_path, source, manifest, files, method_family="spatialglue")


@pytest.mark.anyio
async def test_method_routes_and_fallback(settings, write_h5mu, write_metadata, metadata_values):
    import mudata

    from iscdc.app import create_app
    from iscdc.importer import import_dataset
    from iscdc.spatial_domain_annotation import catalogue_records

    path = write_h5mu()
    data = mudata.read_h5mu(path)
    data.mod["protein"].uns["assay"]["value_type"] = "counts"
    data.write_h5mu(path)
    metadata_values["modalities"]["protein"]["value_type"] = "counts"
    settings = replace(
        settings, spatial_domain_visualization_root=settings.data_root.parent / "domains"
    )
    import_dataset(path, write_metadata(), settings)
    source = catalogue_records(settings)[0]
    source, manifest, files = glue_generation(source)
    root = settings.spatial_domain_visualization_root
    publish_generation(root, source, manifest, files, method_family="spatialglue")
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/databases/{source['dataset_id']}")
        buttons = ModeButtons(response.text).buttons
        assert buttons["spatialglue"]["aria-pressed"] == "true"
        assert "disabled" in buttons["spatial_domain"]
        assert "SpatialGlue + Leiden" in response.text and "igraph" in response.text
        url = (
            f"/databases/{source['dataset_id']}/spatial-domain-visualization/"
            "spatialglue/run-1/sample_0"
        )
        for encoding in ("identity", "gzip", "br"):
            response = await client.get(url, headers={"Accept-Encoding": encoding})
            assert response.status_code == 200
            payload = response.content
            if not payload.startswith(b"ISCDCSD\0"):
                # httpx's Brotli decoder is optional in the website environment.
                from iscdc.cell_type_visualization import _identity_payload

                payload = _identity_payload(encoding, payload, 32 + 10 * source["n_obs"])
            assert payload.startswith(b"ISCDCSD\0")
            assert (
                await client.head(url, headers={"Accept-Encoding": encoding})
            ).status_code == 200
        assert (await client.get(url.replace("/spatialglue/", "/rna/"))).status_code == 404
        assert (await client.get(url.replace("/spatialglue/", "/unknown/"))).status_code == 404
        assert (await client.get(url.replace("run-1", "other"))).status_code == 404
        public = await client.get(f"/api/databases/{source['dataset_id']}")
        assert "SpatialGlue" not in public.text
    base = domain_directory(root, source["dataset_id"], "spatialglue")
    (base / "generations/run-1/features/protein.txt.gz").write_bytes(b"broken")
    assert load_spatial_domain_visualizations(root, [source], method_family="spatialglue") == {}
