from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import httpx
import pytest

from iscdc import cell_type_visualization as ct
from iscdc.spatial_domain_annotation import (
    DomainInferenceError,
    eligibility,
    estimated_memory_bytes,
    load_parameters,
)
from iscdc.spatial_domain_visualization import (
    DomainVisualizationError,
    assignment_bytes,
    build_point_representations,
    decode_points,
    encode_points,
    file_record,
    load_spatial_domain_visualization,
    load_spatial_domain_visualizations,
    method_for_resolution,
    obs_order_sha256,
    publish_generation,
)


def generation(record=None):
    record = record or {
        "dataset_id": "example",
        "dataset_type": "full",
        "sha256": "a" * 64,
        "n_obs": 3,
        "sample_ids": ["section A"],
        "coordinate_dimensions": 2,
        "spatial_unit": "single_cell",
        "coordinate_unit": "pixel",
    }
    n = record["n_obs"]
    ids = [f"cell_{i + 1}" for i in range(n)]
    codes = [1] * (n - 1) + [0]
    key, sample_id = "sample_0", record["sample_ids"][0]
    payload = encode_points(range(n), range(n), codes)
    files = {
        "assignments.tsv.gz": assignment_bytes(
            ids, [sample_id] * n, codes, [""] * (n - 1) + ["zero_counts"]
        )
    }
    reps = {}
    for encoding, content in build_point_representations(payload).items():
        name = f"points/{key}{ct._ENCODING_SUFFIXES[encoding]}"
        files[name] = content
        reps[encoding] = {
            **file_record(name, content),
            "encoding": encoding,
            "content_size": len(payload),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        }
    report = {
        "report_version": 1,
        "dataset_id": record["dataset_id"],
        "generation_id": "run-1",
        "source_sha256": record["sha256"],
        "status": "passed",
        "samples": {
            sample_id: {
                "n_domains": 1,
                "analyzed": n - 1,
                "not_analyzed": 1,
                "parameters": {"resolution": 1.0},
                "preprocessing": {"input": "RNA counts"},
                "warnings": [],
                "domains": {"1": {"count": n - 1, "spatial_components": 1}},
                "selected_genes": ["gene_A", "gene_B"],
                "selected_genes_sha256": obs_order_sha256(["gene_A", "gene_B"]),
            }
        },
    }
    files["report.json"] = json.dumps(report).encode()
    manifest = {
        "manifest_version": 1,
        "dataset_id": record["dataset_id"],
        "generation_id": "run-1",
        "generated_at": "2026-09-19T00:00:00+00:00",
        "source": {
            "sha256": record["sha256"],
            "obs_order_sha256": obs_order_sha256(ids),
            "observation_count": n,
            "sample_ids": [sample_id],
            "spatial_unit": record["spatial_unit"],
        },
        "method": method_for_resolution(record["spatial_unit"]),
        "coordinates": {"system": "cartesian", "unit": record["coordinate_unit"], "y_axis": "up"},
        "samples": [
            {
                "key": key,
                "id": sample_id,
                "count": n,
                "bounds": [0.0, 0.0, float(n - 1), float(n - 1)],
                "representations": reps,
                "categories": [
                    {"code": 1, "label": "Domain 1", "color": "#112233", "count": n - 1},
                    {"code": 0, "label": "Not analyzed", "color": "#889999", "count": 1},
                ],
            }
        ],
        "report": file_record("report.json", files["report.json"]),
        "assignments": file_record("assignments.tsv.gz", files["assignments.tsv.gz"]),
        "provenance": {
            "environment_lock_sha256": "b" * 64,
            "packages": {"pybanksy": "1.3.5"},
            "parameters": {"resolution": 1.0},
            "input_modality": "rna",
        },
    }
    return record, manifest, files


def test_domain_format_cannot_be_mistaken_for_cell_types():
    payload = encode_points([1, 2], [3, 4], [1, 0])
    assert decode_points(payload).type_ids == (1, 0)
    with pytest.raises(ct.CellTypeVisualizationError, match="magic"):
        ct.decode_points(payload)
    with pytest.raises(DomainVisualizationError, match="magic"):
        decode_points(ct.encode_points([1], [1], [0]))


@pytest.mark.parametrize(
    "resolution,method",
    [("single_cell", "BANKSY"), ("near_cellular", "GraphST"), ("spot_level", "GraphST")],
)
def test_method_uses_resolution_not_point_count(resolution, method):
    assert method_for_resolution(resolution) == method
    record = {
        "spatial_unit": resolution,
        "coordinate_dimensions": 2,
        "n_obs": 700000,
        "modalities": {"rna": {"value_type": "counts"}},
    }
    assert eligibility(record) is None
    record["modalities"]["rna"]["value_type"] = "normalized"
    assert eligibility(record) == "rna_preprocessing_not_verified"
    record["modalities"] = {}
    assert eligibility(record) == "missing_rna"


def test_legacy_units_and_unknown_parameters_rejected(tmp_path):
    with pytest.raises(DomainVisualizationError):
        method_for_resolution("bin")
    path = tmp_path / "config.yaml"
    path.write_text("defaults: {method: BANKSY}\n")
    with pytest.raises(DomainInferenceError):
        load_parameters(path, "example")
    path.write_text(
        "defaults: {resolution: 0.5}\ndatasets:\n  example:\n"
        "    samples:\n      s1: {resolution: 0.8}\n"
    )
    assert load_parameters(path, "example", "s1")["resolution"] == 0.8
    assert load_parameters(path, "example", "s2")["resolution"] == 0.5
    assert estimated_memory_bytes(20000, 3000, "GraphST") > 32 * 2**30


def test_atomic_generation_roundtrip_and_source_binding(tmp_path):
    record, manifest, files = generation()
    snapshot = publish_generation(tmp_path, record, manifest, files)
    assert snapshot.samples["sample_0"].count == 3
    assert snapshot.manifest["method"] == "BANKSY"
    assert load_spatial_domain_visualization(tmp_path, record).generation_id == "run-1"
    with pytest.raises(DomainVisualizationError, match="immutable"):
        publish_generation(tmp_path, record, manifest, files)
    stale = {**record, "sha256": "f" * 64}
    assert load_spatial_domain_visualizations(tmp_path, [stale]) == {}
    ct.publish_failure(tmp_path, record["dataset_id"], "resource budget exceeded")
    assert load_spatial_domain_visualizations(tmp_path, [record]) == {}
    assert (tmp_path / "example/generations/run-1/manifest.json").exists()


@pytest.mark.parametrize(
    "mutation", ["method", "counts", "labels", "order", "confidence", "unsafe"]
)
def test_inconsistent_artifacts_never_become_visible(tmp_path, mutation):
    record, manifest, files = generation()
    if mutation == "method":
        manifest["method"] = "GraphST"
    elif mutation == "counts":
        manifest["samples"][0]["categories"][0]["count"] = 7
    elif mutation == "labels":
        manifest["samples"][0]["categories"][0]["label"] = "T cell"
    elif mutation == "order":
        manifest["source"]["obs_order_sha256"] = "f" * 64
    elif mutation == "confidence":
        path = manifest["samples"][0]["representations"]["identity"]["path"]
        payload = b"ISCDCSD\0" + ct.encode_points([0, 1, 2], [0, 1, 2], [1, 1, 0], [1, 1, 0])[8:]
        files[path] = payload
        manifest["samples"][0]["representations"]["identity"].update(file_record(path, payload))
    else:
        files["../escape"] = b"bad"
    with pytest.raises(DomainVisualizationError):
        publish_generation(tmp_path, record, manifest, files)
    assert not (tmp_path / "example/status.json").exists()


def test_corrupt_one_dataset_does_not_hide_others(tmp_path):
    record, manifest, files = generation()
    publish_generation(tmp_path, record, manifest, files)
    bad = {**record, "dataset_id": "bad"}
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad/status.json").write_text('{"state":"failure"}')
    assert list(load_spatial_domain_visualizations(tmp_path, [bad, record])) == ["example"]


@pytest.mark.anyio
async def test_domain_only_page_and_internal_endpoint_leave_public_api_unchanged(
    settings,
    write_h5mu,
    write_metadata,
):
    from iscdc.app import create_app
    from iscdc.importer import import_dataset
    from iscdc.spatial_domain_annotation import catalogue_records

    settings = replace(
        settings, spatial_domain_visualization_root=settings.data_root.parent / "domains"
    )
    import_dataset(write_h5mu(), write_metadata(), settings)
    record = catalogue_records(settings)[0]
    record, manifest, files = generation(record)
    publish_generation(settings.spatial_domain_visualization_root, record, manifest, files)
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        page = await client.get(f"/databases/{record['dataset_id']}")
        assert page.status_code == 200
        assert "Spatial domains" in page.text
        assert 'id="spatial-domain-method-modal"' in page.text
        assert 'id="cell-type-method-modal"' not in page.text
        path = f"/databases/{record['dataset_id']}/spatial-domain-visualization/run-1/sample_0"
        points = await client.get(path, headers={"Accept-Encoding": "identity"})
        assert points.status_code == 200
        assert points.headers["content-type"] == "application/vnd.iscdc.spatial-domain-points"
        assert decode_points(points.content).point_count == record["n_obs"]
        assert (await client.head(path)).status_code == 200
        assert (await client.get(path.replace("run-1", "wrong"))).status_code == 404
        assert (
            await client.get(path, headers={"Accept-Encoding": "identity;q=0, gzip;q=0, br;q=0"})
        ).status_code == 406
        public = await client.get(f"/api/databases/{record['dataset_id']}")
        assert public.status_code == 200
        assert "spatial_domain" not in json.dumps(public.json())


def test_resource_preflight_rejects_before_matrix_allocation(monkeypatch):
    import sys
    from types import SimpleNamespace

    from iscdc.spatial_domain_annotation import DEFAULTS, check_resources

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=32 * 2**30)),
    )
    with pytest.raises(DomainInferenceError, match="no downsampling"):
        check_resources(20000, 3000, "GraphST", DEFAULTS)


@pytest.mark.anyio
async def test_both_modes_and_corrupt_domain_preserve_cell_types(
    settings,
    write_h5mu,
    write_metadata,
):
    from test_app import _publish_test_cell_type_visualization

    from iscdc.app import create_app
    from iscdc.importer import import_dataset
    from iscdc.spatial_domain_annotation import catalogue_records

    settings = replace(
        settings,
        cell_type_visualization_root=settings.data_root.parent / "cells",
        spatial_domain_visualization_root=settings.data_root.parent / "domains",
    )
    imported = import_dataset(write_h5mu(), write_metadata(), settings)
    _publish_test_cell_type_visualization(settings.cell_type_visualization_root, imported.sha256)
    record, manifest, files = generation(catalogue_records(settings)[0])
    snapshot = publish_generation(
        settings.spatial_domain_visualization_root, record, manifest, files
    )
    for corrupt in (False, True):
        if corrupt:
            snapshot.samples["sample_0"].representations["identity"].path.write_bytes(b"broken")
        app = create_app(settings)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(f"/databases/{record['dataset_id']}")
        assert response.status_code == 200
        assert 'id="cell-type-method-modal"' in response.text
        assert ('id="spatial-domain-method-modal"' in response.text) is (not corrupt)
        assert 'value="cell_type"' in response.text
        assert ('value="spatial_domain"' in response.text) is (not corrupt)


def test_observation_digest_uses_unambiguous_project_framing():
    from iscdc.cell_type_annotation import _obs_order_sha256

    assert obs_order_sha256(["a\nb", "c"]) != obs_order_sha256(["a", "b\nc"])
    assert obs_order_sha256(["cell_1", "cell_2"]) == _obs_order_sha256(["cell_1", "cell_2"])


def test_worker_receives_seed_and_cpu_headroom_before_python_starts(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from iscdc import spatial_domain_annotation as domain

    captured = {}

    class FinishedProcess:
        pid = 12345
        returncode = 0

        def __init__(self, command, *, env):
            captured.update(command=command, env=env)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def poll(self):
            return self.returncode

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "iscdc-spatial-domain")
    monkeypatch.delenv("ISCDC_DOMAIN_WORKER", raising=False)
    monkeypatch.setattr(domain.os, "sched_getaffinity", lambda pid: set(range(16)))
    monkeypatch.setattr(domain.subprocess, "Popen", FinishedProcess)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda pid: None))
    args = SimpleNamespace(
        command="generate-spatial-domain-visualization",
        dataset_id="example",
        config=None,
        output_root=tmp_path,
        force=False,
    )
    assert domain.execute_cli(args) == 0
    assert captured["env"]["PYTHONHASHSEED"] == "42"
    assert captured["env"]["ISCDC_DOMAIN_WORKER"] == "1"
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        assert captured["env"][variable] == "8"
