from __future__ import annotations

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import anndata as ad
import h5py
import httpx
import mudata as md
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from iscdc.app import create_app
from iscdc.importer import import_dataset
from iscdc.molecular_prepare import (
    audit_generation,
    catalogue,
    prepare_batch,
    prepare_generation,
    publish_batch,
    sha256,
)
from iscdc.molecular_visualization import (
    HEADER,
    MAGIC,
    encode_array,
    feature_search,
    load_publication,
    read_vector,
    safe_path,
)


def _decode(payload):
    magic, version, kind, count, name = HEADER.unpack(payload[:32])
    assert magic == MAGIC and version == 1
    dtype = np.dtype(name.rstrip(b"\0").decode())
    values = np.frombuffer(payload, dtype=dtype, offset=32, count=count)
    states = np.frombuffer(payload, dtype=np.uint8, offset=32 + count * dtype.itemsize)
    return kind, values, states


def _source(settings, encoding):
    x = np.array([[5, 0, -2], [2**53 + 1, 7, 0]], dtype=np.int64)
    if encoding == "csr":
        x = sparse.csr_matrix(x)
    elif encoding == "csc":
        x = sparse.csc_matrix(x)
    rna = ad.AnnData(
        x,
        obs=pd.DataFrame(index=["b", "a"]),
        var=pd.DataFrame(
            {"gene_symbol": ["shared", "shared", 'a/%_"']}, index=["ENSG1", "ENSG2", "chr1:1-2"]
        ),
    )
    protein = ad.AnnData(
        np.array([[0.123456789012345, np.nan], [np.inf, -3.0]]),
        obs=pd.DataFrame(index=["a", "c"]),
        var=pd.DataFrame(index=["CD3", "CD4"]),
    )
    rna.uns["assay"] = {"value_type": "counts"}
    protein.uns["assay"] = {"value_type": "background_corrected_intensity"}
    data = md.MuData({"rna": rna, "protein": protein})
    data.obs["sample_id"] = ["B" if i == "c" else "A" for i in data.obs_names]
    data.obsm["spatial"] = np.arange(data.n_obs * 2).reshape(-1, 2).astype(float)
    directory = settings.data_root / "molecular_test"
    directory.mkdir(parents=True)
    path = directory / "dataset.h5mu"
    data.write_h5mu(path)
    return {
        "dataset_id": "molecular_test",
        "storage_dir": "molecular_test",
        "sha256": sha256(path),
        "n_obs": data.n_obs,
        "coordinate_dimensions": 2,
        "sample_ids": ["A", "B"],
    }, data


@pytest.mark.parametrize("encoding", ["dense", "csr", "csc"])
def test_prepare_preserves_values_alignment_precision_and_missingness(settings, tmp_path, encoding):
    record, source = _source(settings, encoding)
    root = tmp_path / "prepared"
    item = prepare_generation(settings, record, root)
    manifest = audit_generation(settings, record, root, item)
    modalities = {m["name"]: m for m in manifest["modalities"]}
    directory = root / item["directory"]
    for sample in manifest["samples"]:
        names = source.obs_names[source.obs.sample_id == sample["id"]]
        for name, modality in modalities.items():
            adata = source.mod[name]
            for feature in range(adata.n_vars):
                kind, values, states = _decode(
                    read_vector(directory, modality, sample["key"], feature)
                )
                assert kind == 2
                for i, obs in enumerate(names):
                    if obs not in adata.obs_names:
                        assert states[i] == 1
                    else:
                        expected = adata[obs, feature].X
                        expected = (expected.toarray() if sparse.issparse(expected) else expected)[
                            0, 0
                        ]
                        if np.isfinite(expected):
                            assert values[i] == expected and states[i] == 0
                        else:
                            assert states[i] == 2
    index = directory / modalities["rna"]["index"]
    assert [r["id"] for r in feature_search(index, "SHARED", 0, 50)["items"]] == ["ENSG1", "ENSG2"]
    assert feature_search(index, 'a/%_"', 0, 50)["items"][0]["id"] == "chr1:1-2"
    assert len(feature_search(index, "EN", 0, 50)["items"]) == 2
    page = feature_search(index, "", 0, 1)
    assert page["nextOffset"] == 1
    assert feature_search(index, "", 2, 1)["nextOffset"] is None
    assert feature_search(index, "not present", 0, 50)["items"] == []
    exact_page = feature_search(index, "shared", 0, 1)
    assert exact_page["items"][0]["id"] == "ENSG1" and exact_page["nextOffset"] == 1
    assert feature_search(index, "shared", 1, 1)["items"][0]["id"] == "ENSG2"
    assert feature_search(index, "shared", 2, 1)["items"] == []
    assert feature_search(index, "ENSG", 1, 1)["items"][0]["id"] == "ENSG2"


def test_binary_golden_vector_and_unsafe_paths(tmp_path):
    payload = encode_array(np.array([2**53 + 1], dtype=np.int64), kind=2, states=[0])
    assert payload[:16].hex() == "49534344434d4f000100020001000000"
    assert _decode(payload)[1][0] == 2**53 + 1
    for value in ("../outside", "/outside"):
        with pytest.raises(ValueError):
            safe_path(tmp_path, value)
    with pytest.raises(ValueError):
        encode_array(np.array([1]), kind=2, states=[3])


def test_audit_detects_corruption_and_source_change(settings, tmp_path):
    record, _ = _source(settings, "csr")
    root = tmp_path / "prepared"
    item = prepare_generation(settings, record, root)
    source = settings.data_root / record["storage_dir"] / "dataset.h5mu"
    with h5py.File(source, "r+") as handle:
        handle.attrs["changed"] = True
    with pytest.raises(ValueError, match="Source checksum"):
        audit_generation(settings, record, root, item)
    manifest = json.loads((root / item["directory"] / "manifest.json").read_text())
    artifact = root / item["directory"] / manifest["modalities"][0]["matrix"]
    with artifact.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="artifact checksum"):
        audit_generation(settings, record, root, item, check_source=False)


def _published(settings, write_h5mu, write_metadata, tmp_path):
    settings = replace(
        settings, analytics_enabled=False, molecular_visualization_root=tmp_path / "published"
    )
    import_dataset(write_h5mu(), write_metadata(), settings)
    prepared = tmp_path / "prepared"
    batch = prepare_batch(settings, prepared)
    assert batch["status"] == "prepared"
    publication = publish_batch(settings, prepared, settings.molecular_visualization_root)
    record = catalogue(settings)[0]
    item = publication["datasets"][record["dataset_id"]]
    manifest = json.loads(
        (settings.molecular_visualization_root / item["directory"] / "manifest.json").read_text()
    )
    return settings, prepared, manifest


@pytest.mark.anyio
async def test_page_and_routes_use_only_prepared_artifacts(
    settings, write_h5mu, write_metadata, tmp_path, monkeypatch
):
    settings, _, manifest = _published(settings, write_h5mu, write_metadata, tmp_path)
    real_open = h5py.File

    def guarded_open(path, *args, **kwargs):
        assert not str(path).endswith(".h5mu"), "Website must not open a source matrix"
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", guarded_open)
    # Startup must not open even the prepared value matrices.
    monkeypatch.setattr(h5py, "File", lambda *a, **k: pytest.fail("HDF5 opened during startup"))
    app = create_app(settings)
    monkeypatch.setattr(h5py, "File", guarded_open)
    did, generation = manifest["dataset_id"], manifest["generation_id"]
    prefix = f"/databases/{did}/molecular-visualization/{generation}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        page = await client.get(f"/databases/{did}")
        assert page.status_code == 200
        assert 'data-visualization-mode="molecular" aria-pressed="true"' in page.text
        assert "Molecular distribution" in page.text
        search = await client.get(prefix + "/modalities/rna/features", params={"q": "gene"})
        assert len(search.json()["items"]) == 2
        assert (await client.get(prefix + "/modalities/rna/features?limit=101")).status_code == 422
        overflow = await client.get(prefix + "/modalities/rna/features?offset=99999999999999999999")
        assert overflow.status_code == 422
        assert (await client.get(prefix + "/modalities/rna/features?offset=999999")).json()[
            "items"
        ] == []
        coordinates = await client.get(prefix + "/samples/s0/coordinates")
        assert HEADER.unpack(coordinates.content[:32])[2:4] == (1, 2)
        route = prefix + "/samples/s0/modalities/rna/features/0"
        values = await client.get(route, headers={"Accept-Encoding": "gzip"})
        assert values.status_code == 200 and values.headers["content-encoding"] == "gzip"
        np.testing.assert_array_equal(_decode(values.content)[1], [1, 0])
        assert (await client.head(route)).status_code == 200
        assert (
            await client.get(
                route, headers={"If-None-Match": values.headers["etag"], "Accept-Encoding": "gzip"}
            )
        ).status_code == 304
        for suffix in (
            "/samples/unknown/coordinates",
            "/samples/s0/modalities/rna/features/999",
            "/samples/s0/modalities/unknown/features/0",
            "/samples/s0/modalities/rna/features/-1",
        ):
            assert (await client.get(prefix + suffix)).status_code == 404
        assert (await client.get(route.replace(generation, "old"))).status_code == 404
        assert (
            await client.get(route, headers={"Accept-Encoding": "identity;q=0,gzip;q=0"})
        ).status_code == 406


def test_publication_failure_preserves_previous_index(
    settings, write_h5mu, write_metadata, tmp_path, monkeypatch
):
    settings, prepared, manifest = _published(settings, write_h5mu, write_metadata, tmp_path)
    root = settings.molecular_visualization_root
    before = (root / "publication.json").read_bytes()
    # Failed audit must never update the publication pointer.
    (prepared / "batch.json").write_text('{"scope":"all","records":[],"datasets":{}}')
    with pytest.raises(ValueError, match="Catalogue changed"):
        publish_batch(settings, prepared, root)
    assert (root / "publication.json").read_bytes() == before
    assert load_publication(
        root, [SimpleNamespace(dataset_type="full", dataset_id=manifest["dataset_id"])]
    )
    (root / "publication.json").write_text("broken")
    assert load_publication(root, []) == {}


def test_resume_reuses_verified_generation(settings, write_h5mu, write_metadata, tmp_path):
    import_dataset(write_h5mu(), write_metadata(), settings)
    root = tmp_path / "prepared"
    first = prepare_batch(settings, root)
    second = prepare_batch(settings, root)
    assert first["datasets"] == second["datasets"]
    record = catalogue(settings)[0]
    source = settings.data_root / record["storage_dir"] / "dataset.h5mu"
    with h5py.File(source, "r+") as handle:
        handle.attrs["changed"] = True
    assert prepare_batch(settings, root)["status"] == "failed"


def test_copy_failure_cannot_replace_publication(
    settings, write_h5mu, write_metadata, tmp_path, monkeypatch
):
    settings, prepared, _ = _published(settings, write_h5mu, write_metadata, tmp_path)
    root = settings.molecular_visualization_root
    before = (root / "publication.json").read_bytes()
    record = catalogue(settings)[0]
    batch = json.loads((prepared / "batch.json").read_text())
    batch["datasets"][record["dataset_id"]] = prepare_generation(settings, record, prepared)
    (prepared / "batch.json").write_text(json.dumps(batch))

    def fail_copy(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(shutil, "copytree", fail_copy)
    with pytest.raises(OSError, match="disk failure"):
        publish_batch(settings, prepared, root)
    assert (root / "publication.json").read_bytes() == before
