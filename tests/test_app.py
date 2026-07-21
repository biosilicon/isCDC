from __future__ import annotations

import httpx
import pytest

from iscdc.app import create_app
from iscdc.importer import import_dataset

pytestmark = pytest.mark.anyio


def _app_with_dataset(settings, write_h5mu, write_metadata):  # noqa: ANN202
    import_dataset(write_h5mu(), write_metadata(), settings)
    return create_app(settings)


async def test_pages_and_filters(settings, write_h5mu, write_metadata):
    app = _app_with_dataset(settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/")).status_code == 200
        response = await client.get("/datasets?q=kidney&organism=Homo%20sapiens&modality=rna")
        assert response.status_code == 200
        assert "Test RNA and protein dataset" in response.text
        assert "1 matching dataset" in response.text

        empty = await client.get("/datasets?tissue=brain")
        assert "No matching datasets" in empty.text


async def test_detail_and_missing_page(settings, write_h5mu, write_metadata):
    app = _app_with_dataset(settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get("/datasets/test_rna_protein")
        assert detail.status_code == 200
        assert "Test assay" in detail.text
        assert (await client.get("/datasets/unknown")).status_code == 404


async def test_api_list_detail_and_pagination(settings, write_h5mu, write_metadata):
    app = _app_with_dataset(settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/datasets?technology=Test%20assay&limit=1&offset=0")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["dataset_id"] == "test_rna_protein"
        assert set(payload["items"][0]["downloads"]) == {
            "h5mu",
            "metadata",
            "manifest",
            "validation",
            "checksum",
        }
        assert (await client.get("/api/datasets/test_rna_protein")).status_code == 200
        assert (await client.get("/api/datasets/unknown")).status_code == 404


async def test_downloads_use_safe_fixed_names(settings, write_h5mu, write_metadata):
    app = _app_with_dataset(settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for kind in ("h5mu", "metadata", "manifest", "validation", "checksum"):
            response = await client.get(f"/datasets/test_rna_protein/downloads/{kind}")
            assert response.status_code == 200
            assert "test_rna_protein" in response.headers["content-disposition"]
        assert (await client.get("/datasets/test_rna_protein/downloads/unknown")).status_code == 404
