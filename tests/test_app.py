from __future__ import annotations

from copy import deepcopy
from urllib.parse import quote

import httpx
import mudata as md
import numpy as np
import pytest
import yaml
from sqlalchemy import select

from iscdc.analytics import VisitEvent
from iscdc.app import create_app
from iscdc.database import create_database_engine, create_session_factory
from iscdc.importer import import_dataset
from iscdc.models import Dataset, Modality
from iscdc.splitter import spatial_split

pytestmark = pytest.mark.anyio


def _normalize(value):  # noqa: ANN001, ANN202
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_normalize(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _write_product_metadata(path, destination):  # noqa: ANN001, ANN202
    product = md.read_h5mu(path)
    try:
        values = {
            "database": _normalize(product.uns["database"]),
            "sample_ids": sorted(set(product.obs["sample_id"].astype(str))),
            "modalities": {
                name: _normalize(adata.uns["assay"]) for name, adata in product.mod.items()
            },
            "title": f"Derived {product.uns['database']['dataset_type']} dataset",
            "description": "A deterministic derived dataset used for web tests.",
            "keywords": ["derived", "test"],
            "license": None,
            "publication": None,
        }
    finally:
        product.file.close()
    destination.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return destination


def _app_with_database(settings, write_h5mu, write_metadata):  # noqa: ANN202
    import_dataset(write_h5mu(), write_metadata(), settings)
    return create_app(settings)


def _app_with_challenge(
    tmp_path,
    settings,
    write_h5mu,
    write_metadata,
    *,
    split_id="web_split_v1",
    challenge_type="same_slice",
    imported_sides=("train", "test"),
):  # noqa: ANN202
    source_path = write_h5mu()
    import_dataset(source_path, write_metadata(), settings)
    config_path = tmp_path / "web-spatial.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.1",
                "split_id": split_id,
                "challenge_type": challenge_type,
                "feature_merge_policy": "preserve",
                "source": str(source_path),
                "output_dir": "web-derived-output",
                "train": {"dataset_id": "web_train"},
                "test": {
                    "dataset_id": "web_test",
                    "regions": [
                        {
                            "sample_id": "sample_01",
                            "x_min": 2,
                            "x_max": 2,
                            "y_min": 3,
                            "y_max": 3,
                        }
                    ],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    train_path, test_path = spatial_split(config_path)
    products = {
        "train": (train_path, tmp_path / "web-train.metadata.yaml"),
        "test": (test_path, tmp_path / "web-test.metadata.yaml"),
    }
    for side in imported_sides:
        product_path, metadata_path = products[side]
        import_dataset(product_path, _write_product_metadata(product_path, metadata_path), settings)
    return create_app(settings)


async def test_home_and_database_pages_use_new_entry_points(
    settings, write_h5mu, write_metadata
):
    app = _app_with_database(settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        home = await client.get("/")
        assert home.status_code == 200
        assert "Browse databases" in home.text
        assert "Browse challenges" in home.text

        response = await client.get(
            "/databases?q=kidney&organism=Homo%20sapiens&modality=rna"
        )
        assert response.status_code == 200
        assert "Test RNA and protein dataset" in response.text
        assert "1 matching database" in response.text

        empty = await client.get("/databases?tissue=brain")
        assert "No matching databases" in empty.text


async def test_histone_modality_is_imported_and_filterable(
    metadata_values, settings, write_h5mu, write_metadata
):
    values = deepcopy(metadata_values)
    values["database"]["histone_mark"] = "H3K27me3"
    values["modalities"]["histone"] = values["modalities"].pop("protein")
    values["title"] = "Test RNA and histone dataset"
    import_dataset(
        write_h5mu(second_modality_name="histone"),
        write_metadata(values),
        settings,
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/databases", params={"modality": "histone"})
        assert page.status_code == 200
        assert "Test RNA and histone dataset" in page.text

        response = await client.get("/api/databases", params={"modality": "histone"})
        assert response.status_code == 200
        assert response.json()["items"][0]["modalities"][0]["name"] == "histone"


async def test_database_detail_rejects_non_full_and_missing_entries(
    tmp_path, settings, write_h5mu, write_metadata
):
    app = _app_with_challenge(tmp_path, settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get("/databases/test_rna_protein")
        assert detail.status_code == 200
        assert "Test assay" in detail.text
        assert (await client.get("/databases/web_train")).status_code == 404
        assert (await client.get("/databases/unknown")).status_code == 404


async def test_challenge_groups_pair_and_shows_complete_file_metadata(
    tmp_path, settings, write_h5mu, write_metadata
):
    app = _app_with_challenge(tmp_path, settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/challenges")
        assert listing.status_code == 200
        assert "1 matching challenge" in listing.text
        assert listing.text.count('class="card dataset-card"') == 1
        assert "web_split_v1" in listing.text
        assert "Same slice" in listing.text
        assert "Derived train dataset" in listing.text
        assert "Derived test dataset" in listing.text

        all_filters = await client.get(
            "/challenges",
            params={
                "q": "",
                "organism": "",
                "tissue": "",
                "modality": "",
                "technology": "",
                "spatial_unit": "",
                "challenge_type": "",
            },
        )
        assert all_filters.status_code == 200
        assert "1 matching challenge" in all_filters.text

        detail = await client.get("/challenges/web_split_v1")
        assert detail.status_code == 200
        assert "Training data" in detail.text
        assert "Test data" in detail.text
        assert "Feature merge policy" in detail.text
        assert "Same slice" in detail.text
        assert "/downloads/web_train/h5mu" in detail.text
        assert "/downloads/web_test/h5mu" in detail.text

    analytics = app.state.analytics
    assert analytics is not None
    with analytics.session_factory() as session:
        event_types = session.scalars(select(VisitEvent.event_type)).all()
    assert "challenge_detail_view" in event_types


async def test_challenge_filter_matches_one_side_but_returns_both(
    tmp_path, settings, write_h5mu, write_metadata
):
    _app_with_challenge(tmp_path, settings, write_h5mu, write_metadata)
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        test = session.get(Dataset, "web_test")
        assert test is not None
        test.tissue = "lung"
        session.commit()
    engine.dispose()

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/challenges?tissue=lung")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["train"]["dataset_id"] == "web_train"
        assert payload["items"][0]["test"]["dataset_id"] == "web_test"

        matching = await client.get("/api/challenges?challenge_type=same_slice")
        assert matching.status_code == 200
        assert matching.json()["total"] == 1
        assert matching.json()["items"][0]["challenge_type"] == "same_slice"
        empty = await client.get("/api/challenges?challenge_type=cross_subject")
        assert empty.status_code == 200
        assert empty.json()["total"] == 0


@pytest.mark.parametrize(
    ("imported_sides", "status", "missing_text"),
    [
        (("test",), "missing_train", "Training data is not available"),
        (("train",), "missing_test", "Test data is not available"),
    ],
)
async def test_incomplete_challenge_is_visible(
    tmp_path,
    settings,
    write_h5mu,
    write_metadata,
    imported_sides,
    status,
    missing_text,
):
    app = _app_with_challenge(
        tmp_path,
        settings,
        write_h5mu,
        write_metadata,
        imported_sides=imported_sides,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = (await client.get("/api/challenges/web_split_v1")).json()
        assert payload["status"] == status
        assert payload["challenge_type"] == "same_slice"
        detail = await client.get("/challenges/web_split_v1")
        assert detail.status_code == 200
        assert missing_text in detail.text


async def test_duplicate_challenge_side_is_an_integrity_error(
    tmp_path, settings, write_h5mu, write_metadata
):
    _app_with_challenge(
        tmp_path,
        settings,
        write_h5mu,
        write_metadata,
        imported_sides=("train",),
    )
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        original = session.get(Dataset, "web_train")
        assert original is not None
        duplicate = Dataset(
            dataset_id="web_train_duplicate",
            schema_version=original.schema_version,
            dataset_type=original.dataset_type,
            title=original.title,
            description=original.description,
            source=deepcopy(original.source),
            organism=deepcopy(original.organism),
            tissue=deepcopy(original.tissue),
            spatial_unit=original.spatial_unit,
            coordinate_unit=original.coordinate_unit,
            pairing_type=original.pairing_type,
            derivation=deepcopy(original.derivation),
            split_id=original.split_id,
            sample_ids=deepcopy(original.sample_ids),
            keywords=deepcopy(original.keywords),
            license=deepcopy(original.license),
            publication=deepcopy(original.publication),
            additional_metadata=deepcopy(original.additional_metadata),
            n_obs=original.n_obs,
            coordinate_dimensions=original.coordinate_dimensions,
            file_size=original.file_size,
            sha256=original.sha256,
            storage_dir="web_train_duplicate",
            validation_warning_count=original.validation_warning_count,
            imported_at=original.imported_at,
            modalities=[
                Modality(
                    name=modality.name,
                    technology=deepcopy(modality.technology),
                    value_type=modality.value_type,
                    n_obs=modality.n_obs,
                    n_vars=modality.n_vars,
                )
                for modality in original.modalities
            ],
        )
        session.add(duplicate)
        session.commit()
    engine.dispose()

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/challenges")
        assert page.status_code == 500
        assert "multiple train datasets" in page.text
        api = await client.get("/api/challenges")
        assert api.status_code == 500
        assert "multiple train datasets" in api.json()["detail"]


async def test_derived_file_without_split_id_is_an_integrity_error(
    tmp_path, settings, write_h5mu, write_metadata
):
    _app_with_challenge(
        tmp_path,
        settings,
        write_h5mu,
        write_metadata,
        imported_sides=("train",),
    )
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        train = session.get(Dataset, "web_train")
        assert train is not None
        train.split_id = None
        session.commit()
    engine.dispose()

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/challenges")
        assert page.status_code == 500
        assert "does not define a split_id" in page.text
        api = await client.get("/api/challenges")
        assert api.status_code == 500
        assert "does not define a split_id" in api.json()["detail"]


@pytest.mark.parametrize("mutation", ["missing", "mismatch"])
async def test_invalid_challenge_type_is_an_integrity_error(
    mutation, tmp_path, settings, write_h5mu, write_metadata
):
    _app_with_challenge(tmp_path, settings, write_h5mu, write_metadata)
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        test = session.get(Dataset, "web_test")
        assert test is not None and test.derivation is not None
        derivation = deepcopy(test.derivation)
        if mutation == "missing":
            derivation.pop("challenge_type")
        else:
            derivation["challenge_type"] = "cross_subject"
        test.derivation = derivation
        session.commit()
    engine.dispose()

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/challenges")
        assert response.status_code == 500
        expected = "valid challenge_type" if mutation == "missing" else "inconsistent"
        assert expected in response.json()["detail"]


async def test_special_split_id_is_addressable(
    tmp_path, settings, write_h5mu, write_metadata
):
    split_id = "nested/split name"
    app = _app_with_challenge(
        tmp_path,
        settings,
        write_h5mu,
        write_metadata,
        split_id=split_id,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get(f"/challenges/{quote(split_id)}")
        assert detail.status_code == 200
        assert split_id in detail.text
        api = await client.get(f"/api/challenges/{quote(split_id)}")
        assert api.status_code == 200
        assert api.json()["split_id"] == split_id


async def test_new_apis_and_pagination(settings, write_h5mu, write_metadata):
    app = _app_with_database(settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/databases?technology=Test%20assay&limit=1&offset=0")
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
        assert (await client.get("/api/databases/test_rna_protein")).status_code == 200
        assert (await client.get("/api/databases/unknown")).status_code == 404
        assert (await client.get("/api/challenges/unknown")).status_code == 404


async def test_downloads_use_new_route_and_safe_fixed_names(
    settings, write_h5mu, write_metadata
):
    app = _app_with_database(settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for kind in ("h5mu", "metadata", "manifest", "validation", "checksum"):
            response = await client.get(f"/downloads/test_rna_protein/{kind}")
            assert response.status_code == 200
            assert "test_rna_protein" in response.headers["content-disposition"]
        assert (await client.get("/downloads/test_rna_protein/unknown")).status_code == 404


async def test_old_dataset_routes_are_removed(settings, write_h5mu, write_metadata):
    app = _app_with_database(settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/datasets")).status_code == 404
        assert (await client.get("/datasets/test_rna_protein")).status_code == 404
        assert (await client.get("/api/datasets")).status_code == 404
        assert (await client.get("/api/datasets/test_rna_protein")).status_code == 404


async def test_database_api_and_filters_preserve_multivalue_metadata(
    settings, write_h5mu, write_metadata
):
    import_dataset(write_h5mu(), write_metadata(), settings)
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        database = session.get(Dataset, "test_rna_protein")
        assert database is not None
        database.source = ["SOURCE_A", "SOURCE_B"]
        database.organism = ["Homo sapiens", "Mus musculus"]
        database.tissue = ["kidney", "lung"]
        database.modalities[0].technology = ["Test assay", "Second assay"]
        session.commit()
    engine.dispose()

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/databases?organism=Mus%20musculus&technology=Second%20assay"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["source"] == ["SOURCE_A", "SOURCE_B"]
        assert payload["items"][0]["organism"] == ["Homo sapiens", "Mus musculus"]
