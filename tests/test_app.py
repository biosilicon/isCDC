from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime
from urllib.parse import quote

import httpx
import mudata as md
import numpy as np
import pytest
import yaml
from sqlalchemy import select

from iscdc.analytics import VisitEvent
from iscdc.app import create_app
from iscdc.cell_type_visualization import (
    POINT_MEDIA_TYPE,
    build_point_representations,
    encode_points,
    publish_failure,
    publish_generation,
)
from iscdc.database import create_database_engine, create_session_factory
from iscdc.importer import import_dataset
from iscdc.models import Dataset, Modality
from iscdc.repository import CatalogueFilters, list_challenges
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
            "publication": None,
        }
    finally:
        product.file.close()
    destination.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return destination


def _app_with_database(settings, write_h5mu, write_metadata):  # noqa: ANN202
    import_dataset(write_h5mu(), write_metadata(), settings)
    return create_app(settings)


def _publish_test_cell_type_visualization(  # noqa: ANN001, ANN202
    root, source_sha, *, inferred=False
):
    dataset_id = "test_rna_protein"
    generation_id = "test-generation-v1"
    confidence = [0.91, 0.72] if inferred else None
    points = encode_points([0.0, 2.0], [1.0, 3.0], [0, 1], confidence)
    report_document = {
        "report_version": 1,
        "dataset_id": dataset_id,
        "generation_id": generation_id,
        "source_sha256": source_sha,
        "status": "passed",
        "quality_control": (
            {"calibrated": True, "ece": 0.0143, "shared_genes": 402}
            if inferred
            else {"aligned": True}
        ),
        "thresholds": (
            {
                "max_ece": 0.1,
                "min_marker_agreement": None,
                "require_calibration": True,
            }
            if inferred
            else {}
        ),
        "warnings": [],
    }
    report = json.dumps(report_document, sort_keys=True).encode() + b"\n"
    files = {"report.json": report}
    if inferred:
        files["inference.h5"] = b"\x89HDF\r\n\x1a\nopaque inference output"

    def file_record(path, payload):  # noqa: ANN001, ANN202
        return {
            "path": path,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    representations = {}
    suffixes = {"identity": ".bin", "gzip": ".bin.gz", "br": ".bin.br"}
    encoded_points = build_point_representations(points)
    for encoding, encoded in encoded_points.items():
        path = f"samples/sample_01{suffixes[encoding]}"
        files[path] = encoded
        representations[encoding] = {
            **file_record(path, encoded),
            "encoding": encoding,
            "content_size": len(points),
            "content_sha256": hashlib.sha256(points).hexdigest(),
        }
    manifest = {
        "manifest_version": 1,
        "dataset_id": dataset_id,
        "generation_id": generation_id,
        "generated_at": "2026-08-18T12:00:00+00:00",
        "source": {
            "sha256": source_sha,
            "obs_order_sha256": "a" * 64,
            "observation_count": 2,
            "coordinate_dimensions": 2,
            "sample_ids": ["sample_01"],
        },
        "annotation": {
            "kind": "inferred" if inferred else "source",
            "method": "SingleR" if inferred else "mdata.obs[cell_type]",
        },
        "coordinates": {"system": "global", "unit": "micrometer", "y_axis": "down"},
        "categories": [
            {
                "type_id": 0,
                "label": "T cell",
                "color": "#3366CC",
                "count": 1,
                "state": "biological",
            },
            {
                "type_id": 1,
                "label": "B cell",
                "color": "#DC3912",
                "count": 1,
                "state": "biological",
            },
        ],
        "samples": [
            {
                "key": "sample_01",
                "id": "sample_01",
                "count": 2,
                "bounds": [0.0, 1.0, 2.0, 3.0],
                "category_counts": {"0": 1, "1": 1},
                "representations": representations,
            }
        ],
        "report": file_record("report.json", report),
        "provenance": {
            "environment_lock_sha256": "b" * 64,
            "references": (
                [
                    {
                        "id": "census_test_reference",
                        "version": "2026-08-22.1",
                        "sha256": "c" * 64,
                    }
                ]
                if inferred
                else []
            ),
            "parameters": (
                {"cores": 4, "curation_status": "curated", "exclusive": False}
                if inferred
                else {}
            ),
        },
    }
    if inferred:
        for category, ontology_id in zip(
            manifest["categories"], ("CL:0000084", "CL:0000236"), strict=True
        ):
            category["cell_ontology_id"] = ontology_id
        manifest["inference"] = file_record("inference.h5", files["inference.h5"])
    publish_generation(root, manifest, files)
    return generation_id, points, encoded_points


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
                "schema_version": "1.2",
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


def _write_difficulty_report(settings, scores):  # noqa: ANN001, ANN202
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        challenges, total = list_challenges(
            session, CatalogueFilters(), offset=0, limit=1_000_000
        )
        assert len(challenges) == total
        rows = []
        successful_scores = [score for score in scores.values() if score is not None]
        for challenge in challenges:
            score = scores.get(challenge.split_id)
            common = {
                "split_id": challenge.split_id,
                "challenge_type": challenge.challenge_type,
                "input_modality": "rna",
            }
            if score is None:
                rows.append({**common, "status": "failed"})
                continue
            assert challenge.train is not None and challenge.test is not None
            lower_count = sum(value < score for value in successful_scores)
            percentile = (
                100 * lower_count / (len(successful_scores) - 1)
                if len(successful_scores) > 1
                else 0.0
            )
            rows.append(
                {
                    **common,
                    "status": "success",
                    "train": {
                        "dataset_id": challenge.train.dataset_id,
                        "sha256": challenge.train.sha256,
                    },
                    "test": {
                        "dataset_id": challenge.test.dataset_id,
                        "sha256": challenge.test.sha256,
                    },
                    "mean_auroc": score,
                    "domain_shift_score": max(0.0, min(1.0, 2 * (score - 0.5))),
                    "difficulty_percentile": percentile,
                }
            )
    engine.dispose()
    report = {
        "report_version": "1.0",
        "method_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "parameters": {"input_modality": "rna"},
        "challenge_count": len(rows),
        "success_count": sum(row["status"] == "success" for row in rows),
        "failure_count": sum(row["status"] == "failed" for row in rows),
        "challenges": rows,
    }
    path = settings.database_path.parent / "challenge_difficulty.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _clone_challenge(settings, source_split_id, split_id, suffix):  # noqa: ANN001, ANN202
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        sources = list(
            session.scalars(select(Dataset).where(Dataset.split_id == source_split_id)).all()
        )
        assert len(sources) == 2
        for source in sources:
            derivation = deepcopy(source.derivation)
            assert derivation is not None
            derivation["split_id"] = split_id
            clone = Dataset(
                dataset_id=f"{source.dataset_id}_{suffix}",
                entry_id=split_id,
                schema_version=source.schema_version,
                dataset_type=source.dataset_type,
                title=f"{source.title} {suffix}",
                description=source.description,
                source=deepcopy(source.source),
                organism=deepcopy(source.organism),
                tissue=deepcopy(source.tissue),
                spatial_unit=source.spatial_unit,
                coordinate_unit=source.coordinate_unit,
                pairing_type=source.pairing_type,
                derivation=derivation,
                split_id=split_id,
                sample_ids=deepcopy(source.sample_ids),
                keywords=deepcopy(source.keywords),
                publication=deepcopy(source.publication),
                additional_metadata=deepcopy(source.additional_metadata),
                n_obs=source.n_obs,
                coordinate_dimensions=source.coordinate_dimensions,
                file_size=source.file_size,
                sha256=source.sha256,
                storage_dir=f"{source.storage_dir}_{suffix}",
                validation_warning_count=source.validation_warning_count,
                imported_at=source.imported_at,
                modalities=[
                    Modality(
                        name=modality.name,
                        technology=deepcopy(modality.technology),
                        value_type=modality.value_type,
                        n_obs=modality.n_obs,
                        n_vars=modality.n_vars,
                    )
                    for modality in source.modalities
                ],
            )
            session.add(clone)
        session.commit()
    engine.dispose()


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
        assert "1 matching entry" in response.text
        assert "By entry" in response.text
        assert "By dataset" in response.text

        empty = await client.get("/databases?tissue=brain")
        assert "No matching entries" in empty.text

        datasets = await client.get("/databases?view=datasets&tissue=brain")
        assert "No matching databases" in datasets.text


async def test_database_entry_pages_and_api_group_all_slides_after_member_filter(
    settings, write_h5mu, write_metadata
):
    import_dataset(write_h5mu(), write_metadata(), settings)
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        source = session.get(Dataset, "test_rna_protein")
        assert source is not None
        clone = Dataset(
            dataset_id="test_rna_protein_second_slide",
            entry_id=source.entry_id,
            schema_version=source.schema_version,
            dataset_type="full",
            title="Second slide in the same entry",
            description="A second grouped slide.",
            source=deepcopy(source.source),
            organism=deepcopy(source.organism),
            tissue="heart",
            spatial_unit=source.spatial_unit,
            coordinate_unit=source.coordinate_unit,
            pairing_type=source.pairing_type,
            derivation=None,
            split_id=None,
            sample_ids=["sample_02"],
            keywords=deepcopy(source.keywords),
            publication=deepcopy(source.publication),
            additional_metadata=deepcopy(source.additional_metadata),
            n_obs=7,
            coordinate_dimensions=source.coordinate_dimensions,
            file_size=17,
            sha256="f" * 64,
            storage_dir="test_rna_protein_second_slide",
            validation_warning_count=0,
            imported_at=source.imported_at,
            modalities=[
                Modality(
                    name=modality.name,
                    technology=deepcopy(modality.technology),
                    value_type=modality.value_type,
                    n_obs=7,
                    n_vars=modality.n_vars,
                )
                for modality in source.modalities
            ],
        )
        session.add(clone)
        session.commit()
    engine.dispose()

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        home = await client.get("/")
        listing = await client.get("/databases?tissue=heart")
        detail = await client.get("/databases/entries/TEST001")
        api_list = await client.get("/api/database-entries?tissue=heart")
        api_detail = await client.get("/api/database-entries/TEST001")
        dataset_listing = await client.get("/databases?view=datasets")

    assert "1" in home.text
    assert "2 slides" in home.text
    assert "2 slides" in listing.text
    assert "Test RNA and protein dataset" in listing.text
    assert "Second slide in the same entry" in listing.text
    assert detail.status_code == 200
    assert "Datasets in this entry" in detail.text
    assert api_list.json()["total"] == 1
    assert api_list.json()["items"][0]["slide_count"] == 2
    assert len(api_list.json()["items"][0]["datasets"]) == 2
    assert api_detail.json()["total_observations"] == 9
    assert "2 matching datasets" in dataset_listing.text


async def test_file_metadata_omits_license_and_exposes_import_date_only(
    settings, write_h5mu, write_metadata
):
    import_dataset(write_h5mu(), write_metadata(), settings)
    imported_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        database = session.get(Dataset, "test_rna_protein")
        assert database is not None
        database.imported_at = imported_at
        session.commit()
    engine.dispose()

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get("/databases/test_rna_protein")
        detail_api = await client.get("/api/databases/test_rna_protein")
        list_api = await client.get("/api/databases")
        openapi = await client.get("/openapi.json")

    assert detail.status_code == 200
    assert "License" not in detail.text
    assert "Entry ID" in detail.text
    assert "TEST001" in detail.text
    assert "2026-01-02" in detail.text
    assert "03:04:05" not in detail.text

    for payload in (detail_api.json(), list_api.json()["items"][0]):
        assert "license" not in payload
        assert payload["entry_id"] == "TEST001"
        assert payload["imported_at"] == "2026-01-02"

    response_schema = openapi.json()["components"]["schemas"]["DataFileResponse"]
    assert "license" not in response_schema["properties"]
    assert response_schema["properties"]["imported_at"]["format"] == "date"


async def test_database_pages_show_matching_thumbnail(
    tmp_path, settings, write_h5mu, write_metadata
):
    static_dir = tmp_path / "static"
    thumbnail_dir = static_dir / "database_thumbnails"
    thumbnail_dir.mkdir(parents=True)
    stylesheet_content = b".database-thumbnail-list img { max-height: 4rem; }"
    (static_dir / "styles.css").write_bytes(stylesheet_content)
    thumbnail_content = b"test webp content"
    thumbnail_path = thumbnail_dir / "test_rna_protein.webp"
    thumbnail_path.write_bytes(thumbnail_content)
    thumbnail_settings = replace(settings, static_dir=static_dir)
    app = _app_with_database(thumbnail_settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/databases")
        detail = await client.get("/databases/test_rna_protein")
        thumbnail = await client.get(
            "/static/database_thumbnails/test_rna_protein.webp"
        )

    expected_url = "/static/database_thumbnails/test_rna_protein.webp"
    expected_styles_version = hashlib.sha256(stylesheet_content).hexdigest()[:12]
    for page in (listing, detail):
        assert page.status_code == 200
        assert f"/static/styles.css?v={expected_styles_version}" in page.text
        assert expected_url in page.text
        assert "Thumbnail for Test RNA and protein dataset" in page.text
        assert 'loading="lazy"' in page.text
        assert 'decoding="async"' in page.text
    assert "database-thumbnail-list" in listing.text
    assert "database-thumbnail-detail" in detail.text
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/webp"
    assert thumbnail.content == thumbnail_content


async def test_database_pages_omit_thumbnail_when_unavailable(
    settings, write_h5mu, write_metadata
):
    app = _app_with_database(settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/databases")
        detail = await client.get("/databases/test_rna_protein")

    assert listing.status_code == 200
    assert detail.status_code == 200
    assert "database-thumbnail" not in listing.text
    assert "database-thumbnail" not in detail.text


async def test_cell_type_visualization_is_conditional_internal_and_content_negotiated(
    tmp_path, settings, write_h5mu, write_metadata
):
    visualization_root = tmp_path / "cell-type-visualizations"
    visualization_settings = replace(
        settings, cell_type_visualization_root=visualization_root
    )
    imported = import_dataset(
        write_h5mu(), write_metadata(), visualization_settings
    )
    baseline_app = create_app(visualization_settings)
    baseline_transport = httpx.ASGITransport(app=baseline_app)
    async with httpx.AsyncClient(
        transport=baseline_transport, base_url="http://test"
    ) as client:
        baseline_page = await client.get("/databases/test_rna_protein")
        baseline_api = (await client.get("/api/databases/test_rna_protein")).json()
    assert "cell-type-visualization" not in baseline_page.text

    generation_id, identity, encoded_points = _publish_test_cell_type_visualization(
        visualization_root, imported.sha256
    )
    app = create_app(visualization_settings)
    transport = httpx.ASGITransport(app=app)
    endpoint = (
        "/databases/test_rna_protein/cell-type-visualization/"
        f"{generation_id}/sample_01"
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get("/databases/test_rna_protein")
        listing = await client.get("/databases")
        api = await client.get("/api/databases/test_rna_protein")
        openapi = await client.get("/openapi.json")
        point_file = await client.get(endpoint, headers={"Accept-Encoding": "identity"})
        gzip_file = await client.get(endpoint, headers={"Accept-Encoding": "gzip"})
        brotli_file = await client.get(endpoint, headers={"Accept-Encoding": "br"})
        head = await client.head(endpoint, headers={"Accept-Encoding": "identity"})
        unacceptable = await client.get(
            endpoint,
            headers={"Accept-Encoding": "br;q=0,gzip;q=0,identity;q=0"},
        )
        missing_generation = await client.get(
            endpoint.replace(generation_id, "not-current")
        )
        missing_sample = await client.get(endpoint.replace("sample_01", "other"))

    assert detail.status_code == 200
    assert 'id="cell-type-visualization"' in detail.text
    assert 'data-bs-target="#cell-type-method-modal"' in detail.text
    assert detail.text.count('id="cell-type-method-modal"') == 1
    assert "cell_type_visualization.js?v=" in detail.text
    assert endpoint in detail.text
    assert "cell-type-visualization" not in listing.text
    assert api.json() == baseline_api
    assert not any(
        "cell-type-visualization" in path for path in openapi.json()["paths"]
    )
    method_modal = detail.text.split('id="cell-type-method-modal"', maxsplit=1)[1]
    assert "mdata.obs[cell_type]" in method_modal
    assert "Existing annotation file; no computational inference was performed." in method_modal
    assert "Reference ID" not in method_modal
    assert "Runtime parameters" not in method_modal
    assert "QC publication thresholds" not in method_modal
    assert "QC results" not in method_modal
    assert "Confidence" not in method_modal
    assert method_modal.count('data-bs-dismiss="modal"') == 1
    for response, encoding in ((point_file, None), (gzip_file, "gzip"), (brotli_file, "br")):
        assert response.status_code == 200
        # httpx always decodes gzip, while Brotli decoding depends on the optional
        # client extra. Accept only the canonical content or the exact stored bytes.
        expected_encoding = "identity" if encoding is None else encoding
        assert response.content in {identity, encoded_points[expected_encoding]}
        assert response.headers["content-type"].startswith(POINT_MEDIA_TYPE)
        assert response.headers.get("content-encoding") == encoding
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert response.headers["vary"] == "Accept-Encoding"
    assert head.status_code == 200
    assert head.content == b""
    assert unacceptable.status_code == 406
    assert missing_generation.status_code == 404
    assert missing_sample.status_code == 404

    analytics = app.state.analytics
    assert analytics is not None
    with analytics.session_factory() as session:
        endpoint_events = session.scalars(
            select(VisitEvent).where(
                VisitEvent.path.like("%/cell-type-visualization/%")
            )
        ).all()
    assert endpoint_events == []


async def test_inferred_cell_type_method_modal_shows_validated_audit_details(
    tmp_path, settings, write_h5mu, write_metadata
):
    visualization_root = tmp_path / "cell-type-visualizations"
    visualization_settings = replace(
        settings, cell_type_visualization_root=visualization_root
    )
    imported = import_dataset(
        write_h5mu(), write_metadata(), visualization_settings
    )
    _publish_test_cell_type_visualization(
        visualization_root, imported.sha256, inferred=True
    )
    app = create_app(visualization_settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        detail = await client.get("/databases/test_rna_protein")

    assert detail.status_code == 200
    assert detail.text.count('id="cell-type-method-modal"') == 1
    method_modal = detail.text.split('id="cell-type-method-modal"', maxsplit=1)[1]
    for expected in (
        "SingleR",
        "Reference ID",
        "census_test_reference",
        "2026-08-22.1",
        "Runtime parameters",
        "cores",
        "curation_status",
        "curated",
        "QC publication thresholds",
        "max_ece",
        "min_marker_agreement",
        "Not configured",
        "QC results",
        "Passed",
        "shared_genes",
        "402",
    ):
        assert expected in method_modal
    assert "Existing annotation file" not in method_modal
    assert "Confidence" not in method_modal


async def test_latest_cell_type_failure_withdraws_old_generation_after_restart(
    tmp_path, settings, write_h5mu, write_metadata
):
    visualization_root = tmp_path / "cell-type-visualizations"
    visualization_settings = replace(
        settings, cell_type_visualization_root=visualization_root
    )
    imported = import_dataset(
        write_h5mu(), write_metadata(), visualization_settings
    )
    generation_id, _, _ = _publish_test_cell_type_visualization(
        visualization_root, imported.sha256
    )
    old_app = create_app(visualization_settings)
    publish_failure(
        visualization_root,
        "test_rna_protein",
        "calibration quality gate failed",
    )
    restarted_app = create_app(visualization_settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=old_app), base_url="http://test"
    ) as client:
        old_page = await client.get("/databases/test_rna_protein")
        old_points = await client.get(
            "/databases/test_rna_protein/cell-type-visualization/"
            f"{generation_id}/sample_01",
            headers={"Accept-Encoding": "identity"},
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=restarted_app), base_url="http://test"
    ) as client:
        new_page = await client.get("/databases/test_rna_protein")
        new_points = await client.get(
            "/databases/test_rna_protein/cell-type-visualization/"
            f"{generation_id}/sample_01",
            headers={"Accept-Encoding": "identity"},
        )

    assert 'id="cell-type-visualization"' in old_page.text
    assert old_points.status_code == 200
    assert "cell-type-visualization" not in new_page.text
    assert "Visualization unavailable" not in new_page.text
    assert new_points.status_code == 404


async def test_detail_template_tolerates_missing_auxiliary_index_during_reload(
    settings, write_h5mu, write_metadata
):
    app = _app_with_database(settings, write_h5mu, write_metadata)
    app.state.templates.env.globals.pop("auxiliary_files_by_dataset", None)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get("/databases/test_rna_protein")

    assert detail.status_code == 200
    assert "Auxiliary files" not in detail.text


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
        assert response.json()["items"][0]["modality_count"] == 2
        assert "2 modalities" not in page.text


async def test_three_modality_database_is_annotated(
    metadata_values, settings, write_h5mu, write_metadata
):
    values = deepcopy(metadata_values)
    values["database"]["pairing_type"] = "partially_shared"
    values["modalities"]["metabolite"] = {
        "technology": "Xenium",
        "value_type": "intensity",
    }
    import_dataset(
        write_h5mu("partially_shared", include_third_modality=True),
        write_metadata(values),
        settings,
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/databases")
        detail = await client.get("/databases/test_rna_protein")
        api = await client.get("/api/databases/test_rna_protein")

    assert "3 modalities" in listing.text
    assert "3 modalities" in detail.text
    assert "Modality count" in detail.text
    assert api.json()["modality_count"] == 3


async def test_database_detail_rejects_non_full_and_missing_entries(
    tmp_path, settings, write_h5mu, write_metadata
):
    app = _app_with_challenge(tmp_path, settings, write_h5mu, write_metadata)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get("/databases/test_rna_protein")
        assert detail.status_code == 200
        assert "Xenium" in detail.text
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

        api_detail = await client.get("/api/challenges/web_split_v1")
        assert api_detail.status_code == 200
        for side in ("train", "test"):
            file_payload = api_detail.json()[side]
            assert "license" not in file_payload
            assert date.fromisoformat(file_payload["imported_at"]).isoformat() == file_payload[
                "imported_at"
            ]

    assert "License" not in detail.text

    analytics = app.state.analytics
    assert analytics is not None
    with analytics.session_factory() as session:
        event_types = session.scalars(select(VisitEvent.event_type)).all()
    assert "challenge_detail_view" in event_types


async def test_multisource_challenge_shows_every_sample_source_in_page_and_api(
    tmp_path, settings, write_h5mu, write_metadata
):
    _app_with_challenge(tmp_path, settings, write_h5mu, write_metadata)
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        first_source = session.get(Dataset, "test_rna_protein")
        train = session.get(Dataset, "web_train")
        assert first_source is not None
        assert train is not None and train.derivation is not None
        first_source.title = "First source database"
        second_source = Dataset(
            dataset_id="second_source_database",
            entry_id="second_source_database",
            schema_version=first_source.schema_version,
            dataset_type="full",
            title="Second source database",
            description=first_source.description,
            source=deepcopy(first_source.source),
            organism=deepcopy(first_source.organism),
            tissue=deepcopy(first_source.tissue),
            spatial_unit=first_source.spatial_unit,
            coordinate_unit=first_source.coordinate_unit,
            pairing_type=first_source.pairing_type,
            derivation=None,
            split_id=None,
            sample_ids=["sample_02"],
            keywords=deepcopy(first_source.keywords),
            publication=deepcopy(first_source.publication),
            additional_metadata=deepcopy(first_source.additional_metadata),
            n_obs=first_source.n_obs,
            coordinate_dimensions=first_source.coordinate_dimensions,
            file_size=first_source.file_size,
            sha256=first_source.sha256,
            storage_dir="second_source_database",
            validation_warning_count=first_source.validation_warning_count,
            imported_at=first_source.imported_at,
            modalities=[
                Modality(
                    name=modality.name,
                    technology=deepcopy(modality.technology),
                    value_type=modality.value_type,
                    n_obs=modality.n_obs,
                    n_vars=modality.n_vars,
                )
                for modality in first_source.modalities
            ],
        )
        derivation = deepcopy(train.derivation)
        derivation["construction_type"] = "composite"
        derivation["source_dataset_ids"] = [
            "test_rna_protein",
            "second_source_database",
        ]
        train.derivation = derivation
        train.sample_ids = [
            "test_rna_protein::sample_01",
            "second_source_database::sample_02",
        ]
        session.add(second_source)
        session.commit()
    engine.dispose()

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get("/challenges/web_split_v1")
        api_detail = await client.get("/api/challenges/web_split_v1")
        api_listing = await client.get("/api/challenges")

    assert detail.status_code == 200
    assert "Original sample" in detail.text
    assert "test_rna_protein::sample_01" in detail.text
    assert "second_source_database::sample_02" in detail.text
    assert "First source database" in detail.text
    assert "Second source database" in detail.text
    assert "/databases/test_rna_protein" in detail.text
    assert "/databases/second_source_database" in detail.text

    expected = [
        {
            "sample_id": "test_rna_protein::sample_01",
            "source_sample_id": "sample_01",
            "source_database_id": "test_rna_protein",
            "source_database_title": "First source database",
            "source": "TEST001",
        },
        {
            "sample_id": "second_source_database::sample_02",
            "source_sample_id": "sample_02",
            "source_database_id": "second_source_database",
            "source_database_title": "Second source database",
            "source": "TEST001",
        },
    ]
    assert api_detail.status_code == 200
    assert api_detail.json()["train"]["sample_sources"] == expected
    assert api_detail.json()["test"]["sample_sources"] == []
    assert api_listing.status_code == 200
    assert api_listing.json()["items"][0]["train"]["sample_sources"] == expected


async def test_challenge_difficulty_is_shown_in_pages_api_and_method_modal(
    tmp_path, settings, write_h5mu, write_metadata
):
    _app_with_challenge(tmp_path, settings, write_h5mu, write_metadata)
    _write_difficulty_report(settings, {"web_split_v1": 0.75})
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/challenges")
        detail = await client.get("/challenges/web_split_v1")
        api_listing = await client.get("/api/challenges")
        api_detail = await client.get("/api/challenges/web_split_v1")

    expected = {
        "mean_auroc": 0.75,
        "domain_shift_score": 0.5,
        "difficulty_percentile": 0.0,
    }
    for page in (listing, detail):
        assert page.status_code == 200
        assert "Domain AUROC" in page.text
        assert "0.750" in page.text
        assert "Shift score" in page.text
        assert "0.500" in page.text
        assert "0.0%" in page.text
        assert page.text.count('id="difficulty-method-modal"') == 1
        assert "train–test distribution-separability proxy" in page.text
        assert "not an absolute difficulty score" in page.text
        modal = page.text.split('id="difficulty-method-modal"', maxsplit=1)[1]
        assert modal.count('data-bs-dismiss="modal"') == 1
        assert ">Close</button>" in modal
    assert api_listing.json()["items"][0]["difficulty"] == expected
    assert api_detail.json()["difficulty"] == expected


async def test_challenge_difficulty_sorting_precedes_pagination_and_places_missing_last(
    tmp_path, settings, write_h5mu, write_metadata
):
    _app_with_challenge(tmp_path, settings, write_h5mu, write_metadata)
    _clone_challenge(settings, "web_split_v1", "low_shift", "low")
    _clone_challenge(settings, "web_split_v1", "unavailable_shift", "unavailable")
    _write_difficulty_report(
        settings,
        {
            "web_split_v1": 0.8,
            "low_shift": 0.6,
            "unavailable_shift": None,
        },
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ascending = await client.get("/api/challenges?sort=difficulty_asc")
        descending = await client.get("/api/challenges?sort=difficulty_desc")
        second_descending = await client.get(
            "/api/challenges?sort=difficulty_desc&limit=1&offset=1"
        )
        listing = await client.get("/challenges?sort=difficulty_desc")
        invalid = await client.get("/api/challenges?sort=unknown")

    assert [item["split_id"] for item in ascending.json()["items"]] == [
        "low_shift",
        "web_split_v1",
        "unavailable_shift",
    ]
    assert [item["split_id"] for item in descending.json()["items"]] == [
        "web_split_v1",
        "low_shift",
        "unavailable_shift",
    ]
    assert second_descending.json()["total"] == 3
    assert second_descending.json()["items"][0]["split_id"] == "low_shift"
    assert descending.json()["items"][-1]["difficulty"] is None
    assert listing.text.index("web_split_v1") < listing.text.index("low_shift")
    assert listing.text.index("low_shift") < listing.text.index("unavailable_shift")
    assert 'option value="difficulty_desc" selected' in listing.text
    assert invalid.status_code == 422


async def test_challenge_difficulty_snapshot_is_fail_open_and_loaded_only_at_startup(
    tmp_path, settings, write_h5mu, write_metadata
):
    _app_with_challenge(tmp_path, settings, write_h5mu, write_metadata)
    report_path = _write_difficulty_report(settings, {"web_split_v1": 0.7})
    app = create_app(settings)
    _write_difficulty_report(settings, {"web_split_v1": 0.9})

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        unchanged = await client.get("/api/challenges/web_split_v1")
    assert unchanged.json()["difficulty"]["mean_auroc"] == 0.7

    refreshed_app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=refreshed_app), base_url="http://test"
    ) as client:
        refreshed = await client.get("/api/challenges/web_split_v1")
    assert refreshed.json()["difficulty"]["mean_auroc"] == 0.9

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["challenges"][0]["train"]["sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    stale_app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=stale_app), base_url="http://test"
    ) as client:
        stale_page = await client.get("/challenges")
        stale_api = await client.get("/api/challenges/web_split_v1")
    assert stale_page.status_code == 200
    assert "Unavailable" in stale_page.text
    assert stale_api.status_code == 200
    assert stale_api.json()["difficulty"] is None


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
            entry_id=original.entry_id,
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
        response = await client.get("/api/databases?technology=Xenium&limit=1&offset=0")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["dataset_id"] == "test_rna_protein"
        assert payload["items"][0]["entry_id"] == "TEST001"
        assert payload["items"][0]["sample_sources"] == []
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
        legacy = await client.get(
            "/api/databases?technology=10x%20Genomics%20Xenium%20In%20Situ"
        )
        assert legacy.status_code == 200
        assert legacy.json()["total"] == 0


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
        database.modalities[0].technology = ["Xenium", "SPOTS"]
        session.commit()
    engine.dispose()

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/databases?organism=Mus%20musculus&technology=SPOTS"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["source"] == ["SOURCE_A", "SOURCE_B"]
        assert payload["items"][0]["organism"] == ["Homo sapiens", "Mus musculus"]
