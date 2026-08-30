from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import mudata as md
import numpy as np
import pytest
import yaml
from sqlalchemy import func, select

from iscdc.auxiliary import register_auxiliary_file
from iscdc.database import create_database_engine, create_session_factory
from iscdc.importer import DatasetImportError, import_dataset
from iscdc.models import Dataset
from iscdc.splitter import compose_split, spatial_split


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
            "description": "A deterministic derived dataset used for import validation.",
            "keywords": ["derived", "test"],
            "publication": None,
        }
    finally:
        product.file.close()
    destination.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return destination


def test_import_creates_catalogue_and_provenance_files(settings, write_h5mu, write_metadata):
    source = write_h5mu()
    result = import_dataset(source, write_metadata(), settings)

    destination = settings.data_root / "test_rna_protein"
    assert result.destination == destination
    assert {path.name for path in destination.iterdir()} == {
        "dataset.h5mu",
        "metadata.yaml",
        "manifest.json",
        "validation_report.json",
        "checksum.sha256",
    }
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert result.sha256 == expected_hash
    assert (destination / "checksum.sha256").read_text().startswith(expected_hash)
    assert "license" not in yaml.safe_load(
        (destination / "metadata.yaml").read_text(encoding="utf-8")
    )
    assert json.loads((destination / "validation_report.json").read_text())["valid"]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["manifest_version"] == "1.1"
    assert manifest["auxiliary_files"] == []

    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        assert session.scalar(select(func.count()).select_from(Dataset)) == 1
        dataset = session.get(Dataset, "test_rna_protein")
        assert dataset is not None
        assert {modality.name for modality in dataset.modalities} == {"rna", "protein"}
    engine.dispose()


def test_duplicate_dataset_is_rejected(settings, write_h5mu, write_metadata):
    source = write_h5mu()
    metadata = write_metadata()
    import_dataset(source, metadata, settings)

    with pytest.raises(DatasetImportError, match="already indexed"):
        import_dataset(source, metadata, settings)


def test_replace_atomically_updates_dataset_and_preserves_auxiliary_files(
    tmp_path, settings, write_h5mu, write_metadata
):
    source = write_h5mu()
    metadata = write_metadata()
    original = import_dataset(source, metadata, settings)
    auxiliary_source = tmp_path / "slide.svs"
    auxiliary_source.write_bytes(b"small deterministic slide")
    register_auxiliary_file(
        original.dataset_id,
        auxiliary_source,
        settings,
        auxiliary_id="he_wsi",
        label="H&E whole-slide image",
        source_url="https://example.org/slide.svs",
        media_type="image/tiff",
    )

    replacement = md.read_h5mu(source)
    replacement_path = tmp_path / "replacement.h5mu"
    try:
        replacement.mod["rna"].X[0, 0] = 99
        replacement.write_h5mu(replacement_path)
    finally:
        replacement.file.close()

    result = import_dataset(replacement_path, metadata, settings, replace=True)

    assert result.sha256 != original.sha256
    destination = settings.data_root / original.dataset_id
    manifest = json.loads((destination / "manifest.json").read_text())
    assert [entry["id"] for entry in manifest["auxiliary_files"]] == ["he_wsi"]
    assert (destination / "auxiliary" / "slide.svs").read_bytes() == auxiliary_source.read_bytes()
    staging_root = settings.data_root / ".staging"
    assert not staging_root.exists() or list(staging_root.iterdir()) == []
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        assert session.scalar(select(func.count()).select_from(Dataset)) == 1
        assert session.get(Dataset, original.dataset_id).sha256 == result.sha256
    engine.dispose()


def test_failed_replace_leaves_original_dataset_untouched(
    metadata_values, settings, write_h5mu, write_metadata
):
    source = write_h5mu()
    metadata = write_metadata()
    original = import_dataset(source, metadata, settings)
    invalid_values = deepcopy(metadata_values)
    invalid_values["database"]["tissue"] = "lung"

    with pytest.raises(DatasetImportError, match="validation failed"):
        import_dataset(
            source,
            write_metadata(invalid_values, "invalid.yaml"),
            settings,
            replace=True,
        )

    destination = settings.data_root / original.dataset_id
    stored_hash = hashlib.sha256((destination / "dataset.h5mu").read_bytes()).hexdigest()
    assert stored_hash == original.sha256
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        assert session.get(Dataset, original.dataset_id).sha256 == original.sha256
    engine.dispose()


def test_failed_validation_leaves_no_dataset(metadata_values, settings, write_h5mu, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["tissue"] = "lung"

    with pytest.raises(DatasetImportError, match="validation failed"):
        import_dataset(write_h5mu(), write_metadata(values), settings)

    assert not (settings.data_root / "test_rna_protein").exists()
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        assert session.scalar(select(func.count()).select_from(Dataset)) == 0
    engine.dispose()


def test_imports_derived_sides_after_their_full_source(
    tmp_path, settings, write_h5mu, write_metadata
):
    source_path = write_h5mu()
    import_dataset(source_path, write_metadata(), settings)
    config_path = tmp_path / "spatial.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.2",
                "split_id": "import_split_v1",
                "challenge_type": "same_slice",
                "feature_merge_policy": "preserve",
                "source": str(source_path),
                "output_dir": "derived-output",
                "train": {"dataset_id": "derived_train"},
                "test": {
                    "dataset_id": "derived_test",
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
    train_metadata = _write_product_metadata(train_path, tmp_path / "train.metadata.yaml")
    test_metadata = _write_product_metadata(test_path, tmp_path / "test.metadata.yaml")

    import_dataset(train_path, train_metadata, settings)
    import_dataset(test_path, test_metadata, settings)

    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        train = session.get(Dataset, "derived_train")
        test = session.get(Dataset, "derived_test")
        assert train is not None and train.dataset_type == "train"
        assert test is not None and test.dataset_type == "test"
        assert train.split_id == test.split_id == "import_split_v1"
        assert train.derivation["challenge_type"] == "same_slice"
        assert train.derivation["source_dataset_ids"] == ["test_rna_protein"]
    engine.dispose()


def test_derived_import_requires_source_first(tmp_path, settings, write_h5mu):
    source_path = write_h5mu()
    config_path = tmp_path / "spatial.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.2",
                "split_id": "missing_source_split",
                "challenge_type": "same_slice",
                "feature_merge_policy": "preserve",
                "source": str(source_path),
                "output_dir": "missing-source-output",
                "train": {"dataset_id": "missing_source_train"},
                "test": {
                    "dataset_id": "missing_source_test",
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
            }
        ),
        encoding="utf-8",
    )
    train_path, _ = spatial_split(config_path)
    metadata_path = _write_product_metadata(train_path, tmp_path / "missing.metadata.yaml")

    with pytest.raises(DatasetImportError, match="imported first"):
        import_dataset(train_path, metadata_path, settings)


def test_composite_import_preserves_multivalue_source_and_technology(
    tmp_path, settings, write_h5mu
):
    template_path = write_h5mu()
    source_paths = []
    for dataset_id, technology in (
        ("composite_a", "Xenium"),
        ("composite_b", "SPOTS"),
        ("composite_c", "Spatial-CITE-seq"),
    ):
        source = md.read_h5mu(template_path)
        try:
            source.uns["database"]["dataset_id"] = dataset_id
            source.uns["database"]["source"] = f"SOURCE-{dataset_id}"
            for adata in source.mod.values():
                adata.uns["assay"]["technology"] = technology
            path = tmp_path / f"{dataset_id}.h5mu"
            source.write_h5mu(path)
        finally:
            source.file.close()
        metadata_path = _write_product_metadata(path, tmp_path / f"{dataset_id}.metadata.yaml")
        import_dataset(path, metadata_path, settings)
        source_paths.append(path)

    config_path = tmp_path / "compose.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.2",
                "split_id": "composite_import_v1",
                "challenge_type": "cross_subject",
                "feature_merge_policy": "preserve",
                "output_dir": "composite-output",
                "train": {
                    "dataset_id": "composite_train",
                    "sources": [str(source_paths[0]), str(source_paths[1])],
                    "reference_dataset_id": None,
                },
                "test": {
                    "dataset_id": "composite_test",
                    "sources": [str(source_paths[2])],
                    "reference_dataset_id": None,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    train_path, test_path = compose_split(config_path)
    train_metadata = _write_product_metadata(train_path, tmp_path / "composite-train.yaml")
    test_metadata = _write_product_metadata(test_path, tmp_path / "composite-test.yaml")
    import_dataset(train_path, train_metadata, settings)
    import_dataset(test_path, test_metadata, settings)

    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        train = session.get(Dataset, "composite_train")
        assert train is not None
        assert train.source == ["SOURCE-composite_a", "SOURCE-composite_b"]
        assert all(
            modality.technology == ["Xenium", "SPOTS"] for modality in train.modalities
        )
    engine.dispose()
