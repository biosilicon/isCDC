from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml

import iscdc.catalogue_migration as catalogue_migration
from iscdc.catalogue_migration import (
    CatalogueV4MigrationError,
    CatalogueV4MigrationResult,
    finalize_catalogue_v4_migration,
    inspect_catalogue_v3,
    migrate_catalogue_v4,
)
from iscdc.importer import import_dataset


def _prepare_v3_catalogue(settings, write_h5mu, write_metadata) -> tuple[Path, str]:
    result = import_dataset(write_h5mu(), write_metadata(), settings)
    metadata_path = result.destination / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["license"] = {
        "name": "Test License",
        "identifier": "LicenseRef-Test",
        "url": "https://example.com/license",
    }
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute("ALTER TABLE datasets ADD COLUMN license JSON")
        connection.execute(
            "UPDATE datasets SET license = ?",
            (json.dumps(metadata["license"]),),
        )
        connection.execute(
            "UPDATE catalogue_metadata SET value = '3' WHERE key = 'schema_version'"
        )
        connection.commit()
    return metadata_path, result.sha256


def _write_supplemental_metadata(root: Path, name: str, license_value) -> Path:  # noqa: ANN001
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"title": "Supplemental", "license": license_value}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _catalogue_state(path: Path) -> tuple[str, set[str]]:
    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT value FROM catalogue_metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(datasets)")}
    return str(version), columns


def test_catalogue_v4_migration_dry_run_and_finalize(
    tmp_path, settings, write_h5mu, write_metadata
):
    metadata_path, h5mu_sha256 = _prepare_v3_catalogue(
        settings, write_h5mu, write_metadata
    )
    temp_root = tmp_path / "temp"
    exp_root = tmp_path / "exp"
    temp_metadata = _write_supplemental_metadata(
        temp_root, "dataset/metadata.yaml", None
    )
    exp_metadata = _write_supplemental_metadata(
        exp_root,
        "dataset.metadata.yaml",
        {"name": "Experiment License"},
    )
    catalogue_before = settings.database_path.read_bytes()
    metadata_before = metadata_path.read_bytes()

    inventory = migrate_catalogue_v4(
        settings, temp_root=temp_root, exp_root=exp_root, dry_run=True
    )

    assert inventory == inspect_catalogue_v3(
        settings, temp_root=temp_root, exp_root=exp_root
    )
    assert inventory.dataset_count == 1
    assert inventory.formal_metadata_count == 1
    assert inventory.supplemental_metadata_count == 2
    assert inventory.license_field_count == 3
    assert inventory.non_null_license_count == 2
    assert settings.database_path.read_bytes() == catalogue_before
    assert metadata_path.read_bytes() == metadata_before

    result = migrate_catalogue_v4(settings, temp_root=temp_root, exp_root=exp_root)

    assert isinstance(result, CatalogueV4MigrationResult)
    assert result.backup_path.is_dir()
    assert result.report_path.is_file()
    version, columns = _catalogue_state(settings.database_path)
    assert version == "4"
    assert "license" not in columns
    for path in (metadata_path, temp_metadata, exp_metadata):
        assert "license" not in yaml.safe_load(path.read_text(encoding="utf-8"))
    assert (
        catalogue_migration._sha256(settings.data_root / "test_rna_protein" / "dataset.h5mu")
        == h5mu_sha256
    )

    removed = finalize_catalogue_v4_migration(result.report_path, settings)

    assert removed == result.backup_path
    assert not removed.exists()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["backup_deleted_at"] is not None


def test_catalogue_v4_migration_rolls_back_partial_metadata_activation(
    tmp_path, settings, write_h5mu, write_metadata, monkeypatch
):
    metadata_path, _h5mu_sha256 = _prepare_v3_catalogue(
        settings, write_h5mu, write_metadata
    )
    temp_root = tmp_path / "temp"
    exp_root = tmp_path / "exp"
    _write_supplemental_metadata(temp_root, "dataset/metadata.yaml", None)
    _write_supplemental_metadata(exp_root, "dataset.metadata.yaml", None)
    original_metadata = metadata_path.read_bytes()
    original_replace = catalogue_migration._atomic_copy_replace
    calls = 0

    def fail_second_replacement(source, destination):  # noqa: ANN001, ANN202
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected activation failure")
        original_replace(source, destination)

    monkeypatch.setattr(
        catalogue_migration, "_atomic_copy_replace", fail_second_replacement
    )

    with pytest.raises(CatalogueV4MigrationError, match="rolled back"):
        migrate_catalogue_v4(settings, temp_root=temp_root, exp_root=exp_root)

    version, columns = _catalogue_state(settings.database_path)
    assert version == "3"
    assert "license" in columns
    assert metadata_path.read_bytes() == original_metadata


def test_catalogue_v4_migration_rejects_license_inside_h5mu(
    tmp_path, settings, write_h5mu, write_metadata
):
    _prepare_v3_catalogue(settings, write_h5mu, write_metadata)
    h5mu_path = settings.data_root / "test_rna_protein" / "dataset.h5mu"
    import h5py

    with h5py.File(h5mu_path, "r+") as handle:
        handle["uns/database"].create_dataset("license", data="not allowed")

    with pytest.raises(CatalogueV4MigrationError, match="MuData contains unsupported"):
        migrate_catalogue_v4(
            settings,
            temp_root=tmp_path / "temp",
            exp_root=tmp_path / "exp",
            dry_run=True,
        )
