from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import h5py
import mudata as md
import pytest
import yaml

import iscdc.entry_migration as entry_migration
from iscdc.entry_migration import (
    EntryIdMigrationError,
    EntryIdMigrationResult,
    finalize_entry_id_migration,
    inspect_entry_id_migration,
    migrate_entry_ids,
)
from iscdc.importer import import_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _downgrade_to_v4(settings, destination: Path) -> None:  # noqa: ANN001
    h5mu_path = destination / "dataset.h5mu"
    with h5py.File(h5mu_path, "r+") as handle:
        del handle["uns/database/entry_id"]
    sha256 = _sha256(h5mu_path)
    file_size = h5mu_path.stat().st_size

    metadata_path = destination / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    del metadata["database"]["entry_id"]
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["database"]["entry_id"]
    manifest["files"]["h5mu"].update(size=file_size, sha256=sha256)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "checksum.sha256").write_text(f"{sha256}  dataset.h5mu\n", encoding="utf-8")

    with sqlite3.connect(settings.database_path) as connection:
        connection.execute("DROP INDEX ix_datasets_entry_id")
        connection.execute("ALTER TABLE datasets DROP COLUMN entry_id")
        connection.execute("UPDATE datasets SET file_size = ?, sha256 = ?", (file_size, sha256))
        connection.execute("UPDATE catalogue_metadata SET value = '4' WHERE key = 'schema_version'")
        connection.commit()


def _prepare_v4(settings, write_h5mu, write_metadata) -> Path:  # noqa: ANN001
    result = import_dataset(write_h5mu(), write_metadata(), settings)
    _downgrade_to_v4(settings, result.destination)
    return result.destination


def _catalogue_state(path: Path) -> tuple[str, set[str], tuple[str, int, str]]:
    with sqlite3.connect(path) as connection:
        version = str(
            connection.execute(
                "SELECT value FROM catalogue_metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
        )
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(datasets)")}
        row = connection.execute(
            "SELECT entry_id, file_size, sha256 FROM datasets"
            if "entry_id" in columns
            else "SELECT '', file_size, sha256 FROM datasets"
        ).fetchone()
    return version, columns, (str(row[0]), int(row[1]), str(row[2]))


def _catalogue_dump(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return "\n".join(connection.iterdump())


def test_entry_id_migration_dry_run_migrate_and_finalize(settings, write_h5mu, write_metadata):
    destination = _prepare_v4(settings, write_h5mu, write_metadata)
    auxiliary = destination / "auxiliary" / "keep.bin"
    auxiliary.parent.mkdir()
    auxiliary.write_bytes(b"must remain untouched")
    catalogue_before = settings.database_path.read_bytes()
    files_before = {
        name: (destination / name).read_bytes() for name in entry_migration.ARTIFACT_NAMES
    }

    inventory = migrate_entry_ids(settings, dry_run=True)

    assert inventory == inspect_entry_id_migration(settings)
    assert inventory.as_dict() == {
        "dataset_count": 1,
        "full_count": 1,
        "train_count": 0,
        "test_count": 0,
        "unique_entry_count": 1,
        "h5mu_count": 1,
        "h5mu_bytes": (destination / "dataset.h5mu").stat().st_size,
        "metadata_count": 1,
        "manifest_count": 1,
        "checksum_count": 1,
    }
    assert settings.database_path.read_bytes() == catalogue_before
    assert {
        name: (destination / name).read_bytes() for name in entry_migration.ARTIFACT_NAMES
    } == files_before

    result = migrate_entry_ids(settings)

    assert isinstance(result, EntryIdMigrationResult)
    assert result.backup_path.is_dir()
    assert result.report_path.is_file()
    version, columns, row = _catalogue_state(settings.database_path)
    assert version == "5"
    assert "entry_id" in columns
    assert row[0] == "test_rna_protein"
    h5mu_path = destination / "dataset.h5mu"
    assert row[1:] == (h5mu_path.stat().st_size, _sha256(h5mu_path))
    with h5py.File(h5mu_path, "r") as handle:
        assert handle["uns/database/entry_id"].asstr()[()] == "test_rna_protein"
        assert handle["uns/database/entry_id"].attrs["encoding-type"] == "string"
        assert handle["uns/database/entry_id"].attrs["encoding-version"] == "0.2.0"
    migrated = md.read_h5mu(h5mu_path)
    assert migrated.uns["database"]["entry_id"] == "test_rna_protein"
    metadata = yaml.safe_load((destination / "metadata.yaml").read_text())
    manifest = json.loads((destination / "manifest.json").read_text())
    assert metadata["database"]["entry_id"] == "test_rna_protein"
    assert manifest["database"]["entry_id"] == "test_rna_protein"
    assert manifest["files"]["h5mu"]["sha256"] == row[2]
    assert (destination / "checksum.sha256").read_text() == (f"{row[2]}  dataset.h5mu\n")
    assert auxiliary.read_bytes() == b"must remain untouched"

    removed = finalize_entry_id_migration(result.report_path, settings)

    assert removed == result.backup_path
    assert not removed.exists()
    report = json.loads(result.report_path.read_text())
    assert report["backup_deleted_at"] is not None


def test_entry_id_migration_rolls_back_all_activated_files(
    settings, write_h5mu, write_metadata, monkeypatch
):
    destination = _prepare_v4(settings, write_h5mu, write_metadata)
    original_catalogue = _catalogue_dump(settings.database_path)
    original_files = {
        name: (destination / name).read_bytes() for name in entry_migration.ARTIFACT_NAMES
    }

    def fail_verification(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise EntryIdMigrationError("injected verification failure")

    monkeypatch.setattr(entry_migration, "_verify_active", fail_verification)

    with pytest.raises(EntryIdMigrationError, match="was rolled back"):
        migrate_entry_ids(settings)

    assert _catalogue_dump(settings.database_path) == original_catalogue
    assert {
        name: (destination / name).read_bytes() for name in entry_migration.ARTIFACT_NAMES
    } == original_files
    version, columns, _row = _catalogue_state(settings.database_path)
    assert version == "4"
    assert "entry_id" not in columns


def test_entry_id_derivation_uses_split_id_and_safe_fallback():
    base = {"dataset_id": "full_1", "dataset_type": "full", "split_id": "ignored"}
    assert entry_migration._entry_id(base) == "full_1"
    assert (
        entry_migration._entry_id(
            {"dataset_id": "train_1", "dataset_type": "train", "split_id": "challenge_1"}
        )
        == "challenge_1"
    )
    assert (
        entry_migration._entry_id(
            {"dataset_id": "test_1", "dataset_type": "test", "split_id": None}
        )
        == "test_1"
    )
    with pytest.raises(EntryIdMigrationError, match="Unsafe entry_id"):
        entry_migration._entry_id(
            {"dataset_id": "test_1", "dataset_type": "test", "split_id": "../unsafe"}
        )
