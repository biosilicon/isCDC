from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from datetime import UTC, datetime

import h5py
import mudata as md
import pytest
import yaml

from iscdc.database import (
    CatalogueMetadata,
    create_database_engine,
    create_session_factory,
    initialize_database,
)
from iscdc.models import Dataset, Modality
from iscdc.schema_migration import (
    MigrationInventory,
    MigrationResult,
    SchemaMigrationError,
    finalize_schema_1_2_migration,
    migrate_schema_1_2,
)


def _set_schema_version(path, value, *, dataset_id=None):  # noqa: ANN001, ANN202
    with h5py.File(path, "r+") as handle:
        handle["uns/database/schema_version"][()] = value
        if dataset_id is not None:
            handle["uns/database/dataset_id"][()] = dataset_id


def test_schema_migration_trims_two_modality_data_and_finalizes_backup(
    tmp_path,
    monkeypatch,
    metadata_values,
    settings,
    write_h5mu,
):
    data_root = settings.data_root
    data_root.mkdir(parents=True)
    dataset_id = "legacy_partial"
    directory = data_root / dataset_id
    directory.mkdir()
    legacy_h5mu = write_h5mu(
        "partially_shared", name="legacy-partial-source.h5mu"
    )
    shutil.copy2(legacy_h5mu, directory / "dataset.h5mu")
    _set_schema_version(
        directory / "dataset.h5mu", "1.1", dataset_id=dataset_id
    )

    metadata = deepcopy(metadata_values)
    metadata["database"].update(
        {
            "schema_version": "1.1",
            "dataset_id": dataset_id,
            "pairing_type": "partially_shared",
        }
    )
    (directory / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    imported_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    sha256 = hashlib.sha256((directory / "dataset.h5mu").read_bytes()).hexdigest()
    file_size = (directory / "dataset.h5mu").stat().st_size
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "imported_at": imported_at.isoformat(),
                "database": metadata["database"],
                "files": {
                    "h5mu": {
                        "name": "dataset.h5mu",
                        "size": file_size,
                        "sha256": sha256,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "validation_report.json").write_text("{}\n", encoding="utf-8")
    (directory / "checksum.sha256").write_text(
        f"{sha256}  dataset.h5mu\n", encoding="utf-8"
    )

    engine = create_database_engine(settings.database_path)
    initialize_database(engine)
    session_factory = create_session_factory(engine)
    with session_factory() as session:
        session.get(CatalogueMetadata, "schema_version").value = "2"
        session.add(
            Dataset(
                dataset_id=dataset_id,
                schema_version="1.1",
                dataset_type="full",
                title=metadata["title"],
                description=metadata["description"],
                source=metadata["database"]["source"],
                organism=metadata["database"]["organism"],
                tissue=metadata["database"]["tissue"],
                spatial_unit=metadata["database"]["spatial_unit"],
                coordinate_unit=metadata["database"]["coordinate_unit"],
                pairing_type="partially_shared",
                derivation=None,
                split_id=None,
                sample_ids=metadata["sample_ids"],
                keywords=metadata["keywords"],
                license=None,
                publication=None,
                additional_metadata={},
                n_obs=3,
                coordinate_dimensions=2,
                file_size=file_size,
                sha256=sha256,
                storage_dir=dataset_id,
                validation_warning_count=0,
                imported_at=imported_at,
                modalities=[
                    Modality(
                        name=name,
                        technology="Test assay",
                        value_type=values["value_type"],
                        n_obs=2,
                        n_vars=2 if name == "rna" else 1,
                    )
                    for name, values in metadata["modalities"].items()
                ],
            )
        )
        session.commit()
    engine.dispose()

    exp_root = tmp_path / "exp"
    exp_root.mkdir()
    exp_h5mu = write_h5mu(name="legacy-exp-source.h5mu")
    exp_target = exp_root / "xenium_human_rcc_ffpe_rna_protein.h5mu"
    shutil.copy2(exp_h5mu, exp_target)
    _set_schema_version(exp_target, "1.1")
    (exp_root / "xenium_human_rcc_ffpe_rna_protein_vertical_split.yaml").write_text(
        yaml.safe_dump({"schema_version": "1.1"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        "iscdc.schema_migration.EXPECTED_DATASET_TYPES",
        {"full": 1, "train": 0, "test": 0},
    )
    monkeypatch.setattr(
        "iscdc.schema_migration.EXPECTED_PARTIAL_DATASET_IDS", {dataset_id}
    )

    inventory = migrate_schema_1_2(settings, exp_root=exp_root, dry_run=True)
    assert isinstance(inventory, MigrationInventory)
    assert inventory.partial_dataset_ids == (dataset_id,)

    result = migrate_schema_1_2(settings, exp_root=exp_root)
    assert isinstance(result, MigrationResult)
    assert result.backup_path.is_dir()
    assert (result.backup_path / "datasets" / dataset_id).is_dir()

    migrated = md.read_h5mu(data_root / dataset_id / "dataset.h5mu")
    try:
        assert migrated.uns["database"]["schema_version"] == "1.2"
        assert migrated.uns["database"]["pairing_type"] == "same_unit"
        assert migrated.n_obs == 1
        assert {adata.n_obs for adata in migrated.mod.values()} == {1}
    finally:
        migrated.file.close()
    migrated_metadata = yaml.safe_load(
        (data_root / dataset_id / "metadata.yaml").read_text(encoding="utf-8")
    )
    assert migrated_metadata["database"]["schema_version"] == "1.2"
    assert migrated_metadata["database"]["pairing_type"] == "same_unit"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["datasets"][dataset_id]["removed_observation_ids"] == [
        "cell_1",
        "cell_3",
    ]
    assert report["backup_deleted_at"] is None

    migrated_path = data_root / dataset_id / "dataset.h5mu"
    original_bytes = migrated_path.read_bytes()
    tampered_bytes = bytearray(original_bytes)
    tampered_bytes[-1] ^= 1
    migrated_path.write_bytes(tampered_bytes)
    with pytest.raises(SchemaMigrationError, match="checksum mismatch"):
        finalize_schema_1_2_migration(result.report_path, settings)
    assert result.backup_path.is_dir()
    migrated_path.write_bytes(original_bytes)

    removed = finalize_schema_1_2_migration(result.report_path, settings)

    assert removed == result.backup_path
    assert not result.backup_path.exists()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["backup_deleted_at"] is not None
