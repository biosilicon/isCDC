from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
from sqlalchemy import func, select

from iscdc.database import create_database_engine, create_session_factory
from iscdc.importer import DatasetImportError, import_dataset
from iscdc.models import Dataset


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
    assert json.loads((destination / "validation_report.json").read_text())["valid"]

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
