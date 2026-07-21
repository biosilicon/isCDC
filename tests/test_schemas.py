from __future__ import annotations

from copy import deepcopy

import pytest

from iscdc.schemas import MetadataLoadError, load_metadata


def test_metadata_accepts_nullable_license_and_publication(write_metadata):
    metadata = load_metadata(write_metadata())

    assert metadata.license is None
    assert metadata.publication is None


def test_metadata_preserves_additional_database_fields(metadata_values, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["disease"] = "renal cell carcinoma"

    metadata = load_metadata(write_metadata(values))

    assert metadata.additional_database_values() == {"disease": "renal cell carcinoma"}


def test_metadata_rejects_unsafe_dataset_id(metadata_values, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["dataset_id"] = "../escape"

    with pytest.raises(MetadataLoadError, match="dataset_id"):
        load_metadata(write_metadata(values))


def test_metadata_requires_nullable_keys(metadata_values, write_metadata):
    values = deepcopy(metadata_values)
    del values["license"]

    with pytest.raises(MetadataLoadError, match="license"):
        load_metadata(write_metadata(values))
