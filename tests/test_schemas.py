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


def test_metadata_rejects_schema_v10(metadata_values, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["schema_version"] = "1.0"

    with pytest.raises(MetadataLoadError, match="schema_version"):
        load_metadata(write_metadata(values))


def test_metadata_validates_derived_relationships_and_lists(metadata_values, write_metadata):
    values = deepcopy(metadata_values)
    values["database"].update(
        {
            "dataset_id": "derived_train",
            "dataset_type": "train",
            "source": ["SOURCE_A", "SOURCE_B"],
            "organism": ["Homo sapiens", "Mus musculus"],
            "tissue": ["kidney", "lung"],
            "derivation": {
                "construction_type": "composite",
                "source_dataset_ids": ["full_a", "full_b"],
                "split_id": "split_v1",
                "selection_description": "All assigned observations.",
                "feature_merge_policy": "intersection",
                "processing_description": "Aligned to common features.",
            },
        }
    )
    values["modalities"]["rna"]["technology"] = ["Tech A", "Tech B"]

    metadata = load_metadata(write_metadata(values))

    assert metadata.database.dataset_type == "train"
    assert metadata.database.source == ["SOURCE_A", "SOURCE_B"]
    assert metadata.modalities["rna"].technology == ["Tech A", "Tech B"]


def test_full_metadata_rejects_derivation(metadata_values, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["derivation"] = {
        "construction_type": "subset",
        "source_dataset_ids": ["source_full"],
        "split_id": "split_v1",
        "selection_description": "Invalid full derivation.",
        "feature_merge_policy": "preserve",
        "processing_description": "None.",
    }

    with pytest.raises(MetadataLoadError, match="must not define derivation"):
        load_metadata(write_metadata(values))


def test_full_metadata_accepts_explicit_empty_derivation(metadata_values, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["derivation"] = {}

    metadata = load_metadata(write_metadata(values))

    assert metadata.database.derivation is None
