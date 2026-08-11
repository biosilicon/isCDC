from __future__ import annotations

from copy import deepcopy

import mudata as md
import pytest

from iscdc.schemas import load_metadata
from iscdc.validation import validate_h5mu, validate_train_test_pair


def test_valid_same_unit_dataset_passes(write_h5mu, write_metadata):
    outcome = validate_h5mu(write_h5mu(), load_metadata(write_metadata()))

    assert outcome.valid
    assert outcome.n_obs == 2
    assert set(outcome.modalities) == {"rna", "protein"}


def test_histone_is_a_standard_modality(metadata_values, write_h5mu, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["histone_mark"] = "H3K27me3"
    values["modalities"]["histone"] = values["modalities"].pop("protein")

    outcome = validate_h5mu(
        write_h5mu(second_modality_name="histone"),
        load_metadata(write_metadata(values)),
    )

    assert outcome.valid
    assert set(outcome.modalities) == {"rna", "histone"}
    assert "nonstandard_modality_name" not in {issue.code for issue in outcome.warnings}


def test_specific_histone_mark_is_not_a_standard_modality(
    metadata_values, write_h5mu, write_metadata
):
    values = deepcopy(metadata_values)
    values["modalities"]["h3k27me3"] = values["modalities"].pop("protein")

    outcome = validate_h5mu(
        write_h5mu(second_modality_name="h3k27me3"),
        load_metadata(write_metadata(values)),
    )

    assert outcome.valid
    assert "nonstandard_modality_name" in {issue.code for issue in outcome.warnings}


def test_three_modality_partially_shared_dataset_passes(
    metadata_values, write_h5mu, write_metadata
):
    values = deepcopy(metadata_values)
    values["database"]["pairing_type"] = "partially_shared"
    values["modalities"]["metabolite"] = {
        "technology": "Test assay",
        "value_type": "intensity",
    }

    outcome = validate_h5mu(
        write_h5mu("partially_shared", include_third_modality=True),
        load_metadata(write_metadata(values)),
    )

    assert outcome.valid
    assert outcome.n_obs == 4


def test_two_modality_partially_shared_dataset_is_rejected(write_h5mu):
    outcome = validate_h5mu(write_h5mu("partially_shared"))

    assert not outcome.valid
    assert "two_modality_pairing_required" in {issue.code for issue in outcome.errors}


def test_unpaired_dataset_is_rejected(write_h5mu):
    outcome = validate_h5mu(write_h5mu("unpaired"))

    assert not outcome.valid
    assert "invalid_database_metadata" in {issue.code for issue in outcome.errors}


def test_pairing_mismatch_is_rejected(metadata_values, write_h5mu, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["pairing_type"] = "partially_shared"
    values["modalities"]["metabolite"] = {
        "technology": "Test assay",
        "value_type": "intensity",
    }

    outcome = validate_h5mu(
        write_h5mu(
            declared_pairing_type="partially_shared", include_third_modality=True
        ),
        load_metadata(write_metadata(values)),
    )

    assert not outcome.valid
    assert "partially_shared_mismatch" in {issue.code for issue in outcome.errors}


def test_missing_assay_is_rejected(write_h5mu, write_metadata):
    outcome = validate_h5mu(write_h5mu(include_assay=False), load_metadata(write_metadata()))

    assert not outcome.valid
    assert "missing_assay" in {issue.code for issue in outcome.errors}


@pytest.mark.parametrize(
    ("writer_options", "expected_code"),
    [
        ({"include_protein": False}, "too_few_modalities"),
        ({"include_spatial": False}, "missing_spatial"),
        ({"missing_rna_x": True}, "missing_x"),
        ({"duplicate_rna_features": True}, "duplicate_var_names"),
    ],
)
def test_invalid_structures_are_rejected(writer_options, expected_code, write_h5mu, write_metadata):
    outcome = validate_h5mu(write_h5mu(**writer_options), load_metadata(write_metadata()))

    assert not outcome.valid
    assert expected_code in {issue.code for issue in outcome.errors}


def test_yaml_database_mismatch_is_rejected(metadata_values, write_h5mu, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["tissue"] = "lung"

    outcome = validate_h5mu(write_h5mu(), load_metadata(write_metadata(values)))

    assert not outcome.valid
    assert "database_metadata_mismatch" in {issue.code for issue in outcome.errors}


def test_yaml_sample_mismatch_is_rejected(metadata_values, write_h5mu, write_metadata):
    values = deepcopy(metadata_values)
    values["sample_ids"] = ["different_sample"]

    outcome = validate_h5mu(write_h5mu(), load_metadata(write_metadata(values)))

    assert not outcome.valid
    assert "sample_ids_mismatch" in {issue.code for issue in outcome.errors}


@pytest.mark.parametrize(
    ("coordinate_unit", "expects_nonstandard_warning"),
    [("array_index", False), ("grid_step", True)],
)
def test_coordinate_unit_vocabulary_controls_warning(
    coordinate_unit,
    expects_nonstandard_warning,
    metadata_values,
    write_h5mu,
    write_metadata,
):
    metadata_values["database"]["coordinate_unit"] = coordinate_unit

    outcome = validate_h5mu(write_h5mu(), load_metadata(write_metadata()))

    assert outcome.valid
    warning_codes = {issue.code for issue in outcome.warnings}
    assert ("nonstandard_coordinate_unit" in warning_codes) is expects_nonstandard_warning


def test_train_test_pair_rejects_overlapping_source_observations(tmp_path, write_h5mu):
    source_path = write_h5mu()
    product_paths = {}
    for dataset_type in ("train", "test"):
        product = md.read_h5mu(source_path)
        try:
            product.obs["source_dataset_id"] = "test_rna_protein"
            product.obs["source_obs_id"] = product.obs_names.astype(str)
            product.uns["database"].update(
                {
                    "dataset_id": f"overlap_{dataset_type}",
                    "dataset_type": dataset_type,
                    "derivation": {
                        "construction_type": "subset",
                        "source_dataset_ids": ["test_rna_protein"],
                        "split_id": "overlap_split",
                        "challenge_type": "same_slice",
                        "selection_description": "The same observations on both sides.",
                        "feature_merge_policy": "preserve",
                        "processing_description": "Values are unchanged.",
                    },
                }
            )
            path = tmp_path / f"{dataset_type}.h5mu"
            product.write_h5mu(path)
            product_paths[dataset_type] = path
        finally:
            product.file.close()

    outcome = validate_train_test_pair(product_paths["train"], product_paths["test"])

    assert not outcome.valid
    assert "train_test_source_overlap" in {issue.code for issue in outcome.errors}


def test_train_test_pair_rejects_challenge_type_mismatch(tmp_path, write_h5mu):
    source_path = write_h5mu()
    product_paths = {}
    for dataset_type, challenge_type in (
        ("train", "same_slice"),
        ("test", "cross_subject"),
    ):
        product = md.read_h5mu(source_path)
        try:
            product.obs["source_dataset_id"] = "test_rna_protein"
            product.obs["source_obs_id"] = [
                f"{dataset_type}_{name}" for name in product.obs_names.astype(str)
            ]
            product.uns["database"].update(
                {
                    "dataset_id": f"mismatch_{dataset_type}",
                    "dataset_type": dataset_type,
                    "derivation": {
                        "construction_type": "subset",
                        "source_dataset_ids": ["test_rna_protein"],
                        "split_id": "mismatch_split",
                        "challenge_type": challenge_type,
                        "selection_description": "Disjoint synthetic observations.",
                        "feature_merge_policy": "preserve",
                        "processing_description": "Values are unchanged.",
                    },
                }
            )
            path = tmp_path / f"mismatch_{dataset_type}.h5mu"
            product.write_h5mu(path)
            product_paths[dataset_type] = path
        finally:
            product.file.close()

    outcome = validate_train_test_pair(product_paths["train"], product_paths["test"])

    assert not outcome.valid
    assert "challenge_type_mismatch" in {issue.code for issue in outcome.errors}
