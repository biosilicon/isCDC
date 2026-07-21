from __future__ import annotations

from copy import deepcopy

import pytest

from iscdc.schemas import load_metadata
from iscdc.validation import validate_h5mu


def test_valid_same_unit_dataset_passes(write_h5mu, write_metadata):
    outcome = validate_h5mu(write_h5mu(), load_metadata(write_metadata()))

    assert outcome.valid
    assert outcome.n_obs == 2
    assert set(outcome.modalities) == {"rna", "protein"}


def test_valid_partially_shared_dataset_passes(metadata_values, write_h5mu, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["pairing_type"] = "partially_shared"

    outcome = validate_h5mu(write_h5mu("partially_shared"), load_metadata(write_metadata(values)))

    assert outcome.valid
    assert outcome.n_obs == 3


def test_valid_unpaired_dataset_passes(metadata_values, write_h5mu, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["pairing_type"] = "unpaired"

    outcome = validate_h5mu(write_h5mu("unpaired"), load_metadata(write_metadata(values)))

    assert outcome.valid
    assert outcome.n_obs == 4


def test_pairing_mismatch_is_rejected(metadata_values, write_h5mu, write_metadata):
    values = deepcopy(metadata_values)
    values["database"]["pairing_type"] = "partially_shared"

    outcome = validate_h5mu(
        write_h5mu(declared_pairing_type="partially_shared"),
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
