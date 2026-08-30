from __future__ import annotations

from copy import deepcopy

import mudata as md
import pandas as pd
import pytest

from iscdc.schemas import load_metadata
from iscdc.validation import (
    CELL_TYPE_PROVENANCE_KEY,
    CELL_TYPE_PROVENANCE_VERSION,
    UNANNOTATED_CELL_TYPE,
    validate_h5mu,
    validate_train_test_pair,
)


def test_valid_same_unit_dataset_passes(write_h5mu, write_metadata):
    outcome = validate_h5mu(write_h5mu(), load_metadata(write_metadata()))

    assert outcome.valid
    assert outcome.n_obs == 2
    assert set(outcome.modalities) == {"rna", "protein"}


def test_h5mu_rejects_technology_outside_controlled_vocabulary(tmp_path, write_h5mu):
    source_path = write_h5mu()
    mdata = md.read_h5mu(source_path)
    try:
        mdata.mod["rna"].uns["assay"]["technology"] = "10x Genomics Xenium In Situ"
        invalid_path = tmp_path / "unsupported-technology.h5mu"
        mdata.write_h5mu(invalid_path)
    finally:
        mdata.file.close()

    outcome = validate_h5mu(invalid_path)

    assert not outcome.valid
    assert "unsupported_technology" in {issue.code for issue in outcome.errors}


def _write_cell_type_variant(path, destination, values, provenance=None) -> None:  # noqa: ANN001
    mdata = md.read_h5mu(path)
    try:
        mdata.obs["cell_type"] = values
        if provenance is not None:
            mdata.uns[CELL_TYPE_PROVENANCE_KEY] = provenance
        mdata.write_h5mu(destination)
    finally:
        mdata.file.close()


def _partial_cell_type_provenance(
    *, annotated_count: int = 1, unannotated_count: int = 1
) -> dict:
    return {
        "version": CELL_TYPE_PROVENANCE_VERSION,
        "unannotated_label": UNANNOTATED_CELL_TYPE,
        "sources": {
            "test_rna_protein": {
                "source_file": "source_cell_groups.csv",
                "source_url": "https://example.org/source_cell_groups.csv",
                "source_sha256": "1" * 64,
                "observation_id_column": "cell_id",
                "label_column": "group",
                "annotated_count": annotated_count,
                "unannotated_count": unannotated_count,
            }
        },
    }


def test_optional_cell_type_categorical_passes(tmp_path, write_h5mu, write_metadata):
    path = tmp_path / "with-cell-type.h5mu"
    _write_cell_type_variant(
        write_h5mu(),
        path,
        pd.Categorical(["T cell", "B cell"], categories=["T cell", "B cell"]),
    )

    outcome = validate_h5mu(path, load_metadata(write_metadata()))

    assert outcome.valid


def test_partial_source_cell_type_with_unannotated_provenance_passes(
    tmp_path, write_h5mu, write_metadata
):
    path = tmp_path / "partial-source-cell-type.h5mu"
    _write_cell_type_variant(
        write_h5mu(),
        path,
        pd.Categorical(
            ["T cell", UNANNOTATED_CELL_TYPE],
            categories=["T cell", UNANNOTATED_CELL_TYPE],
        ),
        _partial_cell_type_provenance(),
    )

    outcome = validate_h5mu(path, load_metadata(write_metadata()))

    assert outcome.valid


@pytest.mark.parametrize(
    ("categories", "provenance", "expected_code"),
    [
        (
            ["T cell", UNANNOTATED_CELL_TYPE],
            None,
            "missing_cell_type_provenance",
        ),
        (
            [UNANNOTATED_CELL_TYPE, "T cell"],
            _partial_cell_type_provenance(),
            "nonterminal_unannotated_cell_type",
        ),
        (
            ["T cell", UNANNOTATED_CELL_TYPE],
            _partial_cell_type_provenance(annotated_count=2, unannotated_count=1),
            "cell_type_provenance_count_mismatch",
        ),
    ],
)
def test_invalid_partial_source_cell_type_is_rejected(
    tmp_path,
    categories,
    provenance,
    expected_code,
    write_h5mu,
    write_metadata,
):
    path = tmp_path / f"invalid-{expected_code}.h5mu"
    _write_cell_type_variant(
        write_h5mu(),
        path,
        pd.Categorical(
            ["T cell", UNANNOTATED_CELL_TYPE],
            categories=categories,
        ),
        provenance,
    )

    outcome = validate_h5mu(path, load_metadata(write_metadata()))

    assert not outcome.valid
    assert expected_code in {issue.code for issue in outcome.errors}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_file", "../source_cell_groups.csv"),
        ("source_url", "file:///tmp/source_cell_groups.csv"),
        ("source_sha256", "A" * 64),
    ],
)
def test_unsafe_partial_source_provenance_is_rejected(
    tmp_path, field, value, write_h5mu, write_metadata
):
    provenance = _partial_cell_type_provenance()
    provenance["sources"]["test_rna_protein"][field] = value
    path = tmp_path / f"unsafe-{field}.h5mu"
    _write_cell_type_variant(
        write_h5mu(),
        path,
        pd.Categorical(
            ["T cell", UNANNOTATED_CELL_TYPE],
            categories=["T cell", UNANNOTATED_CELL_TYPE],
        ),
        provenance,
    )

    outcome = validate_h5mu(path, load_metadata(write_metadata()))

    assert not outcome.valid
    assert "invalid_cell_type_provenance" in {
        issue.code for issue in outcome.errors
    }


def test_orphan_cell_type_provenance_is_rejected(
    tmp_path, write_h5mu, write_metadata
):
    path = tmp_path / "orphan-cell-type-provenance.h5mu"
    _write_cell_type_variant(
        write_h5mu(),
        path,
        pd.Categorical(["T cell", "B cell"]),
        _partial_cell_type_provenance(),
    )

    outcome = validate_h5mu(path, load_metadata(write_metadata()))

    assert not outcome.valid
    assert "orphan_cell_type_provenance" in {
        issue.code for issue in outcome.errors
    }


@pytest.mark.parametrize(
    ("values", "expected_code"),
    [
        ([1, 2], "invalid_cell_type_dtype"),
        (pd.Categorical(["T cell", None]), "null_cell_type"),
        (pd.Categorical(["T cell", " B cell"]), "noncanonical_cell_type_label"),
        (
            pd.Categorical(
                ["T cell", "B cell"], categories=["T cell", "B cell", "Unused"]
            ),
            "unused_cell_type_category",
        ),
        (
            pd.Categorical(["T cell", "B cell"], ordered=True),
            "ordered_cell_type",
        ),
    ],
)
def test_invalid_optional_cell_type_is_rejected(
    tmp_path, values, expected_code, write_h5mu, write_metadata
):
    path = tmp_path / f"invalid-{expected_code}.h5mu"
    _write_cell_type_variant(write_h5mu(), path, values)

    outcome = validate_h5mu(path, load_metadata(write_metadata()))

    assert not outcome.valid
    assert expected_code in {issue.code for issue in outcome.errors}


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
        "technology": "Xenium",
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
        "technology": "Xenium",
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
