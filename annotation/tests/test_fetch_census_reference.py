import h5py
import pandas as pd
from scipy import sparse

from annotation.fetch_census_reference import (
    _target_count_depth_metadata,
    collapse_duplicate_genes,
    select_partitions,
)


def _recipe() -> dict:
    return {
        "holdout_donor": "holdout",
        "sampling": {
            "seed": 17,
            "min_train_cells_per_type": 2,
            "min_holdout_cells_per_type": 2,
            "max_train_cells_per_type": 2,
            "max_holdout_cells_per_type": 2,
        },
    }


def test_partition_selection_is_deterministic_and_excludes_generic_terms():
    rows = []
    join_id = 1
    for donor in ("train_a", "train_b", "holdout"):
        for label, ontology in (("T cell", "CL:0000084"), ("B cell", "CL:0000236")):
            for _ in range(4):
                rows.append(
                    {
                        "soma_joinid": join_id,
                        "donor_id": donor,
                        "cell_type": label,
                        "cell_type_ontology_term_id": ontology,
                    }
                )
                join_id += 1
    rows.extend(
        [
            {
                "soma_joinid": join_id,
                "donor_id": "train_a",
                "cell_type": "unknown",
                "cell_type_ontology_term_id": "unknown",
            },
            {
                "soma_joinid": join_id + 1,
                "donor_id": "holdout",
                "cell_type": "cell",
                "cell_type_ontology_term_id": "CL:0000000",
            },
        ]
    )
    frame = pd.DataFrame(rows)

    first = select_partitions(frame, _recipe())
    second = select_partitions(frame.sample(frac=1, random_state=3), _recipe())

    assert first["soma_joinid"].tolist() == second["soma_joinid"].tolist()
    assert first["partition"].value_counts().to_dict() == {"train": 4, "holdout": 4}
    assert set(first["cell_type_ontology_term_id"]) == {"CL:0000084", "CL:0000236"}


def test_duplicate_gene_symbols_are_merged_without_densifying():
    matrix = sparse.csr_matrix([[1, 2, 0], [0, 3, 4]])
    var = pd.DataFrame(
        {
            "feature_id": ["ENSMUSG1", "ENSMUSG2", "ENSMUSG3"],
            "feature_name": ["A", "A", "B"],
        }
    )

    collapsed, genes = collapse_duplicate_genes(matrix, var, "feature_name")

    assert sparse.issparse(collapsed)
    assert genes["gene_id"].tolist() == ["A", "B"]
    assert collapsed.toarray().tolist() == [[3, 0], [3, 4]]
    assert genes["source_feature_id"].tolist() == ["ENSMUSG1,ENSMUSG2", "ENSMUSG3"]


def test_target_count_depth_summary_reads_sparse_rows_without_densifying(tmp_path):
    path = tmp_path / "matrix.h5"
    matrix = sparse.csr_matrix([[1, 0, 2], [0, 0, 0], [4, 5, 0], [0, 2, 0]])
    with h5py.File(path, "w") as handle:
        group = handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["shape"] = matrix.shape
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
    with h5py.File(path, "r") as handle:
        summary = _target_count_depth_metadata(handle["X"], 20)

    assert summary["observation_count"] == 4
    assert summary["nonzero_observation_count"] == 3
    assert summary["zero_observation_count"] == 1
    assert summary["quantiles"]["q50"] == 2.5
    assert summary["deterministic_positive_depth_sample"] == [2, 3, 9]
