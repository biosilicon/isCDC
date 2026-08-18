#!/usr/bin/env python3
"""Summarize pinned CELLxGENE Census datasets for reference curation."""

from __future__ import annotations

import argparse
import json

import cellxgene_census

TILEDB_CONFIG = {
    "sm.compute_concurrency_level": 3,
    "sm.io_concurrency_level": 3,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True, choices=("Homo sapiens", "Mus musculus"))
    parser.add_argument("--filter", required=True)
    parser.add_argument("--release", default="2025-11-08")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    value_filter = f"is_primary_data == True and ({args.filter})"
    with cellxgene_census.open_soma(
        census_version=args.release, tiledb_config=TILEDB_CONFIG
    ) as census:
        observations = cellxgene_census.get_obs(
            census,
            args.species,
            value_filter=value_filter,
            column_names=["dataset_id", "donor_id", "cell_type_ontology_term_id"],
        )
        datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    summaries = []
    for dataset_id, frame in observations.groupby("dataset_id", observed=True, sort=True):
        by_type = frame.groupby("cell_type_ontology_term_id", observed=True)[
            "donor_id"
        ].nunique()
        metadata = datasets[datasets["dataset_id"] == dataset_id].iloc[0]
        summaries.append(
            {
                "dataset_id": str(dataset_id),
                "dataset_version_id": str(metadata["dataset_version_id"]),
                "collection_id": str(metadata["collection_id"]),
                "title": str(metadata["dataset_title"]),
                "citation": str(metadata["citation"]),
                "cells": int(len(frame)),
                "donors": int(frame["donor_id"].nunique()),
                "cell_types": int(frame["cell_type_ontology_term_id"].nunique()),
                "cell_types_in_multiple_donors": int((by_type >= 2).sum()),
                "donor_counts": {
                    str(key): int(value)
                    for key, value in frame["donor_id"]
                    .astype(str)
                    .value_counts()
                    .head(20)
                    .items()
                },
            }
        )
    summaries.sort(
        key=lambda item: (
            item["cell_types_in_multiple_donors"], item["donors"], item["cells"]
        ),
        reverse=True,
    )
    print(json.dumps(summaries[: args.limit], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
