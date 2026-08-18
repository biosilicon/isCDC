#!/usr/bin/env python3
"""List categorical CELLxGENE Census observation values for filter curation."""

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
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--contains", default="")
    parser.add_argument("--release", default="2025-11-08")
    args = parser.parse_args()
    columns = ("tissue", "tissue_general", "disease", "development_stage")
    with cellxgene_census.open_soma(
        census_version=args.release, tiledb_config=TILEDB_CONFIG
    ) as census:
        observations = cellxgene_census.get_obs(
            census,
            args.species,
            value_filter=f"dataset_id == '{args.dataset_id}'",
            column_names=list(columns),
        )
    needle = args.contains.casefold()
    result = {}
    for column in columns:
        series = observations[column]
        values = series.cat.categories if hasattr(series, "cat") else series.unique()
        result[column] = sorted(
            str(value) for value in values if needle in str(value).casefold()
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
