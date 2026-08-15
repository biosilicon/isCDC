from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .auxiliary import MANIFEST_VERSION
from .schemas import MetadataDocument
from .validation import ValidationOutcome


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_manifest(
    metadata: MetadataDocument,
    outcome: ValidationOutcome,
    file_size: int,
    sha256: str,
    imported_at: datetime,
    auxiliary_files: list[dict] | None = None,
) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": metadata.database.dataset_id,
        "imported_at": imported_at.isoformat(),
        "database": metadata.database_values(),
        "sample_ids": metadata.sample_ids,
        "n_obs": outcome.n_obs,
        "coordinate_dimensions": outcome.coordinate_dimensions,
        "modalities": {
            name: {
                "technology": summary.technology,
                "value_type": summary.value_type,
                "n_obs": summary.n_obs,
                "n_vars": summary.n_vars,
            }
            for name, summary in sorted(outcome.modalities.items())
        },
        "files": {
            "h5mu": {
                "name": "dataset.h5mu",
                "size": file_size,
                "sha256": sha256,
            },
            "metadata": {"name": "metadata.yaml"},
            "validation_report": {"name": "validation_report.json"},
            "checksum": {"name": "checksum.sha256"},
        },
        "auxiliary_files": auxiliary_files or [],
    }


def write_dataset_artifacts(
    directory: Path,
    metadata: MetadataDocument,
    outcome: ValidationOutcome,
    file_size: int,
    sha256: str,
    imported_at: datetime,
    checked_at: datetime,
    auxiliary_files: list[dict] | None = None,
) -> None:
    write_json(
        directory / "manifest.json",
        build_manifest(
            metadata,
            outcome,
            file_size,
            sha256,
            imported_at,
            auxiliary_files=auxiliary_files,
        ),
    )
    write_json(
        directory / "validation_report.json",
        outcome.report(checked_at.isoformat(), "dataset.h5mu"),
    )
    (directory / "checksum.sha256").write_text(
        f"{sha256}  dataset.h5mu\n", encoding="utf-8"
    )
