"""Load previously validated publication metadata for website startup.

These readers do not audit artifacts, decode points, or compare source hashes.
Publication and offline audit commands retain their strict loaders.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from . import cell_type_visualization as ct
from .auxiliary import AuxiliaryFile
from .difficulty_snapshot import ChallengeDifficulty, DifficultySnapshot
from .spatial_domain_visualization import DomainVisualization
from .spatial_thumbnails import DIRECTORY, LABELS
from .spatialglue_config import MODALITIES

LOG = logging.getLogger(__name__)
READ_ERRORS = (OSError, ValueError, KeyError, TypeError)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _generation(base):
    try:
        status = _json(base / "status.json")
    except FileNotFoundError:
        return None
    if status["state"] != "success":
        return None
    directory = base / "generations" / status["generation_id"]
    manifest = _json(directory / "manifest.json")
    report = _json(directory / manifest["report"]["path"])
    return directory, manifest, report


def _samples(directory, manifest, *, domain=False):
    samples = {}
    for sample in manifest["samples"]:
        representations = {
            encoding: ct.PointRepresentation(
                directory / record["path"],
                encoding,
                record["size"],
                record["sha256"],
                record["content_size"],
                record["content_sha256"],
            )
            for encoding, record in sample["representations"].items()
        }
        counts = (
            {category["code"]: category["count"] for category in sample["categories"]}
            if domain
            else {int(code): count for code, count in sample["category_counts"].items()}
        )
        samples[sample["key"]] = ct.CellTypeSample(
            sample["key"],
            sample["id"],
            sample["count"],
            tuple(sample["bounds"]),
            MappingProxyType(counts),
            MappingProxyType(representations),
        )
    return MappingProxyType(samples)


def load_cell_type_visualizations(root: Path, datasets):
    result = {}
    for dataset in datasets:
        dataset_id = ct._dataset_value(dataset, "dataset_id")
        try:
            current = _generation(root / dataset_id)
            if current is None:
                continue
            directory, manifest, report = current
            coordinates = manifest["coordinates"]
            result[dataset_id] = ct.CellTypeVisualization(
                dataset_id,
                directory.name,
                manifest["generated_at"],
                manifest["annotation"]["kind"],
                manifest["annotation"]["method"],
                coordinates["system"],
                coordinates["unit"],
                coordinates["y_axis"],
                _samples(directory, manifest),
                tuple(ct.CellTypeCategory(**category) for category in manifest["categories"]),
                directory / "manifest.json",
                directory / manifest["report"]["path"],
                directory / manifest["inference"]["path"] if "inference" in manifest else None,
                MappingProxyType(report),
                MappingProxyType(manifest["provenance"]),
            )
        except READ_ERRORS as exc:
            LOG.warning("Unable to read cell-type metadata for %s: %s", dataset_id, exc)
    return result


def _domain(base, dataset_id):
    current = _generation(base)
    if current is None:
        return None
    directory, manifest, report = current
    return DomainVisualization(
        dataset_id,
        directory.name,
        MappingProxyType(manifest),
        MappingProxyType(report),
        _samples(directory, manifest, domain=True),
    )


def load_spatial_domain_visualizations(root: Path, datasets, *, method_family="rna"):
    result = {}
    for dataset in datasets:
        dataset_id = ct._dataset_value(dataset, "dataset_id")
        base = root / dataset_id
        if method_family == "spatialglue":
            base = base / "methods" / "spatialglue"
        try:
            snapshot = _domain(base, dataset_id)
            if snapshot is not None:
                result[dataset_id] = snapshot
        except READ_ERRORS as exc:
            LOG.warning(
                "Unable to read %s domain metadata for %s: %s", method_family, dataset_id, exc
            )
    return result


def load_spatialglue_combinations(root: Path, datasets):
    result = {}
    for dataset in datasets:
        dataset_id = ct._dataset_value(dataset, "dataset_id")
        base = root / dataset_id / "methods" / "spatialglue"
        parent = base / "combinations"
        snapshots = {}
        for status in sorted(parent.glob("*/status.json")):
            try:
                snapshot = _domain(status.parent, dataset_id)
                if snapshot is not None:
                    snapshots[status.parent.name] = snapshot
            except READ_ERRORS as exc:
                LOG.warning("Unable to read SpatialGLUE metadata %s: %s", status, exc)
        try:
            legacy = _domain(base, dataset_id)
            if legacy is not None:
                key = "__".join(legacy.manifest["provenance"]["input_modalities"])
                # An explicit failed combination still supersedes its legacy result.
                if not (parent / key / "status.json").exists():
                    snapshots.setdefault(key, legacy)
        except READ_ERRORS as exc:
            LOG.warning("Unable to read legacy SpatialGLUE metadata for %s: %s", dataset_id, exc)
        if snapshots:
            result[dataset_id] = dict(
                sorted(
                    snapshots.items(),
                    key=lambda item: tuple(
                        MODALITIES.index(m)
                        for m in item[1].manifest["provenance"]["input_modalities"]
                    ),
                )
            )
    return result


def load_auxiliary_files(dataset_dir: Path, dataset_id: str):
    manifest = _json(dataset_dir / "manifest.json")
    return tuple(
        sorted(
            (
                AuxiliaryFile(
                    entry["id"],
                    entry["label"],
                    entry["name"],
                    entry["media_type"],
                    entry["size"],
                    entry["sha256"],
                    entry["source_url"],
                    datetime.fromisoformat(entry["retrieved_at"]),
                    dataset_dir / entry["name"],
                )
                for entry in manifest.get("auxiliary_files", [])
            ),
            key=lambda item: item.auxiliary_id,
        )
    )


def load_difficulty_snapshot(path: Path):
    report = _json(path)
    return DifficultySnapshot(
        report["report_version"],
        report["method_version"],
        datetime.fromisoformat(report["generated_at"]),
        report["parameters"]["input_modality"],
        MappingProxyType(
            {
                row["split_id"]: ChallengeDifficulty(
                    row["mean_auroc"],
                    row["domain_shift_score"],
                    row["difficulty_percentile"],
                )
                for row in report["challenges"]
                if row["status"] == "success"
            }
        ),
    )


def discover_spatial_thumbnails(settings, records):
    result = {}
    directory = settings.static_dir / DIRECTORY
    for record in records:
        if record["dataset_type"] != "full":
            continue
        dataset_id = record["dataset_id"]
        try:
            manifest = _json(directory / f"{dataset_id}.json")
            result[dataset_id] = {
                "path": f"{DIRECTORY}/{dataset_id}.webp",
                "label": LABELS[manifest["kind"]],
                "kind": manifest["kind"],
                "local_field": manifest["local_field"],
            }
        except FileNotFoundError:
            continue
        except READ_ERRORS as exc:
            LOG.warning("Unable to read spatial preview metadata for %s: %s", dataset_id, exc)
    return result
