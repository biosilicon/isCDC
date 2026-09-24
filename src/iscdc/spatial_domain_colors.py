"""Read precomputed display colors without scanning observations or solving matches."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from . import cell_type_visualization as ct

LOG = logging.getLogger(__name__)
SNAPSHOT_NAME = "spatial_domain_colors.json"
METHOD = "maximum-overlap-one-to-one-v1"


def snapshot_binding(snapshot) -> dict[str, str]:
    """Fingerprint loaded metadata, including source, assignments and palette."""
    return {
        "generation_id": snapshot.generation_id,
        "manifest_sha256": hashlib.sha256(ct._canonical_json(dict(snapshot.manifest))).hexdigest(),
    }


def entries_digest(entries) -> str:
    return hashlib.sha256(ct._canonical_json(entries)).hexdigest()


def load_domain_color_overrides(path: Path, rna_snapshots, glue_combinations):
    """Read the offline-approved palette; do not validate or recompute at startup."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        result = {}
        for entry in document["entries"]:
            dataset_id = entry["dataset_id"]
            combination_id = entry["combination_id"]
            if dataset_id not in rna_snapshots or combination_id not in glue_combinations.get(
                dataset_id, {}
            ):
                continue
            result.setdefault(dataset_id, {})[combination_id] = {
                sample_id: {int(code): color for code, color in colors.items()}
                for sample_id, colors in entry["samples"].items()
            }
        return result
    except (ValueError, TypeError, KeyError, OSError) as exc:
        LOG.warning("Unable to read domain colors from %s: %s", path, exc)
        return {}
