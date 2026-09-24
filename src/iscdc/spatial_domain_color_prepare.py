"""Offline preparation of the generation-bound domain display color snapshot."""

from __future__ import annotations

import colorsys
import csv
import gzip
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import cell_type_visualization as ct
from .spatial_domain_colors import METHOD, SNAPSHOT_NAME, entries_digest, snapshot_binding
from .spatial_domain_visualization import DomainVisualization, domain_directory


def _extra_colors(start: int):
    # Extend the inference palette without importing its scientific runtime.
    for code in range(start, start + 1024):
        rgb = colorsys.hsv_to_rgb((code * 0.61803398875) % 1, 0.65, 0.82)
        yield "#" + "".join(f"{round(value * 255):02X}" for value in rgb)
    # Fixed saturation/value has finitely many rounded RGB colors. This odd
    # multiplier traverses all RGB values if an unusually large palette exhausts it.
    for code in range(0x1000000):
        yield f"#{((start + code) * 0x9E3779) % 0x1000000:06X}"


def match_domain_colors(rna_categories, glue_categories, overlaps) -> dict[int, str]:
    """Maximize unchanged cell colors with positive-overlap, one-to-one matches.

    ``overlaps`` maps (RNA code, SpatialGLUE code) to a shared observation count.
    Sort codes before solving so manifest/category iteration order cannot break ties.
    Code zero is excluded even when supplied by a caller.
    """
    rna = {c["code"]: c["color"] for c in rna_categories if c["code"] > 0}
    glue = sorted(c["code"] for c in glue_categories if c["code"] > 0)
    rna_codes = sorted(rna)
    colors = {}
    if rna_codes and glue:
        weights = np.array(
            [[overlaps.get((r, g), 0) for g in glue] for r in rna_codes], dtype=np.int64
        )
        rows, columns = linear_sum_assignment(weights, maximize=True)
        for row, column in zip(rows, columns):
            if weights[row, column] > 0:
                colors[glue[column]] = rna[rna_codes[row]]

    reserved = {c["color"].upper() for c in rna_categories}
    reserved.update(c["color"].upper() for c in glue_categories if c["code"] == 0)
    candidates = _extra_colors(max([*rna_codes, *glue], default=0) + 1)
    for code in glue:
        if code not in colors:
            color = next(c for c in candidates if c not in reserved)
            colors[code] = color
            reserved.add(color)
    return colors


def _assignment_path(root: Path, snapshot: DomainVisualization, family: str) -> Path:
    manifest = snapshot.manifest
    combination = (
        manifest["provenance"]["combination_id"] if manifest["manifest_version"] == 3 else None
    )
    directory = (
        domain_directory(root, snapshot.dataset_id, family, combination_id=combination)
        / "generations"
        / snapshot.generation_id
    )
    path, _, _ = ct._validate_file_record(
        directory,
        manifest["assignments"],
        "color alignment assignments",
        required_path="assignments.tsv.gz",
    )
    return path


def _snapshot_colors(root, rna, glue):
    if rna.dataset_id != glue.dataset_id or rna.manifest["source"] != glue.manifest["source"]:
        raise ValueError("RNA and SpatialGLUE source bindings differ")
    rna_samples = {s["id"]: s for s in rna.manifest["samples"]}
    glue_samples = {s["id"]: s for s in glue.manifest["samples"]}
    if rna_samples.keys() != glue_samples.keys():
        raise ValueError("RNA and SpatialGLUE samples differ")
    overlaps = {sample_id: Counter() for sample_id in rna_samples}
    counts = Counter()
    with (
        gzip.open(_assignment_path(root, rna, "rna"), "rt", encoding="utf-8", newline="") as left,
        gzip.open(
            _assignment_path(root, glue, "spatialglue"), "rt", encoding="utf-8", newline=""
        ) as right,
    ):
        readers = [csv.DictReader(stream, delimiter="\t") for stream in (left, right)]
        for r, g in zip_longest(*readers):
            if (
                r is None
                or g is None
                or (r["observation_id"], r["sample_id"]) != (g["observation_id"], g["sample_id"])
            ):
                raise ValueError("RNA and SpatialGLUE observation identities/order differ")
            sample_id = r["sample_id"]
            counts[sample_id] += 1
            rna_code, glue_code = int(r["domain_id"]), int(g["domain_id"])
            if rna_code > 0 and glue_code > 0:
                overlaps[sample_id][rna_code, glue_code] += 1
    for sample_id, sample in rna_samples.items():
        if (
            counts[sample_id] != sample["count"]
            or counts[sample_id] != glue_samples[sample_id]["count"]
        ):
            raise ValueError("RNA and SpatialGLUE sample coverage differs")
    return {
        sample_id: match_domain_colors(
            sample["categories"], glue_samples[sample_id]["categories"], overlaps[sample_id]
        )
        for sample_id, sample in rna_samples.items()
    }


def build_domain_color_snapshot(root, rna_snapshots, glue_combinations):
    """Compute mappings offline, retaining per-combination failures for the operator."""
    entries, failures = [], []
    for dataset_id, combinations in sorted(glue_combinations.items()):
        rna = rna_snapshots.get(dataset_id)
        if rna is None:
            continue
        for combination_id, glue in sorted(combinations.items()):
            identity = {"dataset_id": dataset_id, "combination_id": combination_id}
            try:
                samples = _snapshot_colors(root, rna, glue)
                entries.append(
                    {
                        **identity,
                        "rna": snapshot_binding(rna),
                        "spatialglue": snapshot_binding(glue),
                        "samples": {
                            s: {str(c): color for c, color in colors.items()}
                            for s, colors in samples.items()
                        },
                    }
                )
            except (ValueError, TypeError, KeyError, OSError, EOFError, csv.Error) as exc:
                failures.append({**identity, "error": str(exc)})
    return {
        "snapshot_version": 1,
        "method": METHOD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "entries_sha256": entries_digest(entries),
        "failures": failures,
    }


def write_domain_color_snapshot(path, document):
    """Replace only a complete display snapshot, never inference sidecars."""
    if document["failures"]:
        raise ValueError(
            f"Color preparation failed; keeping previous snapshot: {document['failures']}"
        )
    ct._atomic_json(path, document)


def prepare_catalogue_colors(settings, *, output=None, domain_root=None):
    """Prepare published pairs, releasing each dataset's validated data in turn."""
    import json
    from time import perf_counter

    from .spatial_domain_annotation import catalogue_records
    from .spatial_domain_visualization import (
        load_spatial_domain_visualizations,
        load_spatialglue_combinations,
    )

    root = (domain_root or settings.spatial_domain_visualization_root).resolve(strict=True)
    records = catalogue_records(settings)
    document = build_domain_color_snapshot(root, {}, {})
    started = perf_counter()
    for index, record in enumerate(records, 1):
        rna = load_spatial_domain_visualizations(root, [record])
        if not rna:
            continue
        glue = load_spatialglue_combinations(root, [record])
        partial = build_domain_color_snapshot(root, rna, glue)
        document["entries"].extend(partial["entries"])
        document["failures"].extend(partial["failures"])
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(records)}",
                    "dataset_id": record["dataset_id"],
                    "mapped_combinations": len(partial["entries"]),
                    "failures": partial["failures"],
                    "elapsed_seconds": round(perf_counter() - started, 3),
                }
            ),
            flush=True,
        )
    if (domain_root or settings.spatial_domain_visualization_root).resolve(strict=True) != root:
        raise ValueError("Domain release changed during preparation; previous colors retained")
    document["entries_sha256"] = entries_digest(document["entries"])
    output = output or settings.database_path.parent / SNAPSHOT_NAME
    write_domain_color_snapshot(output, document)
    summary = {
        "output": str(output),
        "datasets": len({e["dataset_id"] for e in document["entries"]}),
        "combinations": len(document["entries"]),
        "samples": sum(len(e["samples"]) for e in document["entries"]),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    print(json.dumps(summary), flush=True)
    return summary
