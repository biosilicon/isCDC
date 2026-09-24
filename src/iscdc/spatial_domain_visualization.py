"""Versioned, read-only spatial-domain snapshots; no scientific runtime imports."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from . import cell_type_visualization as ct

POINT_MAGIC = b"ISCDCSD\0"
POINT_MEDIA_TYPE = "application/vnd.iscdc.spatial-domain-points"
METHODS = {"single_cell": "BANKSY", "near_cellular": "GraphST", "spot_level": "GraphST"}
LOG = logging.getLogger(__name__)
DomainVisualizationError = ct.CellTypeVisualizationError


def method_for_resolution(resolution: str) -> str:
    try:
        return METHODS[resolution]
    except KeyError as exc:
        raise DomainVisualizationError(
            "An explicit biological spatial resolution is required"
        ) from exc


def encode_points(x, y, labels) -> bytes:
    return POINT_MAGIC + ct.encode_points(x, y, labels)[8:]


def decode_points(payload: bytes) -> ct.PointData:
    if payload[:8] != POINT_MAGIC:
        raise DomainVisualizationError("Invalid spatial-domain point magic")
    result = ct.decode_points(ct.POINT_MAGIC + payload[8:])
    if result.confidence is not None:
        raise DomainVisualizationError("Spatial domains must not declare confidence")
    return result


def build_point_representations(payload: bytes) -> dict[str, bytes]:
    decode_points(payload)
    return {
        "identity": payload,
        "gzip": gzip.compress(payload, compresslevel=9, mtime=0),
        "br": ct._compress_brotli(payload),
    }


def obs_order_sha256(ids) -> str:
    digest = hashlib.sha256()
    for obs_id in ids:
        encoded = str(obs_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def file_record(name: str, payload: bytes) -> dict:
    return {"path": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


@dataclass(frozen=True)
class DomainVisualization:
    dataset_id: str
    generation_id: str
    manifest: Mapping[str, Any]
    report: Mapping[str, Any]
    samples: Mapping[str, ct.CellTypeSample]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DomainVisualizationError(message)


def domain_directory(
    root: Path, dataset_id: str, method_family: str = "rna", *, combination_id=None
) -> Path:
    base = root / ct._safe_name(dataset_id, "dataset_id")
    _require(method_family in {"rna", "spatialglue"}, "Unknown domain method family")
    _require(not base.is_symlink(), "Unsafe domain directory")
    if method_family == "spatialglue":
        base = base / "methods" / "spatialglue"
        _require(not base.parent.is_symlink() and not base.is_symlink(), "Unsafe method directory")
    if combination_id is not None:
        from .spatialglue_config import combination_id as canonical_combination

        _require(method_family == "spatialglue", "Only SpatialGlue has combinations")
        _require(
            canonical_combination(combination_id.split("__")) == combination_id,
            "Invalid combination identity",
        )
        base = base / "combinations" / combination_id
        _require(not base.parent.is_symlink(), "Unsafe combination directory")
        _require(not base.parent.is_symlink() and not base.is_symlink(), "Unsafe method directory")
    return base


def _load_generation(
    directory: Path, dataset: object, method_family: str = "rna", combination_id=None
) -> DomainVisualization:
    m = ct._read_json(directory / "manifest.json", "domain manifest")
    ct._strict_object(
        m,
        "domain manifest",
        required={
            "manifest_version",
            "dataset_id",
            "generation_id",
            "generated_at",
            "source",
            "method",
            "coordinates",
            "samples",
            "assignments",
            "report",
            "provenance",
        },
    )
    expect = ct._dataset_expectation(dataset)
    _require(expect.dataset_type == "full", "Only full Databases support spatial domains")
    multimodal = method_family == "spatialglue"
    _require(
        m["manifest_version"] in ((2, 3) if multimodal else (1,)),
        "Unsupported domain manifest version",
    )
    _require(m["dataset_id"] == expect.dataset_id, "Domain dataset ID mismatch")
    _require(m["generation_id"] == directory.name, "Domain generation ID mismatch")
    ct._timestamp(m["generated_at"], "generated_at")
    src = ct._strict_object(
        m["source"],
        "source",
        required={
            "sha256",
            "obs_order_sha256",
            "observation_count",
            "sample_ids",
            "spatial_unit",
        },
    )
    _require(src["sha256"] == expect.source_sha256, "Stale domain source checksum")
    ct._sha256(src["obs_order_sha256"], "source.obs_order_sha256")
    _require(src["observation_count"] == expect.observation_count, "Observation count mismatch")
    _require(src["sample_ids"] == list(expect.sample_ids), "Source sample mismatch")
    _require(expect.coordinate_dimensions == 2, "Spatial domains require 2D coordinates")
    resolution = ct._dataset_value(dataset, "spatial_unit")
    _require(src["spatial_unit"] == resolution, "Spatial resolution mismatch")
    if not multimodal:
        _require(m["method"] == method_for_resolution(resolution), "Incorrect domain method")
    coord = ct._strict_object(
        m["coordinates"], "coordinates", required={"system", "unit", "y_axis"}
    )
    _require(
        coord["system"] == "cartesian" and coord["y_axis"] in {"up", "down"},
        "Unsupported domain coordinate convention",
    )
    ct._string(coord["unit"], "coordinate unit")
    expected_unit = ct._dataset_value(dataset, "coordinate_unit")
    if expected_unit is not None:
        _require(coord["unit"] == expected_unit, "Coordinate unit mismatch")
    provenance = ct._strict_object(
        m["provenance"],
        "provenance",
        required={
            "environment_lock_sha256",
            "packages",
            "parameters",
        }
        | (
            {
                "input_modalities",
                "unused_modalities",
                "input_value_types",
                "device",
                "adapter_sha256",
            }
            if multimodal
            else {"input_modality"}
        ),
        optional=(
            {"combination_id", "recipe_version", "partial_run"}
            if multimodal
            else {"adapter_sha256"}
        ),
    )
    ct._sha256(provenance["environment_lock_sha256"], "environment_lock_sha256")
    if "adapter_sha256" in provenance:
        ct._sha256(provenance["adapter_sha256"], "adapter_sha256")
    if multimodal:
        from .spatialglue_config import modality_records, select_modalities

        modalities = select_modalities(dataset, provenance["input_modalities"])
        _require(modalities == provenance["input_modalities"], "Incorrect modality order")
        if m["manifest_version"] == 3:
            from .spatialglue_config import combination_id as canonical_combination

            _require(
                combination_id == canonical_combination(modalities)
                and provenance.get("combination_id") == combination_id,
                "Combination identity mismatch",
            )
            _require(provenance.get("recipe_version") == 1, "Unknown preprocessing recipe version")
        else:
            _require(combination_id is None, "Legacy result cannot masquerade as a combination")
        available = modality_records(dataset)
        _require(
            provenance["unused_modalities"] == sorted(set(available) - set(modalities)),
            "Unused modality mismatch",
        )
        _require(
            provenance["input_value_types"]
            == {name: available[name]["value_type"] for name in modalities},
            "Input value type mismatch",
        )
        _require(
            m["method"] == ("SpatialGlue_3M" if len(modalities) == 3 else "SpatialGlue"),
            "Incorrect SpatialGlue model",
        )
        _require(
            isinstance(provenance["device"], dict)
            and str(provenance["device"].get("device", "")).startswith("cuda:")
            and bool(provenance["device"].get("uuid")),
            "Missing CUDA provenance",
        )
    else:
        _require(provenance["input_modality"] == "rna", "Expected RNA input")
    _require(
        isinstance(provenance["packages"], dict) and isinstance(provenance["parameters"], dict),
        "Invalid provenance",
    )
    report_path, _, _ = ct._validate_file_record(
        directory, m["report"], "report", required_path="report.json"
    )
    report = ct._read_json(report_path, "report")
    _require(
        report.get("report_version") == m["manifest_version"] and report.get("status") == "passed",
        "Invalid domain report",
    )
    for key, expected in (
        ("dataset_id", expect.dataset_id),
        ("generation_id", directory.name),
        ("source_sha256", expect.source_sha256),
    ):
        _require(report.get(key) == expected, f"Report {key} mismatch")
    _require(isinstance(report.get("samples"), dict), "Missing sample diagnostics")
    ct._canonical_json(report)
    assignment_path, _, _ = ct._validate_file_record(
        directory, m["assignments"], "assignments", required_path="assignments.tsv.gz"
    )
    assignments: dict[str, list[tuple[str, int, str]]] = {s: [] for s in expect.sample_ids}
    obs_ids: list[str] = []
    with gzip.open(assignment_path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require(
            reader.fieldnames == ["observation_id", "sample_id", "domain_id", "reason"],
            "Invalid assignment columns",
        )
        for row in reader:
            _require(len(obs_ids) < expect.observation_count, "Too many assignments")
            obs_id = ct._string(row["observation_id"], "observation_id")
            _require(row["sample_id"] in assignments, "Foreign assignment sample")
            code = int(row["domain_id"])
            _require(code >= 0 and (row["reason"] == "") == (code > 0), "Invalid analysis status")
            _require(
                row["reason"]
                in (
                    {"", "missing_modality", "zero_counts"}
                    if multimodal
                    else {"", "missing_rna", "zero_counts"}
                ),
                "Unknown exclusion reason",
            )
            assignments[row["sample_id"]].append((obs_id, code, row["reason"]))
            obs_ids.append(obs_id)
    _require(
        len(obs_ids) == expect.observation_count and len(set(obs_ids)) == len(obs_ids),
        "Missing or duplicate domain assignments",
    )
    _require(obs_order_sha256(obs_ids) == src["obs_order_sha256"], "Assignment order mismatch")
    if expect.obs_order_sha256:
        _require(
            src["obs_order_sha256"] == expect.obs_order_sha256, "Source observation order mismatch"
        )
    _require(isinstance(m["samples"], list), "Invalid domain samples")
    samples = {}
    seen_ids = []
    for sample in m["samples"]:
        ct._strict_object(
            sample,
            "sample",
            required={
                "key",
                "id",
                "count",
                "bounds",
                "categories",
                "representations",
            },
        )
        key = ct._safe_name(sample["key"], "sample.key", key=True)
        sample_id = sample["id"]
        _require(
            key not in samples and sample_id in assignments and sample_id not in seen_ids,
            "Duplicate or foreign sample",
        )
        seen_ids.append(sample_id)
        rows = assignments[sample_id]
        count = ct._integer(sample["count"], "sample.count", minimum=1)
        _require(count == len(rows), "Sample count mismatch")
        bounds = tuple(ct._finite(v, "bounds") for v in sample["bounds"])
        _require(len(bounds) == 4, "Invalid bounds")
        categories = sample["categories"]
        _require(isinstance(categories, list) and bool(categories), "Missing domain categories")
        codes, counts = set(), {}
        for category in categories:
            ct._strict_object(category, "category", required={"code", "label", "color", "count"})
            code = ct._integer(category["code"], "code")
            _require(code < 65536 and code not in codes, "Invalid or duplicate domain code")
            codes.add(code)
            expected_label = f"Domain {code}" if code else "Not analyzed"
            _require(category["label"] == expected_label, "Invalid domain label")
            _require(
                isinstance(category["color"], str)
                and ct._COLOR.fullmatch(category["color"]) is not None,
                "Invalid domain color",
            )
            counts[code] = ct._integer(category["count"], "category.count", minimum=1)
        observed = {}
        for _, code, _ in rows:
            observed[code] = observed.get(code, 0) + 1
        _require(counts == observed, "Assignment category counts mismatch")
        _require(set(sample["representations"]) == {"identity", "gzip", "br"}, "Missing encodings")
        representations = {}
        canonical = None
        for encoding, record in sample["representations"].items():
            ct._strict_object(
                record,
                "representation",
                required={
                    "path",
                    "size",
                    "sha256",
                    "encoding",
                    "content_size",
                    "content_sha256",
                },
            )
            _require(record["encoding"] == encoding, "Encoding mismatch")
            path, size, sha = ct._validate_file_record(
                directory, {k: record[k] for k in ("path", "size", "sha256")}, "point file"
            )
            _require(record["content_size"] == 32 + 10 * count, "Invalid point content size")
            payload = ct._identity_payload(encoding, path.read_bytes(), record["content_size"])
            _require(
                hashlib.sha256(payload).hexdigest() == record["content_sha256"],
                "Point digest mismatch",
            )
            if canonical is None:
                canonical = payload
                points = decode_points(payload)
                _require(points.point_count == count, "Point count mismatch")
                _require(
                    points.type_ids == tuple(row[1] for row in rows), "Point/assignment mismatch"
                )
                actual_bounds = (min(points.x), min(points.y), max(points.x), max(points.y))
                _require(actual_bounds == bounds, "Point bounds mismatch")
            else:
                _require(payload == canonical, "Encoding content mismatch")
            representations[encoding] = ct.PointRepresentation(
                path, encoding, size, sha, record["content_size"], record["content_sha256"]
            )
        _require(sample_id in report["samples"], "Sample report missing")
        diagnostic = report["samples"][sample_id]
        _require(isinstance(diagnostic, dict), "Invalid sample report")
        for name in ("parameters", "preprocessing", "domains"):
            _require(isinstance(diagnostic.get(name), dict), f"Missing report {name}")
        _require(
            isinstance(diagnostic.get("warnings"), list)
            and all(isinstance(w, str) for w in diagnostic["warnings"]),
            "Invalid warnings",
        )
        _require(
            diagnostic.get("analyzed") == count - counts.get(0, 0)
            and diagnostic.get("not_analyzed") == counts.get(0, 0),
            "QC coverage mismatch",
        )
        _require(diagnostic.get("n_domains") == len(codes - {0}), "QC domain count mismatch")
        _require(set(diagnostic["domains"]) == {str(c) for c in codes - {0}}, "QC domains mismatch")
        for code, domain in diagnostic["domains"].items():
            _require(
                isinstance(domain, dict) and domain.get("count") == counts[int(code)],
                "QC domain size mismatch",
            )
            components = ct._integer(
                domain.get("spatial_components"), "spatial_components", minimum=1
            )
            _require(components <= domain["count"], "Invalid spatial component count")
        genes = diagnostic.get("selected_genes")
        _require(
            isinstance(genes, list)
            and len(genes)
            >= (
                0
                if multimodal and "rna" not in modalities
                else (1 if m["manifest_version"] == 3 else 2)
            )
            and all(isinstance(g, str) and g.strip() == g and g for g in genes),
            "Invalid selected RNA features",
        )
        _require(
            len(set(genes)) == len(genes)
            and obs_order_sha256(genes) == diagnostic.get("selected_genes_sha256"),
            "Selected feature digest mismatch",
        )
        if multimodal:
            _validate_multimodal_sample(directory, diagnostic, modalities, rows, count)
        samples[key] = ct.CellTypeSample(
            key,
            sample_id,
            count,
            bounds,
            MappingProxyType(counts),
            MappingProxyType(representations),
        )
    _require(seen_ids == list(expect.sample_ids), "Sample coverage mismatch")
    _require(set(report["samples"]) == set(expect.sample_ids), "Report sample coverage mismatch")
    return DomainVisualization(
        expect.dataset_id,
        directory.name,
        MappingProxyType(m),
        MappingProxyType(report),
        MappingProxyType(samples),
    )


def _validate_multimodal_sample(directory, diagnostic, modalities, rows, count):
    features = diagnostic.get("features")
    _require(
        isinstance(features, dict) and set(features) == set(modalities), "Missing modality features"
    )
    for name, feature in features.items():
        path, _, _ = ct._validate_file_record(directory, feature["file"], f"{name} features")
        expected_count = ct._integer(feature["count"], "feature count", minimum=1)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            names = []
            for line in stream:
                _require(len(names) < expected_count, "Too many modality features")
                names.append(ct._string(line.rstrip("\n"), "feature ID"))
        _require(
            len(names) == expected_count and len(set(names)) == expected_count,
            "Invalid modality feature identities",
        )
        _require(
            obs_order_sha256(names) == feature["ids_sha256"], "Modality feature digest mismatch"
        )
        if name == "rna":
            _require(names == diagnostic["selected_genes"], "RNA feature list mismatch")
    coverage = diagnostic.get("coverage", {})
    for reason in ("missing_modality", "zero_counts"):
        _require(
            coverage.get(reason) == sum(row[2] == reason for row in rows),
            "Pairing exclusion count mismatch",
        )
    _require(set(coverage.get("modalities", {})) == set(modalities), "Missing modality coverage")
    for values in coverage["modalities"].values():
        missing = ct._integer(values.get("missing"), "missing modality count")
        zero = ct._integer(values.get("zero_counts"), "zero modality count")
        _require(missing + zero <= count, "Invalid modality coverage")
    clustering = diagnostic.get("clustering", {})
    _require(
        clustering.get("flavor") == "igraph"
        and clustering.get("directed") is False
        and clustering.get("n_iterations") == -1,
        "Invalid SpatialGlue clustering",
    )
    _require(
        diagnostic["parameters"].get("input_modalities") == modalities,
        "Parameter modality mismatch",
    )
    for digest in diagnostic.get("stage_sha256", {}).values():
        ct._sha256(digest, "stage_sha256")
    ct._validate_file_record(directory, diagnostic["joint_embedding"], "joint embedding")


def load_spatial_domain_visualization(
    root: Path, dataset: object, *, method_family: str = "rna", combination_id=None
) -> DomainVisualization:
    dataset_id = ct._safe_name(ct._dataset_value(dataset, "dataset_id"), "dataset_id")
    base = domain_directory(root, dataset_id, method_family, combination_id=combination_id)
    _require(not base.is_symlink(), "Unsafe domain directory")
    status = ct._read_json(base / "status.json", "domain status")
    _require(
        status.get("status_version") == 1 and status.get("state") == "success",
        "No successful domain generation",
    )
    _require(status.get("dataset_id") == dataset_id, "Status dataset mismatch")
    generation = ct._safe_name(status.get("generation_id"), "generation_id")
    directory = base / "generations" / generation
    _require(
        not directory.is_symlink() and not directory.parent.is_symlink(),
        "Unsafe generation directory",
    )
    manifest_path = ct._safe_file(directory, "manifest.json", "manifest")
    _require(
        ct._file_digest(manifest_path)[1] == status.get("manifest_sha256"),
        "Manifest digest mismatch",
    )
    return _load_generation(directory, dataset, method_family, combination_id)


def load_spatial_domain_visualizations(
    root: Path, datasets, *, method_family: str = "rna"
) -> dict[str, DomainVisualization]:
    result = {}
    for dataset in datasets:
        dataset_id = ct._dataset_value(dataset, "dataset_id")
        try:
            if not (domain_directory(root, dataset_id, method_family) / "status.json").exists():
                continue
            result[dataset_id] = load_spatial_domain_visualization(
                root, dataset, method_family=method_family
            )
        except (ValueError, TypeError, KeyError, OSError, EOFError, csv.Error) as exc:
            LOG.warning("Ignoring %s spatial-domain sidecar %s: %s", method_family, dataset_id, exc)
    return result


def publish_generation(
    root: Path,
    dataset: object,
    manifest: dict,
    files: Mapping[str, bytes],
    *,
    method_family: str = "rna",
    combination_id=None,
) -> DomainVisualization:
    dataset_id = ct._safe_name(manifest["dataset_id"], "dataset_id")
    generation = ct._safe_name(manifest["generation_id"], "generation_id")
    base = domain_directory(root, dataset_id, method_family, combination_id=combination_id)
    target = base / "generations"
    _require(not base.is_symlink() and not target.is_symlink(), "Unsafe publication directory")
    target.mkdir(parents=True, exist_ok=True)
    _require(not (target / generation).exists(), "Generations are immutable")
    staging_parent = Path(tempfile.mkdtemp(prefix=".staging-", dir=target))
    staging = staging_parent / generation
    staging.mkdir()
    try:
        for name, payload in files.items():
            relative = Path(name)
            _require(
                not relative.is_absolute()
                and all(ct._SAFE_NAME.fullmatch(p) for p in relative.parts),
                "Unsafe artifact path",
            )
            _require(name != "manifest.json", "Reserved artifact name")
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        encoded = ct._canonical_json(manifest) + b"\n"
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _load_generation(staging, dataset, method_family, combination_id)
        os.rename(staging, target / generation)
        ct._atomic_json(
            base / "status.json",
            {
                "status_version": 1,
                "state": "success",
                "dataset_id": dataset_id,
                "generation_id": generation,
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return load_spatial_domain_visualization(
            root, dataset, method_family=method_family, combination_id=combination_id
        )
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def assignment_bytes(ids, sample_ids, codes, reasons) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(["observation_id", "sample_id", "domain_id", "reason"])
    writer.writerows(zip(ids, sample_ids, codes, reasons, strict=True))
    return gzip.compress(stream.getvalue().encode(), mtime=0)


def publish_method_failure(
    root, dataset_id, error, *, method_family, details=None, combination_id=None
):
    base = domain_directory(root, dataset_id, method_family, combination_id=combination_id)
    failure = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex
    parent = base / "failures"
    _require(not parent.is_symlink(), "Unsafe failure directory")
    directory = parent / failure
    directory.mkdir(parents=True)
    report = {
        "dataset_id": dataset_id,
        "failure_id": failure,
        "status": "failed",
        "method_family": method_family,
        "error": str(error)[:2000],
        "details": details or {},
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    ct._atomic_json(directory / "report.json", report)
    ct._atomic_json(
        base / "status.json",
        {
            "status_version": 1,
            "state": "failure",
            "dataset_id": dataset_id,
            "failure_id": failure,
            "updated_at": report["failed_at"],
            "report": file_record(
                f"failures/{failure}/report.json", (directory / "report.json").read_bytes()
            ),
        },
    )


def load_spatialglue_combinations(root, datasets):
    """Read each independent current combination, including a legacy result as fallback."""
    from .spatialglue_config import MODALITIES
    from .spatialglue_config import combination_id as canonical_combination

    result = {}
    for dataset in datasets:
        dataset_id = ct._dataset_value(dataset, "dataset_id")
        base = domain_directory(root, dataset_id, "spatialglue")
        snapshots = {}
        parent = base / "combinations"
        if parent.exists() and not parent.is_symlink():
            for path in sorted(parent.iterdir()):
                try:
                    snapshot = load_spatial_domain_visualization(
                        root, dataset, method_family="spatialglue", combination_id=path.name
                    )
                    snapshots[path.name] = snapshot
                except (ValueError, TypeError, KeyError, OSError, EOFError, csv.Error) as exc:
                    LOG.warning(
                        "Ignoring SpatialGlue combination %s/%s: %s", dataset_id, path.name, exc
                    )
        if (base / "status.json").exists():
            try:
                old = load_spatial_domain_visualization(root, dataset, method_family="spatialglue")
                key = canonical_combination(old.manifest["provenance"]["input_modalities"])
                # A newer failed status must not resurrect an older successful result.
                if not (parent / key / "status.json").exists():
                    snapshots.setdefault(key, old)
            except (ValueError, TypeError, KeyError, OSError, EOFError, csv.Error) as exc:
                LOG.warning("Ignoring legacy SpatialGlue %s: %s", dataset_id, exc)
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
