"""Offline spatial previews and small, fail-open startup manifests.

These previews describe measured coverage, not histological segmentation. No catalogue
or H5MU data are rewritten. Rendering parameters are part of the versioned contract.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .auxiliary import load_auxiliary_files
from .config import Settings
from .thumbnails import ThumbnailGenerationError

VERSION = 1
METHOD = "spatial-signal-v1"
DIRECTORY = "database_thumbnails/spatial"
PARAMETERS = {
    "max_dimension": 640,
    "padding_fraction": 0.04,
    "percentiles": [1, 99],
    "rna_opacity_power": 2,
    "density_neighbors": 16,
    "radius_spacing_fraction": 0.45,
    "radius_pixels": [0.75, 6.0],
    "supersampling": 4,
    "palette": "viridis-10-linear",
    "quality": 85,
    "webp_method": 6,
    "overlap": "mean_scaled_signal",
    "display_axes": "x-right-y-down",
}
# Fixed Viridis anchors; no plotting library or browser runtime is required.
PALETTE = np.array(
    [
        [68, 1, 84],
        [72, 40, 120],
        [62, 73, 137],
        [49, 104, 142],
        [38, 130, 142],
        [31, 158, 137],
        [53, 183, 121],
        [110, 206, 88],
        [181, 222, 43],
        [253, 231, 37],
    ],
    dtype=float,
)
LABELS = {"rna_signal": "RNA signal", "point_density": "Spatial point density"}
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_id(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", value):
        raise ThumbnailGenerationError("Unsafe dataset ID")


def database_records(settings: Settings) -> list[dict]:
    """Never initialize or migrate the catalogue from a preview command."""
    with sqlite3.connect(settings.database_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        version = db.execute(
            "SELECT value FROM catalogue_metadata WHERE key='schema_version'"
        ).fetchone()
        if version is None or version[0] != "5":
            raise ThumbnailGenerationError("Spatial previews require catalogue schema 5")
        return [
            dict(row)
            for row in db.execute(
                "SELECT *, EXISTS(SELECT 1 FROM modalities m "
                "WHERE m.dataset_id=datasets.dataset_id "
                "AND m.name='rna') AS has_rna FROM datasets "
                "WHERE dataset_type='full' ORDER BY dataset_id"
            )
        ]


def _source(record: dict, settings: Settings) -> Path:
    _safe_id(record["dataset_id"])
    directory = (settings.data_root / record["storage_dir"]).resolve()
    if directory.parent != settings.data_root.resolve():
        raise ThumbnailGenerationError("Unsafe dataset directory")
    path = directory / "dataset.h5mu"
    if path.is_symlink() or not path.is_file():
        raise ThumbnailGenerationError("Missing or unsafe H5MU")
    return path


def skip_reason(record: dict, settings: Settings) -> str | None:
    source = _source(record, settings)
    if any(
        item.auxiliary_id == "he_wsi"
        for item in load_auxiliary_files(source.parent, record["dataset_id"])
    ):
        return "registered_he_wsi"
    native = settings.static_dir / "database_thumbnails" / (record["dataset_id"] + ".webp")
    if native.exists():
        return "native_thumbnail"
    return None


def row_totals(matrix) -> np.ndarray:  # noqa: ANN001
    """Stream dense, CSR or CSC HDF5 counts without densifying the matrix."""
    import h5py

    def check(values: np.ndarray) -> None:
        if not np.isfinite(values).all() or (values < 0).any():
            raise ThumbnailGenerationError("RNA counts must be finite and nonnegative")
        if not np.equal(values, np.floor(values)).all():
            raise ThumbnailGenerationError("RNA counts must be integral")

    if isinstance(matrix, h5py.Dataset):
        if matrix.ndim != 2:
            raise ThumbnailGenerationError("RNA matrix must be two-dimensional")
        result = np.zeros(matrix.shape[0], dtype=np.float64)
        for start in range(0, len(result), 2048):
            block = matrix[start : start + 2048]
            check(block)
            result[start : start + len(block)] = block.sum(axis=1, dtype=np.float64)
    else:
        encoding = matrix.attrs.get("encoding-type")
        shape = matrix.attrs.get("shape")
        if encoding not in {"csr_matrix", "csc_matrix"} or len(shape) != 2:
            raise ThumbnailGenerationError("Unsupported sparse RNA encoding")
        result = np.zeros(shape[0], dtype=np.float64)
        indptr = matrix["indptr"][:]
        if encoding == "csr_matrix":
            for start in range(0, len(result), 2048):
                end = min(start + 2048, len(result))
                values = matrix["data"][indptr[start] : indptr[end]]
                check(values)
                # reduceat cannot represent empty rows; cumulative differences can.
                cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
                result[start:end] = (
                    cumulative[indptr[start + 1 : end + 1] - indptr[start]]
                    - cumulative[indptr[start:end] - indptr[start]]
                )
        else:
            for start in range(0, len(matrix["data"]), 1_000_000):
                values = matrix["data"][start : start + 1_000_000]
                check(values)
                indices = matrix["indices"][start : start + len(values)]
                np.add.at(result, indices, values)
    if not np.isfinite(result).all():
        raise ThumbnailGenerationError("Nonfinite RNA totals")
    return result


def read_inputs(path: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    import h5py
    from anndata.io import read_elem

    with h5py.File(path, "r") as handle:
        obs = read_elem(handle["obs"])
        xy = np.asarray(handle["obsm/spatial"], dtype=np.float64)
        if xy.shape != (len(obs), 2) or not np.isfinite(xy).all():
            raise ThumbnailGenerationError("Expected finite, observation-aligned 2D coordinates")
        if not obs.index.is_unique or obs["sample_id"].isna().any():
            raise ThumbnailGenerationError("Invalid observation IDs or samples")
        if obs["sample_id"].nunique() != 1:
            raise ThumbnailGenerationError("Multiple sample coordinate systems are unsupported")
        database = read_elem(handle["uns/database"])
        audit = {"obs_ids": np.asarray(obs.index, dtype=str), "source_xy": xy.copy()}
        if "rna" in handle["mod"]:
            rna = handle["mod/rna"]
            assay = read_elem(rna["uns/assay"])
            if assay.get("value_type") != "counts":
                raise ThumbnailGenerationError("RNA previews require declared raw counts")
            rna_obs = read_elem(rna["obs"])
            positions = obs.index.get_indexer(rna_obs.index)
            if not rna_obs.index.is_unique or (positions < 0).any():
                raise ThumbnailGenerationError("RNA observation alignment failed")
            totals = row_totals(rna["X"])
            if len(totals) != len(positions):
                raise ThumbnailGenerationError("RNA matrix and observation axis disagree")
            # Partially shared files show precisely the RNA-measured observations.
            xy = xy[positions]
            audit["obs_ids"] = audit["obs_ids"][positions]
            audit["source_xy"] = xy.copy()
            audit["top_level_positions"] = positions
            var = read_elem(rna["var"])
            audit["feature_ids"] = np.asarray(var.index, dtype=str)
            audit["rna_row_positions"] = np.arange(len(totals))
            kind = "rna_signal"
        else:
            totals = np.empty(0)
            audit["top_level_positions"] = np.arange(len(obs))
            kind = "point_density"
        transform = [1.0, 1.0]
        if database.get("coordinate_unit") == "array_index":
            if database.get("entry_id") == "S065":
                if (
                    not np.equal(xy, np.floor(xy)).all()
                    or len(np.unique(np.mod(xy[:, 0] + xy[:, 1], 2))) != 1
                ):
                    raise ThumbnailGenerationError("S065 Visium index geometry failed validation")
                transform = [1.0, float(np.sqrt(3))]
            elif database.get("dataset_id") != "GSE213264_human_gbm_spatial_citeseq":
                raise ThumbnailGenerationError("Unreviewed array-index geometry")
        xy = xy * transform
        info = {
            "kind": kind,
            "sample_id": str(obs["sample_id"].iloc[0]),
            "local_field": database.get("entry_id") == "S051",
            "coordinate_scale": transform,
            "coordinate_unit": database["coordinate_unit"],
            "top_level_n_obs": len(obs),
            "plotted_n_obs": len(xy),
            "embedded_dataset_id": database["dataset_id"],
        }
        return xy, totals, info, audit


def scale_signal(signal: np.ndarray) -> tuple[np.ndarray, list[float]]:
    logged = np.log1p(signal)
    lower, upper = np.percentile(logged, PARAMETERS["percentiles"])
    if upper <= lower:
        # A constant positive signal must not turn an entire sampled tissue invisible.
        scaled = np.where(signal > 0, 0.5, 0.0)
    else:
        scaled = np.clip((logged - lower) / (upper - lower), 0, 1)
    return scaled, [float(lower), float(upper)]


def render_preview(
    xy: np.ndarray,
    signal: np.ndarray,
    kind: str,
) -> tuple[Image.Image, dict, dict]:
    from scipy.spatial import cKDTree

    if kind not in LABELS or xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ThumbnailGenerationError("Invalid preview coordinates or kind")
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    if len(unique) < 2 or np.any(np.ptp(unique, axis=0) <= 0):
        raise ThumbnailGenerationError("Coordinates do not span a two-dimensional view")
    tree = cKDTree(unique)
    nearest = tree.query(unique, k=2)[0][:, 1]
    spacing = float(np.median(nearest))
    k = min(PARAMETERS["density_neighbors"], len(unique) - 1)
    if kind == "point_density":
        distance = tree.query(unique, k=k + 1)[0][:, -1]
        signal = (k / (np.pi * distance**2))[inverse]
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape != (len(xy),) or not np.isfinite(signal).all() or (signal < 0).any():
        raise ThumbnailGenerationError("Invalid spatial signal")
    scaled, bounds = scale_signal(signal)
    span = np.ptp(xy, axis=0)
    padding = max(span) * PARAMETERS["padding_fraction"]
    extent = span + 2 * padding
    scale = (PARAMETERS["max_dimension"] - 1) / max(extent)
    dimensions = np.maximum(1, np.rint(extent * scale).astype(int) + 1)
    radius = float(
        np.clip(
            spacing * scale * PARAMETERS["radius_spacing_fraction"], *PARAMETERS["radius_pixels"]
        )
    )
    ss = PARAMETERS["supersampling"]
    centers = np.rint((xy - xy.min(axis=0) + padding) * scale * ss).astype(int)
    width, height = map(int, dimensions * ss)
    sums = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.uint32)
    radius_hi = radius * ss
    for dy in range(-int(np.ceil(radius_hi)), int(np.ceil(radius_hi)) + 1):
        for dx in range(-int(np.ceil(radius_hi)), int(np.ceil(radius_hi)) + 1):
            if dx * dx + dy * dy > radius_hi * radius_hi:
                continue
            x, y = centers[:, 0] + dx, centers[:, 1] + dy
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            np.add.at(sums, (y[valid], x[valid]), scaled[valid])
            np.add.at(weights, (y[valid], x[valid]), 1)
    occupied = weights > 0
    values = sums[occupied] / weights[occupied]
    colors = np.column_stack(
        [
            np.interp(values, np.linspace(0, 1, len(PALETTE)), PALETTE[:, channel])
            for channel in range(3)
        ]
    )
    alpha = (
        values ** PARAMETERS["rna_opacity_power"] if kind == "rna_signal" else np.ones(len(values))
    )
    if kind == "rna_signal" and bounds[0] == bounds[1]:
        alpha = (values > 0).astype(float)
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    pixels[occupied] = np.rint(colors * alpha[:, None] + 255 * (1 - alpha[:, None]))
    image = Image.fromarray(pixels).resize(tuple(dimensions), Image.Resampling.LANCZOS)
    return (
        image,
        {
            "log1p_bounds": bounds,
            "nearest_spacing": spacing,
            "radius_px": radius,
            "dimensions": dimensions.tolist(),
            "density_neighbors_used": k,
            "coordinate_bounds": [xy.min(axis=0).tolist(), xy.max(axis=0).tolist()],
            "pixels_per_coordinate_unit": float(scale),
            "padding": float(padding),
        },
        {"display_xy": xy, "signal": signal, "scaled_signal": scaled},
    )


def _publish(image: Image.Image, manifest: dict, directory: Path, *, force: bool) -> None:
    """Lock publication, stage both files, and restore the previous pair on failure.

    The manifest is the commit marker. A crash between replacements leaves a checksum
    mismatch, which discovery hides rather than showing mismatched image semantics.
    """
    if directory.is_symlink():
        raise ThumbnailGenerationError("Unsafe preview directory")
    directory.mkdir(parents=True, exist_ok=True)
    dataset_id = manifest["dataset_id"]
    _safe_id(dataset_id)
    paths = [directory / f"{dataset_id}.{extension}" for extension in ("webp", "json")]
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if any(path.is_symlink() for path in paths):
            raise ThumbnailGenerationError("Unsafe preview destination")
        if not force and any(path.exists() for path in paths):
            raise ThumbnailGenerationError("Preview exists; use --force to replace it")
        previous = [path.read_bytes() if path.exists() else None for path in paths]
        with tempfile.TemporaryDirectory(prefix=f".{dataset_id}-", dir=directory) as temporary:
            staged = [Path(temporary) / path.name for path in paths]
            image.save(staged[0], format="WEBP", quality=85, method=6)
            with Image.open(staged[0]) as check:
                check.load()
                if check.format != "WEBP" or check.mode != "RGB" or check.size != image.size:
                    raise ThumbnailGenerationError("Generated image failed validation")
            manifest["image_sha256"] = sha256_file(staged[0])
            staged[1].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            for path in staged:
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            replaced = []
            try:
                for source, destination in zip(staged, paths, strict=True):
                    os.replace(source, destination)
                    replaced.append(destination)
                os.fsync(descriptor)
            except OSError:
                for index, path in enumerate(paths):
                    if path not in replaced:
                        continue
                    if previous[index] is None:
                        path.unlink(missing_ok=True)
                    else:
                        restore = Path(temporary) / f"restore-{index}"
                        restore.write_bytes(previous[index])
                        os.replace(restore, path)
                raise
    finally:
        os.close(descriptor)


def generate_spatial_thumbnail(
    record: dict,
    settings: Settings,
    *,
    output_dir: Path | None = None,
    audit_dir: Path | None = None,
    force: bool = False,
) -> dict:
    reason = skip_reason(record, settings)
    if reason:
        raise ThumbnailGenerationError(f"Use existing image workflow: {reason}")
    source = _source(record, settings)
    dataset_id = record["dataset_id"]
    destination = output_dir if output_dir is not None else settings.static_dir / DIRECTORY
    if output_dir is None and destination.resolve() != settings.static_dir.resolve() / DIRECTORY:
        raise ThumbnailGenerationError("Unsafe preview directory")
    if not force and any(
        (destination / f"{dataset_id}.{ext}").exists() for ext in ("webp", "json")
    ):
        raise ThumbnailGenerationError("Preview exists; use --force to replace it")
    source_stat = source.stat()
    if sha256_file(source) != record["sha256"]:
        raise ThumbnailGenerationError("Actual H5MU checksum differs from catalogue")
    xy, signal, info, audit = read_inputs(source)
    if info.pop("embedded_dataset_id") != dataset_id:
        raise ThumbnailGenerationError("Embedded dataset ID differs from catalogue")
    image, rendered, arrays = render_preview(xy, signal, info["kind"])
    current_stat = source.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (
        current_stat.st_size,
        current_stat.st_mtime_ns,
    ):
        image.close()
        raise ThumbnailGenerationError("H5MU changed during generation")
    manifest = {
        "manifest_version": VERSION,
        "method": METHOD,
        "dataset_id": dataset_id,
        "source_sha256": record["sha256"],
        "parameters": PARAMETERS,
        **info,
        **rendered,
    }
    audit_root = (
        audit_dir
        if audit_dir is not None
        else (settings.database_path.parent / "spatial_thumbnail_audits")
    )
    audit_root.mkdir(parents=True, exist_ok=True)
    # Content-addressed audit names retain earlier runs across force replacements.
    array_digest = hashlib.sha256()
    for key, value in sorted({**audit, **arrays}.items()):
        array_digest.update(key.encode())
        array_digest.update(np.ascontiguousarray(value).tobytes())
    manifest["audit_arrays_sha256"] = array_digest.hexdigest()
    audit_path = audit_root / f"{dataset_id}-{array_digest.hexdigest()}.npz"
    if not audit_path.exists():
        with audit_path.open("xb") as stream:
            np.savez_compressed(stream, **audit, **arrays)
    try:
        _publish(image, manifest, destination, force=force)
    finally:
        image.close()
    return {
        "dataset_id": dataset_id,
        "destination": str(destination / f"{dataset_id}.webp"),
        "audit": str(audit_path),
        **manifest,
    }


def discover_spatial_thumbnails(settings: Settings, records: list[dict]) -> dict[str, dict]:
    result = {}
    directory = settings.static_dir / DIRECTORY
    if (
        directory.resolve() != settings.static_dir.resolve() / DIRECTORY
        or directory.is_symlink()
        or not directory.is_dir()
    ):
        return result
    for record in records:
        if record["dataset_type"] != "full":
            continue
        dataset_id = record["dataset_id"]
        try:
            _safe_id(dataset_id)
            path = directory / f"{dataset_id}.webp"
            sidecar = directory / f"{dataset_id}.json"
            if not path.exists() and not sidecar.exists():
                continue
            if path.is_symlink() or sidecar.is_symlink() or sidecar.stat().st_size > 32768:
                raise ValueError("Unsafe spatial preview files")
            manifest = json.loads(sidecar.read_text())
            if (
                type(manifest["manifest_version"]) is not int
                or manifest["manifest_version"] != VERSION
                or manifest["method"] != METHOD
                or manifest["dataset_id"] != dataset_id
                or manifest["source_sha256"] != record["sha256"]
                or manifest["parameters"] != PARAMETERS
                or manifest["kind"] not in LABELS
                or (manifest["kind"] == "rna_signal") != bool(record["has_rna"])
                or type(manifest["local_field"]) is not bool
                or manifest["local_field"] != (record["entry_id"] == "S051")
                or manifest["top_level_n_obs"] != record["n_obs"]
                or type(manifest["plotted_n_obs"]) is not int
                or not 1 <= manifest["plotted_n_obs"] <= record["n_obs"]
                or path.stat().st_size > 8_000_000
                or sha256_file(path) != manifest["image_sha256"]
            ):
                raise ValueError("Invalid or stale spatial preview manifest")
            with Image.open(path) as image:
                if (
                    image.format != "WEBP"
                    or image.mode != "RGB"
                    or list(image.size) != manifest["dimensions"]
                    or min(image.size) < 1
                    or max(image.size) != 640
                ):
                    raise ValueError("Invalid spatial preview image")
                image.load()
            result[dataset_id] = {
                "path": f"{DIRECTORY}/{dataset_id}.webp",
                "label": LABELS[manifest["kind"]],
                "kind": manifest["kind"],
                "local_field": manifest["local_field"],
            }
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            ThumbnailGenerationError,
            Image.DecompressionBombError,
        ):
            logger.warning("Ignoring invalid spatial preview for %s", dataset_id, exc_info=True)
    return result
