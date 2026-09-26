"""Read-only, generation-bound molecular visualization serving.

Source matrices are never opened here. All scientific validation belongs to the
offline preparation module; startup reads only the small publication index.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import sqlite3
import struct
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response

LOG = logging.getLogger(__name__)
VERSION = 1
MAGIC = b"ISCDCMO\0"
HEADER = struct.Struct("<8sHHI16s")
MEDIA_TYPE = "application/vnd.iscdc.molecular-array"
SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")
DTYPES = {
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
}


def safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Unsafe molecular artifact path")
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Unsafe molecular artifact path")
    return path


def encode_array(values, *, kind: int, states=None) -> bytes:
    import numpy as np

    values = np.asarray(values)
    dtype = values.dtype.name
    if dtype not in DTYPES or kind not in {1, 2}:
        raise ValueError("Unsupported molecular array")
    count = values.shape[-1]
    if kind == 1 and (values.shape != (2, count) or dtype != "float64"):
        raise ValueError("Coordinates must be a 2 × N Float64 array")
    if kind == 2 and (values.ndim != 1 or np.shape(states) != (count,)):
        raise ValueError("Values and states must have equal length")
    body = values.astype(values.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
    if kind == 2:
        states = np.asarray(states, dtype=np.uint8)
        if np.any(states > 2):
            raise ValueError("Unknown molecular measurement state")
        body += states.tobytes()
    return HEADER.pack(MAGIC, VERSION, kind, count, dtype.encode()) + body


def load_publication(root: Path | None, datasets) -> dict:
    if root is None:
        return {}
    try:
        index = json.loads((root / "publication.json").read_text())
        if index["version"] != VERSION or not isinstance(index["datasets"], dict):
            raise ValueError("Unsupported molecular publication version")
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    known = {d.dataset_id for d in datasets if d.dataset_type == "full"}
    result = {}
    for dataset_id, record in index["datasets"].items():
        if dataset_id not in known:
            continue
        try:
            directory = safe_path(root, record["directory"])
            manifest = json.loads((directory / "manifest.json").read_text())
            if (
                manifest["version"] != VERSION
                or manifest["dataset_id"] != dataset_id
                or not manifest["samples"]
                or not manifest["modalities"]
                or not SAFE_KEY.fullmatch(manifest["generation_id"])
            ):
                raise ValueError("Invalid molecular manifest")
            for sample in manifest["samples"]:
                if (
                    not SAFE_KEY.fullmatch(sample["key"])
                    or not isinstance(sample["id"], str)
                    or not isinstance(sample["count"], int)
                    or sample["count"] < 1
                ):
                    raise ValueError("Invalid molecular sample metadata")
                safe_path(directory, sample["coordinates"])
            for modality in manifest["modalities"]:
                if (
                    not SAFE_KEY.fullmatch(modality["name"])
                    or not isinstance(modality["n_vars"], int)
                    or modality["n_vars"] < 1
                    or not isinstance(modality["first_feature"], dict)
                ):
                    raise ValueError("Invalid molecular modality metadata")
                safe_path(directory, modality["matrix"])
                safe_path(directory, modality["index"])
            result[dataset_id] = {**manifest, "directory": directory}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOG.warning("Molecular visualization unavailable for %s: %s", dataset_id, exc)
    return result


def view_config(snapshot: dict, request: Request) -> dict:
    base = str(
        request.url_for(
            "molecular_coordinates",
            dataset_id=snapshot["dataset_id"],
            generation_id=snapshot["generation_id"],
            sample_key="_",
        )
    ).rsplit("/samples/_/coordinates", 1)[0]
    return {
        "kind": "molecular",
        "viewId": "molecular",
        "methodFamily": "molecular",
        "label": "Stored molecular value",
        "title": "Molecular distribution",
        "methodModal": "#molecular-method-modal",
        "annotationKind": "molecular",
        "datasetId": snapshot["dataset_id"],
        "generationId": snapshot["generation_id"],
        "yAxis": snapshot["y_axis"],
        "categories": [],
        "baseUrl": base,
        "modalities": [
            {k: m[k] for k in ("name", "value_type", "n_vars", "first_feature")}
            for m in snapshot["modalities"]
        ],
        "samples": [
            {
                "key": s["key"],
                "id": s["id"],
                "count": s["count"],
                "url": f"{base}/samples/{s['key']}/coordinates",
            }
            for s in snapshot["samples"]
        ],
        "initialSampleKey": snapshot["samples"][0]["key"],
    }


def feature_search(path: Path, q: str, offset: int, limit: int) -> dict:
    query = q.strip().casefold()
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True) as db:
        db.row_factory = sqlite3.Row
        if not query:
            sql, args = "SELECT * FROM features ORDER BY idx LIMIT ? OFFSET ?", [limit + 1, offset]
        elif len(query) < 3:
            # Indexable prefix ranges, including literal SQL wildcard characters.
            sql = """SELECT * FROM features WHERE idx IN (
                SELECT idx FROM features WHERE folded_id >= ? AND folded_id < ?
                UNION SELECT idx FROM features WHERE folded_label >= ? AND folded_label < ?
            ) ORDER BY (folded_id = ?) DESC, (folded_label = ?) DESC, idx LIMIT ? OFFSET ?"""
            args = [
                query,
                query + chr(0x10FFFF),
                query,
                query + chr(0x10FFFF),
                query,
                query,
                limit + 1,
                offset,
            ]
        else:
            # Resolve exact matches separately. Ordering the entire FTS join by
            # computed ranks materializes hundreds of thousands of ATAC hits.
            # FTS can instead stream rowids and stop as soon as a page is full.
            exact_count = db.execute(
                "SELECT count(*) FROM features WHERE folded_id=? OR folded_label=?",
                (query, query),
            ).fetchone()[0]
            rows = (
                db.execute(
                    "SELECT * FROM features WHERE folded_id=? OR folded_label=? "
                    "ORDER BY (folded_id=?) DESC, idx LIMIT ? OFFSET ?",
                    (query, query, query, limit + 1, offset),
                ).fetchall()
                if offset < exact_count
                else []
            )
            remaining = limit + 1 - len(rows)
            if remaining:
                rows.extend(
                    db.execute(
                        "SELECT f.* FROM search JOIN features f ON f.idx=search.rowid "
                        "WHERE search MATCH ? AND NOT (folded_id=? OR folded_label=?) "
                        "ORDER BY search.rowid LIMIT ? OFFSET ?",
                        (
                            '"' + query.replace('"', '""') + '"',
                            query,
                            query,
                            remaining,
                            max(0, offset - exact_count),
                        ),
                    ).fetchall()
                )
        if len(query) < 3:
            rows = db.execute(sql, args).fetchall()
    return {
        "items": [
            {"key": str(r["idx"]), "id": r["feature_id"], "label": r["label"]} for r in rows[:limit]
        ],
        "offset": offset,
        "nextOffset": offset + limit if len(rows) > limit else None,
    }


def read_vector(directory: Path, modality: dict, sample_key: str, feature: int) -> bytes:
    import h5py
    import numpy as np

    with h5py.File(safe_path(directory, modality["matrix"]), "r") as handle:
        rows = handle[f"samples/{sample_key}"][:]
        present = rows >= 0
        if handle.attrs["encoding"] == "csc":
            start, end = handle["indptr"][feature : feature + 2]
            column = np.zeros(int(handle.attrs["n_obs"]), dtype=handle["data"].dtype)
            column[handle["indices"][int(start) : int(end)]] = handle["data"][int(start) : int(end)]
        else:
            column = handle["values"][feature, :]
        values = np.zeros(len(rows), dtype=column.dtype)
        values[present] = column[rows[present]]
        states = np.where(present, 0, 1).astype(np.uint8)
        states[present & ~np.isfinite(values)] = 2
        return encode_array(values, kind=2, states=states)


class MolecularService:
    """Bound expensive reads and retain at most 128 MiB of encoded responses."""

    def __init__(self, snapshots: dict):
        self.snapshots = snapshots
        self.slots = threading.BoundedSemaphore(4)
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.cache_bytes = 0

    def resolve(self, dataset_id, generation_id, sample_key=None, modality_name=None):
        snapshot = self.snapshots.get(dataset_id)
        if snapshot is None or snapshot["generation_id"] != generation_id:
            raise HTTPException(404, "Molecular visualization not found")
        sample = next((s for s in snapshot["samples"] if s["key"] == sample_key), None)
        modality = next((m for m in snapshot["modalities"] if m["name"] == modality_name), None)
        if sample_key is not None and sample is None:
            raise HTTPException(404, "Molecular sample not found")
        if modality_name is not None and modality is None:
            raise HTTPException(404, "Molecular modality not found")
        return snapshot, sample, modality

    def vector(self, snapshot, sample, modality, feature, encoding):
        key = (
            snapshot["dataset_id"],
            snapshot["generation_id"],
            sample["key"],
            modality["name"],
            feature,
            encoding,
        )
        with self.slots:
            with self.lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key]
            value = read_vector(snapshot["directory"], modality, sample["key"], feature)
            if encoding == "gzip":
                value = gzip.compress(value, compresslevel=1, mtime=0)
            with self.lock:
                if key not in self.cache:
                    self.cache[key] = value
                    self.cache_bytes += len(value)
                while self.cache_bytes > 128 * 1024**2:
                    self.cache_bytes -= len(self.cache.popitem(last=False)[1])
            return value


def install_routes(application, snapshots: dict, preferred_encoding) -> None:
    service = MolecularService(snapshots)
    application.state.molecular_visualizations = snapshots
    router = APIRouter(
        prefix="/databases/{dataset_id}/molecular-visualization/{generation_id}",
        include_in_schema=False,
    )

    @router.get("/modalities/{modality_name}/features", name="molecular_features")
    def features(
        dataset_id: str,
        generation_id: str,
        modality_name: str,
        q: str = Query("", max_length=256),
        offset: int = Query(0, ge=0, le=2**31 - 1),
        limit: int = Query(50, ge=1, le=100),
    ):
        snapshot, _, modality = service.resolve(
            dataset_id, generation_id, modality_name=modality_name
        )
        if offset >= modality["n_vars"]:
            return {"items": [], "offset": offset, "nextOffset": None}
        try:
            with service.slots:
                return feature_search(
                    safe_path(snapshot["directory"], modality["index"]), q, offset, limit
                )
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise HTTPException(404, "Molecular feature index unavailable") from exc

    @router.api_route(
        "/samples/{sample_key}/coordinates", methods=["GET", "HEAD"], name="molecular_coordinates"
    )
    def coordinates(request: Request, dataset_id: str, generation_id: str, sample_key: str):
        snapshot, sample, _ = service.resolve(dataset_id, generation_id, sample_key)
        path = safe_path(snapshot["directory"], sample["coordinates"])
        if not path.is_file():
            raise HTTPException(404, "Molecular coordinates unavailable")
        return FileResponse(
            path,
            media_type=MEDIA_TYPE,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Content-Type-Options": "nosniff",
                "ETag": '"' + sample["sha256"] + '"',
            },
        )

    @router.api_route(
        "/samples/{sample_key}/modalities/{modality_name}/features/{feature_key}",
        methods=["GET", "HEAD"],
        name="molecular_values",
    )
    def values(
        request: Request,
        dataset_id: str,
        generation_id: str,
        sample_key: str,
        modality_name: str,
        feature_key: str,
    ):
        snapshot, sample, modality = service.resolve(
            dataset_id, generation_id, sample_key, modality_name
        )
        if not re.fullmatch(r"0|[1-9][0-9]{0,9}", feature_key):
            raise HTTPException(404, "Molecular feature not found")
        feature = int(feature_key)
        if feature >= modality["n_vars"]:
            raise HTTPException(404, "Molecular feature not found")
        encoding = preferred_encoding(request.headers.get("accept-encoding"), {"identity", "gzip"})
        if encoding is None:
            raise HTTPException(406, "No acceptable molecular encoding")
        identity = json.dumps(
            [dataset_id, generation_id, sample_key, modality_name, feature_key, encoding]
        ).encode()
        etag = '"' + hashlib.sha256(identity).hexdigest() + '"'
        headers = {
            "ETag": etag,
            "Cache-Control": "public, max-age=31536000, immutable",
            "Vary": "Accept-Encoding",
            "X-Content-Type-Options": "nosniff",
        }
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        try:
            payload = service.vector(snapshot, sample, modality, feature, encoding)
        except (OSError, KeyError, ValueError, IndexError) as exc:
            raise HTTPException(404, "Molecular values unavailable") from exc
        if encoding != "identity":
            headers["Content-Encoding"] = encoding
        headers["Content-Length"] = str(len(payload))
        return Response(
            b"" if request.method == "HEAD" else payload, media_type=MEDIA_TYPE, headers=headers
        )

    application.include_router(router)
