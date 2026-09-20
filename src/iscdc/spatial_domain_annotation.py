"""Offline RNA spatial-domain inference. The website never imports algorithm packages."""

from __future__ import annotations

import argparse
import colorsys
import fcntl
import hashlib
import importlib.metadata
import json
import os
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import cell_type_visualization as ct
from .config import PROJECT_ROOT, Settings
from .spatial_domain_resources import managed_cpu_ids, resource_guard
from .spatial_domain_visualization import (
    assignment_bytes,
    build_point_representations,
    decode_points,
    encode_points,
    file_record,
    load_spatial_domain_visualization,
    method_for_resolution,
    obs_order_sha256,
    publish_generation,
)

ADAPTER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEFAULT_CONFIG = PROJECT_ROOT / "assets" / "spatial_domain" / "defaults.yaml"
LOCK_PATH = PROJECT_ROOT / "annotation" / "spatial_domain" / "requirements.lock.txt"
DEFAULTS = {
    "seed": 42,
    "resolution": 1.0,
    "n_pcs": 20,
    "expression_neighbors": 50,
    "n_top_genes": 3000,
    "target_sum": 10000,
    "threads": 40,
    "memory_budget_gb": 128,
    "banksy_lambda": 0.8,
    "banksy_neighbors": 30,
    "banksy_max_m": 1,
    "graphst_neighbors": 3,
    "graphst_epochs": 600,
    "graphst_learning_rate": 0.001,
    "graphst_dim_output": 64,
    "graphst_threads": 8,
    "y_axis": "up",
}


class DomainInferenceError(ValueError):
    pass


def enable_h5mu_null_reader():
    """Backport AnnData 0.12's HDF5 None reader without changing algorithm packages."""
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec

    spec = IOSpec("null", "0.1.0")
    if not _REGISTRY.has_read(h5py.Dataset, spec):
        _REGISTRY.register_read(h5py.Dataset, spec)(_read_h5mu_null)


def _read_h5mu_null(element, *, _reader):
    # The upstream writer stores None as an HDF5 null dataspace, not a numeric array.
    if element.shape is not None:
        raise DomainInferenceError("Invalid HDF5 null encoding: expected null dataspace")
    return None


def select_variable_genes(data, n_top_genes):
    import scanpy as sc

    attempts = []
    for span in (0.3, 0.5, 0.75, 1.0):
        try:
            sc.pp.highly_variable_genes(
                data, flavor="seurat_v3", n_top_genes=n_top_genes, span=span
            )
        except ValueError as exc:
            if not any(
                marker in str(exc).lower()
                for marker in ("svddc failed", "near singularities", "reciprocal condition number")
            ):
                raise
            attempts.append({"span": span, "error": str(exc)})
            print(f"HVG seurat_v3 span={span} numerical failure: {exc}", flush=True)
        else:
            attempts.append({"span": span, "status": "success"})
            print(f"HVG seurat_v3 selected span={span}; attempts={len(attempts)}", flush=True)
            return {"selected_span": span, "attempts": attempts}
    raise DomainInferenceError(f"Seurat v3 HVG failed for all LOESS spans: {attempts}")


def load_parameters(path: Path | None, dataset_id: str, sample_id: str | None = None) -> dict:
    params = dict(DEFAULTS)
    if path:
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict) or set(config) - {"defaults", "datasets"}:
            raise DomainInferenceError("Config must contain defaults and/or datasets")
        datasets = config.get("datasets", {})
        if not isinstance(datasets, dict):
            raise DomainInferenceError("datasets must be a mapping")
        overrides = datasets.get(dataset_id, {})
        if not isinstance(overrides, dict) or set(overrides) - {"parameters", "samples"}:
            raise DomainInferenceError("Dataset config accepts parameters and samples")
        if not isinstance(overrides.get("samples", {}), dict):
            raise DomainInferenceError("samples must be a mapping")
        for values in [
            config.get("defaults", {}),
            overrides.get("parameters", {}),
            overrides.get("samples", {}).get(sample_id, {}) if sample_id else {},
        ]:
            if not isinstance(values, dict) or set(values) - set(DEFAULTS):
                raise DomainInferenceError("Unknown spatial-domain parameter")
            params.update(values)
    for key, value in params.items():
        if key == "y_axis":
            if value not in {"up", "down"}:
                raise DomainInferenceError("y_axis must be up or down")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not 0 <= value < float("inf")
        ):
            raise DomainInferenceError(f"Invalid non-negative parameter {key}")
        elif isinstance(DEFAULTS[key], int) and not isinstance(value, int):
            raise DomainInferenceError(f"{key} must be an integer")
        elif value == 0 and key not in {"seed", "banksy_max_m", "banksy_lambda"}:
            raise DomainInferenceError(f"{key} must be positive")
    if params["seed"] >= 2**32 or params["n_top_genes"] < 2:
        raise DomainInferenceError("seed must be below 2**32 and n_top_genes at least 2")
    if params["banksy_lambda"] > 1 or params["banksy_max_m"] not in {0, 1}:
        raise DomainInferenceError("BANKSY requires lambda in [0,1] and max_m in {0,1}")
    return params


def catalogue_records(settings: Settings, ids: list[str] | None = None) -> list[dict]:
    with sqlite3.connect(f"file:{settings.database_path.resolve()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        records = []
        for row in conn.execute(
            "SELECT * FROM datasets WHERE dataset_type='full' ORDER BY dataset_id"
        ):
            if ids is not None and row["dataset_id"] not in ids:
                continue
            record = dict(row)
            record["sample_ids"] = json.loads(record["sample_ids"])
            record["modalities"] = {
                r["name"]: dict(r)
                for r in conn.execute(
                    "SELECT * FROM modalities WHERE dataset_id=?", (row["dataset_id"],)
                )
            }
            records.append(record)
    if ids is not None and set(ids) != {r["dataset_id"] for r in records}:
        raise DomainInferenceError("Unknown Database ID or non-full dataset")
    return records


def eligibility(record: dict) -> str | None:
    method_for_resolution(record["spatial_unit"])
    if record["coordinate_dimensions"] != 2:
        return "requires_2d_coordinates"
    rna = record["modalities"].get("rna")
    if rna is None:
        return "missing_rna"
    if rna["value_type"] != "counts":
        return "rna_preprocessing_not_verified"
    return None


def estimated_memory_bytes(n_obs: int, n_vars: int, method: str) -> int:
    # Includes upstream dense copies, graph symmetrization, autograd and PCA workspace.
    features = n_vars
    if method == "GraphST":
        return 16 * n_obs * n_obs * 8 + 24 * n_obs * features * 8 + 2**30
    return 24 * n_obs * features * 8 + n_obs * 100 * 16 + 2**30


def check_resources(n_obs: int, n_vars: int, method: str, params: dict) -> int:
    import psutil

    estimate = estimated_memory_bytes(n_obs, min(n_vars, params["n_top_genes"]), method)
    available = max(0, psutil.virtual_memory().available - 8 * 2**30)
    budget = min(int(params["memory_budget_gb"] * 2**30), available)
    if estimate > budget:
        raise DomainInferenceError(
            f"Resource preflight: estimated {estimate / 2**30:.1f} GiB exceeds "
            f"available budget {budget / 2**30:.1f} GiB; no downsampling performed"
        )
    return estimate


def prepare_rna(matrix, obs_ids, coordinates, params):
    import anndata as ad
    import numpy as np
    import scanpy as sc
    from scipy import sparse

    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    values = matrix.data
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise DomainInferenceError("RNA counts must be finite non-negative integers")
    valid = np.asarray(matrix.sum(axis=1)).ravel() > 0
    matrix = matrix[valid]
    means = np.asarray(matrix.mean(axis=0)).ravel()
    variance = np.asarray(matrix.power(2).mean(axis=0)).ravel() - means**2
    genes = np.flatnonzero((means > 0) & (variance > 1e-12))
    if matrix.shape[0] < 3 or len(genes) < 2:
        raise DomainInferenceError("Insufficient variable RNA features or nonzero observations")
    data = ad.AnnData(matrix[:, genes], obs={"id": np.asarray(obs_ids)[valid]})
    data.obs_names = np.asarray(obs_ids)[valid]
    data.obsm["spatial"] = np.asarray(coordinates)[valid]
    data.uns["domain_hvg"] = {"selected_span": None, "attempts": []}
    if len(genes) > params["n_top_genes"]:
        data.uns["domain_hvg"] = select_variable_genes(data, params["n_top_genes"])
        mask = data.var["highly_variable"].to_numpy()
        genes = genes[mask]
        data = data[:, mask].copy()
    data.var["highly_variable"] = True  # Explicitly preprocessed; GraphST must not repeat it.
    sc.pp.normalize_total(data, target_sum=params["target_sum"])
    sc.pp.log1p(data)
    return data, valid, genes


def matrix_sha256(matrix) -> str:
    """Hash numeric stages in bounded row blocks for cross-process replay diagnostics."""
    import numpy as np
    from scipy import sparse

    digest = hashlib.sha256()
    digest.update(f"{matrix.shape}:{matrix.dtype}".encode())
    if sparse.issparse(matrix):
        canonical = matrix.tocsr()
        if not canonical.has_canonical_format:
            canonical = canonical.copy()
            canonical.sum_duplicates()
            canonical.sort_indices()
        for component in (canonical.data, canonical.indices, canonical.indptr):
            digest.update(memoryview(np.ascontiguousarray(component)))
        return digest.hexdigest()
    for start in range(0, matrix.shape[0], 256):
        block = matrix[start : start + 256]
        if hasattr(block, "toarray"):
            block = block.toarray()
        digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()


def run_algorithm(data, method: str, params: dict):
    import random

    import numpy as np
    import scanpy as sc
    from sklearn.decomposition import PCA

    random.seed(params["seed"])
    np.random.seed(params["seed"])
    stages = {
        "preprocessed": matrix_sha256(data.X),
        "coordinates": matrix_sha256(data.obsm["spatial"]),
    }
    n = data.n_obs
    if method == "BANKSY":
        from banksy.embed_banksy import generate_banksy_matrix
        from banksy.initialize_banksy import initialize_banksy

        data.obs["x"] = data.obsm["spatial"][:, 0]
        data.obs["y"] = data.obsm["spatial"][:, 1]
        banksy = initialize_banksy(
            data,
            ("x", "y", "spatial"),
            num_neighbours=min(params["banksy_neighbors"], n - 1),
            nbr_weight_decay="scaled_gaussian",
            max_m=params["banksy_max_m"],
            plt_edge_hist=False,
            plt_nbr_weights=False,
            plt_agf_angles=False,
            plt_theta=False,
        )
        _, augmented = generate_banksy_matrix(
            data, banksy, [params["banksy_lambda"]], params["banksy_max_m"], verbose=False
        )
        matrix = augmented.X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
    else:
        import torch
        from GraphST.GraphST import GraphST
        from GraphST.preprocess import construct_interaction

        torch.set_num_threads(min(params["threads"], params["graphst_threads"]))
        torch.use_deterministic_algorithms(True)
        sc.pp.scale(data, zero_center=False, max_value=10)
        construct_interaction(data, n_neighbors=min(params["graphst_neighbors"], n - 1))
        model = GraphST(
            data,
            device=torch.device("cpu"),
            epochs=params["graphst_epochs"],
            learning_rate=params["graphst_learning_rate"],
            dim_output=params["graphst_dim_output"],
            random_seed=params["seed"],
            datatype="10X",
        )
        trained = model.train()
        matrix = trained.obsm["emb"]
    stages["method_features"] = matrix_sha256(matrix)
    if not np.isfinite(matrix).all():
        raise DomainInferenceError("Algorithm produced non-finite features")
    dims = min(params["n_pcs"], n - 1, matrix.shape[1] - 1)
    representation = PCA(
        n_components=dims, svd_solver="randomized", random_state=params["seed"]
    ).fit_transform(matrix)
    if not np.isfinite(representation).all() or not np.any(np.var(representation, axis=0) > 0):
        raise DomainInferenceError("Algorithm produced degenerate PCA coordinates")
    stages["pca"] = matrix_sha256(representation)
    data.obsm["domain_pca"] = representation
    sc.pp.neighbors(
        data,
        n_neighbors=min(params["expression_neighbors"], n - 1),
        use_rep="domain_pca",
        random_state=params["seed"],
    )
    stages["connectivities"] = matrix_sha256(data.obsp["connectivities"])
    data.uns["domain_stage_sha256"] = stages
    sc.tl.leiden(
        data,
        resolution=params["resolution"],
        random_state=params["seed"],
        key_added="domain",
        flavor="leidenalg",
        n_iterations=-1,
    )
    labels = data.obs["domain"].astype(str).to_numpy()
    ordered = sorted(set(labels), key=lambda k: min(data.obs_names[labels == k]))
    remap = {label: i + 1 for i, label in enumerate(ordered)}
    return np.array([remap[label] for label in labels], dtype=np.uint16), dims


def _diagnostics(coords, labels):
    import numpy as np
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import kneighbors_graph

    graph = kneighbors_graph(coords, n_neighbors=min(6, len(coords) - 1), include_self=False)
    graph = graph.maximum(graph.T)
    domains = {}
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        domains[str(int(label))] = {
            "count": len(idx),
            "spatial_components": int(
                connected_components(graph[idx][:, idx], directed=False, return_labels=False)
            ),
        }
    return domains


def _color(code: int) -> str:
    if code == 0:
        return "#8A929B"
    rgb = colorsys.hsv_to_rgb((code * 0.61803398875) % 1, 0.65, 0.82)
    return "#" + "".join(f"{round(v * 255):02X}" for v in rgb)


def generate(
    record: dict, settings: Settings, output_root: Path, config: Path | None, *, force=False
):
    """Run inside a resource-limited offline process; never mutate source files."""
    import mudata
    import numpy as np
    from scipy import sparse
    from threadpoolctl import threadpool_limits

    dataset_id = record["dataset_id"]
    ct._safe_name(dataset_id, "dataset_id")
    if (output_root / dataset_id / "status.json").exists() and not force:
        raise DomainInferenceError(
            "Existing domain status; use --force to create another generation"
        )
    started = time.monotonic()
    mdata = None
    try:
        reason = eligibility(record)
        if reason:
            raise DomainInferenceError(reason)
        params = load_parameters(config, dataset_id)
        if not LOCK_PATH.is_file():
            raise DomainInferenceError("Missing locked spatial-domain environment")
        for line in LOCK_PATH.read_text().splitlines():
            if not line or line.startswith(("#", "--")):
                continue
            name, version = line.split("==", 1)
            if importlib.metadata.version(name) != version:
                raise DomainInferenceError(f"Environment lock mismatch: {name} requires {version}")
        path = settings.data_root / record["storage_dir"] / "dataset.h5mu"
        if ct._file_digest(path)[1] != record["sha256"]:
            raise DomainInferenceError("Source checksum differs from catalogue")
        enable_h5mu_null_reader()
        mdata = mudata.read_h5mu(path, backed="r")
        ids = np.asarray(mdata.obs_names.astype(str))
        if len(ids) != record["n_obs"] or len(set(ids)) != len(ids):
            raise DomainInferenceError("Invalid top-level observation IDs")
        samples = mdata.obs["sample_id"].astype(str).to_numpy()
        coordinates = np.asarray(mdata.obsm["spatial"])
        if coordinates.shape != (len(ids), 2) or not np.isfinite(coordinates).all():
            raise DomainInferenceError("Invalid two-dimensional coordinates")
        rna = mdata.mod["rna"]
        rna_ids = rna.obs_names.astype(str)
        if not rna_ids.is_unique or not rna.var_names.is_unique or not set(rna_ids) <= set(ids):
            raise DomainInferenceError("Invalid RNA observation or feature identities")
        lookup = {value: i for i, value in enumerate(rna_ids)}
        codes = np.zeros(len(ids), dtype=np.uint16)
        reasons = np.full(len(ids), "missing_rna", dtype=object)
        sample_reports, sample_docs, files = {}, [], {}
        method = method_for_resolution(record["spatial_unit"])
        for sample_index, sample_id in enumerate(record["sample_ids"]):
            sp = load_parameters(config, dataset_id, sample_id)
            if any(sp[k] != params[k] for k in ("threads", "memory_budget_gb", "y_axis")):
                raise DomainInferenceError(
                    "Sample overrides cannot change resource limits or coordinate orientation"
                )
            all_idx = np.flatnonzero(samples == sample_id)
            rna_idx = np.array([i for i in all_idx if ids[i] in lookup], dtype=int)
            if not len(rna_idx):
                raise DomainInferenceError(f"Sample {sample_id} has no RNA observations")
            estimate = check_resources(len(rna_idx), rna.n_vars, method, sp)
            matrix = rna.X[[lookup[ids[i]] for i in rna_idx]]
            if not sparse.issparse(matrix):
                matrix = sparse.csr_matrix(matrix)
            runtime_params = {**sp, "threads": min(sp["threads"], len(os.sched_getaffinity(0)))}
            with threadpool_limits(limits=runtime_params["threads"]):
                data, valid, genes = prepare_rna(
                    matrix, ids[rna_idx], coordinates[rna_idx], runtime_params
                )
                labels, dims = run_algorithm(data, method, runtime_params)
                diagnostics = _diagnostics(data.obsm["spatial"], labels)
            analyzed_idx = rna_idx[valid]
            codes[analyzed_idx] = labels
            reasons[rna_idx] = "zero_counts"
            reasons[analyzed_idx] = ""
            counts = {
                int(code): int((codes[all_idx] == code).sum()) for code in np.unique(codes[all_idx])
            }
            selected_genes = [str(rna.var_names[i]) for i in genes]
            sample_reports[sample_id] = {
                "parameters": runtime_params,
                "stage_sha256": data.uns.get("domain_stage_sha256", {}),
                "n_pcs_actual": dims,
                "analyzed": len(analyzed_idx),
                "not_analyzed": len(all_idx) - len(analyzed_idx),
                "n_domains": len(diagnostics),
                "domains": diagnostics,
                "selected_genes": selected_genes,
                "selected_genes_sha256": obs_order_sha256(selected_genes),
                "estimated_peak_bytes": estimate,
                "warnings": ["Only one spatial domain detected"] if len(diagnostics) == 1 else [],
                "preprocessing": {
                    "input": "RNA counts",
                    "hvg": "seurat_v3 when features > n_top_genes",
                    "hvg_fit": data.uns["domain_hvg"],
                    "normalization": "normalize_total then log1p",
                    "scaling": "BANKSY component z-scores"
                    if method == "BANKSY"
                    else "scale zero_center=False max_value=10",
                },
            }
            if len(data.uns["domain_hvg"]["attempts"]) > 1:
                sample_reports[sample_id]["warnings"].append(
                    "Seurat v3 LOESS span increased after numerical fitting failure; see hvg_fit"
                )
            key = f"sample_{sample_index}"
            payload = encode_points(
                coordinates[all_idx, 0].tolist(),
                coordinates[all_idx, 1].tolist(),
                codes[all_idx].tolist(),
            )
            point = decode_points(payload)
            reps = {}
            for encoding, content in build_point_representations(payload).items():
                suffix = {"identity": ".bin", "gzip": ".bin.gz", "br": ".bin.br"}[encoding]
                name = f"points/{key}{suffix}"
                files[name] = content
                reps[encoding] = {
                    **file_record(name, content),
                    "encoding": encoding,
                    "content_size": len(payload),
                    "content_sha256": hashlib.sha256(payload).hexdigest(),
                }
            sample_docs.append(
                {
                    "key": key,
                    "id": sample_id,
                    "count": len(all_idx),
                    "bounds": [min(point.x), min(point.y), max(point.x), max(point.y)],
                    "categories": [
                        {
                            "code": c,
                            "label": f"Domain {c}" if c else "Not analyzed",
                            "color": _color(c),
                            "count": n,
                        }
                        for c, n in counts.items()
                    ],
                    "representations": reps,
                }
            )
        if set(samples) != set(record["sample_ids"]):
            raise DomainInferenceError("Source samples disagree with catalogue")
        if ct._file_digest(path)[1] != record["sha256"]:
            raise DomainInferenceError("Source changed during inference")
        now = datetime.now(timezone.utc)
        generation = now.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        report = {
            "report_version": 1,
            "dataset_id": dataset_id,
            "generation_id": generation,
            "source_sha256": record["sha256"],
            "status": "passed",
            "input_reader": "locked mudata/anndata with HDF5 null 0.1.0 compatibility",
            "samples": sample_reports,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cpu_seconds": (
                resource.getrusage(resource.RUSAGE_SELF).ru_utime
                + resource.getrusage(resource.RUSAGE_SELF).ru_stime
            ),
            "cpu_affinity_limit": len(os.sched_getaffinity(0)),
            "validation_scope": "Computational integrity, not biological ground-truth validation",
        }
        files["report.json"] = ct._canonical_json(report) + b"\n"
        files["assignments.tsv.gz"] = assignment_bytes(ids, samples, codes, reasons)
        packages = {
            p: importlib.metadata.version(p)
            for p in ("pybanksy", "GraphST", "scanpy", "numpy", "scipy", "torch", "leidenalg")
        }
        manifest = {
            "manifest_version": 1,
            "dataset_id": dataset_id,
            "generation_id": generation,
            "generated_at": now.isoformat(),
            "source": {
                "sha256": record["sha256"],
                "obs_order_sha256": obs_order_sha256(ids),
                "observation_count": len(ids),
                "sample_ids": record["sample_ids"],
                "spatial_unit": record["spatial_unit"],
            },
            "method": method,
            "coordinates": {
                "system": "cartesian",
                "unit": record["coordinate_unit"],
                "y_axis": params["y_axis"],
            },
            "samples": sample_docs,
            "assignments": file_record("assignments.tsv.gz", files["assignments.tsv.gz"]),
            "report": file_record("report.json", files["report.json"]),
            "provenance": {
                "environment_lock_sha256": ct._file_digest(LOCK_PATH)[1],
                "packages": packages,
                "adapter_sha256": ADAPTER_SHA256,
                "parameters": params,
                "input_modality": "rna",
            },
        }
        snapshot = publish_generation(output_root, record, manifest, files)
        return {
            "dataset_id": dataset_id,
            "state": "success",
            "generation_id": snapshot.generation_id,
            "method": method,
            "elapsed_seconds": report["elapsed_seconds"],
            "peak_rss_bytes": report["peak_rss_bytes"],
        }
    except Exception as exc:
        ct.publish_failure(
            output_root,
            dataset_id,
            str(exc)[:2000] or type(exc).__name__,
            stage="spatial_domain",
            category="inference",
            details={"exception": type(exc).__name__},
        )
        raise
    finally:
        if mdata is not None:
            mdata.file.close()


def add_cli_commands(subparsers):
    generate_parser = subparsers.add_parser(
        "generate-spatial-domain-visualization",
        help="Infer RNA spatial domains offline; requires the isolated spatial-domain environment.",
    )
    generate_parser.add_argument("dataset_id")
    generate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate_parser.add_argument("--output-root", type=Path)
    generate_parser.add_argument("--force", action="store_true")
    audit = subparsers.add_parser(
        "audit-spatial-domain-visualizations", help="Read-only eligibility and artifact audit."
    )
    audit.add_argument("dataset_ids", nargs="*")
    audit.add_argument("--all", action="store_true")
    audit.add_argument("--output-root", type=Path)


def execute_cli(args):
    settings = Settings.from_environment()
    root = args.output_root or settings.spatial_domain_visualization_root
    if args.command == "audit-spatial-domain-visualizations":
        if not args.all and not args.dataset_ids:
            raise DomainInferenceError("Specify Database IDs or --all")
        results = []
        for record in catalogue_records(settings, None if args.all else args.dataset_ids):
            row = {"dataset_id": record["dataset_id"]}
            try:
                row.update(
                    method=method_for_resolution(record["spatial_unit"]),
                    input_status=eligibility(record) or "eligible",
                )
            except ValueError:
                row.update(method=None, input_status="resolution_not_classified")
            try:
                snapshot = load_spatial_domain_visualization(root, record)
                row.update(state="success", generation_id=snapshot.generation_id)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                row.update(state="unavailable", reason=str(exc))
            results.append(row)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    if (
        Path(sys.prefix).name != "iscdc-spatial-domain"
        and os.environ.get("CONDA_DEFAULT_ENV") != "iscdc-spatial-domain"
    ):
        raise DomainInferenceError(
            "Run generation in the isolated iscdc-spatial-domain environment"
        )
    params = load_parameters(args.config, args.dataset_id)
    if os.environ.get("ISCDC_DOMAIN_WORKER") != "1":
        env = os.environ.copy()
        env["ISCDC_DOMAIN_WORKER"] = "1"
        env["PYTHONHASHSEED"] = str(params["seed"])
        available = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
        threads = min(params["threads"], max(1, available - 8))
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            env[key] = str(threads)
        command = [
            sys.executable,
            "-m",
            "iscdc.spatial_domain_annotation",
            "generate-spatial-domain-visualization",
            args.dataset_id,
            "--output-root",
            str(root),
        ]
        if args.config:
            command += ["--config", str(args.config)]
        if args.force:
            command.append("--force")
        import psutil

        env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iscdc-domain-mpl"))
        env.setdefault("NUMBA_THREADING_LAYER", "workqueue")
        budget = int(params["memory_budget_gb"] * 2**30)
        with subprocess.Popen(command, env=env) as process:
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    rss = monitored.memory_info().rss + sum(
                        child.memory_info().rss for child in monitored.children(recursive=True)
                    )
                    if rss > budget:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        ct.publish_failure(
                            root,
                            args.dataset_id,
                            "Runtime memory budget exceeded",
                            stage="resources",
                            category="resource_limit",
                            details={"observed_rss_bytes": rss, "budget_bytes": budget},
                        )
                        return 1
                except psutil.NoSuchProcess:
                    pass
                time.sleep(1)
            return process.returncode
    if hasattr(os, "sched_getaffinity"):
        leased = managed_cpu_ids()
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(
            0,
            leased if leased is not None else cpus[: min(params["threads"], max(1, len(cpus) - 8))],
        )
    os.nice(5)
    record = catalogue_records(settings, [args.dataset_id])[0]
    directory = root / args.dataset_id
    if directory.is_symlink():
        raise DomainInferenceError("Unsafe domain directory")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DomainInferenceError("Another spatial-domain job owns this dataset") from exc
        with resource_guard():
            print(
                json.dumps(
                    generate(record, settings, root, args.config, force=args.force), indent=2
                )
            )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_cli_commands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    try:
        return execute_cli(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Spatial domain error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
