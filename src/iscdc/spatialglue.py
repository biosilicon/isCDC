"""Offline paired multi-omics inference using the official SpatialGlue trainers."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import importlib.metadata
import io
import os
import random
import resource
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import cell_type_visualization as ct
from . import spatialglue_config as config
from .spatial_domain_annotation import (
    _color,
    _diagnostics,
    enable_h5mu_null_reader,
    matrix_sha256,
)
from .spatial_domain_visualization import (
    assignment_bytes,
    build_point_representations,
    decode_points,
    domain_directory,
    encode_points,
    file_record,
    obs_order_sha256,
    publish_generation,
    publish_method_failure,
)
from .spatialglue_preprocessing import prepare_modalities  # noqa: F401
from .spatialglue_resources import check_resources, cuda_info, gpu_guard
from .spatialglue_runtime import CUDA_PEAKS, TIMINGS, phase


def adapter_sha256():
    digest = hashlib.sha256()
    for name in (
        "spatialglue.py",
        "spatialglue_config.py",
        "spatialglue_resources.py",
        "spatialglue_cli.py",
        "spatialglue_preprocessing.py",
        "spatialglue_graph.py",
        "spatialglue_runtime.py",
        "spatialglue_cache.py",
        "spatial_domain_annotation.py",
        "spatial_domain_visualization.py",
    ):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def validate_environment():
    packages = {}
    for line in config.LOCK_PATH.read_text().splitlines():
        if not line or line.startswith(("#", "--")):
            continue
        name, version = line.split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != version:
            raise ValueError(f"SpatialGlue environment lock mismatch: {name} requires {version}")
        packages[name] = actual
    return packages


def paired_sample(mdata, modalities, indices, ids, value_types=None, *, record=None):
    """Return all input matrices aligned by ID, with absence separate from zero counts."""
    import numpy as np
    from scipy import sparse

    matrices, feature_ids, coverage = {}, {}, {}
    present = np.ones(len(indices), dtype=bool)
    nonzero = np.ones(len(indices), dtype=bool)
    for name in modalities:
        adata = mdata.mod[name]
        if not adata.obs_names.is_unique or not adata.var_names.is_unique:
            raise ValueError(f"Duplicate {name} observation or feature IDs")
        if not set(adata.obs_names) <= set(ids):
            raise ValueError(f"Foreign {name} observation IDs")
        names = np.asarray(adata.var_names.astype(str))
        if any(not value or value != value.strip() or "\n" in value for value in names):
            raise ValueError(f"Invalid {name} feature IDs")
        positions = adata.obs_names.get_indexer(ids[indices])
        exists = positions >= 0
        # Backed sparse datasets support row indexing; empty rows never masquerade as data.
        rows = sparse.csr_matrix(adata.X[positions[exists]], dtype=np.float64)
        values = rows.data
        value_type = (value_types or {}).get(name, "counts")
        if value_type in {"counts", "binary"} and (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError(f"{name} counts must be finite non-negative integers")
        if value_types and value_types[name] == "binary" and (values > 1).any():
            raise ValueError(f"{name} binary input contains values other than 0/1")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name} values")
        if value_type in {"intensity", "binary"} and (values < 0).any():
            raise ValueError(f"Negative {name} {value_type}")
        if (
            name == "methylation"
            and config.recipe(record or {}, name, value_type) == "methylation_fraction_v1"
            and ((values < 0).any() or (values > 1).any())
        ):
            raise ValueError("Methylation fraction outside [0, 1]")
        # Count zero rows before transforms; signed continuous values must not cancel.
        totals = np.asarray(abs(rows).sum(axis=1)).ravel()
        good = np.zeros(len(indices), dtype=bool)
        good[exists] = totals > 0
        present &= exists
        if name in {"rna", "translatome"}:
            nonzero &= good
        matrices[name] = (rows, exists)
        feature_ids[name] = names
        coverage[name] = {"missing": int((~exists).sum()), "zero_counts": int((totals == 0).sum())}
    valid = present & nonzero
    reasons = np.full(len(indices), "missing_modality", dtype=object)
    reasons[present] = "zero_counts"
    reasons[valid] = ""
    if valid.sum() < 3:
        raise ValueError("Insufficient jointly present, nonzero observations")
    aligned = {name: rows[valid[exists]] for name, (rows, exists) in matrices.items()}
    return (
        aligned,
        feature_ids,
        valid,
        reasons,
        {
            "missing_modality": int((~present).sum()),
            "zero_counts": int((present & ~nonzero).sum()),
            "modalities": coverage,
        },
    )


def train(prepared, params):
    import numpy as np
    import torch

    from .spatialglue_graph import adjacent_matrix_preprocessing, exact_neighbors, spatial_neighbors

    module = "SpatialGlue_3M" if len(prepared) == 3 else "SpatialGlue"
    implementation = importlib.import_module(f"{module}.SpatialGlue_pyG")
    trainer = implementation.Train_SpatialGlue
    data = {}
    with phase("graph", params):
        first = next(iter(prepared.values()))
        spatial = spatial_neighbors(first.obsm["spatial"], params["spatial_neighbors"])
        for i, adata in enumerate(prepared.values(), 1):
            adata.obsp["glue_spatial"] = spatial
            adata.obsm["adj_feature"] = exact_neighbors(
                adata.obsm["feat"],
                np.asarray(adata.obs_names),
                params["feature_neighbors"],
                metric=adata.uns["feature_metric"],
                device=params["device"],
                workspace_mb=params["graph_workspace_mb"],
                informative=adata.obs["feature_informative"].to_numpy(),
            )
            data[f"adata_omics{i}"] = adata
    with phase("training", params):
        # Only substitute the upstream dense adjacency adapter; model/loss stay official.
        previous = implementation.adjacent_matrix_preprocessing
        implementation.adjacent_matrix_preprocessing = adjacent_matrix_preprocessing
        try:
            model = trainer(
                data,
                device=torch.device(params["device"]),
                random_seed=params["seed"],
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                epochs=params["epochs"],
                dim_output=params["dim_output"],
                weight_factors=params["weight_factors"],
            )
        finally:
            implementation.adjacent_matrix_preprocessing = previous
        # The two-modality constructor overrides epochs and weights for its default SPOTS
        # datatype. Reapply the resolved recipe before train() to honor configuration.
        model.epochs = params["epochs"]
        model.weight_factors = list(params["weight_factors"])
        output = model.train()
        embedding = output["SpatialGlue"]
        if (
            embedding.shape != (next(iter(prepared.values())).n_obs, params["dim_output"])
            or not np.isfinite(embedding).all()
        ):
            raise ValueError("SpatialGlue produced invalid joint representation")
        stages = {}
        for i, (name, adata) in enumerate(prepared.items(), 1):
            stages[f"{name}_feature_graph"] = matrix_sha256(adata.obsm["adj_feature"])
            for kind in ("spatial", "feature"):
                graph = model.adj[f"adj_{kind}_omics{i}"].to_sparse_coo().coalesce().cpu()
                stages[f"{name}_{kind}_graph_indices"] = matrix_sha256(graph.indices().numpy())
                stages[f"{name}_{kind}_graph_values"] = matrix_sha256(graph.values().numpy())
        del model
        return embedding, stages


def cluster(embedding, ids, params):
    import anndata as ad
    import numpy as np
    import scanpy as sc
    from sklearn.decomposition import PCA

    dims = min(params["n_pcs"], len(ids) - 1, embedding.shape[1] - 1)
    representation = PCA(
        n_components=dims, svd_solver="randomized", random_state=params["seed"]
    ).fit_transform(embedding)
    if not np.isfinite(representation).all() or not np.any(np.var(representation, axis=0) > 0):
        raise ValueError("SpatialGlue produced degenerate PCA coordinates")
    data = ad.AnnData(representation)
    data.obs_names = ids
    data.obsm["domain_pca"] = representation
    previous_jobs = sc.settings.n_jobs
    try:
        sc.settings.n_jobs = min(params["threads"], len(os.sched_getaffinity(0)))
        sc.pp.neighbors(
            data,
            n_neighbors=min(params["expression_neighbors"], len(ids) - 1),
            use_rep="domain_pca",
            random_state=params["seed"],
        )
    finally:
        sc.settings.n_jobs = previous_jobs
    sc.tl.leiden(
        data,
        resolution=params["resolution"],
        random_state=params["seed"],
        flavor="igraph",
        directed=False,
        n_iterations=-1,
        key_added="domain",
    )
    labels = data.obs["domain"].astype(str).to_numpy()
    ordered = sorted(set(labels), key=lambda key: min(ids[labels == key]))
    if len(ordered) >= 65536:
        raise ValueError("Too many spatial domains for point encoding")
    remap = {label: i + 1 for i, label in enumerate(ordered)}
    return (
        np.array([remap[label] for label in labels], dtype=np.uint16),
        dims,
        {
            "joint_embedding": matrix_sha256(embedding),
            "pca": matrix_sha256(representation),
            "connectivities": matrix_sha256(data.obsp["connectivities"]),
        },
    )


def sample_points(key, sample_id, coordinates, codes, files):
    import numpy as np

    payload = encode_points(coordinates[:, 0].tolist(), coordinates[:, 1].tolist(), codes.tolist())
    point = decode_points(payload)
    reps = {}
    for encoding, content in build_point_representations(payload).items():
        name = f"points/{key}{ct._ENCODING_SUFFIXES[encoding]}"
        files[name] = content
        reps[encoding] = {
            **file_record(name, content),
            "encoding": encoding,
            "content_size": len(payload),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "key": key,
        "id": sample_id,
        "count": len(codes),
        "bounds": [min(point.x), min(point.y), max(point.x), max(point.y)],
        "categories": [
            {
                "code": int(c),
                "label": f"Domain {c}" if c else "Not analyzed",
                "color": _color(int(c)),
                "count": int((codes == c).sum()),
            }
            for c in np.unique(codes)
        ],
        "representations": reps,
    }


def generate(record, settings, output_root, config_path, *, force=False, modalities=None):
    import h5py
    import mudata
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    dataset_id = record["dataset_id"]
    params = config.load_parameters(config_path, dataset_id)
    config.eligibility(record, params)
    modalities = config.select_modalities(record, modalities or params["input_modalities"])
    combination = config.combination_id(modalities)
    base = domain_directory(output_root, dataset_id, "spatialglue", combination_id=combination)
    if (base / "status.json").exists() and not force:
        raise ValueError("Existing SpatialGlue status; use --force to create another generation")
    started, mdata = time.monotonic(), None
    try:
        params = config.load_parameters(config_path, dataset_id)
        params = config.load_parameters(config_path, dataset_id, modalities=modalities)
        packages = validate_environment()
        info = cuda_info(params)
        path = settings.data_root / record["storage_dir"] / "dataset.h5mu"
        if ct._file_digest(path)[1] != record["sha256"]:
            raise ValueError("Source checksum differs from catalogue")
        enable_h5mu_null_reader()
        mdata = mudata.read_h5mu(path, backed="r")
        ids = np.asarray(mdata.obs_names.astype(str))
        coordinates = np.asarray(mdata.obsm["spatial"])
        samples = mdata.obs["sample_id"].astype(str).to_numpy()
        if len(ids) != record["n_obs"] or len(set(ids)) != len(ids):
            raise ValueError("Invalid top-level observation IDs")
        if coordinates.shape != (len(ids), 2) or not np.isfinite(coordinates).all():
            raise ValueError("Invalid two-dimensional coordinates")
        if set(samples) != set(record["sample_ids"]):
            raise ValueError("Source samples disagree with catalogue")
        if not set(modalities) <= set(mdata.mod):
            raise ValueError("Source modalities disagree with catalogue")
        with h5py.File(path, "r") as source:
            input_bytes = 0
            for name in modalities:
                matrix = source[f"mod/{name}/X"]
                arrays = [matrix] if isinstance(matrix, h5py.Dataset) else list(matrix.values())
                input_bytes += sum(a.size * a.dtype.itemsize for a in arrays)
        codes = np.zeros(len(ids), dtype=np.uint16)
        reasons = np.full(len(ids), "missing_modality", dtype=object)
        reports, docs, files = {}, [], {}
        with gpu_guard(info):
            for index, sample_id in enumerate(record["sample_ids"]):
                sample_started = time.monotonic()
                sp = config.load_parameters(
                    config_path, dataset_id, sample_id, modalities=modalities
                )
                sp["source_sha256"] = record["sha256"]
                sp["input_value_types"] = {
                    m: record["modalities"][m]["value_type"] for m in modalities
                }
                sp["preprocessing_recipes"] = {
                    m: config.recipe(record, m, sp["input_value_types"][m]) for m in modalities
                }
                if not os.environ.get("ISCDC_GLUE_REQUEST_FD"):
                    sp["threads"] = min(sp["threads"], len(os.sched_getaffinity(0)))
                random.seed(sp["seed"])
                np.random.seed(sp["seed"])
                torch.manual_seed(sp["seed"])
                torch.cuda.manual_seed_all(sp["seed"])
                torch.set_num_threads(sp["threads"])
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                all_idx = np.flatnonzero(samples == sample_id)
                # Bound reading/preprocessing before materializing large modality matrices.
                resources = check_resources(
                    len(all_idx),
                    {
                        name: min(mdata.mod[name].n_vars, sp["n_top_genes"])
                        if name == "rna"
                        else mdata.mod[name].n_vars
                        for name in modalities
                    },
                    sp,
                    input_bytes=input_bytes,
                )
                with threadpool_limits(limits=sp["threads"]):
                    with phase("pairing", sp):
                        matrices, names, valid, sample_reasons, coverage = paired_sample(
                            mdata,
                            modalities,
                            all_idx,
                            ids,
                            {m: record["modalities"][m]["value_type"] for m in modalities},
                            record=record,
                        )
                    analyzed = all_idx[valid]
                    with phase("preprocessing", sp):
                        prepared, selected, notes, stages = prepare_modalities(
                            matrices, names, ids[analyzed], coordinates[analyzed], sp
                        )
                    del matrices
                    embedding, graph_stages = train(prepared, sp)
                    del prepared
                    with phase("clustering", sp):
                        labels, dims, cluster_stages = cluster(embedding, ids[analyzed], sp)
                        diagnostic = _diagnostics(coordinates[analyzed], labels)
                codes[analyzed], reasons[all_idx] = labels, sample_reasons
                key = f"sample_{index}"
                features = {}
                for name in modalities:
                    filename = f"features/{key}_{name}.txt.gz"
                    files[filename] = gzip.compress(
                        ("\n".join(selected[name]) + "\n").encode(), mtime=0
                    )
                    features[name] = {
                        "count": len(selected[name]),
                        "ids_sha256": obs_order_sha256(selected[name]),
                        "file": file_record(filename, files[filename]),
                    }
                stream = io.BytesIO()
                np.save(stream, embedding, allow_pickle=False)
                filename = f"embeddings/{key}.npy"
                files[filename] = stream.getvalue()
                reports[sample_id] = {
                    "parameters": sp,
                    "preprocessing": notes,
                    "features": features,
                    "coverage": coverage,
                    "analyzed": len(analyzed),
                    "not_analyzed": len(all_idx) - len(analyzed),
                    "n_domains": len(diagnostic),
                    "domains": diagnostic,
                    "n_pcs_actual": dims,
                    "selected_genes": selected.get("rna", []),
                    "selected_genes_sha256": obs_order_sha256(selected.get("rna", [])),
                    "stage_sha256": {
                        **stages,
                        **graph_stages,
                        **cluster_stages,
                        "coordinates": matrix_sha256(coordinates[analyzed]),
                        "analyzed_ids": obs_order_sha256(ids[analyzed]),
                    },
                    "joint_embedding": file_record(filename, files[filename]),
                    "clustering": {
                        "flavor": "igraph",
                        "directed": False,
                        "n_iterations": -1,
                        "resolution": sp["resolution"],
                        "seed": sp["seed"],
                    },
                    "resources": {
                        **resources,
                        "peak_cuda_allocated_bytes": CUDA_PEAKS["allocated"],
                        "peak_cuda_reserved_bytes": CUDA_PEAKS["reserved"],
                    },
                    "elapsed_seconds": time.monotonic() - sample_started,
                    "warnings": ["Only one spatial domain detected"]
                    if len(diagnostic) == 1
                    else [],
                }
                print(
                    f"SpatialGlue sample {sample_id}: {len(analyzed)} analyzed, "
                    f"{len(diagnostic)} domains; {reports[sample_id]['resources']}",
                    flush=True,
                )
                docs.append(
                    sample_points(key, sample_id, coordinates[all_idx], codes[all_idx], files)
                )
        if ct._file_digest(path)[1] != record["sha256"]:
            raise ValueError("Source changed during inference")
        now = datetime.now(timezone.utc)
        generation = now.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        report = {
            "report_version": 3,
            "dataset_id": dataset_id,
            "generation_id": generation,
            "source_sha256": record["sha256"],
            "status": "passed",
            "samples": reports,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "phase_seconds": dict(TIMINGS),
            "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            "validation_scope": "Computational integrity, not biological ground-truth validation",
        }
        files["report.json"] = ct._canonical_json(report) + b"\n"
        files["assignments.tsv.gz"] = assignment_bytes(ids, samples, codes, reasons)
        manifest = {
            "manifest_version": 3,
            "dataset_id": dataset_id,
            "generation_id": generation,
            "generated_at": now.isoformat(),
            "method": "SpatialGlue_3M" if len(modalities) == 3 else "SpatialGlue",
            "source": {
                "sha256": record["sha256"],
                "obs_order_sha256": obs_order_sha256(ids),
                "observation_count": len(ids),
                "sample_ids": record["sample_ids"],
                "spatial_unit": record["spatial_unit"],
            },
            "coordinates": {
                "system": "cartesian",
                "unit": record["coordinate_unit"],
                "y_axis": params["y_axis"],
            },
            "samples": docs,
            "assignments": file_record("assignments.tsv.gz", files["assignments.tsv.gz"]),
            "report": file_record("report.json", files["report.json"]),
            "provenance": {
                "combination_id": combination,
                "recipe_version": 1,
                "partial_run": len(modalities) != len(record["modalities"]),
                "environment_lock_sha256": ct._file_digest(config.LOCK_PATH)[1],
                "packages": packages,
                "adapter_sha256": adapter_sha256(),
                "parameters": params,
                "input_modalities": modalities,
                "unused_modalities": sorted(set(record["modalities"]) - set(modalities)),
                "input_value_types": {m: record["modalities"][m]["value_type"] for m in modalities},
                "device": info,
            },
        }
        snapshot = publish_generation(
            output_root,
            record,
            manifest,
            files,
            method_family="spatialglue",
            combination_id=combination,
        )
        return {
            "dataset_id": dataset_id,
            "combination_id": combination,
            "state": "success",
            "method": manifest["method"],
            "generation_id": snapshot.generation_id,
            "elapsed_seconds": report["elapsed_seconds"],
        }
    except Exception as exc:
        publish_method_failure(
            output_root, dataset_id, exc, method_family="spatialglue", combination_id=combination
        )
        raise
    finally:
        if mdata is not None and mdata.file.is_open:
            mdata.file.close()
