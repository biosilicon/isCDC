"""Source-aware SpatialGLUE recipes; all transforms operate on inference copies."""

from __future__ import annotations

import importlib

from . import spatialglue_config as config
from .spatial_domain_annotation import matrix_sha256, select_variable_genes


def scaled_pca(matrix, dimensions, seed):
    """Centered, unit-variance randomized PCA without materializing a dense n x p copy."""
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from sklearn.utils.extmath import svd_flip
    from sklearn.utils.sparsefuncs import mean_variance_axis

    means, var = mean_variance_axis(matrix, axis=0)
    scale = np.sqrt(np.maximum(var, 0) * matrix.shape[0] / max(1, matrix.shape[0] - 1))
    scale[scale == 0] = 1
    if matrix.shape[1] <= 2:
        return (matrix.toarray() - means) / scale

    def forward(right):
        right = np.asarray(right).reshape(matrix.shape[1], -1) / scale[:, None]
        return matrix @ right - means @ right

    def backward(right):
        right = np.asarray(right).reshape(matrix.shape[0], -1)
        return (matrix.T @ right - means[:, None] * right.sum(axis=0)) / scale[:, None]

    operator = LinearOperator(
        matrix.shape,
        matmat=forward,
        rmatmat=backward,
        matvec=forward,
        rmatvec=backward,
        dtype=np.float64,
    )
    transpose = matrix.shape[0] < matrix.shape[1]
    op = operator.T if transpose else operator
    width = min(dimensions + 10, min(op.shape))
    q = np.random.RandomState(seed).normal(size=(op.shape[1], width))
    for _ in range(5):
        q = np.linalg.qr(op @ q, mode="reduced")[0]
        q = np.linalg.qr(op.T @ q, mode="reduced")[0]
    q = np.linalg.qr(op @ q, mode="reduced")[0]
    left, singular, right = np.linalg.svd((op.T @ q).T, full_matrices=False)
    u = q @ left
    if transpose:
        u, right = right.T, u.T
    u, right = svd_flip(u, right)
    return u[:, :dimensions] * singular[:dimensions]


def _prepare_modalities(matrices, feature_ids, ids, coordinates, params):
    import anndata as ad
    import numpy as np
    import scanpy as sc
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    from sklearn.utils.sparsefuncs import mean_variance_axis

    prepared, selected, notes, stages = {}, {}, {}, {}
    value_types = params.get("input_value_types", {})
    recipes = params.get("preprocessing_recipes", {})
    for name, original in matrices.items():
        stages[f"{name}_input"] = matrix_sha256(original)
        matrix = sparse.csr_matrix(original, dtype=np.float64)
        means, variance = mean_variance_axis(matrix, axis=0)
        tolerance = np.finfo(float).eps * 8 * np.maximum(means**2 + variance, 1e-300)
        keep = np.flatnonzero(variance > tolerance)
        if not len(keep):
            raise ValueError(f"No variable features for {name}")
        value_type = value_types.get(name, "counts")
        recipe = recipes.get(name) or config.recipe({}, name, value_type)
        data = ad.AnnData(matrix[:, keep].copy())
        data.obs_names, data.var_names = ids, feature_ids[name][keep]
        note = {"recipe": recipe, "source_value_type": value_type}
        zero = np.asarray(abs(data.X).sum(axis=1)).ravel() == 0
        if recipe == "transcript_counts_v1":
            fit = {"selected_span": None, "attempts": []}
            if data.n_vars > params["n_top_genes"]:
                fit = select_variable_genes(data, params["n_top_genes"])
                data = data[:, data.var["highly_variable"]].copy()
            sc.pp.normalize_total(data, target_sum=params["target_sum"])
            sc.pp.log1p(data)
            note.update(
                value_processing="normalize_total, log1p, z-score, PCA",
                hvg="seurat_v3",
                hvg_fit=fit,
            )
        elif recipe == "protein_clr_v1":
            logged = data.X.copy()
            logged.data = np.log1p(logged.data)
            denominator = np.exp(np.asarray(logged.sum(axis=1)).ravel() / data.n_vars)
            data.X = data.X.multiply(1 / denominator[:, None]).tocsr()
            data.X.data = np.log1p(data.X.data)
            note["value_processing"] = "per-observation CLR, z-score, PCA"
        elif recipe == "protein_asinh_v1":
            x = data.X.tocsc()
            scales = []
            for col in range(x.shape[1]):
                values = x.data[x.indptr[col] : x.indptr[col + 1]]
                nonzero = abs(values[values != 0])
                scale = float(np.median(nonzero)) if len(nonzero) else 1.0
                values[:] = np.arcsinh(values / scale)
                scales.append(scale)
            data.X = x.tocsr()
            note.update(
                value_processing="feature-median absolute nonzero scale, asinh, z-score, PCA",
                asinh_scales=scales,
            )
        elif recipe == "msi_intensity_v1":
            sc.pp.normalize_total(data, target_sum=params["target_sum"])
            sc.pp.log1p(data)
            note["value_processing"] = "total intensity normalization, log1p, z-score, PCA"
        elif recipe == "receptor_counts_v1":
            data.X.data = np.log1p(data.X.data)
            note["value_processing"] = "log1p, sparse SVD; retain undetected rows"
        elif recipe == "receptor_binary_v1":
            note["value_processing"] = "binary presence, sparse SVD; retain undetected rows"
        elif recipe == "epigenome_lsi_v1":
            note["value_processing"] = "TF-IDF, L1 normalization, log1p, LSI; drop first"
        elif recipe == "methylation_residual_v1":
            note.update(
                value_processing="signed source residuals retained, z-score, PCA",
                source_value_semantics="signed MethSCAn residual",
                comparability_warning="Not numerically comparable to methylation fractions",
            )
        else:
            note["value_processing"] = "source values retained, z-score, PCA"
        dimensions = min(params["input_dims"], len(ids) - 1, data.n_vars)
        if recipe == "epigenome_lsi_v1" and data.n_vars > 2:
            active = np.flatnonzero(~zero)
            dimensions = min(dimensions, len(active) - 1, data.n_vars - 1)
            if dimensions < 1:
                raise ValueError(f"Insufficient epigenome signal for {name}")
            subset = data[active].copy()
            module = "SpatialGlue_3M" if len(matrices) == 3 else "SpatialGlue"
            preprocessing = importlib.import_module(f"{module}.preprocess")
            preprocessing.lsi(
                subset,
                n_components=dimensions + 1,
                use_highly_variable=False,
                random_state=params["seed"],
            )
            features = np.zeros((len(ids), dimensions), dtype=np.float64)
            features[active] = subset.obsm["X_lsi"]
        elif recipe.startswith("receptor_"):
            if data.n_vars <= 2:
                features = data.X.toarray()
            else:
                features = TruncatedSVD(
                    n_components=dimensions, random_state=params["seed"]
                ).fit_transform(data.X)
        else:
            features = scaled_pca(data.X, dimensions, params["seed"])
        if not np.isfinite(features).all() or not np.any(np.var(features, axis=0) > 0):
            raise ValueError(f"Non-finite or degenerate features for {name}")
        informative = ~zero
        metric = "euclidean" if features.shape[1] <= 2 else "correlation"
        if metric == "correlation":
            informative &= np.std(features, axis=1) > 0
        # Only reduced features and graph metadata remain live during model training.
        reduced = ad.AnnData(sparse.csr_matrix((len(ids), 0)))
        reduced.obs_names = ids
        reduced.obsm["spatial"] = coordinates
        reduced.obsm["feat"] = features.astype(np.float32)
        reduced.obs["feature_informative"] = informative
        reduced.uns["feature_metric"] = metric
        prepared[name] = reduced
        selected[name] = list(data.var_names.astype(str))
        note.update(
            dimensions=features.shape[1],
            feature_metric=metric,
            observed_zero_rows=int(zero.sum()),
            feature_graph_rows=int(informative.sum()),
        )
        notes[name] = note
        stages[f"{name}_features"] = matrix_sha256(reduced.obsm["feat"])
    return prepared, selected, notes, stages


def prepare_modalities(matrices, feature_ids, ids, coordinates, params):
    import json

    import anndata as ad
    import numpy as np
    from scipy import sparse

    from .spatial_domain_visualization import obs_order_sha256
    from .spatialglue_cache import cache_key, cached_artifact, commit_cache

    prepared, selected, notes, stages = {}, {}, {}, {}
    for name, matrix in matrices.items():
        key = cache_key(
            "preprocessing",
            {
                "input": matrix_sha256(matrix),
                "ids": obs_order_sha256(ids),
                "features": obs_order_sha256(feature_ids[name]),
                "name": name,
                "parameters": {
                    k: params.get(k)
                    for k in ("seed", "n_top_genes", "input_dims", "target_sum", "source_sha256")
                },
                "value_type": params.get("input_value_types", {}).get(name),
                "recipe": params.get("preprocessing_recipes", {}).get(name),
            },
            __file__,
        )
        with cached_artifact(key) as (target, hit):
            if hit:
                with np.load(target, allow_pickle=False) as cached:
                    reduced = ad.AnnData(sparse.csr_matrix((len(ids), 0)))
                    reduced.obs_names = ids
                    reduced.obsm["spatial"] = coordinates
                    reduced.obsm["feat"] = cached["features"]
                    reduced.obs["feature_informative"] = cached["informative"]
                    reduced.uns["feature_metric"] = str(cached["metric"])
                    prepared[name] = reduced
                    selected[name] = cached["selected"].tolist()
                    notes[name] = json.loads(str(cached["notes"]))
                    stages.update(json.loads(str(cached["stages"])))
                print(f"SpatialGlue preprocessing cache hit: {name} {key}", flush=True)
            else:
                result, chosen, note, stage = _prepare_modalities(
                    {name: matrix}, {name: feature_ids[name]}, ids, coordinates, params
                )
                prepared.update(result)
                selected.update(chosen)
                notes.update(note)
                stages.update(stage)
                if target:
                    value = result[name]
                    commit_cache(
                        target,
                        lambda path: np.savez(
                            path,
                            features=value.obsm["feat"],
                            informative=value.obs["feature_informative"].to_numpy(),
                            metric=np.array(value.uns["feature_metric"]),
                            selected=np.array(chosen[name]),
                            notes=np.array(json.dumps(note[name])),
                            stages=np.array(json.dumps(stage)),
                        ),
                    )
    return prepared, selected, notes, stages
