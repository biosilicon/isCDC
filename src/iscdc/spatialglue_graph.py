"""Sparse adjacency and bounded, exact GPU neighbors for SpatialGLUE."""

from __future__ import annotations


def _ordered_topk(distances, indices, k):
    import torch

    # Candidate arrays have at most 2*k columns. Stable secondary order is the ID rank.
    order = torch.argsort(indices, dim=1, stable=True)
    distances = distances.gather(1, order)
    indices = indices.gather(1, order)
    order = torch.argsort(distances, dim=1, stable=True)[:, :k]
    return distances.gather(1, order), indices.gather(1, order)


def _exact_neighbors(
    features,
    ids,
    k,
    *,
    metric="correlation",
    device="cuda:0",
    workspace_mb=128,
    informative=None,
    query_indices=None,
):
    """Exact all-reference scan; float64 distances, no n*n allocation or approximation.

    Optional queries are for independent large-graph verification. Returned row/column
    indices refer to the original observation axis. Uninformative rows have no edges.
    """
    import time

    import numpy as np
    import torch
    from scipy import sparse

    n = len(features)
    valid = np.ones(n, dtype=bool) if informative is None else np.asarray(informative, bool)
    active = np.flatnonzero(valid)
    active = active[np.argsort(np.asarray(ids)[active], kind="stable")]
    k = min(k, max(0, len(active) - 1))
    if k == 0:
        return sparse.csr_matrix((n, n), dtype=np.float32)
    array = np.asarray(features[active], dtype=np.float64)
    if metric == "correlation":
        array -= array.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(array, axis=1, keepdims=True)
        if (norm == 0).any():
            raise ValueError("Undefined correlation distance on informative row")
        array /= norm
    elif metric != "euclidean":
        raise ValueError("Unsupported exact neighbor metric")
    reference = torch.as_tensor(array, dtype=torch.float64, device=device)
    squared = (reference * reference).sum(dim=1)
    query_ranks = np.arange(len(active))
    if query_indices is not None:
        query_ranks = query_ranks[np.isin(active, query_indices)]
    if len(query_ranks) == 0:
        return sparse.csr_matrix((n, n), dtype=np.float32)
    # Distance + tie workspace, with explicit headroom for topk implementation buffers.
    qsize = min(512, len(query_ranks))
    rsize = max(k + 1, min(16384, int(workspace_mb * 2**20 / max(1, qsize) / 40)))
    out_rows, out_cols = [], []
    last_progress = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(query_ranks), max(1, qsize)):
            ranks_np = query_ranks[start : start + qsize]
            ranks = torch.as_tensor(ranks_np, device=device)
            query = reference[ranks]
            best_d = torch.full((len(ranks), k), float("inf"), device=device, dtype=torch.float64)
            best_i = torch.full((len(ranks), k), len(active), device=device, dtype=torch.int64)
            for offset in range(0, len(active), rsize):
                stop = min(offset + rsize, len(active))
                dot = query @ reference[offset:stop].T
                distance = (
                    1 - dot
                    if metric == "correlation"
                    else (squared[ranks, None] + squared[None, offset:stop] - 2 * dot)
                )
                distance.clamp_(min=0)
                local = ranks - offset
                own = (local >= 0) & (local < stop - offset)
                distance[torch.arange(len(ranks), device=device)[own], local[own]] = float("inf")
                take = min(k, stop - offset)
                values, indices = torch.topk(distance, take, largest=False, sorted=True)
                indices += offset
                boundary = values[:, -1:]
                # topk does not define tie order. Replace its boundary ties with the
                # lexicographically first IDs from the entire reference block.
                rank_grid = torch.arange(offset, stop, device=device).expand(len(ranks), -1)
                ties = torch.where(distance == boundary, rank_grid, len(active))
                tie_i = torch.topk(ties, take, largest=False, sorted=True).values
                tie_d = torch.where(tie_i < len(active), boundary, float("inf"))
                values = torch.where(values < boundary, values, float("inf"))
                values, indices = _ordered_topk(
                    torch.cat((values, tie_d), 1), torch.cat((indices, tie_i), 1), take
                )
                best_d, best_i = _ordered_topk(
                    torch.cat((best_d, values), 1), torch.cat((best_i, indices), 1), k
                )
            neighbors = best_i.cpu().numpy()
            if (neighbors >= len(active)).any():
                raise ValueError("Incomplete exact neighbor graph")
            out_rows.append(np.repeat(active[ranks_np], k))
            out_cols.append(active[neighbors.ravel()])
            if time.monotonic() - last_progress >= 60:
                print(
                    f"SpatialGlue exact graph: {start + len(ranks_np)}/{len(query_ranks)} "
                    f"queries, {len(active)} references, {metric}",
                    flush=True,
                )
                last_progress = time.monotonic()
    rows, cols = np.concatenate(out_rows), np.concatenate(out_cols)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))


def spatial_neighbors(coordinates, k):
    import numpy as np
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors

    n = len(coordinates)
    k = min(k, n - 1)
    nearest = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(coordinates)
    indices = nearest.kneighbors(coordinates, return_distance=False)
    cols = np.concatenate([row[row != i][:k] for i, row in enumerate(indices)])
    return sparse.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), cols)), shape=(n, n))


def normalize_graph(matrix):
    """Exactly the upstream binary union, add-I and D^-1/2 A D^-1/2 operations."""
    import numpy as np
    from scipy import sparse

    graph = (matrix + matrix.T).tocsr().astype(np.float64)
    graph.data[:] = np.minimum(graph.data, 1)
    graph = graph + sparse.eye(graph.shape[0], format="csr")
    degree = np.asarray(graph.sum(axis=1)).ravel() ** -0.5
    graph = sparse.diags(degree) @ graph @ sparse.diags(degree)
    graph = graph.astype(np.float32).tocsr()
    graph.sort_indices()
    return graph


def torch_graph(matrix):
    import torch

    # CSR sparse/dense multiplication avoids the unordered COO atomic accumulation.
    graph = normalize_graph(matrix)
    return torch.sparse_csr_tensor(
        torch.from_numpy(graph.indptr.astype("int64")),
        torch.from_numpy(graph.indices.astype("int64")),
        torch.from_numpy(graph.data),
        size=graph.shape,
    )


def adjacent_matrix_preprocessing(*modalities):
    result = {}
    for i, data in enumerate(modalities, 1):
        result[f"adj_spatial_omics{i}"] = torch_graph(data.obsp["glue_spatial"])
        result[f"adj_feature_omics{i}"] = torch_graph(data.obsm["adj_feature"])
    return result


def exact_neighbors(
    features,
    ids,
    k,
    *,
    metric="correlation",
    device="cuda:0",
    workspace_mb=128,
    informative=None,
    query_indices=None,
):
    from scipy import sparse

    from .spatial_domain_annotation import matrix_sha256
    from .spatial_domain_visualization import obs_order_sha256
    from .spatialglue_cache import cache_key, cached_artifact, commit_cache

    if query_indices is not None:
        return _exact_neighbors(
            features,
            ids,
            k,
            metric=metric,
            device=device,
            workspace_mb=workspace_mb,
            informative=informative,
            query_indices=query_indices,
        )
    key = cache_key(
        "exact_graph",
        {
            "features": matrix_sha256(features),
            "ids": obs_order_sha256(ids),
            "k": k,
            "metric": metric,
            "workspace_mb": workspace_mb,
            "informative": matrix_sha256(informative) if informative is not None else None,
        },
        __file__,
    )
    with cached_artifact(key) as (target, hit):
        if hit:
            print(f"SpatialGlue exact graph cache hit: {key}", flush=True)
            return sparse.load_npz(target)
        graph = _exact_neighbors(
            features,
            ids,
            k,
            metric=metric,
            device=device,
            workspace_mb=workspace_mb,
            informative=informative,
        )
        if target:
            commit_cache(target, lambda path: sparse.save_npz(path, graph, compressed=False))
        return graph
