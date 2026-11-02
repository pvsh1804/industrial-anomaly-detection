"""Sparse normal-only graph selectors; paper equations (1), (2), and (4)."""
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite matrix with at least two rows")
    return x


def knn_graph(x, neighbors=10, bandwidth=None):
    x = matrix(x)
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    k = min(neighbors, len(x) - 1)
    distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
    # Remove self by ID, not by position: duplicate rows can reorder ties.
    rows, cols, squared = [], [], []
    for i, (ds, ids) in enumerate(zip(distances, indices)):
        take = ids != i
        rows.extend([i] * k)
        cols.extend(ids[take][:k])
        squared.extend(ds[take][:k] ** 2)
    squared = np.asarray(squared)
    positive = squared[squared > 0]
    t = float(bandwidth if bandwidth is not None else (np.median(positive) if len(positive) else 1.0))
    if t <= 0:
        raise ValueError("Heat-kernel bandwidth must be positive")
    graph = sparse.csr_matrix((np.exp(-squared / t), (rows, cols)), shape=(len(x), len(x)))
    return graph.maximum(graph.T)


def consistency_graph(original, transformed, neighbors=10, pair_weight=1.0):
    original, transformed = matrix(original), matrix(transformed)
    if original.shape != transformed.shape or pair_weight < 0:
        raise ValueError("Photometric features must correspond exactly; pair weight must be nonnegative")
    joined = np.concatenate([original, transformed])
    graph = knn_graph(joined, neighbors)
    n = len(original)
    rows = np.arange(n)
    pairs = sparse.csr_matrix((np.full(n, pair_weight), (rows, rows + n)), shape=graph.shape)
    return joined, graph + pairs + pairs.T


def laplacian_scores(x, graph):
    x = matrix(x)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    if graph.shape != (len(x), len(x)) or degree.sum() <= 0:
        raise ValueError("Invalid graph")
    centered = x - (degree @ x) / degree.sum()
    denominator = np.sum(degree[:, None] * centered ** 2, axis=0)
    numerator = np.sum(centered * (degree[:, None] * centered - graph @ centered), axis=0)
    result = np.full(x.shape[1], np.inf)
    good = denominator > np.finfo(float).eps * np.maximum(1, np.sum(degree[:, None] * x ** 2, axis=0))
    result[good] = np.maximum(numerator[good], 0) / denominator[good]
    return result


def spec_scores(x, graph):
    """Equation (4), identity spectral function, trivial component removed.

    Algebraically equivalent to degree-centered Laplacian score. Keeping an
    independent implementation lets tests check the paper's equivalence.
    No dense eigendecomposition is needed for this SPEC variant.
    """
    x = matrix(x)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    if graph.shape != (len(x), len(x)) or np.any(degree <= 0):
        raise ValueError("SPEC needs positive graph degrees")
    root = np.sqrt(degree)
    weighted = root[:, None] * x
    norms = np.linalg.norm(weighted, axis=0)
    z = weighted / np.maximum(norms, np.finfo(float).tiny)
    trivial = root / np.linalg.norm(root)
    denominator = 1 - (trivial @ z) ** 2
    normalized_laplacian_z = z - (graph @ (z / root[:, None])) / root[:, None]
    numerator = np.sum(z * normalized_laplacian_z, axis=0)
    scores = np.full(x.shape[1], np.inf)
    good = (denominator > 1e-12) & (norms > 1e-12)
    scores[good] = np.maximum(numerator[good], 0) / denominator[good]
    return scores


def variance_channels(x, quantile=0.1):
    x = matrix(x)
    if not 0 <= quantile < 1:
        raise ValueError("Variance quantile must be in [0, 1)")
    variance = x.var(axis=0)
    ids = np.flatnonzero(variance > np.quantile(variance, quantile))
    if not len(ids):
        raise ValueError("Variance filter removed all channels")
    return ids


def ranked_channels(scores, count):
    valid = np.flatnonzero(np.isfinite(scores))
    if count < 1 or not len(valid):
        raise ValueError("No nonconstant feature channels available")
    return valid[np.argsort(np.asarray(scores)[valid], kind="stable")[:count]]


def correlation_filter(x, ordered, threshold=0.95):
    if not 0 <= threshold <= 1:
        raise ValueError("Correlation threshold must be in [0, 1]")
    z = matrix(x)[:, ordered]
    z = z - z.mean(axis=0)
    norms = np.linalg.norm(z, axis=0)
    z /= np.maximum(norms, 1e-15)
    keep = []
    for i in range(z.shape[1]):
        if norms[i] > 1e-12 and (not keep or np.all(np.abs(z[:, keep].T @ z[:, i]) <= threshold)):
            keep.append(i)
    if not keep:
        raise ValueError("Correlation filter removed all channels")
    return np.asarray(ordered)[keep]
