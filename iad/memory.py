"""Coresets, bounded-memory matching, and exact squared-distance explanations."""
import numpy as np
from sklearn.random_projection import SparseRandomProjection


def greedy_coreset(x, ratio=0.01, max_points=512, seed=42, projection_dim=128):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or not 0 < ratio <= 1 or max_points < 1:
        raise ValueError("Invalid memory-bank input")
    count = min(len(x), max_points, max(1, int(np.ceil(len(x) * ratio))))
    projected = x
    if x.shape[1] > projection_dim:
        projected = SparseRandomProjection(n_components=projection_dim, random_state=seed).fit_transform(x)
    rng = np.random.default_rng(seed)
    current = int(rng.integers(len(x)))
    nearest = np.full(len(x), np.inf)
    selected = []
    for _ in range(count):
        selected.append(current)
        distances = np.sum((projected - projected[current]) ** 2, axis=1)
        nearest = np.minimum(nearest, distances)
        nearest[selected] = -np.inf
        current = int(np.argmax(nearest))
    ids = np.asarray(selected)
    return x[ids].copy(), ids


def nearest_neighbors(x, bank, chunk_size=256, bank_chunk=1024, backend="numpy"):
    x, bank = np.asarray(x, np.float32), np.asarray(bank, np.float32)
    if x.ndim != 2 or bank.ndim != 2 or x.shape[1] != bank.shape[1] or not len(bank):
        raise ValueError("Query and bank must have matching nonempty feature dimensions")
    if not np.isfinite(x).all() or not np.isfinite(bank).all() or min(chunk_size, bank_chunk) < 1:
        raise ValueError("Nonfinite features or invalid chunk size")
    if backend == "faiss":
        import faiss
        index = faiss.IndexFlatL2(bank.shape[1])
        index.add(np.ascontiguousarray(bank))
        _, idx = index.search(np.ascontiguousarray(x), 1)
        idx = idx[:, 0]
    elif backend == "numpy":
        # Float64 distance calculation avoids cancellation affecting NN identity.
        idx = np.zeros(len(x), dtype=np.int64)
        for start in range(0, len(x), chunk_size):
            q = x[start:start + chunk_size].astype(np.float64)
            best = np.full(len(q), np.inf)
            for b in range(0, len(bank), bank_chunk):
                m = bank[b:b + bank_chunk].astype(np.float64)
                squared = np.sum(q*q, axis=1)[:, None] + np.sum(m*m, axis=1)[None] - 2*q @ m.T
                local = np.argmin(squared, axis=1)
                values = squared[np.arange(len(q)), local]
                improve = values < best
                idx[start:start + len(q)][improve] = b + local[improve]
                best[improve] = values[improve]
    else:
        raise ValueError("backend must be numpy or faiss")
    contributions = (x.astype(np.float64) - bank[idx]) ** 2
    return np.sqrt(contributions.sum(axis=1)), idx


def channel_contributions(x, nearest):
    return (np.asarray(x, dtype=np.float64) - np.asarray(nearest, dtype=np.float64)) ** 2
