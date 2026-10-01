"""Shared clustering helpers (extracted from legacy HDBSCAN eval script)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def min_cluster_size_candidates(n: int, max_clusters: int) -> list[int]:
    sizes = set()
    for k in range(2, max_clusters + 1):
        sizes.add(max(5, int(np.ceil(n / k))))
    for frac in (0.05, 0.03, 0.02, 0.01):
        sizes.add(max(5, int(n * frac)))
    for fixed in (3, 5, 10, 15, 20, 25, 30, 50, 75, 100):
        if fixed <= n:
            sizes.add(fixed)
    return sorted(sizes)


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None = None,
    prediction_data: bool = False,
):
    try:
        import hdbscan
    except ImportError as e:
        raise SystemExit("hdbscan not installed. pip install hdbscan") from e

    if min_samples is None:
        min_samples = max(2, min_cluster_size // 2)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
        prediction_data=prediction_data,
    )
    labels = clusterer.fit_predict(l2_normalize(embeddings))
    return labels, clusterer


def cluster_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    mask = labels >= 0
    if mask.sum() < 2 or len(set(labels[mask].tolist())) < 2:
        return None
    n = int(mask.sum())
    return float(
        silhouette_score(
            l2_normalize(embeddings[mask]),
            labels[mask],
            metric="euclidean",
            sample_size=min(5000, n),
        )
    )
