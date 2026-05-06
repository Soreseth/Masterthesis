"""Vectorized Mann-Whitney U and Brunner-Munzel W statistics.

Used by `aggregate_from_pcs.py` (and by extension every per-classifier table
generator) to convert per-paragraph score distributions into a single
per-document statistic that AUROC can be computed against.
"""
import numpy as np
from scipy.stats import rankdata


def mannwhitneyu_u_only(samples, ref):
    """Vectorized MWU returning only the U statistic.
    Handles ties with average ranks.

    Args:
        samples: shape (n_samples, sample_size) -- per-document score arrays.
        ref:     shape (n_ref,)                 -- held-out non-member scores.

    Returns:
        U statistics, shape (n_samples,).
    """
    samples = np.atleast_2d(samples)
    n1 = samples.shape[1]
    n2 = len(ref)
    n = n1 + n2

    ref_tiled = np.tile(ref, (len(samples), 1))
    combined = np.concatenate([samples, ref_tiled], axis=1)

    # Rank within each row.
    order = np.argsort(combined, axis=1)
    ranks = np.empty_like(combined, dtype=float)
    rows = np.arange(len(samples))[:, None]
    ranks[rows, order] = np.arange(1, n + 1)

    # Average ranks for tied values (per row -- scipy convention).
    for i in range(len(combined)):
        unique, inverse, counts = np.unique(
            combined[i], return_inverse=True, return_counts=True)
        if np.any(counts > 1):
            cumsum = np.concatenate([[0], np.cumsum(counts)])
            avg_ranks = (cumsum[:-1] + cumsum[1:] + 1) / 2
            ranks[i] = avg_ranks[inverse]

    R1 = ranks[:, :n1].sum(axis=1)
    U1 = R1 - n1 * (n1 + 1) / 2
    return U1


def brunnermunzel_w_only(samples, ref):
    """Vectorized Brunner-Munzel W statistic.

    Sign convention matches scipy's `scipy.stats.brunnermunzel` (returns y - x).

    Args:
        samples: shape (n_samples, sample_size)
        ref:     shape (n_ref,)

    Returns:
        W statistics, shape (n_samples,).  May contain NaN/inf for degenerate
        inputs (zero variance); callers should filter with `np.isfinite`.
    """
    samples = np.atleast_2d(samples)
    nx = samples.shape[1]
    ny = len(ref)

    W_stats = []
    for sample in samples:
        rankx = rankdata(sample)
        ranky = rankdata(ref)

        combined = np.concatenate([sample, ref])
        rankc = rankdata(combined)
        rankcx = rankc[:nx]
        rankcy = rankc[nx:]

        Sx = np.var(rankcx - rankx, ddof=1)
        Sy = np.var(rankcy - ranky, ddof=1)

        W = nx * ny * (np.mean(rankcy) - np.mean(rankcx))
        W /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)
        W_stats.append(W)

    return np.array(W_stats)
