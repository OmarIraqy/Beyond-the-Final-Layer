"""Query majority voting via reciprocal-rank fusion."""

import numpy as np


def query_majority_vote(distmat: np.ndarray, qf: np.ndarray, window: int = 5) -> np.ndarray:
    """Reciprocal-rank fusion across visually similar query frames.

    For each query i, finds the `window` most similar queries by cosine
    similarity of their L2-normalised features, then fuses all their ranked
    lists via reciprocal rank averaging (1/(rank+1)).

    Args:
        distmat: (num_q, num_g) distance matrix — lower = more similar
        qf:      (num_q, D) L2-normalised query features (numpy float32)
        window:  number of nearest-neighbour queries to include in the vote
    Returns:
        (num_q, num_g) aggregated distance matrix
    """
    num_q = distmat.shape[0]
    qq_sim = np.dot(qf, qf.T)              # (num_q, num_q)
    np.fill_diagonal(qq_sim, -np.inf)      # exclude self
    top_sim = np.argsort(qq_sim, axis=1)[:, -window:]  # (num_q, window)
    ranks = np.argsort(np.argsort(distmat, axis=1), axis=1).astype(np.float32)
    rr_scores = 1.0 / (ranks + 1.0)
    agg_scores = np.empty_like(rr_scores)
    for i in range(num_q):
        group = np.concatenate([[i], top_sim[i]])
        agg_scores[i] = rr_scores[group].mean(axis=0)
    return -agg_scores                     # negate: higher score = lower distance
