"""Semantic confidence bonus for re-ranking.

Reduces distance for gallery images predicted to share the same class as the query.

    d̃_qi = d_qi − 2α · p_i(ŷ_q)

where ŷ_q is the argmax class of the query and p_i(ŷ_q) is the classifier
probability assigned to that class for gallery image i.
"""
from __future__ import annotations

import numpy as np
import torch


def apply_confidence_bonus(
    distmat: np.ndarray,
    q_probs: np.ndarray | torch.Tensor,
    g_probs: np.ndarray | torch.Tensor,
    alpha: float = 0.3,
) -> np.ndarray:
    """Apply semantic confidence bonus to a distance matrix.

    Args:
        distmat: (N_q, N_g) Euclidean distance matrix for L2-normalised features.
        q_probs: (N_q, C) softmax probabilities from the semantic classifier.
        g_probs: (N_g, C) softmax probabilities from the semantic classifier.
        alpha:   Bonus weight (0.1–0.5 typical).

    Returns:
        Modified (N_q, N_g) float32 distance matrix.
    """
    if isinstance(q_probs, torch.Tensor):
        q_probs = q_probs.numpy()
    if isinstance(g_probs, torch.Tensor):
        g_probs = g_probs.numpy()

    q_classes = q_probs.argmax(axis=1)               # (N_q,)
    bonus = g_probs[:, q_classes].T                   # (N_q, N_g)
    return (distmat - 2.0 * alpha * bonus).astype(np.float32)
