"""Distance computation utilities."""

import torch
import numpy as np
import torch.nn.functional as F


def cosine_similarity(qf: np.ndarray, gf: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and gallery features.

    Args:
        qf: [num_query, dim] query features
        gf: [num_gallery, dim] gallery features

    Returns:
        [num_query, num_gallery] similarity matrix (higher = more similar)
    """
    qf_t = torch.from_numpy(qf).float()
    gf_t = torch.from_numpy(gf).float()
    qf_t = F.normalize(qf_t, dim=1)
    gf_t = F.normalize(gf_t, dim=1)
    sim = torch.mm(qf_t, gf_t.t())
    return sim.numpy()


def euclidean_distance(qf: np.ndarray, gf: np.ndarray) -> np.ndarray:
    """Compute euclidean distance between query and gallery features.

    Args:
        qf: [num_query, dim]
        gf: [num_gallery, dim]

    Returns:
        [num_query, num_gallery] distance matrix (lower = more similar)
    """
    qf_t = torch.from_numpy(qf).float()
    gf_t = torch.from_numpy(gf).float()
    dist = torch.cdist(qf_t, gf_t, p=2)
    return dist.numpy()
