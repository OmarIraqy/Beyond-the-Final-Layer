"""Class-consistent gallery masking — filter cross-class matches."""

import numpy as np
from typing import List


def apply_class_mask(
    dist_matrix: np.ndarray,
    query_classes: List[int],
    gallery_classes: List[int],
) -> np.ndarray:
    """Set distance to inf for cross-class query-gallery pairs.

    This ensures each query only retrieves gallery items of the same
    object class (Container, Crosswalk, etc.).

    Args:
        dist_matrix: [num_query, num_gallery] distance (lower = better)
        query_classes: list of class indices for each query
        gallery_classes: list of class indices for each gallery item

    Returns:
        Modified distance matrix with cross-class pairs set to inf.
    """
    dist = dist_matrix.copy()
    q_cls = np.array(query_classes)
    g_cls = np.array(gallery_classes)

    # Vectorized: for each query, mask out gallery items with different class
    for i in range(len(q_cls)):
        if q_cls[i] < 0:
            continue  # unknown class, don't mask
        mask = g_cls != q_cls[i]
        dist[i, mask] = float("inf")

    return dist
