"""K-reciprocal re-ranking (Zhong et al., CVPR 2017).

Standalone implementation — no dependency on PAT.
"""

import numpy as np
from scipy.sparse import csr_matrix


def re_ranking(
    q_g_dist: np.ndarray,
    q_q_dist: np.ndarray,
    g_g_dist: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """K-reciprocal encoding re-ranking.

    Args:
        q_g_dist: [num_query, num_gallery] cosine similarity (NOT distance)
        q_q_dist: [num_query, num_query] cosine similarity
        g_g_dist: [num_gallery, num_gallery] cosine similarity
        k1: k for k-reciprocal neighbors
        k2: k for query expansion
        lambda_value: weight for original distance

    Returns:
        [num_query, num_gallery] final distance (lower = more similar)
    """
    # Convert similarity to distance: dist = 2 - 2*sim
    # Build combined (Q+G) x (Q+G) distance matrix
    num_query = q_g_dist.shape[0]
    num_gallery = q_g_dist.shape[1]
    total = num_query + num_gallery

    original_dist = np.zeros((total, total), dtype=np.float32)
    original_dist[:num_query, num_query:] = 2.0 - 2.0 * q_g_dist
    original_dist[num_query:, :num_query] = original_dist[:num_query, num_query:].T
    original_dist[:num_query, :num_query] = 2.0 - 2.0 * q_q_dist
    original_dist[num_query:, num_query:] = 2.0 - 2.0 * g_g_dist

    # Clamp to non-negative
    original_dist = np.clip(original_dist, 0.0, None)

    # Gaussian kernel
    # (not applied here — we operate on raw distances)

    # Find k-reciprocal neighbors
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1), axis=1)[:, :k1 + 1]

    # Build Jaccard distance using k-reciprocal sets
    V = np.zeros((total, total), dtype=np.float32)

    for i in range(total):
        # Forward k-nearest neighbors
        forward_k_idx = initial_rank[i, :k1 + 1]
        # Find k-reciprocal neighbors
        k_reciprocal_idx = []
        for fk in forward_k_idx:
            fk_forward = initial_rank[fk, :k1 + 1]
            if i in fk_forward:
                k_reciprocal_idx.append(fk)
        k_reciprocal_idx = np.array(k_reciprocal_idx)

        # Expand k-reciprocal set
        k_reciprocal_expansion = list(k_reciprocal_idx)
        for candidate in k_reciprocal_idx:
            candidate_forward = initial_rank[candidate, : int(np.around(k1 / 2.0)) + 1]
            candidate_reciprocal = []
            for cf in candidate_forward:
                cf_forward = initial_rank[cf, : int(np.around(k1 / 2.0)) + 1]
                if candidate in cf_forward:
                    candidate_reciprocal.append(cf)
            candidate_reciprocal = np.array(candidate_reciprocal)
            if len(candidate_reciprocal) > 2.0 / 3.0 * len(candidate_forward):
                for cr in candidate_reciprocal:
                    if cr not in k_reciprocal_expansion:
                        k_reciprocal_expansion.append(cr)

        k_reciprocal_expansion = np.array(k_reciprocal_expansion)

        # Weight by Gaussian kernel
        weight = np.exp(-original_dist[i, k_reciprocal_expansion])
        V[i, k_reciprocal_expansion] = weight / (np.sum(weight) + 1e-12)

    # Query expansion
    if k2 > 0:
        V_qe = np.zeros_like(V)
        for i in range(total):
            V_qe[i] = np.mean(V[initial_rank[i, :k2 + 1]], axis=0)
        V = V_qe

    # Jaccard distance
    # For efficiency, convert to sparse
    invIndex = []
    for i in range(total):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros((num_query, total), dtype=np.float32)
    for i in range(num_query):
        temp_min = np.zeros((1, total), dtype=np.float32)
        nonzero_idx = np.where(V[i] != 0)[0]
        for j in nonzero_idx:
            candidates = invIndex[j]
            temp_min[0, candidates] += np.minimum(V[i, j], V[candidates, j])
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)

    # Final distance: combine Jaccard and original
    final_dist = jaccard_dist[:, num_query:] * (1.0 - lambda_value) + \
                 original_dist[:num_query, num_query:] * lambda_value

    return final_dist
