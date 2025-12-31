# xpbd/patches.py
import numpy as np

def build_patches_knn(
    x0: np.ndarray,
    k: int = 12,
    max_radius: float | None = None,
    sigma: float | None = None,
    reg: float = 1e-6,
):
    """
    Build fixed kNN patches on rest/reference x0 (N,3).

    Returns:
      nbrs:   (N,k) int32 neighbor indices
      w:      (N,k) float32 weights
      r:      (N,k,3) float32 rest offsets r_ij = x0[j]-x0[i]
      Binv:   (N,3,3) float32 inverse of B_i = sum w r r^T (regularized)
      V:      (N,) float32 local "volume scale" (heuristic) for compliance scaling
    """
    assert x0.ndim == 2 and x0.shape[1] == 3
    N = x0.shape[0]

    # brute force distances
    diff = x0[:, None, :] - x0[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)

    nn = np.argpartition(dist2, kth=k, axis=1)[:, :k]  # (N,k)

    # optional radius filtering: if too few survive, we still keep original nn (stable)
    if max_radius is not None:
        r2 = max_radius * max_radius
        for i in range(N):
            keep = nn[i][dist2[i, nn[i]] <= r2]
            if keep.size >= max(3, k // 2):
                # pad back to k if needed
                pad = nn[i][: max(0, k - keep.size)]
                nn[i] = np.concatenate([keep, pad], axis=0)[:k]

    nbrs = nn.astype(np.int32)

    # weights
    if sigma is None:
        # heuristic: median neighbor distance
        d = np.sqrt(np.take_along_axis(dist2, nbrs, axis=1) + 1e-12)
        sigma = float(np.median(d))
        sigma = max(sigma, 1e-6)

    d2 = np.take_along_axis(dist2, nbrs, axis=1)
    w = np.exp(-d2 / (sigma * sigma)).astype(np.float32)

    # rest offsets
    r = (x0[nbrs] - x0[:, None, :]).astype(np.float32)  # (N,k,3)

    # compute Binv per point
    Binv = np.zeros((N, 3, 3), dtype=np.float32)
    V = np.zeros((N,), dtype=np.float32)

    I = np.eye(3, dtype=np.float32)
    for i in range(N):
        Bi = np.zeros((3, 3), dtype=np.float32)
        for kk in range(k):
            rij = r[i, kk][:, None]  # (3,1)
            Bi += w[i, kk] * (rij @ rij.T)
        Bi += reg * I
        Binv[i] = np.linalg.inv(Bi).astype(np.float32)

        # "volume scale" heuristic: cube of mean neighbor length
        mean_len = float(np.mean(np.linalg.norm(r[i], axis=1)) + 1e-9)
        V[i] = (mean_len ** 3)
    print("V stats: min/mean/max =", V.min(), V.mean(), V.max())
    print("V tiny count (<1e-10):", np.sum(V < 1e-10), "/", V.size)

    return nbrs, w, r, Binv, V
