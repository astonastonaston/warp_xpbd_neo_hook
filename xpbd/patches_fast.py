import numpy as np

def build_patches_knn_ckdtree(points, k=12, max_radius=0.02, sigma=None, reg=1e-6, leafsize=32, chunk=20000):
    """
    Memory-safe kNN using scipy cKDTree + optional radius gate.
    Returns:
      nbrs: (N,k) int32  (=-1 if not found / outside radius)
      w:    (N,k) float32
      r0:   (N,k,3) float32  (rest offsets)
      Binv: (N,3,3) float32
      V:    (N,) float32
    """
    from scipy.spatial import cKDTree

    x = np.asarray(points, dtype=np.float32)
    N = x.shape[0]

    tree = cKDTree(x, leafsize=leafsize)

    nbrs = np.full((N, k), -1, dtype=np.int32)
    dist = np.full((N, k), np.inf, dtype=np.float32)

    # query k+1 because self is included at distance 0
    kq = k + 1

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        d, idx = tree.query(x[s:e], k=kq, workers=-1)  # (B,k+1)

        # drop self (first neighbor)
        d = d[:, 1:]
        idx = idx[:, 1:]

        if max_radius is not None and max_radius > 0:
            mask = d <= max_radius
            # write only valid ones; invalid stay -1/inf
            nbrs[s:e][mask] = idx[mask].astype(np.int32)
            dist[s:e][mask] = d[mask].astype(np.float32)
        else:
            nbrs[s:e] = idx.astype(np.int32)
            dist[s:e] = d.astype(np.float32)

    # weights: gaussian on distance or uniform
    if sigma is None:
        # heuristic: sigma from median non-inf distance
        dd = dist[np.isfinite(dist)]
        sigma = float(np.median(dd)) if dd.size else 1e-3

    sigma = max(float(sigma), 1e-12)
    w = np.exp(-(dist**2) / (2.0 * sigma * sigma)).astype(np.float32)
    w[~np.isfinite(dist)] = 0.0

    # r0 offsets
    r0 = np.zeros((N, k, 3), dtype=np.float32)
    valid = nbrs >= 0
    ii, kk = np.where(valid)
    r0[ii, kk] = (x[nbrs[ii, kk]] - x[ii]).astype(np.float32)

    # compute Binv & V per point from weighted covariance of r0
    # B = sum w * (r r^T)
    Binv = np.zeros((N, 3, 3), dtype=np.float32)
    V = np.zeros((N,), dtype=np.float32)

    epsI = reg * np.eye(3, dtype=np.float32)

    for i in range(N):
        wi = w[i]  # (k,)
        ri = r0[i] # (k,3)
        if wi.sum() <= 0:
            Binv[i] = np.eye(3, dtype=np.float32)
            V[i] = 0.0
            continue

        # B = Σ w * r r^T
        B = (ri.T * wi) @ ri  # (3,3)
        B = B + epsI
        try:
            Binv[i] = np.linalg.inv(B).astype(np.float32)
        except np.linalg.LinAlgError:
            Binv[i] = np.linalg.pinv(B).astype(np.float32)

        # simple volume proxy: sqrt(det(B)) (or det, depending your formulation)
        detB = float(np.linalg.det(B))
        V[i] = np.sqrt(max(detB, 0.0))

    return nbrs, w, r0, Binv, V
