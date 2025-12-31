# scripts/make_demo_tissue.py
import os
import argparse
import numpy as np


def make_grid(nx: int, nz: int, spacing: float, y: float = 0.0):
    xs = (np.arange(nx) - (nx - 1) / 2) * spacing
    zs = (np.arange(nz) - (nz - 1) / 2) * spacing
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    Y = np.full_like(X, y, dtype=np.float32)
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    return pts


def farthest_point_sample(points: np.ndarray, m: int, seed: int = 0):
    """
    Simple FPS (O(N*m)) for visualization subsets.
    points: (N,3)
    """
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    if m >= N:
        return np.arange(N, dtype=np.int64)

    idx = np.empty((m,), dtype=np.int64)
    idx[0] = rng.integers(0, N)
    dist2 = np.full((N,), np.inf, dtype=np.float64)

    p0 = points[idx[0]].astype(np.float64)
    d = points.astype(np.float64) - p0[None, :]
    dist2 = np.minimum(dist2, np.sum(d * d, axis=1))

    for k in range(1, m):
        idx[k] = int(np.argmax(dist2))
        pk = points[idx[k]].astype(np.float64)
        d = points.astype(np.float64) - pk[None, :]
        dist2 = np.minimum(dist2, np.sum(d * d, axis=1))

    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="data_demo/tissue_points.npz")
    ap.add_argument("--nx", type=int, default=120, help="grid resolution in x")
    ap.add_argument("--nz", type=int, default=120, help="grid resolution in z")
    ap.add_argument("--spacing", type=float, default=0.0015, help="meters")
    ap.add_argument("--y", type=float, default=0.0, help="plane height (meters)")

    # Optional: specify total points instead of nx/nz (auto makes near-square grid)
    ap.add_argument("--n_points", type=int, default=None,
                    help="if set, overrides nx/nz to approximately match this many points")

    # Optional: produce a *separate* visualization subset file
    ap.add_argument("--viz_points", type=int, default=None,
                    help="if set, also writes a *_viz.npz with only this many points (FPS)")
    ap.add_argument("--viz_method", choices=["fps", "random"], default="fps")
    ap.add_argument("--seed", type=int, default=0)

    # Mass options
    ap.add_argument("--pin_border", action="store_true",
                    help="if set, pins a border ring (inv_mass=0) to reduce global drift")
    ap.add_argument("--border_width", type=int, default=2, help="border width in grid cells when pin_border=True")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    if args.n_points is not None:
        n = int(args.n_points)
        side = int(np.sqrt(n))
        nx = max(2, side)
        nz = max(2, int(np.ceil(n / nx)))
        pts = make_grid(nx, nz, args.spacing, y=args.y)
        # trim to exactly n if you want
        if pts.shape[0] > n:
            pts = pts[:n]
    else:
        pts = make_grid(args.nx, args.nz, args.spacing, y=args.y)

    N = pts.shape[0]

    inv_mass = np.ones((N,), dtype=np.float32)

    # Optional pinning for stability (nice for demos)
    if args.pin_border:
        # infer grid dims if we made a grid; for n_points trimmed this may be imperfect, but ok for demo
        nx = args.nx if args.n_points is None else int(round(np.sqrt(N)))
        nz = args.nz if args.n_points is None else int(np.ceil(N / nx))

        # build mask on the full grid size we *think* we have
        mask = np.zeros((nz, nx), dtype=bool)
        bw = max(1, int(args.border_width))
        mask[:bw, :] = True
        mask[-bw:, :] = True
        mask[:, :bw] = True
        mask[:, -bw:] = True
        # apply to first nx*nz points only (if trimmed)
        pin = mask.reshape(-1)[:N]
        inv_mass[pin] = 0.0

    np.savez_compressed(args.out_path, points=pts, inv_mass=inv_mass)
    print(f"Wrote {args.out_path} points={N}  inv_mass(mean)={inv_mass.mean():.3f}  pinned={int(np.sum(inv_mass==0))}")

    # Optional viz subset
    if args.viz_points is not None:
        m = int(args.viz_points)
        if m <= 0:
            return
        if args.viz_method == "random":
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(N, size=min(m, N), replace=False)
        else:
            sel = farthest_point_sample(pts, min(m, N), seed=args.seed)

        out_viz = args.out_path.replace(".npz", "_viz.npz")
        np.savez_compressed(out_viz, points=pts[sel].astype(np.float32), inv_mass=inv_mass[sel].astype(np.float32))
        print(f"Wrote {out_viz} viz_points={len(sel)}")


if __name__ == "__main__":
    main()

# python scripts/make_demo_tissue.py --nx 100 --nz 100 --spacing 0.0001 --pin_border --border_width 2
# python scripts/make_demo_tissue.py --nx 240 --nz 240 --spacing 0.0009 --viz_points 8000

# square circle demo
# python scripts/make_demo_tissue.py \
#   --nx 100 \
#   --nz 100 \
#   --spacing 0.00015 \
#   --pin_border \
#   --border_width 2 \
#   --out data_demo/tissue_points.npz

# demo tissue for poke drag
# python scripts/make_demo_tissue.py      --nx 140   --nz 140   --spacing 0.0001   --out data_demo/tissue_points_poke_drag.npz --pin_border


