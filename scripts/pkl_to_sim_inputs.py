#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# def choose_frame_key(d, prefer=None):
#     """Pick a key from a dict keyed by frame_id."""
#     keys = sorted(list(d.keys()))
#     if len(keys) == 0:
#         raise RuntimeError("PKL dict has no keys")
#     if prefer is None:
#         return keys[0]
#     if prefer in d:
#         return prefer
#     # allow 'index into sorted keys'
#     if 0 <= prefer < len(keys):
#         return keys[prefer]
#     raise KeyError(f"Requested frame {prefer} not found. Available keys: {keys[:5]}...{keys[-5:]}")


def downsample_points(xyz, rgb=None, max_points=None, seed=0):
    if max_points is None or xyz.shape[0] <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    idx = np.sort(idx)
    xyz2 = xyz[idx]
    rgb2 = rgb[idx] if rgb is not None else None
    return xyz2, rgb2


def pin_border_if_grid(points, border_width=2, eps=1e-12):
    """
    Optional helper: pin points near the XY/XZ boundary of a grid-like sheet.
    This is a heuristic: it assumes your tissue is roughly a planar patch.
    """
    inv_mass = np.ones((points.shape[0],), dtype=np.float32)

    # Find approximate plane: assume y is "height" (like your sim), so border in x-z.
    x = points[:, 0]
    z = points[:, 2]
    xmin, xmax = float(x.min()), float(x.max())
    zmin, zmax = float(z.min()), float(z.max())

    # Estimate spacing by nearest-neighbor-ish using sorted unique coords
    # Works best if the cloud is a grid.
    ux = np.unique(np.round(x, 8))
    uz = np.unique(np.round(z, 8))
    if ux.size < 5 or uz.size < 5:
        # too unstructured; skip pinning
        return inv_mass

    dx = np.median(np.diff(ux))
    dz = np.median(np.diff(uz))
    step = float(max(dx, dz))
    if not np.isfinite(step) or step < eps:
        return inv_mass

    bw = border_width * step + 1e-9
    border = (x < xmin + bw) | (x > xmax - bw) | (z < zmin + bw) | (z > zmax - bw)
    inv_mass[border] = 0.0
    return inv_mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue_pkl", required=True, help="PKL with dict[frame_id] -> {'xyz','rgb',...}")
    ap.add_argument("--tool_pkl", required=True, help="PKL with dict[frame_id] -> {'t','R'}")

    ap.add_argument("--out_tissue_npz", default="data_demo/tissue_points.npz")
    ap.add_argument("--out_tool_npz", default="data_demo/tool_traj.npz")

    ap.add_argument("--tissue_frame", type=int, default=None,
                    help="Frame id to take tissue from. If omitted, uses smallest key.")
    ap.add_argument("--max_points", type=int, default=None,
                    help="Randomly downsample tissue to at most this many points.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pin_border", action="store_true",
                    help="Heuristically pin border points (inv_mass=0) if tissue looks grid-like.")
    ap.add_argument("--border_width", type=int, default=2,
                    help="Border width in grid steps for pin_border.")

    ap.add_argument("--jaw_mode", choices=["none", "always_on", "always_off", "half_on"], default="none",
                    help="Optional jaw signal. 'always_on' -> jaw=0 (grasp_on True with thresh=0.2).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_tissue_npz) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_tool_npz) or ".", exist_ok=True)

    # -------------------------
    # Tissue
    # -------------------------
    tissue = load_pkl(args.tissue_pkl)
    # if not isinstance(tissue, dict):
    #     raise TypeError(f"Expected tissue pkl to be dict, got {type(tissue)}")

    tentry = tissue[args.tissue_frame]
    if "xyz" not in tentry:
        raise KeyError(f"tissue[{args.tissue_frame}] missing 'xyz' key. Keys: {list(tentry.keys())}")

    xyz = np.asarray(tentry["xyz"], dtype=np.float32)
    rgb = None
    if "rgb" in tentry and tentry["rgb"] is not None:
        rgb = np.asarray(tentry["rgb"], dtype=np.float32)
        # if rgb is 0..255 uint8, normalize to 0..1 float
        if rgb.max() > 1.5:
            rgb = (rgb / 255.0).astype(np.float32)

    xyz, rgb = downsample_points(xyz, rgb, args.max_points, args.seed)

    if args.pin_border:
        inv_mass = pin_border_if_grid(xyz, border_width=args.border_width)
    else:
        inv_mass = np.ones((xyz.shape[0],), dtype=np.float32)

    # write tissue npz with RGB for later Open3D visualization
    if rgb is None:
        np.savez_compressed(args.out_tissue_npz, points=xyz, inv_mass=inv_mass)
    else:
        np.savez_compressed(args.out_tissue_npz, points=xyz, inv_mass=inv_mass, rgb=rgb)

    print(f"[OK] Wrote tissue: {args.out_tissue_npz}")
    print(f"     start frame={args.tissue_frame}, points={xyz.shape[0]}, rgb={'yes' if rgb is not None else 'no'}")
    print(f"     inv_mass: mean={float(inv_mass.mean()):.4f} min={float(inv_mass.min()):.1f} max={float(inv_mass.max()):.1f}")

    # -------------------------
    # Tool trajectory
    # -------------------------
    tool = load_pkl(args.tool_pkl)
    if not isinstance(tool, dict):
        raise TypeError(f"Expected tool pkl to be dict, got {type(tool)}")

    keys = sorted(list(tool.keys()))
    T = len(keys)
    R = np.zeros((T, 3, 3), dtype=np.float32)
    t = np.zeros((T, 3), dtype=np.float32)

    for i, k in enumerate(keys):
        entry = tool[k]
        if "R" not in entry or "t" not in entry:
            raise KeyError(f"tool[{k}] must have 'R' and 't'. Keys: {list(entry.keys())}")
        R[i] = np.asarray(entry["R"], dtype=np.float32)
        t[i] = np.asarray(entry["t"], dtype=np.float32)

    # Optional jaw
    jaw = None
    if args.jaw_mode != "none":
        jaw = np.ones((T,), dtype=np.float32)
        if args.jaw_mode == "always_on":
            jaw[:] = 0.0
        elif args.jaw_mode == "always_off":
            jaw[:] = 1.0
        elif args.jaw_mode == "half_on":
            jaw[:] = 1.0
            jaw[T//3:2*T//3] = 0.0

    if jaw is None:
        np.savez_compressed(args.out_tool_npz, R=R, t=t)
    else:
        np.savez_compressed(args.out_tool_npz, R=R, t=t, jaw=jaw)

    print(f"[OK] Wrote tool traj: {args.out_tool_npz}")
    print(f"     frames={T} (keys {keys[0]}..{keys[-1]}), jaw={'yes' if jaw is not None else 'no'}")


if __name__ == "__main__":
    main()

# python scripts/pkl_to_sim_inputs.py \
#   --tissue_pkl /home/nan/Desktop/warp_xpbd_neo_hook/steoeomis_tool_tissue/tissue_pts.pkl \
#   --tool_pkl   /home/nan/Desktop/warp_xpbd_neo_hook/steoeomis_tool_tissue/tool_3d_poses.pkl \
#   --tissue_frame 75 \
#   --max_points 20000 \
#   --jaw_mode always_on \
#   --out_tissue_npz data_demo/tissue_points.npz \
#   --out_tool_npz   data_demo/tool_traj.npz
