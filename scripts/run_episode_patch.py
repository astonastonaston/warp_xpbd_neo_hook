# scripts/run_episode_patch.py
import os
import argparse
import numpy as np

from xpbd.io import load_pointcloud_npz, load_tool_traj_npz, save_episode_metadata, write_ply
# from xpbd.patches import build_patches_knn
from xpbd.patches_fast import build_patches_knn_ckdtree
from xpbd.solver_patch import XPBDPatchSolver

grasp_max_k = 6000

def _try_get_npz(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

def _normalize_rgb(rgb):
    """Return float32 RGB in [0,1] or None."""
    if rgb is None:
        return None
    rgb = np.asarray(rgb)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        return None
    rgb = rgb.astype(np.float32)
    mx = float(np.max(rgb)) if rgb.size else 0.0
    if mx > 1.5:   # likely 0..255
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)

def print_tissue_stats(points, inv_mass):
    print("\n=== Tissue stats ===")
    print("N points:", points.shape[0])

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = points.mean(axis=0)
    extent = maxs - mins

    print("xyz min:", mins)
    print("xyz max:", maxs)
    print("xyz extent:", extent)
    print("xyz center:", center)

    # spacing estimate (nearest neighbor)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(points)
    dists, _ = nn.kneighbors(points[:2000])  # subsample for speed
    spacing = dists[:, 1]
    auto_r = 3.0 * np.percentile(dists, 10)   # or 3x small neighbor distance
    print("suggested grasp_radius ~", auto_r)

    print("approx spacing: mean {:.3e}  min {:.3e}  max {:.3e}".format(
        spacing.mean(), spacing.min(), spacing.max()
    ))

    # inv_mass
    print("inv_mass: mean {:.3f}  min {:.3f}  max {:.3f}".format(
        float(inv_mass.mean()), float(inv_mass.min()), float(inv_mass.max())
    ))
    print("fixed points:", int((inv_mass == 0).sum()))


def pick_grasp_ids(points: np.ndarray, center: np.ndarray, radius: float = 0.01, max_k: int = 300):
    d = np.linalg.norm(points - center[None, :], axis=1)
    ids = np.where(d < radius)[0]
    print("Picked", ids.shape[0], "grasp candidates within radius", radius)
    if ids.shape[0] > max_k:
        ids = ids[np.argsort(d[ids])[:max_k]]
    return ids.astype(np.int32)


def save_frame_npz(out_path: str,
                   x: np.ndarray,
                   v: np.ndarray,
                   tool_R: np.ndarray,
                   tool_t: np.ndarray,
                   grasp_on: bool,
                   res_h: np.ndarray | None = None,
                   res_d: np.ndarray | None = None,
                   detF: np.ndarray | None = None,
                   trFtF: np.ndarray | None = None,
                   tissue_rgb: np.ndarray | None = None):

    v_mag = np.linalg.norm(v, axis=1).astype(np.float32)

    if res_h is None:
        res_h = np.zeros((x.shape[0],), dtype=np.float32)
    if res_d is None:
        res_d = np.zeros((x.shape[0],), dtype=np.float32)
    if detF is None:
        detF = np.ones((x.shape[0],), dtype=np.float32)
    if trFtF is None:
        trFtF = (3.0 * np.ones((x.shape[0],), dtype=np.float32))

    residual = (res_h + res_d).astype(np.float32)

    payload = dict(
        tissue_xyz=x.astype(np.float32),
        tissue_v=v.astype(np.float32),
        v_mag=v_mag,

        res_hydro=res_h.astype(np.float32),
        res_dev=res_d.astype(np.float32),
        detF=detF.astype(np.float32),
        trFtF=trFtF.astype(np.float32),
        residual=residual,

        tool_R=tool_R.astype(np.float32),
        tool_t=tool_t.astype(np.float32),
        grasp_on=np.array([int(grasp_on)], dtype=np.int32),
    )

    if tissue_rgb is not None:
        payload["tissue_rgb"] = tissue_rgb.astype(np.float32)

    np.savez_compressed(out_path, **payload)



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tissue_npz", required=True)
    ap.add_argument("--tool_npz", required=True)
    ap.add_argument("--out_dir", required=True)

    # patch build
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--max_radius", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--reg", type=float, default=1e-6)

    # solver
    ap.add_argument("--dt", type=float, default=1/240.0)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--damping", type=float, default=0.995)

    ap.add_argument("--mu", type=float, default=3.0)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-4)

    ap.add_argument("--grasp_stiffness", type=float, default=0.6)
    ap.add_argument("--attach_stiffness", type=float, default=1.0)
    ap.add_argument("--quasi_static", action="store_true")

    # grasp selection
    ap.add_argument("--grasp_radius", type=float, default=0.01)

    # output controls
    ap.add_argument("--save_every", type=int, default=20, help="write .ply every N frames")
    ap.add_argument("--res_every", type=int, default=1, help="compute residuals every N frames (>=1)")
    ap.add_argument("--save_frame0", action="store_true", help="also save frame_000000 before stepping")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------
    # load inputs
    # ------------------------
    points0, inv_mass = load_pointcloud_npz(args.tissue_npz)

    # Optional: load RGB from tissue npz (if present)
    tissue_rgb = None
    try:
        td = np.load(args.tissue_npz)
        rgb_raw = _try_get_npz(td, ["tissue_rgb", "rgb", "colors", "color"])
        tissue_rgb = _normalize_rgb(rgb_raw)
        if tissue_rgb is not None:
            print("[RGB] found tissue rgb:", tissue_rgb.shape, "range:",
                float(tissue_rgb.min()), float(tissue_rgb.max()))
            if tissue_rgb.shape[0] != points0.shape[0]:
                print("[RGB][WARN] rgb N != points N, ignoring rgb")
                tissue_rgb = None
        else:
            print("[RGB] no rgb found in tissue npz")
    except Exception as e:
        print("[RGB][WARN] failed reading rgb from tissue npz:", repr(e))
        tissue_rgb = None

    print_tissue_stats(points0, inv_mass)
    R, t, jaw = load_tool_traj_npz(args.tool_npz)
    T = int(t.shape[0])

    def print_tool_stats(R, t):
        print("\n=== Tool stats ===")
        print("T frames:", t.shape[0])

        mins = t.min(axis=0)
        maxs = t.max(axis=0)
        extent = maxs - mins

        print("t min:", mins)
        print("t max:", maxs)
        print("t extent:", extent)

        # tool speed
        dt = np.linalg.norm(np.diff(t, axis=0), axis=1)
        print("tool step |Δt|: mean {:.3e}  max {:.3e}".format(
            dt.mean(), dt.max()
        ))

        # rotation magnitude (angle)
        def rot_angle(R):
            tr = np.trace(R)
            return np.arccos(np.clip((tr - 1) * 0.5, -1.0, 1.0))

        ang = np.array([rot_angle(R[i].T @ R[i+1]) for i in range(len(R)-1)])
        print("rot step |Δθ| (rad): mean {:.3e} max {:.3e}".format(
            ang.mean(), ang.max()
        ))
    print_tool_stats(R, t)


    print(points0.shape[0], "points total")
    print("inv_mass mean:", float(np.mean(inv_mass)), "min:", float(np.min(inv_mass)), "max:", float(np.max(inv_mass)))

    # # ------------------------
    # # build patches on reference x0
    # # ------------------------
    # nbrs, w, r0, Binv, V = build_patches_knn(
    #     points0,
    #     k=args.k,
    #     max_radius=args.max_radius,
    #     sigma=args.sigma,
    #     reg=args.reg,
    # )
    nbrs, w, r0, Binv, V = build_patches_knn_ckdtree(
        points0,
        k=args.k,
        max_radius=args.max_radius,
        sigma=args.sigma,
        reg=args.reg,
        chunk=20000,      # tune
        leafsize=32,
    )


    def print_patch_stats(points, nbrs, w, r0, V):
        print("\n=== Patch stats ===")
        k = nbrs.shape[1]

        # neighbor distances
        dists = []
        for i in range(min(2000, points.shape[0])):
            for kk in range(k):
                j = nbrs[i, kk]
                if j >= 0:
                    dists.append(np.linalg.norm(points[j] - points[i]))
        dists = np.array(dists)

        print("k:", k)
        print("neighbor dist: mean {:.3e}  min {:.3e}  max {:.3e}".format(
            dists.mean(), dists.min(), dists.max()
        ))

        print("V (patch volume proxy): min {:.3e} mean {:.3e} max {:.3e}".format(
            V.min(), V.mean(), V.max()
        ))
    print_patch_stats(points0, nbrs, w, r0, V)


    # quick sanity
    print("patch arrays:",
          "nbrs", nbrs.shape,
          "w", w.shape,
          "r0", r0.shape,
          "Binv", Binv.shape,
          "V", V.shape)

    # ------------------------
    # solver
    # ------------------------
    solver = XPBDPatchSolver(
        x0=points0,
        nbrs=nbrs,
        w=w,
        r0=r0,
        Binv=Binv,
        V=V,
        inv_mass=inv_mass,

        dt=args.dt,
        iterations=args.iters,
        damping=args.damping,
        mu=args.mu,
        lam=args.lam,
        gamma=args.gamma,
        eps=args.eps,

        grasp_stiffness=args.grasp_stiffness,
        attach_stiffness=args.attach_stiffness,
        quasi_static=args.quasi_static,
        # gravity=(0,0,0),  # usually best for pure tool-driven demos
    )

    # # attachments from inv_mass==0
    # attach_ids = np.where(inv_mass == 0.0)[0].astype(np.int32)
    # if attach_ids.size > 0 and grasp_ids.size > 0:
    #     fixed_set = set(map(int, attach_ids.tolist()))
    #     overlap = sum((int(i) in fixed_set) for i in grasp_ids)
    #     print(f"[GRASP] overlap with fixed points: {overlap} / {int(grasp_ids.shape[0])}")

    # attachments from inv_mass==0
    attach_ids = np.where(inv_mass == 0.0)[0].astype(np.int32)
    if attach_ids.size > 0:
        print(attach_ids.size, "fixed points (inv_mass==0)")
        solver.set_attachments(attach_ids, points0[attach_ids].astype(np.float32))

    # # grasp region near first tool tip (project y to tissue mean)
    # center = t[0].astype(np.float32).copy()
    # center[1] = float(points0[:, 1].mean())
    # grasp_ids = pick_grasp_ids(points0, center=center, radius=args.grasp_radius, max_k=300)



    d_all = np.linalg.norm(points0 - t[0][None, :], axis=1)
    print("\n=== Tool→tissue distance stats (t0) ===")
    for p in [0, 1, 5, 10, 50, 90, 99, 100]:
        print(f"  p{p:02d}: {np.percentile(d_all, p):.4f} m")
    print("  min idx:", int(np.argmin(d_all)))



    def pick_grasp_ids_nearest(points, tool_pos, radius=0.01, max_k=300):
        d = np.linalg.norm(points - tool_pos[None, :], axis=1)
        nn = int(np.argmin(d))
        center = points[nn]
        ids = np.where(d < radius)[0]
        if ids.shape[0] > max_k:
            ids = ids[np.argsort(d[ids])[:max_k]]
        return ids.astype(np.int32), center, float(d[nn])
    grasp_ids, grasp_center, d_nn = pick_grasp_ids_nearest(
        points0, tool_pos=t[0].astype(np.float32),
        radius=args.grasp_radius, max_k=grasp_max_k
    )
    print("nearest tissue point dist to tool:", d_nn)
    print("grasp_center:", grasp_center)
    print("picked grasp_ids:", grasp_ids.shape[0])
    print(f"[GRASP] grasped points: {int(grasp_ids.shape[0])} / {int(points0.shape[0])} "
      f"({100.0 * grasp_ids.shape[0] / max(1, points0.shape[0]):.2f}%)")
    # report overlap between grasp set and fixed set (should usually be 0)
    if attach_ids.size > 0 and grasp_ids.size > 0:
        fixed_mask = np.zeros((points0.shape[0],), dtype=bool)
        fixed_mask[attach_ids] = True
        overlap = int(fixed_mask[grasp_ids].sum())
        print(f"[GRASP] overlap with fixed points: {overlap} / {int(grasp_ids.shape[0])}")

    if grasp_ids.size > 0:
        d = np.linalg.norm(points0[grasp_ids] - t[0][None, :], axis=1)
        print(f"[GRASP] radius={args.grasp_radius:.4f}  "
            f"d_to_tool: mean={d.mean():.4e} min={d.min():.4e} max={d.max():.4e}")



    if grasp_ids.size > 0:
        solver.set_grasp_region(grasp_ids)
    else:
        print("[WARN] still no grasp points — increase --grasp_radius or check coordinate frames")



    def print_grasp_distance(points, grasp_ids, t0):
        d = np.linalg.norm(points[grasp_ids] - t0[None, :], axis=1)
        print("\n=== Grasp proximity ===")
        print("grasp pts:", grasp_ids.shape[0])
        print("dist to tool t0: mean {:.3e} min {:.3e} max {:.3e}".format(
            d.mean(), d.min(), d.max()
        ))
    print_grasp_distance(points0, grasp_ids, t[0])


    d0 = np.linalg.norm(points0[grasp_ids] - t[0][None,:], axis=1)
    print("dist grasp points to tool at t0: mean", d0.mean(), "min", d0.min(), "max", d0.max())
    solver.set_grasp_region(grasp_ids)

    # metadata
    meta = {
        "tissue_npz": args.tissue_npz,
        "tool_npz": args.tool_npz,
        "N": int(points0.shape[0]),
        "T": int(T),
        "k": int(args.k),
        "max_radius": float(args.max_radius),
        "sigma": None if args.sigma is None else float(args.sigma),
        "reg": float(args.reg),
        "dt": float(args.dt),
        "iters": int(args.iters),
        "damping": float(args.damping),
        "mu": float(args.mu),
        "lam": float(args.lam),
        "gamma": float(args.gamma),
        "eps": float(args.eps),
        "quasi_static": bool(args.quasi_static),
        "grasp_ids_count": int(grasp_ids.shape[0]),
        "save_every": int(args.save_every),
        "res_every": int(args.res_every),
    }
    save_episode_metadata(args.out_dir, meta)

    # ------------------------
    # optionally save frame 0 (rest)
    # ------------------------
    if args.save_frame0:
        x0_sim, v0_sim = solver.get_state_numpy()
        res_h0, res_d0, detF0, trFtF0 = solver.compute_residuals_numpy()
        print("[frame0] hydro mean/max:", float(res_h0.mean()), float(res_h0.max()),
              "dev mean/max:", float(res_d0.mean()), float(res_d0.max()),
              "detF mean/min/max:", float(detF0.mean()), float(detF0.min()), float(detF0.max()))
        save_frame_npz(
            os.path.join(args.out_dir, "frame_000000.npz"),
            x0_sim, v0_sim, R[0], t[0], grasp_on=False,
            res_h=res_h0, res_d=res_d0, detF=detF0, trFtF=trFtF0,
            tissue_rgb=tissue_rgb
        )
        write_ply(os.path.join(args.out_dir, "frame_000000.ply"), x0_sim)

    # ------------------------
    # rollout
    # ------------------------
    cached = None  # (res_h, res_d, detF, trFtF)
    for fi in range(T):
        grasp_on = solver.step(R[fi], t[fi], jaw[fi] if jaw is not None else None)

        x, v = solver.get_state_numpy()

        # residuals (can throttle for speed)
        if (fi % max(1, args.res_every)) == 0 or cached is None:
            res_h, res_d, detF, trFtF = solver.compute_residuals_numpy()
            cached = (res_h, res_d, detF, trFtF)

            if fi % 20 == 0:
                print(f"[{fi:04d}] hydro mean/max {res_h.mean():.3e}/{res_h.max():.3e} | "
                      f"dev mean/max {res_d.mean():.3e}/{res_d.max():.3e} | "
                      f"detF mean {detF.mean():.4f}")

        else:
            res_h, res_d, detF, trFtF = cached

        save_frame_npz(
            os.path.join(args.out_dir, f"frame_{fi:06d}.npz"),
            x, v, R[fi], t[fi], grasp_on,
            res_h=res_h, res_d=res_d, detF=detF, trFtF=trFtF,
            tissue_rgb=tissue_rgb
        )

        if args.save_every > 0 and (fi % args.save_every) == 0:
            write_ply(os.path.join(args.out_dir, f"frame_{fi:06d}.ply"), x)
            print(f"[{fi:04d}/{T}] wrote frame_{fi:06d}.npz/.ply")

    print("Done. Output:", args.out_dir)


if __name__ == "__main__":
    main()


# command for running full tissue pcd + tool traj episode with patch xpbd solver
# python scripts/run_episode_patch.py   --tissue_npz data_demo_all/tissue_points.npz   --tool_npz   data_demo_all/tool_traj.npz   --out_dir    out_episode_patch_all   --k 12 --max_radius 0.03   --grasp_radius 0.1   --dt 0.0041666667 --iters 15   --mu 0.2 --lam 1.0 --eps 1e-4   --quasi_static


# run epi demo (circle or poke drag)
# python scripts/run_episode_patch.py   --tissue_npz data_demo/tissue_points_poke_drag.npz   --tool_npz   data_demo/tool_traj_poke_drag.npz   --out_dir    out_episode_patch_poke_drag   --k 16   --max_radius 0.0005   --dt 0.002   --iters 30   --mu 0.2   --lam 1.0   --eps 1e-4   --quasi_static   --save_every 20   --save_frame0 --grasp_radius 0.05