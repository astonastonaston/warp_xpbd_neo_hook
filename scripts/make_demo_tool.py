# scripts/make_demo_tool.py
import os
import argparse
import numpy as np


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="data_demo/tool_traj.npz")
    ap.add_argument("--T", type=int, default=180)

    # path types
    ap.add_argument("--mode", choices=["circle", "poke_drag"], default="poke_drag")
    ap.add_argument("--radius", type=float, default=0.02, help="for circle mode")
    ap.add_argument("--y", type=float, default=0.008, help="tool tip height (meters)")
    ap.add_argument("--amp", type=float, default=0.015, help="for poke_drag: drag amplitude (meters)")

    # grasp schedule
    ap.add_argument("--grasp", choices=["always_on", "always_off", "on_then_off"], default="always_on")
    ap.add_argument("--on_ratio", type=float, default=0.6, help="fraction of frames with jaw closed for on_then_off")
    ap.add_argument("--jaw_closed", type=float, default=0.0)
    ap.add_argument("--jaw_open", type=float, default=1.0)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    T = int(args.T)
    R = np.zeros((T, 3, 3), dtype=np.float32)
    t = np.zeros((T, 3), dtype=np.float32)
    jaw = np.zeros((T,), dtype=np.float32)

    # --- trajectory ---
    if args.mode == "circle":
        for i in range(T):
            ang = 2 * np.pi * (i / T)
            R[i] = rot_y(0.2 * np.sin(ang))
            t[i] = np.array([args.radius * np.cos(ang), args.y, args.radius * np.sin(ang)], dtype=np.float32)

    else:  # poke_drag: move down a bit then drag in x and z
        # phase splits
        t0 = int(0.15 * T)   # approach
        t1 = int(0.75 * T)   # drag
        t2 = T               # retract

        # start at center, slight approach
        for i in range(T):
            if i < t0:
                u = i / max(1, t0 - 1)
                # approach: y decreases a bit (toward tissue plane), x/z near 0
                x = 0.0
                z = 0.0
                y = args.y + (1.0 - u) * 0.006  # start higher, go down
            elif i < t1:
                u = (i - t0) / max(1, (t1 - t0) - 1)
                # drag: sweep in x and small z wobble
                x = (u - 0.5) * 2.0 * args.amp
                z = 0.4 * args.amp * np.sin(2 * np.pi * u)
                y = args.y
            else:
                u = (i - t1) / max(1, (t2 - t1) - 1)
                # retract up
                x = args.amp
                z = 0.0
                y = args.y + u * 0.010

            ang = 0.15 * np.sin(2 * np.pi * (i / T))
            R[i] = rot_y(ang)
            t[i] = np.array([x, y, z], dtype=np.float32)

    # --- grasp schedule (jaw) ---
    if args.grasp == "always_on":
        jaw[:] = args.jaw_closed
    elif args.grasp == "always_off":
        jaw[:] = args.jaw_open
    else:
        on_T = int(np.clip(args.on_ratio, 0.0, 1.0) * T)
        jaw[:on_T] = args.jaw_closed
        jaw[on_T:] = args.jaw_open

    np.savez_compressed(args.out_path, R=R, t=t, jaw=jaw)
    print(f"Wrote {args.out_path}  T={T}  grasp={args.grasp}  jaw_range=[{jaw.min():.2f},{jaw.max():.2f}]")


if __name__ == "__main__":
    main()

# python scripts/make_demo_tool.py --mode poke_drag --T 240 --grasp always_on --y 0.008 --amp 0.02
# python scripts/make_demo_tool.py --mode poke_drag --T 240 --grasp on_then_off --on_ratio 0.6

# square circle demo
# python scripts/make_demo_tool.py \
#   --mode circle \
#   --T 240 \
#   --radius 0.002 \
#   --y 0.003 \
#   --grasp always_on \
#   --out data_demo/tool_traj.npz

# demo poke drag
# python scripts/make_demo_tool.py   --mode poke_drag   --T 240   --radius 0.002     --out data_demo/tool_traj_poke_drag.npz


