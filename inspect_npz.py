#!/usr/bin/env python3
"""
Inspect the structure and contents of a .npz file.

Usage:
    python inspect_npz.py /home/nan/Desktop/NRDF/examples/noisy_pose.npz
    python inspect_npz.py /home/nan/Desktop/NRDF/outputs/2025-12-27-03-31-05/res.npz
    python inspect_npz.py /home/nan/Desktop/NRDF/examples/noisy_pose.npz --verbose
"""

import argparse
import numpy as np
import textwrap
import pdb

def summarize_array(arr, name="", max_values=5):
    """Print a concise summary of a numpy array."""
    print(f"\n[{name}]")
    print(f"  type     : {type(arr)}")
    print(f"  dtype    : {arr.dtype}")
    print(f"  shape    : {arr.shape}")
    print(f"  ndim     : {arr.ndim}")
    print(f"  size     : {arr.size}")

    if arr.size > 0:
        print(f"  min/max  : {arr.min()} / {arr.max()}")
        flat = arr.ravel()
        print(f"  sample   : {flat[:max_values]}{' ...' if arr.size > max_values else ''}")

def inspect_npz(path, verbose=False):
    print(f"\n📦 Inspecting NPZ file: {path}\n")

    data = np.load(path, allow_pickle=True)

    pdb.set_trace()
    print(f"Keys ({len(data.files)}):")
    for k in data.files:
        print(f"  - {k}")

    for k in data.files:
        val = data[k]
        if isinstance(val, np.ndarray):
            summarize_array(val, name=k)
            if verbose and val.dtype == object:
                print(f"  object entries (first 3):")
                for i, obj in enumerate(val.flat[:3]):
                    print(f"    [{i}] type={type(obj)}")
        else:
            print(f"\n[{k}] Non-array object: type={type(val)}")

    print("\n✅ Done.")

def main():
    parser = argparse.ArgumentParser(description="Inspect .npz file structure")
    parser.add_argument("npz_path", type=str, help="Path to .npz file")
    parser.add_argument("--verbose", action="store_true", help="Show extra info for object arrays")
    args = parser.parse_args()

    inspect_npz(args.npz_path, verbose=args.verbose)

if __name__ == "__main__":
    main()

# python inspect_npz.py /home/nan/Desktop/warp_xpbd_neo_hook/data_demo/tissue_points.npz
# python inspect_npz.py /home/nan/Desktop/warp_xpbd_neo_hook/data_demo/tool_traj.npz
# python inspect_npz.py /home/nan/Desktop/warp_xpbd_neo_hook/out_episode_patch_0001/frame_000000.npz
