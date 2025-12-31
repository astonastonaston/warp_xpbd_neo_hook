# xpbd/io.py
import json
import numpy as np
from pathlib import Path

def load_pointcloud_npz(path: str):
    data = np.load(path, allow_pickle=True)
    pts = data["points"].astype(np.float32)
    inv_mass = data["inv_mass"].astype(np.float32) if "inv_mass" in data else np.ones((pts.shape[0],), dtype=np.float32)
    return pts, inv_mass

def load_tool_traj_npz(path: str):
    data = np.load(path, allow_pickle=True)
    R = data["R"].astype(np.float32)
    t = data["t"].astype(np.float32)
    jaw = data["jaw"].astype(np.float32) if "jaw" in data else np.zeros((t.shape[0],), dtype=np.float32)
    return R, t, jaw

def save_episode_metadata(out_dir: str, meta: dict):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def write_ply(path: str, points: np.ndarray):
    points = points.reshape(-1, 3)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
