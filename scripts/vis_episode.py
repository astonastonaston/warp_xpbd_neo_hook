import os, glob, argparse
import numpy as np
import open3d as o3d
import cv2

# -------------------------
# Loading helpers
# -------------------------
def _as_float32(x):
    return x.astype(np.float32) if x is not None else None

def _try_get(d, keys):
    for k in keys:
        if k in getattr(d, "files", []):
            return d[k]
        if isinstance(d, dict) and k in d:
            return d[k]
    return None

def _normalize_rgb(rgb):
    """Return float RGB in [0,1] with shape (N,3) or None."""
    if rgb is None:
        return None
    rgb = np.asarray(rgb)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        return None
    rgb = rgb.astype(np.float32)
    mx = float(np.max(rgb)) if rgb.size else 0.0
    if mx > 1.5:  # likely 0..255
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)

def load_npz_any(path):
    d = np.load(path)

    pts = _try_get(d, ["tissue_xyz", "points", "xyz"])
    if pts is None:
        raise KeyError(f"{path}: cannot find point array in keys: {getattr(d,'files',[])}")
    pts = pts.astype(np.float32)

    rgb = _try_get(d, ["tissue_rgb", "rgb", "colors", "color"])
    rgb = _normalize_rgb(rgb)

    def get_scalar(key):
        v = _try_get(d, [key])
        return _as_float32(v) if v is not None else None

    ch = {
        "residual": get_scalar("residual"),
        "res_hydro": get_scalar("res_hydro"),
        "res_dev": get_scalar("res_dev"),
        "detF": get_scalar("detF"),
        "trFtF": get_scalar("trFtF"),
        "v_mag": get_scalar("v_mag"),
        "rgb": rgb,
    }

    R = _try_get(d, ["tool_R", "R"])
    t = _try_get(d, ["tool_t", "t"])
    if R is not None and t is not None:
        R = R.astype(np.float64)
        t = t.astype(np.float64)
    else:
        R, t = None, None

    if "grasp_on" in getattr(d, "files", []):
        try:
            ch["grasp_on"] = int(np.array(d["grasp_on"]).reshape(-1)[0])
        except Exception:
            ch["grasp_on"] = 0
    else:
        ch["grasp_on"] = 0

    return pts, ch, R, t


# -------------------------
# Coloring helpers
# -------------------------
def robust_normalize(x, p_low=5.0, p_high=99.0, eps=1e-8):
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    hi = max(hi, lo + eps)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0), lo, hi

def simple_normalize(x, eps=1e-8):
    hi = float(x.max()) if x.size else 1.0
    y = x / (hi + eps)
    return np.clip(y, 0.0, 1.0), 0.0, hi

def make_colors(v01):
    # blue -> red with a bit of green
    return np.stack([v01, 0.2*np.ones_like(v01), 1.0 - v01], axis=1)

def safe_gray(n):
    return np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float64), (n, 1))

def blend_rgb(base_rgb, overlay_rgb, alpha=0.65):
    if base_rgb is None:
        return overlay_rgb
    return np.clip((1.0 - alpha) * base_rgb + alpha * overlay_rgb, 0.0, 1.0)


# -------------------------
# Player
# -------------------------
class Player:
    def __init__(self, paths, fps=30, record_path="sim.mp4", show_tool=True,
                 roi=None, dup=1, point_size=4.0,
                 tool_axis=0.03, tool_tip_radius=0.004):
        self.paths = paths
        self.n = len(paths)
        self.i = 0
        self.playing = False
        self.fps = fps
        self.dt = 1.0 / fps

        # modes
        self.mode = "residual"  # rgb | residual | res_hydro | res_dev | detF | v_mag
        self.use_robust = True
        self.hot_only = False
        self.hot_percent = 10.0

        # visual tuning
        self.point_size = float(point_size)
        self.dup = int(max(1, dup))           # duplicate points for thicker look
        self.roi = roi                         # radius around tool (meters) or None

        # overlay scalar heatmap on RGB if RGB exists
        self.overlay_on_rgb = True
        self.overlay_alpha = 0.65

        self.recording = False
        self.record_path = record_path
        self.record_imgs = []

        self.pc = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=1400, height=900)
        self.vis.add_geometry(self.pc)

        # render options (make it “nicer”)
        opt = self.vis.get_render_option()
        opt.point_size = float(self.point_size)
        opt.background_color = np.asarray([0.02, 0.02, 0.03])  # dark bg
        opt.show_coordinate_frame = False

        self.show_tool = bool(show_tool)
        self.tool_frame_T_prev = np.eye(4)

        self.tool_axis = float(tool_axis)
        self.tool_tip_radius = float(tool_tip_radius)

        if self.show_tool:
            self.tool_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.tool_axis)
            self.tool_tip = o3d.geometry.TriangleMesh.create_sphere(radius=self.tool_tip_radius)
            self.tool_tip.compute_vertex_normals()
            self.tool_tip.paint_uniform_color([1.0, 0.8, 0.1])  # gold
            self.vis.add_geometry(self.tool_frame)
            self.vis.add_geometry(self.tool_tip)

        self._set_frame(0, init_view=True)

        # key callbacks
        self.vis.register_key_callback(ord(" "), self._toggle_play)
        self.vis.register_key_callback(262, self._next_frame)   # Right
        self.vis.register_key_callback(263, self._prev_frame)   # Left
        self.vis.register_key_callback(ord("R"), self._reset_view)
        self.vis.register_key_callback(ord("V"), self._toggle_record)
        self.vis.register_key_callback(ord("Q"), self._quit)
        self.vis.register_key_callback(256, self._quit)         # ESC

        # modes
        self.vis.register_key_callback(ord("0"), lambda vis: self._set_mode("rgb"))
        self.vis.register_key_callback(ord("1"), lambda vis: self._set_mode("residual"))
        self.vis.register_key_callback(ord("2"), lambda vis: self._set_mode("res_hydro"))
        self.vis.register_key_callback(ord("3"), lambda vis: self._set_mode("res_dev"))
        self.vis.register_key_callback(ord("4"), lambda vis: self._set_mode("detF"))
        self.vis.register_key_callback(ord("5"), lambda vis: self._set_mode("v_mag"))

        # toggles
        self.vis.register_key_callback(ord("H"), self._toggle_robust)
        self.vis.register_key_callback(ord("G"), self._toggle_hot_only)
        self.vis.register_key_callback(ord("O"), self._toggle_overlay)

        # point size hotkeys (fast)
        self.vis.register_key_callback(ord("-"), self._pt_smaller_fast)
        self.vis.register_key_callback(ord("="), self._pt_bigger_fast)

        # tool size hotkeys
        self.vis.register_key_callback(ord(","), self._tool_smaller)
        self.vis.register_key_callback(ord("."), self._tool_bigger)

    def _set_mode(self, mode):
        self.mode = mode
        self._set_frame(self.i)
        return False

    def _toggle_robust(self, vis):
        self.use_robust = not self.use_robust
        self._set_frame(self.i)
        return False

    def _toggle_hot_only(self, vis):
        self.hot_only = not self.hot_only
        self._set_frame(self.i)
        return False

    def _toggle_overlay(self, vis):
        self.overlay_on_rgb = not self.overlay_on_rgb
        self._set_frame(self.i)
        return False

    def _pt_smaller_fast(self, vis):
        self.point_size = max(1.0, self.point_size - 1.0)
        self.vis.get_render_option().point_size = float(self.point_size)
        self._set_frame(self.i)
        return False

    def _pt_bigger_fast(self, vis):
        self.point_size = min(20.0, self.point_size + 1.0)
        self.vis.get_render_option().point_size = float(self.point_size)
        self._set_frame(self.i)
        return False

    def _tool_smaller(self, vis):
        if not self.show_tool:
            return False
        self.tool_axis = max(0.005, self.tool_axis * 0.8)
        self.tool_tip_radius = max(0.001, self.tool_tip_radius * 0.8)
        self._rebuild_tool_geoms()
        self._set_frame(self.i)
        return False

    def _tool_bigger(self, vis):
        if not self.show_tool:
            return False
        self.tool_axis = min(0.20, self.tool_axis * 1.25)
        self.tool_tip_radius = min(0.05, self.tool_tip_radius * 1.25)
        self._rebuild_tool_geoms()
        self._set_frame(self.i)
        return False

    def _rebuild_tool_geoms(self):
        # remove & recreate for size change
        self.vis.remove_geometry(self.tool_frame, reset_bounding_box=False)
        self.vis.remove_geometry(self.tool_tip, reset_bounding_box=False)
        self.tool_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.tool_axis)
        self.tool_tip = o3d.geometry.TriangleMesh.create_sphere(radius=self.tool_tip_radius)
        self.tool_tip.compute_vertex_normals()
        self.tool_tip.paint_uniform_color([1.0, 0.8, 0.1])
        self.vis.add_geometry(self.tool_frame, reset_bounding_box=False)
        self.vis.add_geometry(self.tool_tip, reset_bounding_box=False)

    def _set_tool_pose(self, R, t, grasp_on=0):
        if (not self.show_tool) or (R is None) or (t is None):
            return

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        dT = T @ np.linalg.inv(self.tool_frame_T_prev)
        self.tool_frame.transform(dT)
        self.tool_tip.transform(dT)
        self.tool_frame_T_prev = T

        # recolor based on grasp
        if grasp_on:
            self.tool_tip.paint_uniform_color([0.1, 1.0, 0.2])  # green
        else:
            self.tool_tip.paint_uniform_color([1.0, 0.8, 0.1])  # gold

        self.vis.update_geometry(self.tool_frame)
        self.vis.update_geometry(self.tool_tip)

    def _choose_scalar(self, ch):
        if self.mode == "detF":
            if ch["detF"] is None:
                return None, "detF missing"
            return np.abs(ch["detF"] - 1.0), "|detF-1|"
        else:
            x = ch.get(self.mode, None)
            if x is None:
                return None, f"{self.mode} missing"
            return x, self.mode

    def _apply_roi(self, pts, scalar=None, rgb=None, tool_t=None):
        if self.roi is None or tool_t is None:
            return pts, scalar, rgb
        r = float(self.roi)
        d = np.linalg.norm(pts - tool_t[None, :], axis=1)
        mask = d <= r
        pts2 = pts[mask]
        sc2 = scalar[mask] if scalar is not None else None
        rgb2 = rgb[mask] if (rgb is not None and rgb.shape[0] == pts.shape[0]) else None
        return pts2, sc2, rgb2

    def _dup_points(self, pts, colors):
        if self.dup <= 1 or pts.shape[0] == 0:
            return pts, colors
        # duplicate points with tiny jitter so they look “thicker”
        # (pure visualization; does NOT change sim)
        reps = self.dup
        jitter = 0.00015  # 0.15mm
        pts_rep = np.repeat(pts, reps, axis=0)
        colors_rep = np.repeat(colors, reps, axis=0)
        noise = (np.random.randn(*pts_rep.shape).astype(np.float32)) * jitter
        pts_rep = pts_rep + noise
        return pts_rep, colors_rep

    def _set_frame(self, idx, init_view=False):
        idx = int(np.clip(idx, 0, self.n - 1))
        self.i = idx

        pts, ch, R, t = load_npz_any(self.paths[self.i])
        grasp_on = int(ch.get("grasp_on", 0))
        self._set_tool_pose(R, t, grasp_on=grasp_on)

        rgb_frame = ch.get("rgb", None)

        # ROI crop for nicer density
        pts_roi, _, rgb_roi = self._apply_roi(pts, None, rgb_frame, tool_t=t)

        # RGB mode
        if self.mode == "rgb":
            self.pc.points = o3d.utility.Vector3dVector(pts_roi)
            if rgb_roi is not None and rgb_roi.shape[0] == pts_roi.shape[0]:
                colors = rgb_roi.astype(np.float32)
            else:
                colors = safe_gray(pts_roi.shape[0]).astype(np.float32)

            pts_draw, col_draw = self._dup_points(pts_roi, colors)
            self.pc.points = o3d.utility.Vector3dVector(pts_draw)
            self.pc.colors = o3d.utility.Vector3dVector(col_draw.astype(np.float64))

            self.vis.update_geometry(self.pc)
            if init_view:
                self.vis.reset_view_point(True)

            info = "RGB" if rgb_roi is not None else "no rgb"
            roi_info = f"roi={self.roi}" if self.roi is not None else "roi=off"
            print(f"\rFrame {self.i+1}/{self.n}  mode=rgb  {info}  {roi_info}  dup={self.dup}  pt={self.point_size:.1f}  tool={self.tool_axis:.3f}", end="")
            self._maybe_capture()
            return

        # scalar mode
        scalar, label = self._choose_scalar(ch)
        if scalar is None:
            self.pc.points = o3d.utility.Vector3dVector(pts_roi)
            if rgb_roi is not None and rgb_roi.shape[0] == pts_roi.shape[0]:
                colors = rgb_roi.astype(np.float32)
            else:
                colors = safe_gray(pts_roi.shape[0]).astype(np.float32)

            pts_draw, col_draw = self._dup_points(pts_roi, colors)
            self.pc.points = o3d.utility.Vector3dVector(pts_draw)
            self.pc.colors = o3d.utility.Vector3dVector(col_draw.astype(np.float64))

            self.vis.update_geometry(self.pc)
            if init_view:
                self.vis.reset_view_point(True)
            print(f"\n[WARN] {label}")
            self._maybe_capture()
            return

        scalar = scalar.astype(np.float32)

        # ROI crop
        pts_roi, scalar_roi, rgb_roi = self._apply_roi(pts, scalar, rgb_frame, tool_t=t)
        scalar = scalar_roi if scalar_roi is not None else scalar
        pts = pts_roi
        rgb_frame = rgb_roi

        # hot-only
        if self.hot_only and scalar.size > 0:
            thr = np.percentile(scalar, 100.0 - self.hot_percent)
            mask = scalar >= thr
            pts_show = pts[mask]
            sc_show = scalar[mask]
            rgb_show = rgb_frame[mask] if (rgb_frame is not None and rgb_frame.shape[0] == pts.shape[0]) else None
        else:
            pts_show = pts
            sc_show = scalar
            rgb_show = rgb_frame if (rgb_frame is not None and rgb_frame.shape[0] == pts.shape[0]) else None

        # normalize
        if self.use_robust and sc_show.size > 0:
            v01, lo, hi = robust_normalize(sc_show, 5.0, 99.0)
        else:
            v01, lo, hi = simple_normalize(sc_show if sc_show.size else np.array([0.0], np.float32))

        heat = make_colors(v01).astype(np.float32)
        if self.overlay_on_rgb and (rgb_show is not None) and (rgb_show.shape[0] == heat.shape[0]):
            colors = blend_rgb(rgb_show.astype(np.float32), heat, alpha=self.overlay_alpha)
        else:
            colors = heat

        pts_draw, col_draw = self._dup_points(pts_show, colors)

        self.pc.points = o3d.utility.Vector3dVector(pts_draw)
        self.pc.colors = o3d.utility.Vector3dVector(col_draw.astype(np.float64))
        self.vis.update_geometry(self.pc)

        if init_view:
            self.vis.reset_view_point(True)

        roi_info = f"roi={self.roi}" if self.roi is not None else "roi=off"
        msg = (
            f"\rFrame {self.i+1}/{self.n}  mode={label}  "
            f"overlay={self.overlay_on_rgb}  robust={self.use_robust}  hot={self.hot_only}  "
            f"{roi_info}  dup={self.dup}  "
            f"range=[{lo:.2e},{hi:.2e}]  "
            f"mean={float(sc_show.mean()) if sc_show.size else 0.0:.2e} "
            f"max={float(sc_show.max()) if sc_show.size else 0.0:.2e}  "
            f"grasp_on={grasp_on}  pt={self.point_size:.1f}  tool={self.tool_axis:.3f}"
        )
        print(msg, end="")
        self._maybe_capture()

    def _maybe_capture(self):
        if not self.recording:
            return
        img = np.asarray(self.vis.capture_screen_float_buffer(False))
        img = (img * 255).astype(np.uint8)
        self.record_imgs.append(img)

    def _toggle_play(self, vis):
        self.playing = not self.playing
        return False

    def _next_frame(self, vis):
        self.playing = False
        self._set_frame((self.i + 1) % self.n)
        return False

    def _prev_frame(self, vis):
        self.playing = False
        self._set_frame((self.i - 1) % self.n)
        return False

    def _reset_view(self, vis):
        self.vis.reset_view_point(True)
        return False

    def _toggle_record(self, vis):
        self.recording = not self.recording
        if self.recording:
            self.record_imgs = []
            print("\n[REC] ON")
        else:
            print("\n[REC] OFF -> writing...")
            self._write_video()
        return False

    def _write_video(self):
        if len(self.record_imgs) == 0:
            print("[REC] no frames captured.")
            return
        h, w = self.record_imgs[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.record_path, fourcc, self.fps, (w, h))
        for img in self.record_imgs:
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"[REC] Saved: {self.record_path}")

    def _quit(self, vis):
        if self.recording:
            print("\n[REC] quitting -> writing...")
            self._write_video()
        print("\nBye.")
        self.vis.close()
        return False

    def run(self):
        import time
        last = time.time()
        while self.vis.poll_events():
            now = time.time()
            if self.playing and (now - last) >= self.dt:
                self._set_frame((self.i + 1) % self.n)
                last = now
            self.vis.update_renderer()
        self.vis.destroy_window()
        if self.recording:
            self._write_video()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dir", type=str, default=None, help="Directory with frame_*.npz")
    ap.add_argument("--npz", type=str, default=None, help="Single npz to preview")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no_tool", action="store_true")

    # nicer defaults
    ap.add_argument("--point_size", type=float, default=4.0)
    ap.add_argument("--tool_axis", type=float, default=0.03)
    ap.add_argument("--tool_tip", type=float, default=0.004)

    # make it look denser / focused
    ap.add_argument("--roi", type=float, default=None, help="Show only points within this radius (m) around tool tip")
    ap.add_argument("--dup", type=int, default=1, help="Duplicate points for thicker look (visual only)")

    args = ap.parse_args()

    if args.npz is not None:
        assert os.path.exists(args.npz), f"not found: {args.npz}"
        paths = [args.npz]
        record_path = "preview.mp4"
    else:
        assert args.ep_dir is not None, "Provide --ep_dir or --npz"
        paths = sorted(glob.glob(os.path.join(args.ep_dir, "frame_*.npz")))
        assert len(paths) > 0, f"No frames found in {args.ep_dir}"
        record_path = os.path.join(args.ep_dir, "sim.mp4")

    player = Player(
        paths,
        fps=args.fps,
        record_path=record_path,
        show_tool=(not args.no_tool),
        roi=args.roi,
        dup=args.dup,
        point_size=args.point_size,
        tool_axis=args.tool_axis,
        tool_tip_radius=args.tool_tip,
    )

    print("\nControls:")
    print("  Space: Play/Pause   Left/Right: Step   R: Reset view   V: Record   Q/Esc: Quit")
    print("  0: RGB (per-frame) | 1: residual  2: hydro  3: dev  4: |detF-1|  5: speed")
    print("  O: toggle heat overlay on RGB (in scalar modes)")
    print("  H: robust scaling  G: hot-only")
    print("  - / = : point size down/up")
    print("  , / . : tool size down/up")
    print("Tips:")
    print("  Use --roi 0.03 to focus near tool (looks denser), and --dup 2 to thicken points.")
    player.run()

if __name__ == "__main__":
    main()

# python scripts/vis_episode.py \
#   --ep_dir /home/nan/Desktop/warp_xpbd_neo_hook/out_episode_patch_debug \
#   --fps 30 \
#   --point_size 5 \
#   --tool_axis 0.05 \
#   --tool_tip 0.006

# python scripts/vis_episode.py \
#   --ep_dir /home/nan/Desktop/warp_xpbd_neo_hook/out_episode_patch_debug \
#   --fps 30 \
#   --roi 0.04 \
#   --dup 2 \
#   --point_size 6 \
#   --tool_axis 0.06 \
#   --tool_tip 0.008




# viz demo
#  python scripts/vis_episode.py   --ep_dir /home/nan/Desktop/warp_xpbd_neo_hook/out_episode_patch_poke_drag   --fps 30   --point_size 5   --tool_axis 0.003   --tool_tip 0.0005

# viz tool-tissue interaction simulation demo
# python scripts/vis_episode.py   --ep_dir /home/nan/Desktop/warp_xpbd_neo_hook/out_episode_patch_debug   --fps 30   --point_size 5   --tool_axis 0.1   --tool_tip 0.02
