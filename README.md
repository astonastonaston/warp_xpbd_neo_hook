# warp_xpbd_neo_hook

**XPBD-based soft-tissue point-cloud simulation with patch-based Neo-Hookean constraints**

This project demonstrates a **point-cloud deformable tissue simulator** driven by a **moving surgical tool trajectory**, using:

- **Warp** for GPU-accelerated XPBD constraint solving  
- **kNN patch neighborhoods** to construct local deformation gradients  
- **Hydrostatic (volume)** + **Deviatoric (shape)** constraint splitting  
- **Open3D** visualization with residual heatmaps and optional RGB overlays  

It is designed as a **lightweight research demo** for soft-tissue dynamics modeling and visualization.

---

## ⚙️ Environment Setup

### Conda (Recommended)

```bash
cd /home/nan/Desktop/warp_xpbd_neo_hook

conda create -n warp python=3.11 -y
conda activate warp

pip install -U pip
pip install numpy scipy
pip install warp-lang
pip install open3d opencv-python scikit-learn
```

### Verify Warp GPU

```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
```

You should see your CUDA device listed.

---

## 🧩 Input File Formats

### Tissue Point Cloud (`tissue_points.npz`)

Required keys:

- `points` : (N,3) float32 — positions in meters  
- `inv_mass` : (N,) float32  
  - `1.0` → free particle  
  - `0.0` → fixed particle  

Optional:

- `rgb` or `tissue_rgb` : (N,3) uint8 or float — per-point colors

---

### Tool Trajectory (`tool_traj.npz`)

Required keys:

- `R` : (T,3,3) float32 — rotation matrices  
- `t` : (T,3) float32 — translations in meters  

Optional:

- `jaw` : (T,) float32 — grasp signal

---

## 🚀 Quick Demo (Full Pipeline)

### 1️⃣ Generate tissue

```bash
python scripts/make_demo_tissue.py \
  --out_path data_demo/tissue_points.npz \
  --nx 140 --nz 140 --spacing 0.0001 \
  --pin_border --border_width 2
```

### 2️⃣ Generate tool trajectory

```bash
python scripts/make_demo_tool.py \
  --out_path data_demo/tool_traj.npz \
  --mode poke_drag \
  --T 240 --radius 0.002 --y 0.005 \
  --grasp always_on
```

### 3️⃣ Run simulation

```bash
python scripts/run_episode_patch.py \
  --tissue_npz data_demo/tissue_points.npz \
  --tool_npz   data_demo/tool_traj.npz \
  --out_dir    out_episode_patch_demo \
  --k 12 \
  --max_radius 0.01 \
  --dt 0.0041666667 \
  --iters 15 \
  --mu 0.2 \
  --lam 1.0 \
  --eps 1e-4 \
  --quasi_static \
  --save_every 20 \
  --res_every 1 \
  --save_frame0
```

### 4️⃣ Visualize simulation

```bash
python scripts/vis_episode.py \
  --ep_dir out_episode_patch_demo \
  --fps 30
```

---

## 🎮 Visualization Controls

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| ← / → | Step frames |
| R | Reset camera |
| V | Record MP4 |
| Q / Esc | Quit |

### Display Modes

| Key | Mode | Meaning |
|-----|------|---------|
| 0 | RGB | Show stored per-point RGB |
| 1 | residual | res_hydro + res_dev |
| 2 | res_hydro | Volume error |
| 3 | res_dev | Shape distortion |
| 4 | \|det(F) − 1\| | Volume change |
| 5 | v_mag | Velocity magnitude |

### Toggles

| Key | Function |
|-----|----------|
| H | Robust percentile scaling |
| G | Show only top “hot” residual points |
| O | Overlay heatmap on RGB |
| [ / ] | Change point size |

**Tool axes** = coordinate frame  
**Tool sphere** = visualization only (no collision)

---

## 🧠 Physics Model

Each point builds a local deformation gradient **F** from its k-nearest neighbors.

### Hydrostatic (Volume) Constraint

```
C_h = det(F) - 1
res_hydro = |det(F) - 1|
```

### Deviatoric (Shape) Constraint

```
C_d = ||F||_F - sqrt(3)
res_dev = | ||F||_F - sqrt(3) |
```

### Combined Residual

```
residual = res_hydro + res_dev
```

---

## 🔧 Important Parameters

### Patch Construction

| Parameter | Meaning |
|-----------|---------|
| k | neighbors per patch |
| max_radius | search radius |
| sigma | Gaussian weight (optional) |
| reg | numerical regularization |

**Rule of thumb:**  
`max_radius ≈ 3–6 × point spacing`

### Solver Parameters

| Parameter | Meaning |
|-----------|---------|
| dt | timestep |
| iters | XPBD iterations per frame |
| mu | deviatoric stiffness |
| lam | hydrostatic stiffness |
| damping | velocity damping |
| eps | numerical epsilon |
| quasi_static | disables inertia |

---

## 🧲 Grasp Model

A set of points near the tool at frame 0 is selected as a grasp region.  
When `grasp_on=True`, those points are softly attached to tool motion.

*(No collision physics yet — attachment only.)*

---

## ⚠️ Known Limitations

- Tool sphere is visualization only (no collision)  
- CPU kNN patch building limits very large point clouds  
- No self-collision handling  

---

## 🎯 Recommended Demo Preset

```bash
python scripts/make_demo_tissue.py \
  --nx 140 --nz 140 --spacing 0.0001 --pin_border

python scripts/make_demo_tool.py \
  --mode poke_drag --radius 0.002 --y 0.005 --grasp always_on

python scripts/run_episode_patch.py \
  --tissue_npz data_demo/tissue_points.npz \
  --tool_npz data_demo/tool_traj.npz \
  --out_dir out_episode_patch_prof \
  --k 12 --max_radius 0.0006 \
  --dt 0.0041666667 --iters 20 \
  --mu 0.3 --lam 1.2 --eps 1e-4 \
  --quasi_static

python scripts/vis_episode.py --ep_dir out_episode_patch_prof
```

---

## 👤 Author

**Nan Xiao**  
University of Tennessee — AURAS Lab

---

## 🛠 Future Work

- GPU neighbor search (hash grid)  
- True tool–tissue collision constraints  
- Surface mesh extraction  
- Integration with real surgical trajectories  
