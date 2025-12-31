# xpbd/constraints_basic.py
import warp as wp

@wp.kernel
def predict(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    g: wp.vec3,
    dt: float,
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    v[i] = v[i] + g * dt
    x[i] = x[i] + v[i] * dt


@wp.kernel
def update_v(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    dt: float,
    damping: float,
):
    i = wp.tid()
    v[i] = (x[i] - x_prev[i]) / dt
    v[i] = v[i] * damping


@wp.kernel
def apply_anchor_targets(
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    ids: wp.array(dtype=int),
    targets: wp.array(dtype=wp.vec3),
    stiffness: float,
):
    k = wp.tid()
    i = ids[k]
    if inv_mass[i] == 0.0:
        return
    x[i] = x[i] + (targets[k] - x[i]) * stiffness


@wp.kernel
def reset_float(arr: wp.array(dtype=float)):
    i = wp.tid()
    arr[i] = 0.0


@wp.kernel
def reset_vec3(arr: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    arr[i] = wp.vec3(0.0, 0.0, 0.0)
