# xpbd/constraints_patch.py
import warp as wp

@wp.func
def mat33_from_rows(r0: wp.vec3, r1: wp.vec3, r2: wp.vec3):
    return wp.mat33(r0[0], r0[1], r0[2],
                    r1[0], r1[1], r1[2],
                    r2[0], r2[1], r2[2])

@wp.func
def safe_pow(x: float, p: float):
    # Warp has wp.pow; just keep x positive
    return wp.pow(wp.max(x, 1e-12), p)

@wp.func
def clamp(x: float, lo: float, hi: float):
    return wp.min(wp.max(x, lo), hi)

@wp.func
def frob_norm_sq(A: wp.mat33):
    AtA = wp.transpose(A) * A
    return AtA[0,0] + AtA[1,1] + AtA[2,2]

@wp.func
def mat33_transpose(A: wp.mat33):
    return wp.transpose(A)

@wp.func
def mat33_mul(A: wp.mat33, B: wp.mat33):
    return A * B

@wp.func
def mat33_det(A: wp.mat33):
    return wp.determinant(A)

@wp.func
def mat33_inv(A: wp.mat33, eps: float):
    z = 0.0
    Aeps = A + wp.mat33(eps, z, z,
                        z, eps, z,
                        z, z, eps)
    return wp.inverse(Aeps)

@wp.func
def frob_norm(A: wp.mat33, eps: float):
    # sqrt(tr(A^T A))
    AtA = wp.transpose(A) * A
    tr = AtA[0,0] + AtA[1,1] + AtA[2,2]
    return wp.sqrt(tr + eps)

@wp.func
def atomic_add_vec3(arr: wp.array(dtype=wp.vec3), idx: int, v: wp.vec3):
    wp.atomic_add(arr, idx, wp.vec3(v[0], 0.0, 0.0))  # doesn't work: Warp atomic_add for vec3 is limited
    # Workaround: store dx as float3 arrays? -> easiest: use 3 float arrays
    # (We implement dx as 3 float arrays in solver instead.)
    return

@wp.kernel
def xpbd_patch_constraints(
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),

    nbrs: wp.array2d(dtype=int),        # (N,k)
    w: wp.array2d(dtype=float),         # (N,k)
    r0: wp.array2d(dtype=wp.vec3),      # (N,k) rest offsets as vec3
    Binv_rows: wp.array(dtype=wp.vec3), # (N*3,) three rows per point
    V: wp.array(dtype=float),           # (N,)

    lam_h: wp.array(dtype=float),
    lam_d: wp.array(dtype=float),

    dx_x: wp.array(dtype=float),
    dx_y: wp.array(dtype=float),
    dx_z: wp.array(dtype=float),

    dt: float,
    # compliance in "material form" (like your paper): alpha = 1/(stiffness * V)
    # we pass mu and lambda (Lamé) and compute alpha_h, alpha_d per point
    mu: float,
    lam: float,
    gamma: float,       # rest-stability shift for det constraint (default 1.0)
    eps: float,
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return

    # reconstruct Binv (3x3) from stored rows
    br0 = Binv_rows[i*3 + 0]
    br1 = Binv_rows[i*3 + 1]
    br2 = Binv_rows[i*3 + 2]
    Binv = mat33_from_rows(br0, br1, br2)

    # build A = sum w (xj - xi) r_ij^T
    xi = x[i]
    A = wp.mat33(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0)

    k = nbrs.shape[1]
    for kk in range(k):
        j = nbrs[i, kk]
        if j < 0:
            continue
        wij = w[i, kk]
        rij = r0[i, kk]          # vec3
        xj = x[j]
        dij = xj - xi            # vec3

        # outer(dij, rij): dij * rij^T
        A = A + wij * wp.outer(dij, rij)

    F = mat33_mul(A, Binv)

    # # --- constraints ---
    # detF = mat33_det(F)
    # CH = detF - gamma

    # s = frob_norm(F, eps)
    # CD = s - wp.sqrt(3.0)

    # # --- compute dC/dF ---
    # # hydro: d(det)/dF = det(F) * F^{-T}
    # FinvT = mat33_transpose(mat33_inv(F, eps))
    # dCH_dF = detF * FinvT

    # # dev: d(s)/dF = (1/s) * F
    # dCD_dF = (1.0 / (s + eps)) * F

    # ----------------------------
    # Constraints: hydro + dev
    # ----------------------------
    detF = mat33_det(F)
    J = wp.max(detF, 1e-6)         # stabilize

    CH = J - gamma

    # Isochoric deviatoric
    Jm13 = safe_pow(J, -1.0/3.0)
    Fbar = Jm13 * F

    s = frob_norm(Fbar, eps)
    CD = s - wp.sqrt(3.0)

    # ----------------------------
    # Gradients w.r.t F
    # ----------------------------
    FinvT = mat33_transpose(mat33_inv(F, eps))
    dCH_dF = J * FinvT

    # stable approx: ignore d(Jm13)/dF
    dCD_dF = (Jm13 / (s + eps)) * Fbar


    # chain to A: dC/dA = dC/dF * Binv^T
    BinvT = mat33_transpose(Binv)
    dCH_dA = mat33_mul(dCH_dF, BinvT)
    dCD_dA = mat33_mul(dCD_dF, BinvT)

    # gradient w.r.t positions:
    # A = sum w (xj - xi) r^T
    # so ∂C/∂xj = w * (dC/dA) * r
    # and ∂C/∂xi = -sum_j ∂C/∂xj
    gHi = wp.vec3(0.0,0.0,0.0)
    gDi = wp.vec3(0.0,0.0,0.0)

    # store neighbor gradients temporarily in local loop and accumulate norm
    sum_wg2_H = float(0.0)
    sum_wg2_D = float(0.0)

    for kk in range(k):
        j = nbrs[i, kk]
        if j < 0:
            continue
        wij = w[i, kk]
        rij = r0[i, kk]

        # multiply matrix (3x3) by vector rij
        gHj = wij * (dCH_dA * rij)
        gDj = wij * (dCD_dA * rij)

        # account for mass weights
        wj = inv_mass[j]
        if wj > 0.0:
            sum_wg2_H += wj * wp.dot(gHj, gHj)
            sum_wg2_D += wj * wp.dot(gDj, gDj)

        gHi = gHi - gHj
        gDi = gDi - gDj

    wi = inv_mass[i]
    sum_wg2_H += wi * wp.dot(gHi, gHi)
    sum_wg2_D += wi * wp.dot(gDi, gDi)

    # compliance scaling (paper-like): alpha = 1/(stiffness * V)
    Vi = V[i]
    Vi = wp.max(Vi, 1e-12)

    # alpha_h = (1.0 / (lam * Vi)) if lam > 0.0 else 0.0
    # alpha_d = (1.0 / (mu  * Vi)) if mu  > 0.0 else 0.0
    alpha_h = (1.0 / lam) if lam > 0.0 else 0.0
    alpha_d = (1.0 / mu ) if mu  > 0.0 else 0.0


    # XPBD uses alpha_tilde = alpha/(dt^2)
    ah = alpha_h / (dt * dt)
    ad = alpha_d / (dt * dt)

    # --- solve hydro lambda ---
    denom_h = sum_wg2_H + ah
    if denom_h > 1e-12:
        dlam_h = -(CH + ah * lam_h[i]) / denom_h
        lam_h[i] = lam_h[i] + dlam_h

        # apply dx: dx_p = w_p * dlam * grad
        # i
        dx_x[i] += wi * dlam_h * gHi[0]
        dx_y[i] += wi * dlam_h * gHi[1]
        dx_z[i] += wi * dlam_h * gHi[2]
        # neighbors
        for kk in range(k):
            j = nbrs[i, kk]
            if j < 0:
                continue
            wj = inv_mass[j]
            if wj == 0.0:
                continue
            wij = w[i, kk]
            rij = r0[i, kk]
            gHj = wij * (dCH_dA * rij)
            wp.atomic_add(dx_x, j, wj * dlam_h * gHj[0])
            wp.atomic_add(dx_y, j, wj * dlam_h * gHj[1])
            wp.atomic_add(dx_z, j, wj * dlam_h * gHj[2])

    # --- solve dev lambda ---
    denom_d = sum_wg2_D + ad
    if denom_d > 1e-12:
        dlam_d = -(CD + ad * lam_d[i]) / denom_d
        lam_d[i] = lam_d[i] + dlam_d

        dx_x[i] += wi * dlam_d * gDi[0]
        dx_y[i] += wi * dlam_d * gDi[1]
        dx_z[i] += wi * dlam_d * gDi[2]

        for kk in range(k):
            j = nbrs[i, kk]
            if j < 0:
                continue
            wj = inv_mass[j]
            if wj == 0.0:
                continue
            wij = w[i, kk]
            rij = r0[i, kk]
            gDj = wij * (dCD_dA * rij)
            wp.atomic_add(dx_x, j, wj * dlam_d * gDj[0])
            wp.atomic_add(dx_y, j, wj * dlam_d * gDj[1])
            wp.atomic_add(dx_z, j, wj * dlam_d * gDj[2])


@wp.kernel
def apply_dx(
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    dx_x: wp.array(dtype=float),
    dx_y: wp.array(dtype=float),
    dx_z: wp.array(dtype=float),
    omega: float,          # 0..1
    max_step: float,       # e.g. 1e-3 (1mm)
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return

    dx = wp.vec3(dx_x[i], dx_y[i], dx_z[i]) * omega

    # clamp magnitude
    mag = wp.length(dx)
    if mag > max_step:
        dx = dx * (max_step / (mag + 1e-12))

    x[i] = x[i] + dx







@wp.func
def mat33_trace(A: wp.mat33) -> float:
    return A[0,0] + A[1,1] + A[2,2]

# -------------------------------------------------------------------
# Kernel: compute per-point hydro & dev residuals from the SAME F_i
# -------------------------------------------------------------------
@wp.kernel
def compute_patch_residuals(
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),          # ADD THIS if you want to skip fixed points
    nbrs: wp.array2d(dtype=int),
    w: wp.array2d(dtype=float),
    r0: wp.array2d(dtype=wp.vec3),
    Binv_rows: wp.array(dtype=wp.vec3),
    gamma: float,
    eps: float,
    res_hydro: wp.array(dtype=float),
    res_dev: wp.array(dtype=float),
    detF_out: wp.array(dtype=float),
    trFtF_out: wp.array(dtype=float),
):
    i = wp.tid()

    # optional: match solver behavior
    if inv_mass[i] == 0.0:
        res_hydro[i] = 0.0
        res_dev[i] = 0.0
        detF_out[i] = 1.0
        trFtF_out[i] = 3.0
        return

    # reconstruct Binv
    b0 = Binv_rows[i*3 + 0]
    b1 = Binv_rows[i*3 + 1]
    b2 = Binv_rows[i*3 + 2]
    Binv = mat33_from_rows(b0, b1, b2)

    xi = x[i]
    A = wp.mat33(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0)

    kdim = nbrs.shape[1]
    for kk in range(kdim):
        j = nbrs[i, kk]
        if j < 0:
            continue

        wij = w[i, kk]
        if wij <= 0.0:
            continue

        rij = r0[i, kk]
        xj = x[j]
        dij = xj - xi                 # MATCH solver
        A = A + wij * wp.outer(dij, rij)

    F = mat33_mul(A, Binv)

    # J = mat33_det(F)
    # Ch = J - gamma
    # res_hydro[i] = wp.abs(Ch)

    # s = frob_norm(F, eps)            # sqrt(tr(F^T F) + eps)
    # Cd = s - wp.sqrt(3.0)
    # res_dev[i] = wp.abs(Cd)

    detF = mat33_det(F)
    J = wp.max(detF, 1e-6)

    Ch = J - gamma
    res_hydro[i] = wp.abs(Ch)

    Jm13 = safe_pow(J, -1.0/3.0)
    Fbar = Jm13 * F
    s = frob_norm(Fbar, eps)
    Cd = s - wp.sqrt(3.0)
    res_dev[i] = wp.abs(Cd)

    detF_out[i] = detF
    trFtF_out[i] = frob_norm_sq(F)   # optional: use ||F||^2 instead of tr(FtF) explicitly

