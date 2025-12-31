# xpbd/solver_patch.py
import numpy as np
import warp as wp

from .constraints_basic import predict, update_v, apply_anchor_targets, reset_float
from .constraints_patch import xpbd_patch_constraints, apply_dx

wp.init()

class XPBDPatchSolver:
    """
    Mesh-free XPBD using per-point kNN patch deformation gradient F_i,
    enforcing hydrostatic + deviatoric constraints (ICRA / Macklin-style),
    without tets and without edges.

    Optional anchors:
      - attachments (inv_mass==0 points or explicit ids)
      - grasp anchors driven by tool pose
    """

    def __init__(
        self,
        x0: np.ndarray,                # (N,3) initial/reference positions
        nbrs: np.ndarray,              # (N,k)
        w: np.ndarray,                 # (N,k)
        r0: np.ndarray,                # (N,k,3)
        Binv: np.ndarray,              # (N,3,3)
        V: np.ndarray,                 # (N,)
        inv_mass: np.ndarray | None = None,
        gravity=(0.0, -9.81, 0.0),
        dt: float = 1/240.0,
        iterations: int = 10,
        damping: float = 0.995,
        mu: float = 3.0,               # deviatoric stiffness (Lamé μ)
        lam: float = 10.0,             # volumetric stiffness (Lamé λ)
        gamma: float = 1.0,            # hydrostatic rest-stability shift (det(F)-gamma)
        eps: float = 1e-8,
        grasp_stiffness: float = 0.6,
        attach_stiffness: float = 1.0,
        quasi_static: bool = False,    # if True: no velocity integration (helps drift)
    ):
        assert x0.ndim == 2 and x0.shape[1] == 3
        N = x0.shape[0]
        assert nbrs.shape[0] == N
        assert Binv.shape == (N,3,3)

        if inv_mass is None:
            inv_mass = np.ones((N,), dtype=np.float32)

        self.N = N
        self.k = nbrs.shape[1]

        self.dt = float(dt)
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.mu = float(mu)
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.grasp_stiffness = float(grasp_stiffness)
        self.attach_stiffness = float(attach_stiffness)
        self.quasi_static = bool(quasi_static)

        self.g = wp.vec3(float(gravity[0]), float(gravity[1]), float(gravity[2]))

        # state
        self.x = wp.array(x0.astype(np.float32), dtype=wp.vec3)
        self.x_prev = wp.array(x0.astype(np.float32), dtype=wp.vec3)
        self.v = wp.zeros(N, dtype=wp.vec3)
        self.inv_mass = wp.array(inv_mass.astype(np.float32), dtype=float)

        # patch data
        self.nbrs = wp.array2d(nbrs.astype(np.int32), dtype=int)          # (N,k)
        self.w = wp.array2d(w.astype(np.float32), dtype=float)            # (N,k)
        # store r0 as array2d of vec3: (N,k)
        self.r0 = wp.array2d(r0.astype(np.float32), dtype=wp.vec3)        # (N,k)
        self.V = wp.array(V.astype(np.float32), dtype=float)              # (N,)

        # store Binv rows as (N*3,) vec3 for easy indexing
        Binv_rows = np.zeros((N*3, 3), dtype=np.float32)
        Binv_rows[0::3, :] = Binv[:, 0, :]
        Binv_rows[1::3, :] = Binv[:, 1, :]
        Binv_rows[2::3, :] = Binv[:, 2, :]
        self.Binv_rows = wp.array(Binv_rows, dtype=wp.vec3)

        # lambdas (one per point per constraint)
        self.lam_h = wp.zeros(N, dtype=float)
        self.lam_d = wp.zeros(N, dtype=float)

        # dx accumulators (float arrays for atomics)
        self.dx_x = wp.zeros(N, dtype=float)
        self.dx_y = wp.zeros(N, dtype=float)
        self.dx_z = wp.zeros(N, dtype=float)

        # anchors
        self.attach_ids = None
        self.attach_targets = None
        self.grasp_ids = None
        self.grasp_targets = None

        # --- residual outputs for visualization / saving ---
        self.res_hydro = wp.zeros(N, dtype=float)
        self.res_dev   = wp.zeros(N, dtype=float)
        self.detF      = wp.zeros(N, dtype=float)
        self.trFtF     = wp.zeros(N, dtype=float)

    def set_attachments(self, attach_ids: np.ndarray, attach_targets: np.ndarray):
        self.attach_ids = wp.array(attach_ids.astype(np.int32), dtype=int)
        self.attach_targets = wp.array(attach_targets.astype(np.float32), dtype=wp.vec3)

    def set_grasp_region(self, grasp_ids: np.ndarray):
        self.grasp_ids = wp.array(grasp_ids.astype(np.int32), dtype=int)
        zeros = np.zeros((grasp_ids.shape[0], 3), dtype=np.float32)
        self.grasp_targets = wp.array(zeros, dtype=wp.vec3)

    @staticmethod
    def _tool_target_from_pose(R: np.ndarray, t: np.ndarray, local_offset=(0.0, 0.0, 0.0)):
        off = np.array(local_offset, dtype=np.float32)
        return (R @ off + t).astype(np.float32)

    def compute_residuals_numpy(self):
        """
        Computes per-point hydrostatic/dev residuals from the SAME F_i definition
        used by the patch constraints (as long as compute_patch_residuals matches
        the sign convention used in xpbd_patch_constraints).
        Returns:
          res_hydro: (N,)
          res_dev:   (N,)
          detF:      (N,)
          trFtF:     (N,)
        """
        from .constraints_patch import compute_patch_residuals

        wp.launch(
            compute_patch_residuals,
            dim=self.N,
            inputs=[
                self.x, self.inv_mass,
                self.nbrs, self.w, self.r0, self.Binv_rows,
                self.gamma, self.eps,
                self.res_hydro, self.res_dev,
                self.detF, self.trFtF
            ],
        )

        return (
            self.res_hydro.numpy().astype(np.float32),
            self.res_dev.numpy().astype(np.float32),
            self.detF.numpy().astype(np.float32),
            self.trFtF.numpy().astype(np.float32),
        )

    def step(self, R: np.ndarray, t: np.ndarray, jaw: float | None = None, jaw_close_thresh: float = 0.2):
        grasp_on = True if jaw is None else (float(jaw) < float(jaw_close_thresh))
        print("jaw:", jaw, "jaw_thresh:", jaw_close_thresh, "grasp_on:", grasp_on)

        # reset lambdas each frame (you can keep them for warm-start, but start simple)
        wp.launch(reset_float, dim=self.N, inputs=[self.lam_h])
        wp.launch(reset_float, dim=self.N, inputs=[self.lam_d])

        # store prev
        wp.copy(self.x_prev, self.x)

        # predict
        if not self.quasi_static:
            wp.launch(predict, dim=self.N, inputs=[self.x, self.v, self.inv_mass, self.g, self.dt])

        # update grasp targets
        if grasp_on and (self.grasp_ids is not None):
            target = self._tool_target_from_pose(R, t)
            gt = np.repeat(target[None, :], self.grasp_targets.shape[0], axis=0).astype(np.float32)
            self.grasp_targets.assign(gt)

        # solve iterations
        for _ in range(self.iterations):
            # attachments
            if self.attach_ids is not None:
                wp.launch(
                    apply_anchor_targets,
                    dim=self.attach_ids.shape[0],
                    inputs=[self.x, self.inv_mass, self.attach_ids, self.attach_targets, self.attach_stiffness],
                )

            # reset dx
            wp.launch(reset_float, dim=self.N, inputs=[self.dx_x])
            wp.launch(reset_float, dim=self.N, inputs=[self.dx_y])
            wp.launch(reset_float, dim=self.N, inputs=[self.dx_z])

            # patch constraints -> accumulate dx
            wp.launch(
                xpbd_patch_constraints,
                dim=self.N,
                inputs=[
                    self.x, self.inv_mass,
                    self.nbrs, self.w, self.r0, self.Binv_rows, self.V,
                    self.lam_h, self.lam_d,
                    self.dx_x, self.dx_y, self.dx_z,
                    self.dt, self.mu, self.lam, self.gamma, self.eps
                ],
            )
            dx = np.sqrt(
                self.dx_x.numpy()**2 +
                self.dx_y.numpy()**2 +
                self.dx_z.numpy()**2
            )
            # print("[DBG] dx: mean", dx.mean(), "max", dx.max(), "nz", (dx > 1e-12).sum())


            # # apply dx
            # wp.launch(
            #     apply_dx,
            #     dim=self.N,
            #     inputs=[self.x, self.inv_mass, self.dx_x, self.dx_y, self.dx_z],
            # )

            wp.launch(apply_dx, dim=self.N,
                inputs=[self.x, self.inv_mass, self.dx_x, self.dx_y, self.dx_z,
                        0.2,  # omega
                        1e-3  # max_step (1mm)
                    ],)


            # grasp anchors as soft targets (after projection)
            if grasp_on and (self.grasp_ids is not None):
                wp.launch(
                    apply_anchor_targets,
                    dim=self.grasp_ids.shape[0],
                    inputs=[self.x, self.inv_mass, self.grasp_ids, self.grasp_targets, self.grasp_stiffness],
                )

        # update velocity
        if not self.quasi_static:
            wp.launch(update_v, dim=self.N, inputs=[self.x, self.x_prev, self.v, self.dt, self.damping])
        else:
            self.v.zero_()
        return grasp_on

    def get_state_numpy(self):
        return self.x.numpy().astype(np.float32), self.v.numpy().astype(np.float32)
