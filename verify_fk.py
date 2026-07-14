import os
import platform
if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")  # headless software renderer (Linux only)
import numpy as np
import mujoco
from kinematics import ArmFK, BoardFrame, rot_y

m = mujoco.MjModel.from_xml_path("chopstick_crane.xml")
d = mujoco.MjData(m)

arm = ArmFK(p0=(0, 0, 0.5), l1=0.35, l2=0.30, l3=0.15)
# pen_site is offset 0 from 'pen' body which is 0.15 from link3 body origin,
# so effectively the analytic l3 already reaches the site -> compare directly.

rng = np.random.default_rng(0)
max_pos_err = 0.0
max_jac_err = 0.0
pen_site_id = m.site("pen_site").id

for trial in range(200):
    q = rng.uniform(-1.5, 1.5, size=3)
    d.qpos[m.joint("theta1").qposadr[0]] = q[0]
    d.qpos[m.joint("theta2").qposadr[0]] = q[1]
    d.qpos[m.joint("theta3").qposadr[0]] = q[2]
    d.qpos[m.joint("phi").qposadr[0]] = 0.0
    mujoco.mj_forward(m, d)

    p_mj = d.site_xpos[pen_site_id]
    p_an = arm.fk(q)
    err = np.hypot(p_mj[0] - p_an[0], p_mj[2] - p_an[1])
    max_pos_err = max(max_pos_err, err)

    # numeric jacobian of mujoco FK vs analytic jacobian
    eps = 1e-6
    J_num = np.zeros((2, 3))
    for i in range(3):
        qp = q.copy(); qp[i] += eps
        d.qpos[m.joint("theta1").qposadr[0]] = qp[0]
        d.qpos[m.joint("theta2").qposadr[0]] = qp[1]
        d.qpos[m.joint("theta3").qposadr[0]] = qp[2]
        mujoco.mj_forward(m, d)
        p_plus = d.site_xpos[pen_site_id].copy()

        qm = q.copy(); qm[i] -= eps
        d.qpos[m.joint("theta1").qposadr[0]] = qm[0]
        d.qpos[m.joint("theta2").qposadr[0]] = qm[1]
        d.qpos[m.joint("theta3").qposadr[0]] = qm[2]
        mujoco.mj_forward(m, d)
        p_minus = d.site_xpos[pen_site_id].copy()

        J_num[0, i] = (p_plus[0] - p_minus[0]) / (2 * eps)
        J_num[1, i] = (p_plus[2] - p_minus[2]) / (2 * eps)

    J_an = arm.jacobian(q)
    jerr = np.max(np.abs(J_num - J_an))
    max_jac_err = max(max_jac_err, jerr)

print(f"max FK position error over 200 random configs: {max_pos_err:.3e} m")
print(f"max Jacobian element error over 200 random configs: {max_jac_err:.3e}")

# board frame check
board = BoardFrame(p_piv=(0.62, 0, 0.18))
phi_id = m.joint("phi").qposadr[0]
board_body_id = m.body("board").id
max_board_err = 0.0
for phi in np.linspace(-1.0, 1.0, 25):
    d.qpos[phi_id] = phi
    mujoco.mj_forward(m, d)
    that, nhat = board.frame(phi)
    # mujoco board body local +z axis is xmat column 2 (tangent), local -x is xmat column 0 negated (normal)
    xmat = d.xmat[board_body_id].reshape(3, 3)
    that_mj = xmat @ np.array([0, 0, 1.0])
    nhat_mj = xmat @ np.array([-1.0, 0, 0])
    e = max(np.max(np.abs(that - that_mj)), np.max(np.abs(nhat - nhat_mj)))
    max_board_err = max(max_board_err, e)
print(f"max board-frame axis error over phi sweep: {max_board_err:.3e}")
