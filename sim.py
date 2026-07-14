import os
import platform
if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")  # headless software renderer (Linux only)
import argparse
import numpy as np
import mujoco
import imageio

from kinematics import ArmFK, BoardFrame
from controller import DLSController, ForcePIController


def get_contact_normal_force(model, data, geom1_id, geom2_id):
    """Sum |normal force| over all active contacts between the two geoms."""
    Fn = 0.0
    result = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        if {con.geom1, con.geom2} == {geom1_id, geom2_id}:
            mujoco.mj_contactForce(model, data, i, result)
            Fn += abs(result[0])
    return Fn


def run(profile="sweep", T_reach=1.5, T_hold=4.0, T_sweep=29.0, u0=0.15, u1=0.45,
        Fmin=2.0, Fmax=4.0, sine_amp=0.12, out_prefix="sweep"):
    model = mujoco.MjModel.from_xml_path("chopstick_crane.xml")
    data = mujoco.MjData(model)

    arm = ArmFK(p0=(0, 0, 0.5), l1=0.35, l2=0.30, l3=0.15)
    board = BoardFrame(p_piv=(0.62, 0, 0.18))

    # --- solve an initial posture standing off just in front of u0 (no contact yet) ---
    q_mid = np.array([0.3, -1.0, -0.6])
    standoff_ik = DLSController(q_mid=q_mid, kp_task=6.0, lam=0.03, kn=0.5, qdot_max=4.0)
    p_start = board.target_world_xz(0.0, u0, d_pen=-0.03)  # 3 cm off the surface
    q_init = q_mid.copy()
    for _ in range(400):  # offline settle, well clear of the real control loop
        p_cur = arm.fk(q_init)
        J = arm.jacobian(q_init)
        dq, _ = standoff_ik.step(q_init, J, p_start, p_cur)
        q_init = q_init + dq * 0.01
    q_mid = q_init.copy()

    ik = DLSController(q_mid=q_mid, kp_task=6.0, lam=0.05, kn=0.6, qdot_max=3.0)
    fpi = ForcePIController(Fmin=Fmin, Fmax=Fmax, kp=0.003, ki=0.01, d_min=-0.03, d_max=0.02)

    th1_adr = model.joint("theta1").qposadr[0]
    th2_adr = model.joint("theta2").qposadr[0]
    th3_adr = model.joint("theta3").qposadr[0]
    phi_adr = model.joint("phi").qposadr[0]
    pen_gid = model.geom("pen_tip").id
    board_gid = model.geom("board_face").id
    pen_site_id = model.site("pen_site").id

    # start at the commanded mid posture
    data.qpos[th1_adr], data.qpos[th2_adr], data.qpos[th3_adr] = q_mid
    mujoco.mj_forward(model, data)
    q_cmd = q_mid.copy()

    physics_dt = model.opt.timestep
    control_dt = 1.0 / 30.0
    nsub = max(1, round(control_dt / physics_dt))
    control_dt = nsub * physics_dt

    T_total = T_reach + T_hold + T_sweep
    n_steps = int(T_total / control_dt)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.lookat = [0.35, 0, 0.4]
    cam.distance = 1.1
    cam.azimuth = 270
    cam.elevation = -10

    frames = []
    log = {k: [] for k in
           ["t", "u_des", "u_meas", "v_meas", "Fn", "phi", "d_pen",
            "pos_err", "q1", "q2", "q3"]}

    for k in range(n_steps):
        t = k * control_dt

        # ---- profile: desired along-surface contact position u(s), plus its
        # analytic time-derivative for IK feedforward ----
        if t < T_reach + T_hold:
            u_des, u_dot = u0, 0.0
        else:
            s = np.clip((t - T_reach - T_hold) / T_sweep, 0.0, 1.0)
            in_range = 0.0 < (t - T_reach - T_hold) < T_sweep
            if profile == "sweep":
                u_des = u0 + (u1 - u0) * s
                u_dot = (u1 - u0) / T_sweep if in_range else 0.0
            elif profile == "sine":
                u_des = 0.5 * (u0 + u1) + sine_amp * np.sin(2 * np.pi * s)
                u_dot = sine_amp * 2 * np.pi / T_sweep * np.cos(2 * np.pi * s) if in_range else 0.0
            else:  # static
                u_des, u_dot = u0, 0.0

        # ---- read board tilt phi and measured contact force ----
        phi = data.qpos[phi_adr]
        Fn = get_contact_normal_force(model, data, pen_gid, board_gid)

        # ---- reach phase: ramp gently from a 3cm standoff onto the surface
        # before handing off to the force-PI loop, so first contact is soft ----
        if t < T_reach:
            ramp = np.clip(t / T_reach, 0.0, 1.0)
            d_pen = -0.03 * (1.0 - ramp)
            fpi.integral = 0.0  # keep the force loop's integrator quiescent during reach
        else:
            d_pen = fpi.step(Fn, control_dt)

        # ---- compose board FK into the world-frame IK target ----
        p_target = board.target_world_xz(phi, u_des, d_pen)
        that, _ = board.frame(phi)
        p_dot_ff = u_dot * np.array([that[0], that[2]])

        # ---- DLS/Gauss-Newton IK step (redundancy resolved in null space) ----
        p_cur = arm.fk(q_cmd)
        J = arm.jacobian(q_cmd)
        dq, e = ik.step(q_cmd, J, p_target, p_cur, p_dot_ff=p_dot_ff)
        q_cmd = q_cmd + dq * control_dt

        data.ctrl[0], data.ctrl[1], data.ctrl[2] = q_cmd

        for _ in range(nsub):
            mujoco.mj_step(model, data)

        # ---- logging (measured, from actual physics state) ----
        q_actual = np.array([data.qpos[th1_adr], data.qpos[th2_adr], data.qpos[th3_adr]])
        p_pen_actual = arm.fk(q_actual)
        u_meas, v_meas = board.project_point(data.qpos[phi_adr], p_pen_actual)

        log["t"].append(t)
        log["u_des"].append(u_des)
        log["u_meas"].append(u_meas)
        log["v_meas"].append(v_meas)
        log["Fn"].append(Fn)
        log["phi"].append(data.qpos[phi_adr])
        log["d_pen"].append(d_pen)
        log["pos_err"].append(np.linalg.norm(e))
        log["q1"].append(q_actual[0])
        log["q2"].append(q_actual[1])
        log["q3"].append(q_actual[2])

        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    for key in log:
        log[key] = np.array(log[key])

    np.savez(f"{out_prefix}_log.npz", **log)
    imageio.mimsave(f"{out_prefix}.mp4", frames, fps=30, quality=8)
    print(f"Saved {out_prefix}.mp4 ({len(frames)} frames, {len(frames)/30:.1f} s) and {out_prefix}_log.npz")
    print(f"Fn range: [{log['Fn'].min():.2f}, {log['Fn'].max():.2f}] N, "
          f"mean {log['Fn'].mean():.2f} N (target band [{Fmin},{Fmax}])")
    print(f"final tracking |u_des-u_meas|: {abs(log['u_des'][-1]-log['u_meas'][-1])*1000:.2f} mm, "
          f"max pos_err: {log['pos_err'].max()*1000:.2f} mm")
    print(f"board tilt phi range: [{np.degrees(log['phi'].min()):.1f}, {np.degrees(log['phi'].max()):.1f}] deg")
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="sweep", choices=["static", "sweep", "sine"])
    ap.add_argument("--out", default="sweep")
    args = ap.parse_args()
    run(profile=args.profile, out_prefix=args.out)
