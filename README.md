# Planar 3R Arm Tracing on a Passively-Tilting, Spring-Loaded Board

MuJoCo simulation of a 3-link planar arm whose pen tip traces a path along
a board that is itself mounted on a spring-loaded hinge and tilts away
passively under contact pressure. The controller must fold the board's own
(state-dependent) forward kinematics into the arm's inverse-kinematics
target on every timestep, while simultaneously regulating contact force
through the arm's redundant degree of freedom.

## Files

- `chopstick_crane.xml` — MJCF model: 3R arm (revolute joints `theta1/2/3`,
  axis y, planar in x-z) + a hinge-mounted board (`phi`, spring `stiffness`
  + `damping`, i.e. the spring-loaded joint). Collisions are disabled by
  default and re-enabled only between the pen-tip sphere and the board
  face, so the only physical interaction modelled is the pen/board contact.
  The board subsystem uses `gravcomp="1"` so its tilt is driven purely by
  contact torque vs. the spring/damper (see report, Assumptions).
- `kinematics.py` — closed-form planar FK/Jacobian for the arm, and the
  board's moving-frame transform (`BoardFrame`), both validated against
  MuJoCo's own kinematics in `verify_fk.py` (max error ~1e-10).
- `controller.py` — `DLSController` (damped least-squares / Gauss-Newton
  resolved-rate IK with a null-space secondary objective — the "small
  optimization" that exploits the arm's redundant DOF every timestep) and
  `ForcePIController` (regulates commanded penetration depth to keep the
  measured contact normal force in a target band).
- `sim.py` — main loop: profile generation (`static` / `sweep` / `sine`),
  composes the board's FK into the IK target each step, runs the DLS-IK +
  force-PI loop, steps MuJoCo, logs data, and renders the video.
- `make_plots.py` — tracking-error / force / tilt plots from the logged
  `.npz` data.
- `verify_fk.py` — unit check of the analytic kinematics against MuJoCo.

## Running

```bash
pip install mujoco numpy matplotlib imageio imageio-ffmpeg
python3 verify_fk.py            # sanity-check analytic FK/Jacobian
python3 sim.py --profile sweep --out sweep   # run + render (~35 s sim)
python3 make_plots.py           # figures from sweep_log.npz
```

Other profiles: `--profile static` (hold one point, only force is
regulated) and `--profile sine` (oscillate back and forth along the
board). Rendering: on Linux, `sim.py`/`verify_fk.py` auto-select the
headless `osmesa` software renderer; on Windows/macOS they use MuJoCo's
default backend, which works out of the box in a normal desktop Python
install (no extra setup needed) — this is handled automatically, no
edits required.

### Windows + VS Code

1. Install Python from python.org (check "Add python.exe to PATH").
2. Extract this zip, open the folder in VS Code (File → Open Folder).
3. Open a terminal (Terminal → New Terminal) and run:
   ```powershell
   pip install mujoco numpy matplotlib imageio imageio-ffmpeg
   python verify_fk.py
   python sim.py --profile sweep --out sweep
   python make_plots.py
   ```
   (use `py` instead of `python` if that's what your install registered).

## Method summary

At every control step (30 Hz):
1. Read the board's tilt `phi` from `qpos` and the measured contact normal
   force `Fn` (summed from `mj_contactForce` over pen/board contacts).
2. `ForcePIController` turns the `Fn` error (vs. target band) into a
   commanded penetration depth `d_pen`.
3. `BoardFrame.target_world_xz(phi, u_des, d_pen)` composes the board's
   *current* rigid-body transform with the desired along-surface position
   `u_des(t)` (from the chosen profile) and `d_pen`, giving the world-frame
   IK target — this is the "compose the board's FK into the arm's IK
   target" step, done fresh every timestep since `phi` is a state variable.
4. `DLSController.step` solves a damped-least-squares Gauss-Newton update
   for the 3 joint velocities: the primary task is the 2-D pen-tip
   position error (plus a feedforward `u_dot` term along the tangent to
   remove ramp-tracking lag); the left-over null-space DOF is used to pull
   the joints toward mid-range (secondary objective) — this is solved as a
   small closed-form QP every timestep.
5. The resulting joint targets are sent to position actuators, MuJoCo
   steps the physics (arm dynamics + board spring/damper + contact), and
   everything is logged for the plots/video.
