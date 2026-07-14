"""
Resolved-rate (damped least-squares / Gauss-Newton) IK on the 2-D pen-tip
position task, with the one redundant DOF (3 joints, 1-D along-board task +
force task via penetration depth) resolved by a null-space secondary
objective. This is the "small optimization" exploiting redundancy: at every
timestep we solve

    minimize_{dq}  || J dq - Kp_task * e ||^2  +  lam^2 ||dq||^2
    then project the secondary joint-centering velocity into null(J)

which is the standard closed-form solution of a damped weighted least-
squares problem (equivalent to a single Gauss-Newton / Levenberg-Marquardt
step on the position-tracking objective), rather than a black-box IK stack.
"""
import numpy as np


class DLSController:
    def __init__(self, q_mid, kp_task=8.0, lam=0.03, kn=1.0, qdot_max=6.0):
        self.q_mid = np.asarray(q_mid, dtype=float)
        self.kp_task = kp_task
        self.lam = lam
        self.kn = kn
        self.qdot_max = qdot_max

    def step(self, q, J, p_target, p_cur, p_dot_ff=None):
        e = p_target - p_cur  # (2,)
        JT = J.T
        JJt = J @ JT
        # damped pseudo-inverse (Levenberg-Marquardt damping keeps this
        # well-conditioned through the arm's singularities)
        A = JJt + (self.lam ** 2) * np.eye(2)
        J_pinv = JT @ np.linalg.solve(A, np.eye(2))

        # feedforward velocity (e.g. du/dt along the board tangent from the
        # known profile) removes the proportional-control tracking lag that
        # a pure position-error feedback loop has on a ramping target
        ff = p_dot_ff if p_dot_ff is not None else np.zeros(2)
        dq_task = J_pinv @ (self.kp_task * e + ff)

        # secondary objective: pull joints toward mid-range, projected into
        # the task null space so it never disturbs the primary tracking
        dq_secondary = -self.kn * (q - self.q_mid)
        N = np.eye(3) - J_pinv @ J
        dq = dq_task + N @ dq_secondary

        # velocity saturation for safety/smoothness
        norm = np.linalg.norm(dq)
        if norm > self.qdot_max:
            dq *= self.qdot_max / norm
        return dq, e


class ForcePIController:
    """Regulates the commanded penetration depth d_pen so the measured
    contact normal force Fn tracks a target band [Fmin,Fmax] (we drive Fn
    toward the band center with anti-windup)."""

    def __init__(self, Fmin, Fmax, kp=0.006, ki=0.02, d_min=0.0, d_max=0.02):
        self.Fmin, self.Fmax = Fmin, Fmax
        self.Ftarget = 0.5 * (Fmin + Fmax)
        self.kp, self.ki = kp, ki
        self.d_min, self.d_max = d_min, d_max
        self.integral = 0.0
        self.d_pen = 0.0

    def step(self, Fn_meas, dt):
        # only push the integrator toward the band, not a fixed setpoint,
        # so we don't fight the controller once Fn is already inside the band
        if Fn_meas < self.Fmin:
            err = self.Fmin - Fn_meas
        elif Fn_meas > self.Fmax:
            err = self.Fmax - Fn_meas
        else:
            err = 0.0

        self.integral += err * dt
        d_pen = self.kp * err + self.ki * self.integral
        d_pen = np.clip(d_pen, self.d_min, self.d_max)
        # anti-windup: unwind integral if saturated
        if d_pen in (self.d_min, self.d_max):
            self.integral -= err * dt
        self.d_pen = d_pen
        return d_pen
