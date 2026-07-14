"""
Analytic forward kinematics / Jacobian for the planar 3R arm, and the
board (moving-frame) transform. Everything lives in the world x-z plane;
all rotations are about the world y-axis, using the standard right-handed
rotation matrix

    R_y(a) = [[ cos a, 0, sin a],
              [   0,   1,   0  ],
              [-sin a, 0, cos a]]

This is the same convention MuJoCo uses for a hinge joint with axis (0,1,0),
so these closed-form expressions can be checked directly against MuJoCo's
own xpos/xmat outputs (see verify_fk.py).
"""
import numpy as np


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0, c]])


class ArmFK:
    """3R planar arm in the x-z plane, base at p0, link lengths l1,l2,l3."""

    def __init__(self, p0, l1, l2, l3):
        self.p0 = np.asarray(p0, dtype=float)
        self.l1, self.l2, self.l3 = l1, l2, l3

    def fk(self, q):
        """Returns pen-tip world position (x,z) for joint angles q=(th1,th2,th3)."""
        th1, th2, th3 = q
        x0, z0 = self.p0[0], self.p0[2]
        a1 = th1
        a2 = th1 + th2
        a3 = th1 + th2 + th3
        x = x0 + self.l1 * np.cos(a1) + self.l2 * np.cos(a2) + self.l3 * np.cos(a3)
        z = z0 - self.l1 * np.sin(a1) - self.l2 * np.sin(a2) - self.l3 * np.sin(a3)
        return np.array([x, z])

    def jacobian(self, q):
        """2x3 Jacobian d(x,z)/d(th1,th2,th3)."""
        th1, th2, th3 = q
        l1, l2, l3 = self.l1, self.l2, self.l3
        a1 = th1
        a2 = th1 + th2
        a3 = th1 + th2 + th3

        dx1 = -l1 * np.sin(a1) - l2 * np.sin(a2) - l3 * np.sin(a3)
        dx2 = -l2 * np.sin(a2) - l3 * np.sin(a3)
        dx3 = -l3 * np.sin(a3)

        dz1 = -l1 * np.cos(a1) - l2 * np.cos(a2) - l3 * np.cos(a3)
        dz2 = -l2 * np.cos(a2) - l3 * np.cos(a3)
        dz3 = -l3 * np.cos(a3)

        return np.array([[dx1, dx2, dx3],
                          [dz1, dz2, dz3]])


class BoardFrame:
    """Board pivot fixed at p_piv; instantaneous frame set by tilt angle phi.

    At phi=0: tangent that = (0,0,1) (arc-length axis, s runs bottom->top),
              outward normal nhat = (-1,0,0) (faces the arm).
    For general phi, both rotate by R_y(phi).
    """

    def __init__(self, p_piv):
        self.p_piv = np.asarray(p_piv, dtype=float)

    def frame(self, phi):
        R = rot_y(phi)
        that = R @ np.array([0.0, 0.0, 1.0])
        nhat = R @ np.array([-1.0, 0.0, 0.0])
        return that, nhat

    def target_world_xz(self, phi, u, d_pen):
        """World (x,z) target at arc-length u along the board, pushed a
        (commanded) depth d_pen into the board along -nhat (into the surface).
        d_pen > 0 means commanding the pen past the nominal surface (this is
        the lever used to regulate contact force through the compliant
        position servo + contact constraint)."""
        that, nhat = self.frame(phi)
        p = self.p_piv + u * that - d_pen * nhat
        return np.array([p[0], p[2]])

    def project_point(self, phi, world_xz):
        """Inverse: given a world (x,z) point, return (u, v) in the board's
        instantaneous frame, where v is signed distance along +nhat (v>0 = off
        the surface / lost contact, v<0 = penetrating)."""
        that, nhat = self.frame(phi)
        p_world = np.array([world_xz[0], 0.0, world_xz[1]])
        rel = p_world - self.p_piv
        u = rel @ that
        v = rel @ nhat
        return u, v
