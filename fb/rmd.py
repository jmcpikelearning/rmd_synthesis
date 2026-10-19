"""
rmd.py — Robot Mechanism Design toolkit
ME 350R / ME 380R, The University of Texas at Austin

A small, readable library for planar closed-loop mechanisms.

Design rules for this file:
  * Everything is plain numpy. No hidden state, no clever tricks.
  * Function names match the words used in lecture.
  * Every quantity a student derives by hand has a function here with the
    same name, so the hand result can be checked against the code.

Link naming follows the lecture convention (and Norton):

        link 1 = ground   (length d)
        link 2 = crank / input   (length a)
        link 3 = coupler  (length b)
        link 4 = rocker / output (length c)

Angles are measured CCW from the +x axis. The ground link lies along +x,
so theta1 = 0 by choice of frame.

    O2 ---------- d ---------- O4        (ground, link 1)
     |                          |
     a (link 2)          c (link 4)
     |                          |
     A ------- b (link 3) ----- B
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

__all__ = [
    "FourBar", "SliderCrank", "grashof_class", "wrap",
]

TAU = 2.0 * np.pi


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def wrap(theta):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(theta) + np.pi) % TAU - np.pi


def _dedupe(angles, tol=1e-9):
    """Wrap to (-pi, pi], sort, and merge values that differ only by round-off."""
    if len(angles) == 0:
        return np.array([])
    a = np.sort(wrap(np.asarray(angles, dtype=float)))
    keep = [a[0]]
    for x in a[1:]:
        if abs(x - keep[-1]) > tol:
            keep.append(x)
    return np.array(keep)


def _acute(angle):
    """Fold an angle into [0, pi/2] — the 'acute angle between' convention."""
    a = np.abs(wrap(angle))
    return np.where(a > np.pi / 2, np.pi - a, a)


def grashof_class(a, b, c, d):
    """
    Classify a four-bar by the Grashof condition.

    s + l  <  p + q   -> Grashof (class I): some link fully rotates
    s + l  == p + q   -> change point (class III)
    s + l  >  p + q   -> non-Grashof (class II): triple rocker

    Returns a dict with the classification and the specific linkage type.
    """
    L = np.array([d, a, b, c], dtype=float)          # links 1,2,3,4
    names = ["ground", "crank", "coupler", "rocker"]
    s_i = int(np.argmin(L))
    l_i = int(np.argmax(L))
    s, l = L[s_i], L[l_i]
    pq = float(L.sum() - s - l)

    total = s + l
    if np.isclose(total, pq, rtol=1e-12, atol=1e-12):
        cls, kind = "III", "change point"
    elif total < pq:
        cls = "I"
        kind = {
            "ground":  "double crank (drag link)",
            "crank":   "crank-rocker",
            "coupler": "double rocker (Grashof)",
            "rocker":  "rocker-crank",
        }[names[s_i]]
    else:
        cls, kind = "II", "triple rocker"

    return {
        "class": cls,
        "type": kind,
        "shortest": names[s_i],
        "s_plus_l": float(total),
        "p_plus_q": pq,
        "grashof": bool(total < pq),
        # can the *input* link (link 2, length a) turn a full revolution?
        "crank_rotates": bool(total < pq and names[s_i] in ("crank", "ground")),
    }


# --------------------------------------------------------------------------
# four-bar
# --------------------------------------------------------------------------

@dataclass
class FourBar:
    """
    A planar four-bar linkage.

    a : crank  (link 2)   b : coupler (link 3)
    c : rocker (link 4)   d : ground  (link 1)

    branch = +1 (open) or -1 (crossed). The two branches are the two roots
    of the position quadratic — physically, the two ways the linkage can be
    assembled from the same four links.

    CONVENTION (9 Sep 2026): branch=+1 is the assembly drawn in *Kinematics
    Fundamentals* and returned by circuit=+1 in RMD_C4_SliderCrank and
    RMD_C5_FourBar. The two names mean the same thing. Before this date the
    sign was reversed here; any saved result that names a branch should be
    re-run rather than reinterpreted.
    """
    a: float
    b: float
    c: float
    d: float

    # ---------------- classification ----------------

    def grashof(self):
        return grashof_class(self.a, self.b, self.c, self.d)

    # ---------------- position ----------------

    def position(self, theta2, branch=+1):
        """
        Freudenstein closed-form solution.

        Given the crank angle theta2 (rad), return (theta3, theta4) in rad.
        Returns nan where the linkage cannot be assembled — a complex root
        means those link lengths cannot reach that crank angle.
        """
        a, b, c, d = self.a, self.b, self.c, self.d
        t2 = np.asarray(theta2, dtype=float)

        K1 = d / a
        K2 = d / c
        K3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c)

        A = np.cos(t2) - K1 - K2 * np.cos(t2) + K3
        B = -2.0 * np.sin(t2)
        C = K1 - (K2 + 1.0) * np.cos(t2) + K3

        disc4 = B * B - 4.0 * A * C
        bad = disc4 < 0
        root = np.sqrt(np.where(bad, np.nan, disc4))
        theta4 = 2.0 * np.arctan2(-B - branch * root, 2.0 * A)

        K4 = d / b
        K5 = (c * c - d * d - a * a - b * b) / (2.0 * a * b)

        D = np.cos(t2) - K1 + K4 * np.cos(t2) + K5
        E = -2.0 * np.sin(t2)
        F = K1 + (K4 - 1.0) * np.cos(t2) + K5

        disc3 = E * E - 4.0 * D * F
        bad3 = disc3 < 0
        root3 = np.sqrt(np.where(bad3, np.nan, disc3))
        theta3 = 2.0 * np.arctan2(-E - branch * root3, 2.0 * D)

        return wrap(theta3), wrap(theta4)

    def assemblable(self, theta2, branch=+1):
        """True where the linkage can be assembled at this crank angle."""
        t3, _ = self.position(theta2, branch)
        return ~np.isnan(t3)

    def joints(self, theta2, branch=+1):
        """
        Return the four joint locations as complex numbers:
        O2 (origin), A (crank tip), B (rocker tip), O4 (ground pivot).
        """
        t2 = np.asarray(theta2, dtype=float)
        t3, t4 = self.position(t2, branch)
        O2 = np.zeros_like(t2, dtype=complex)
        O4 = np.full_like(O2, self.d + 0j)
        A = self.a * np.exp(1j * t2)
        B = O4 + self.c * np.exp(1j * t4)
        return O2, A, B, O4

    # ---------------- coupler ----------------

    def coupler_point(self, theta2, p, delta=0.0, branch=+1):
        """
        Position of a point P fixed on the coupler.

        p     : distance from joint A to P
        delta : angle of AP measured from the coupler line AB (rad)

        Returns a complex array. Real part is x, imaginary part is y.
        """
        t2 = np.asarray(theta2, dtype=float)
        t3, _ = self.position(t2, branch)
        A = self.a * np.exp(1j * t2)
        return A + p * np.exp(1j * (t3 + delta))

    def coupler_curve(self, p, delta=0.0, branch=+1, n=361):
        """
        The coupler curve: the path traced by point P over one full crank
        revolution. Returns (theta2, P) with P complex.

        This is the central object of planar mechanism design. A Jansen
        linkage is a coupler curve design; so is a Hoeken straight line.
        """
        t2 = np.linspace(0.0, TAU, n)
        return t2, self.coupler_point(t2, p, delta, branch)

    # ---------------- force transmission ----------------

    def transmission_angle(self, theta2, branch=+1):
        """
        Transmission angle mu: the acute angle between coupler and rocker.

        This is the quality of force transfer at joint B. mu = 90 deg is
        perfect; as mu goes to 0 the linkage binds. Most machines are
        designed to keep min(mu) above about 40 degrees.
        """
        t3, t4 = self.position(theta2, branch)
        return _acute(t3 - t4)

    def min_transmission_angle(self, branch=+1, n=721):
        """Worst transmission angle over a full crank revolution (rad)."""
        t2 = np.linspace(0.0, TAU, n)
        mu = self.transmission_angle(t2, branch)
        return float(np.nanmin(mu))

    # -------- the two collinear conditions --------
    #
    # There are exactly two ways a four-bar can go collinear, and they are
    # opposite in every respect. Naming them carelessly is the single most
    # common error in this material, so both are spelled out here.
    #
    #   mA = (r_in / r_out) * (c / a) * sin(mu) / sin(nu)
    #        mu = theta4 - theta3   (angle between links 3 and 4)
    #        nu = theta2 - theta3   (angle between links 2 and 3)
    #
    #   links 2 & 3 collinear  ->  sin(nu) = 0  ->  omega4 = 0, mA -> INFINITY
    #        the ROCKER is at a travel limit.  This is Norton's *toggle*
    #        position and the condition a toggle clamp is built to sit at.
    #        -> output_limit_angles(), alias toggle_angles()
    #
    #   links 3 & 4 collinear  ->  sin(mu) = 0  ->  omega2 = 0, mA -> ZERO
    #        the CRANK is at a travel limit; for a non-Grashof linkage driven
    #        from link 2 these bound the reachable range of theta2.
    #        -> input_limit_angles(), alias dead_center_angles()

    def input_limit_angles(self):
        """
        Crank angles at which links 3 and 4 are collinear.

        The **input link is at a travel limit** — omega2 passes through zero
        while omega4 stays finite, so mechanical advantage goes to ZERO. For a
        non-Grashof linkage driven from link 2, these bound the reachable
        range of theta2.

        Do not confuse with output_limit_angles(), the opposite condition.

        Returns an array of theta2 in rad, possibly empty.
        """
        a, b, c, d = self.a, self.b, self.c, self.d
        out = []
        for sgn in (+1.0, -1.0):
            num = a * a + d * d - (b + sgn * c) ** 2
            den = 2.0 * a * d
            if den == 0:
                continue
            val = num / den
            if -1.0 <= val <= 1.0:
                t = np.arccos(val)
                out.extend([t, -t])
        return _dedupe(out)

    def output_limit_angles(self):
        """
        Crank angles at which links 2 and 3 are collinear.

        The **output link is at a travel limit** — omega4 passes through zero,
        the rocker reverses, and mechanical advantage goes to INFINITY: a small
        input torque produces an arbitrarily large output force, and the
        linkage cannot be driven backwards through the position.

        This is Norton's *toggle* position, and the condition that toggle
        clamps, locking knees and crimping tools are deliberately built to sit
        at. Do not confuse with input_limit_angles(), the opposite condition.

        The geometry: when 2 and 3 are collinear, |O2B| = a + b (extended) or
        |b - a| (folded), so theta2 follows from triangle O2-B-O4 (sides |O2B|,
        c, d) — NOT from triangle O2-A-O4. In the folded case with b > a the
        point B lies on the far side of O2, so the triangle gives theta2 + 180,
        not theta2; that half-turn is applied here. Getting this wrong produces
        spurious roots at which omega4 is not zero.

        Both methods are branch-agnostic geometry: a returned angle is realised
        on ONE branch, so check omega4 with velocity(theta2, 1.0, branch)
        before quoting it for a particular assembly.

        Returns an array of theta2 in rad, possibly empty.
        """
        a, b, c, d = self.a, self.b, self.c, self.d
        out = []
        # (|O2B|, half-turn to apply because B is on the far side of O2)
        cases = [(a + b, 0.0), (abs(b - a), np.pi if b > a else 0.0)]
        for L, flip in cases:
            den = 2.0 * d * L
            if den == 0:
                continue
            val = (d * d + L * L - c * c) / den
            if -1.0 <= val <= 1.0:
                t = np.arccos(val)
                out.extend([t + flip, -t + flip])
        return _dedupe(out)

    # Norton-consistent aliases. toggle_angles() now returns the mA -> INFINITY
    # set (links 2 & 3 collinear). In toolkit v1 it returned the opposite set;
    # the v1 naming was wrong and was corrected on 17 Aug 2026.
    def toggle_angles(self):
        """Alias for output_limit_angles() — Norton's toggle, mA -> infinity."""
        return self.output_limit_angles()

    def dead_center_angles(self):
        """Alias for input_limit_angles() — crank dead points, mA -> zero."""
        return self.input_limit_angles()

    # ---------------- velocity ----------------

    def velocity(self, theta2, omega2=1.0, branch=+1):
        """
        Angular velocities of coupler and rocker, from differentiating the
        vector loop. Returns (omega3, omega4) in rad/s.
        """
        a, b, c = self.a, self.b, self.c
        t2 = np.asarray(theta2, dtype=float)
        t3, t4 = self.position(t2, branch)
        w3 = (a * omega2 / b) * np.sin(t4 - t2) / np.sin(t3 - t4)
        w4 = (a * omega2 / c) * np.sin(t2 - t3) / np.sin(t4 - t3)
        return w3, w4

    def joint_velocities(self, theta2, omega2=1.0, branch=+1):
        """Velocities of A and B as complex numbers (V = j*omega*R)."""
        t2 = np.asarray(theta2, dtype=float)
        t3, t4 = self.position(t2, branch)
        w3, w4 = self.velocity(t2, omega2, branch)
        V_A = 1j * omega2 * self.a * np.exp(1j * t2)
        V_BA = 1j * w3 * self.b * np.exp(1j * t3)
        V_B = 1j * w4 * self.c * np.exp(1j * t4)
        return V_A, V_BA, V_B

    # ---------------- acceleration ----------------

    def acceleration(self, theta2, omega2=1.0, alpha2=0.0, branch=+1):
        """
        Angular accelerations of coupler and rocker. Returns (alpha3, alpha4).

        Uses the A-F coefficient form derived in lecture:
            alpha3 = (C*D - A*F) / (A*E - B*D)
            alpha4 = (C*E - B*F) / (A*E - B*D)
        """
        a, b, c = self.a, self.b, self.c
        t2 = np.asarray(theta2, dtype=float)
        t3, t4 = self.position(t2, branch)
        w3, w4 = self.velocity(t2, omega2, branch)

        A = c * np.sin(t4)
        B = b * np.sin(t3)
        C = (a * alpha2 * np.sin(t2) + a * omega2**2 * np.cos(t2)
             + b * w3**2 * np.cos(t3) - c * w4**2 * np.cos(t4))
        D = c * np.cos(t4)
        E = b * np.cos(t3)
        F = (a * alpha2 * np.cos(t2) - a * omega2**2 * np.sin(t2)
             - b * w3**2 * np.sin(t3) + c * w4**2 * np.sin(t4))

        den = A * E - B * D
        a3 = (C * D - A * F) / den
        a4 = (C * E - B * F) / den
        return a3, a4

    def joint_accelerations(self, theta2, omega2=1.0, alpha2=0.0, branch=+1):
        """Accelerations of A, B relative to A, and B, as complex numbers."""
        t2 = np.asarray(theta2, dtype=float)
        t3, t4 = self.position(t2, branch)
        w3, w4 = self.velocity(t2, omega2, branch)
        a3, a4 = self.acceleration(t2, omega2, alpha2, branch)
        A_A = (1j * alpha2 - omega2**2) * self.a * np.exp(1j * t2)
        A_BA = (1j * a3 - w3**2) * self.b * np.exp(1j * t3)
        A_B = (1j * a4 - w4**2) * self.c * np.exp(1j * t4)
        return A_A, A_BA, A_B

    # ---------------- instant centers and mechanical advantage ----------------

    def instant_center_24(self, theta2, branch=+1):
        """
        Location of I(2,4) on the ground line, as a complex number.

        By Kennedy's rule I(2,4) lies on the line through I(1,2) and I(1,4)
        — that is, the ground line — and also on the line through I(2,3) and
        I(3,4), which is the coupler line AB extended. So it is the
        intersection of the ground line with line AB.

        Where you place this point is what sets your speed ratio, and
        therefore your mechanical advantage.
        """
        _, A, B, _ = self.joints(theta2, branch)
        Ax, Ay = np.real(A), np.imag(A)
        Bx, By = np.real(B), np.imag(B)
        dx, dy = Bx - Ax, By - Ay
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(dy) < 1e-12, np.nan, -Ay / dy)
        return (Ax + t * dx) + 0j

    def mechanical_advantage(self, theta2, r_in=None, r_out=None, branch=+1):
        """
        Force amplification, F_out / F_in.

            mA = |omega_in / omega_out| * (r_in / r_out)

        r_in and r_out are the moment arms at which the input and output
        forces act. They default to the crank and rocker lengths, which is
        the usual textbook case (forces perpendicular to those links).
        """
        r_in = self.a if r_in is None else r_in
        r_out = self.c if r_out is None else r_out
        _, w4 = self.velocity(theta2, 1.0, branch)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(1.0 / w4) * (r_in / r_out)

    # ---------------- static force ----------------

    def input_torque(self, theta2, F_out, r_out=None, branch=+1):
        """
        Input torque required at the crank to produce output force F_out
        applied at moment arm r_out on link 4. Ideal (lossless) case.

            T_in = F_out * r_out * |omega4 / omega2|

        This is the number that sizes your motor. Sweep it over the cycle
        and look at the peak, not the average.
        """
        r_out = self.c if r_out is None else r_out
        _, w4 = self.velocity(theta2, 1.0, branch)
        return np.abs(w4) * F_out * r_out


# --------------------------------------------------------------------------
# slider-crank  (the Build Assignment mechanism)
# --------------------------------------------------------------------------

@dataclass
class SliderCrank:
    """
    Offset slider-crank, in the convention of *Kinematics Fundamentals* Part 4.2
    and Quiz 1 — walk the loop O2 -> A -> B -> O2:

        F1 = a cos(theta2) + b cos(theta3) - x = 0
        F2 = a sin(theta2) + b sin(theta3) - e = 0

    so theta3 is the coupler angle measured from A to B (negative when the crank
    is above the slide), and x is the slider position from O2 along +x, positive
    away from the crank. At a = 40, b = 120, theta2 = 60 deg this gives
    theta3 = -16.7787 deg and x = 134.8913 mm, which is what the notes and the
    quiz key say.

    a : crank      b : coupler      offset : e, perpendicular offset of the slide

    circuit = +1 takes the arcsin root — the assembly drawn in the notes.
    circuit = -1 is the mirror assembly. `branch` is accepted as a synonym so
    older calls keep working.

    CONVENTION (9 Sep 2026): before this date this class reported theta3 along
    B -> A and the slider position with the opposite sign. Any saved number from
    the old version should be re-run, not reinterpreted.
    """
    a: float
    b: float
    offset: float = 0.0

    def position(self, theta2, circuit=+1, *, branch=None):
        """Return (theta3, x) — coupler angle in rad, slider position in mm."""
        if branch is not None:
            circuit = branch
        t2 = np.asarray(theta2, dtype=float)
        s = (self.offset - self.a * np.sin(t2)) / self.b
        s = np.where(np.abs(s) > 1.0, np.nan, s)
        t3 = np.arcsin(s) if circuit > 0 else np.pi - np.arcsin(s)
        x = self.a * np.cos(t2) + self.b * np.cos(t3)
        return wrap(t3), x

    def stroke(self, n=721):
        """Peak-to-peak slider travel over one crank revolution."""
        t2 = np.linspace(0.0, TAU, n)
        _, x = self.position(t2)
        return float(np.nanmax(x) - np.nanmin(x))

    def transmission_angle(self, theta2, circuit=+1, *, branch=None):
        """Acute angle between the coupler and the slide direction."""
        t3, _ = self.position(theta2, circuit, branch=branch)
        return _acute(t3 - np.pi / 2)

    def velocity(self, theta2, omega2=1.0, circuit=+1, *, branch=None):
        """Return (omega3, x_dot). Differentiates F1, F2 above."""
        t2 = np.asarray(theta2, dtype=float)
        t3, _ = self.position(t2, circuit, branch=branch)
        w3 = -(self.a / self.b) * omega2 * np.cos(t2) / np.cos(t3)
        xd = -self.a * omega2 * np.sin(t2) - self.b * w3 * np.sin(t3)
        return w3, xd
