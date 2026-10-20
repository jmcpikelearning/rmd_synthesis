"""Verify a Jansen leg from jansen_opt_torch.py independently, in NumPy at every 0.1 deg (SF 4.6, Step 8).
usage:  python verify_jansen.py [best_jansen.npz | best_jansen.pt]     (default best_jansen.npz)
        python verify_jansen.py --holy                                 (Jansen's own numbers, placed on T)
Only the task constants come from jansen_opt_torch; the kinematics here are written again from scratch:
arccos circle intersections with NO clamp (NaN = the leg cannot assemble there), transmission angles
measured from the joint positions, link lengths re-measured."""
import sys, numpy as np
from jansen_opt_torch import T, SIGMA, N_ST, NAMES, MU_REQ, D2R, MOTION, EQ_QMIN, TRI_MIN

T_C = T.numpy() @ [1, 1j]
T_AREA = 0.5 * np.sum(T_C.real * np.roll(T_C.imag, -1) - np.roll(T_C.real, -1) * T_C.imag)   # in visiting order
JOINTS = "O M Z P1 P2 P3 P4 F".split()
LINKS = [("O", "M", "m"), ("M", "P1", "j"), ("Z", "P1", "b"), ("Z", "P2", "d"), ("P1", "P2", "e"),
         ("M", "P3", "k"), ("Z", "P3", "c"), ("P2", "P4", "f"), ("P3", "P4", "g"), ("P3", "F", "i"),
         ("P4", "F", "h")]
DYADS = [("P1", "M", "Z", "j", "b"), ("P3", "M", "Z", "k", "c"), ("P4", "P2", "P3", "f", "g")]
TRIS = {"bde": ("Z", "P1", "P2"), "ghi": ("P3", "P4", "F")}     # the rigid plates, by their corners

def triangle_np(pose, A, B, C):
    """Shape of the plate A-B-C measured from the joints at the first pose: sides [mm], smallest
    interior angle [deg] and quality q = 4 sqrt(3) Area / sum(side^2) (Area from the cross product)."""
    a, b, c = pose[A][0], pose[B][0], pose[C][0]
    sides = np.abs([b - a, c - b, a - c])
    area = 0.5 * abs(((b - a).conjugate() * (c - a)).imag)
    ang = [abs(np.angle((q - p) / (r - p))) for p, q, r in ((a, b, c), (b, c, a), (c, a, b))]
    return sides, np.degrees(min(ang)), 4 * np.sqrt(3) * area / np.sum(sides ** 2)

def dyad_np(P, Q, lp, lq, sigma):
    """Point X with |XP| = lp, |XQ| = lq (complex numbers); NaN where the circles do not meet."""
    u = Q - P
    with np.errstate(invalid="ignore"):
        gam = np.arccos((lp ** 2 + abs(u) ** 2 - lq ** 2) / (2 * lp * abs(u)))
    return P + lp * np.exp(1j * (np.angle(u) + sigma * gam))

def travel_np(v):
    """Crank travel at the 14 targets: stance k * dth_s, swing gaps by softmax (first logit 0)."""
    ds = v[16]
    q = np.r_[0.0, v[17:]]
    w = np.exp(q - q.max()); w /= w.sum()
    gaps = (2 * np.pi - (N_ST - 1) * ds) * w
    return np.r_[ds * np.arange(N_ST), (N_ST - 1) * ds + np.cumsum(gaps[:-1])]

def pose_np(v, dth, hand):
    """All joints (complex, body frame) at crank travel dth; theta2 = theta0 + hand * MOTION * dth."""
    m, ax, ay, b, c, d, e, f, g, h, i, j, k = v[:13]
    t2 = v[13] + hand * MOTION * np.atleast_1d(np.asarray(dth, float))
    M = m * np.exp(1j * t2)
    Z = complex(-ax, -ay) + 0 * M
    P1 = dyad_np(M, Z, j, b, SIGMA[0])
    P2 = dyad_np(Z, P1, d, e, SIGMA[1])
    P3 = dyad_np(M, Z, k, c, SIGMA[2])
    P4 = dyad_np(P2, P3, f, g, SIGMA[3])
    F = dyad_np(P3, P4, i, h, SIGMA[4])
    H = complex(v[14], v[15])
    body = lambda X: H + hand * X.real + 1j * X.imag
    return dict(zip(JOINTS, [body(X) for X in (0 * M, M, Z, P1, P2, P3, P4, F)]))

def mu_np(pose, X, P, Q):
    """Acute angle at joint X between rods X-P and X-Q, measured from positions [rad]."""
    a = np.abs(np.angle((pose[P] - pose[X]) / (pose[Q] - pose[X])))
    return np.minimum(a, np.pi - a)

def lengths(v):
    return dict(zip(NAMES[:13], v[:13]))

def report(v, hand, step=0.1, verbose=True, cfg=None):
    """cfg: the task/cost numbers the leg was made with (saved by jansen_opt_torch.save); the rules are
    checked against its mu_req, eq_qmin and tri_min, else against the defaults."""
    cfg = cfg or {}
    mu_req, eq_qmin, tri_min = cfg.get("mu_req", MU_REQ / D2R), cfg.get("eq_qmin", EQ_QMIN), cfg.get("tri_min", TRI_MIN)
    v = np.asarray(v, float)
    L = lengths(v)
    dth = np.radians(np.arange(0, 360, step))
    P = pose_np(v, dth, hand)
    out = {}
    out["assembles"] = not any(np.isnan(P[k]).any() for k in JOINTS)
    out["closure"] = max(np.nanmax(np.abs(abs(P[a] - P[b]) - L[n])) for a, b, n in LINKS)
    out["mu_min"] = {X: np.degrees(np.nanmin(mu_np(P, X, A, B))) for X, A, B, _, _ in DYADS}
    out["toggle_margin"] = {X: np.nanmin(np.minimum(L[lp] + L[lq] - abs(P[A] - P[B]),
                                                    abs(P[A] - P[B]) - abs(L[lp] - L[lq])))
                            for X, A, B, lp, lq in DYADS}
    tk = travel_np(v)
    Ck = pose_np(v, tk, hand)["F"]
    miss = abs(Ck - T_C)
    out["miss"] = miss
    out["rms_all"], out["rms_stance"], out["worst"] = (np.sqrt(np.mean(miss ** 2)),
                                                       np.sqrt(np.mean(miss[:N_ST] ** 2)), miss.max())
    st = pose_np(v, np.radians(np.arange(0, np.degrees(tk[N_ST - 1]) + 1e-9, step)), hand)["F"]
    out["stance_Y"] = (st.imag.min(), st.imag.max())
    F = P["F"]
    out["area"] = 0.5 * np.nansum(F.real * np.roll(F.imag, -1) - np.roll(F.real, -1) * F.imag)
    out["lowest_Y"] = np.nanmin(F.imag)
    out["triangles"] = {nm: triangle_np(P, *XYZ) for nm, XYZ in TRIS.items()}
    if verbose:
        print(f"hand {hand:+d} | dth_s {np.degrees(v[16]):.1f} deg -> stance over {np.degrees(tk[N_ST - 1]):.1f} deg of crank")
        print("assembles over the whole turn:", out["assembles"], f"| link-length error {out['closure']:.1e} mm")
        print("mu_min [deg]:", {k: round(x, 2) for k, x in out["mu_min"].items()}, f"(required {mu_req:g})")
        print("toggle margin [mm]:", {k: round(x, 2) for k, x in out["toggle_margin"].items()})
        print(f"miss rms all / stance / worst [mm]: {out['rms_all']:.3f} / {out['rms_stance']:.3f} / {out['worst']:.3f}")
        print("miss per target [mm]:", np.round(miss, 2))
        print(f"stance foot Y range [mm]: {out['stance_Y'][0]:.2f} .. {out['stance_Y'][1]:.2f}")
        print(f"lowest foot Y over the turn [mm]: {out['lowest_Y']:.3f} (must be >= 0: "
              f"{'ok' if out['lowest_Y'] >= -0.01 else 'BELOW THE FLOOR'})")
        print(f"foot loop signed area {out['area']:.0f} mm^2 (T: {T_AREA:+.0f}, same sign = same direction)")
        for nm, (sides, amin, q) in out["triangles"].items():
            flags = ([f"side < {tri_min:g} mm"] if sides.min() < tri_min - 0.01 else []) + \
                    ([f"q < {eq_qmin:g}"] if q < eq_qmin - 1e-3 else [])
            print(f"triangle {nm}: sides {np.round(sides, 1)} mm | smallest angle {amin:.1f} deg | "
                  f"quality q {q:.3f} (1 = equilateral) | {', '.join(flags) or 'ok'}")
    return out

def load(path):
    """(v, hand, cfg) from best_jansen.npz or best_jansen.pt; cfg is None for files saved before it was."""
    if path.endswith(".npz"):
        import json
        d = np.load(path)
        return d["v"], int(d["hand"]), json.loads(str(d["cfg"])) if "cfg" in d else None
    import torch
    d = torch.load(path, weights_only=False)
    return np.asarray(d["v"]), int(d["hand"]), d.get("cfg")

if __name__ == "__main__":
    if "--holy" in sys.argv:
        import jansen_opt_torch as JO
        for hand in (+1, -1):
            JO.set_hand(hand)
            print("Jansen's holy numbers, placed on T:")
            report(JO.holy_v().numpy(), hand)
    else:
        v, hand, cfg = load(sys.argv[1] if len(sys.argv) > 1 else "best_jansen.npz")
        report(v, hand, cfg=cfg)
