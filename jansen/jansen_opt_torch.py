"""
Jansen-linkage walking-leg synthesis with PyTorch + SciPy

Same recipe as leg_opt_torch.py (SF 4.3: residual vector r(v), cost J = r^T r, Levenberg-Marquardt
with exact autograd Jacobians, all starts batched), now for Theo Jansen's leg (KF 11.3, SF 4.9):
8 links, 11 rods, 5 chained circle intersections. 27 numbers instead of 10 means many more valleys
(SF 4.6), so the LM starts come from a global stage first:
  1. Latin hypercube sample of the design box (scipy.stats.qmc), scored in one batch,
  2. differential evolution (scipy.optimize, a GA-family search) started from the best LHS points,
  3. distinct basins picked from the final DE populations -> batched LM, then the best of them
     finished with SciPy least_squares in physical units (exact bounds; LM in z crawls at a bound),
  4. exact Hessian at each minimum (SF 4.4-4.5): strict minimum, flat valley, or saddle.

Design vector v (27), mm and rad:
  m ax ay b c d e f g h i j k   Jansen's letters: crank m, frame pivot Z = (-ax, -ay), 10 rods
  theta0 hX hY                   crank angle at target 0, crank axle O in the body frame
  dth_s q1..q10                  timing: stance crank step, swing-gap softmax logits
Body frame = the frame of the targets T (jansen_opt.py): floor at Y = 0. Rules: transmission angle
>= 40 deg at every dyad, and the foot never below the floor, both over the whole crank turn; the
crank axle O no further forward than the largest target X, and both ground pivots O and Z at least
20 mm above the tallest target.
"""
import math, os, time, argparse
import numpy as np
import torch
from torch.func import vmap, jacrev, jacfwd, hessian
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc, spearmanr

torch.set_default_dtype(torch.float64)          # LM needs float64 (see leg_opt_torch.py)
D2R = math.pi / 180
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ task (hybrid timing)
T_LOOP = torch.tensor([                         # the group's foot loop, copied from jansen_opt.py
    (0, 0), (10, 0), (20, 0), (30, 0),          # floor points
    (40, 2), (45, 5), (50, 10), (55, 20), (55, 30),
    (44, 33.5), (30, 25), (23.5, 17), (15, 10), (5, 5),   # as listed: counter-clockwise
], dtype=torch.float64)
MOTION = -1                                      # -1: the foot goes round clockwise, stance sweeps X 30 -> 0;
                                                 # +1: as listed, counter-clockwise, stance sweeps X 0 -> 30
T = T_LOOP if MOTION > 0 else T_LOOP[[3, 2, 1, 0] + list(range(13, 3, -1))]   # targets in the order visited
N_ST = 4                                         # targets 0..3 at crank travel k * dth_s (constant ground speed)
N_GAP = len(T) - N_ST + 1                        # 11 gaps share the rest of the turn (softmax, first logit = 0)
MU_REQ = 40 * D2R
t_turn = torch.arange(0, 360, 2) * D2R           # full turn for mu_min and assembly, every 2 deg (absolute
                                                 # crank angles: tied to theta0, LM would tune the sampling phase)
TAU_MU, TAU_MM = 0.25 * D2R, 0.25                # soft-min / soft-max temperature over the turn: smooth, so the
                                                 # Hessian exists; conservative by at most tau*ln(180) ~ 1.3 deg
EPS = 1e-12
W_REJ, MARGIN = 10.0, 1.0                        # per mm of violation; keep 1 mm inside the rule
W_GROUND, G_MARGIN = 30.0, 0.05                  # foot below Y = G_MARGIN, per mm (RMS over the turn): a squared
                                                 # penalty always lets a little through, so aim 0.05 mm up
X_BACK = float(T[:, 0].max())                    # crank axle O behind the path: O.x <= the largest target X (55)
PIVOT_CLEAR = 20.0                               # both ground pivots O and Z this far above the tallest target:
Y_PIVOT = float(T[:, 1].max()) + PIVOT_CLEAR     # Y >= 53.5. O is (hX, hY), so its two rules are box bounds;
                                                 # Z.y = hY - ay is a residual (Z_up)

# ------------------------------------------------------------------ mechanism
HOLY = dict(m=15.0, ax=38.0, ay=7.8, b=41.5, c=39.3, d=40.1, e=55.8,     # Jansen's "holy numbers"
            f=39.4, g=36.7, h=65.7, i=49.0, j=50.0, k=61.9)
SIGMA = (-1, +1, +1, -1, +1)     # branches of P1, P2, P3, P4, F: the only signs giving the classic leg
                                 # (P1 above Z, P2 left of Z, P3 below Z, P4 left of P3, foot lowest)
HAND = +1                        # +1: frame pivot Z left of the crank (Jansen's drawing); -1: mirrored
DIR = HAND * MOTION              # crank direction, theta2 = theta0 + DIR * dth

def set_hand(hand):
    """The holy leg traces its foot loop CCW for a CCW crank (signed area +876 mm^2); T_LOOP is listed
    CCW (+958 mm^2) and visited in the MOTION sense. Mirroring flips the loop, so DIR = HAND * MOTION."""
    global HAND, DIR
    HAND, DIR = int(hand), int(hand) * MOTION

# ------------------------------------------------------------------ bounds -> unconstrained z
NAMES = "m ax ay b c d e f g h i j k theta0 hX hY dth_s".split() + [f"q{n}" for n in range(1, N_GAP)]
#                  m   ax   ay    b..k (10)   hX   hY   dth_s       (theta0, q1..q10 are free)
LO = torch.tensor([5., 10., -30.] + [10.] * 10 + [-60., max(30., Y_PIVOT), 10 * D2R])
HI = torch.tensor([30., 80., 30.] + [120.] * 10 + [min(120., X_BACK), 200., 70 * D2R])
Q_MAX = 3.0                                       # DE box for the gap logits (LM may leave it)
DE_LO = np.r_[LO[:13].numpy(), -math.pi, LO[13:].numpy(), [-Q_MAX] * (N_GAP - 1)]
DE_HI = np.r_[HI[:13].numpy(), math.pi, HI[13:].numpy(), [Q_MAX] * (N_GAP - 1)]
NV = len(NAMES)                                   # 27

def unpack(z):
    """z (27,) unconstrained -> v. Box limits via a sigmoid, so LM never leaves them."""
    lo, hi = LO.to(z), HI.to(z)
    s = lo + (hi - lo) * torch.sigmoid(torch.cat([z[..., :13], z[..., 14:17]], -1))
    return torch.cat([s[..., :13], z[..., 13:14], s[..., 13:], z[..., 17:]], -1)

def to_z(v):
    """v (..., 27) -> z, inverse of unpack (for seeding from DE or a known leg)."""
    v = torch.as_tensor(v, dtype=torch.float64)
    lo, hi = LO.to(v), HI.to(v)
    u = ((torch.cat([v[..., :13], v[..., 14:17]], -1) - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
    s = torch.log(u / (1 - u))
    return torch.cat([s[..., :13], v[..., 13:14], s[..., 13:], v[..., 17:]], -1)

# ------------------------------------------------------------------ timing
def travel(v):
    """Crank travel at each target (14,). Stance k * dth_s; the 11 swing gaps split the rest of the
    turn by softmax, so the order is kept and the loop closes by construction. The first logit is
    pinned at 0 (softmax is shift-invariant: a free one would be an exact zero of the Hessian)."""
    ds = v[16]
    q = torch.cat([v.new_zeros(1), v[17:]])
    gaps = (2 * math.pi - (N_ST - 1) * ds) * torch.softmax(q, 0)
    stance = ds * torch.arange(N_ST).to(v)
    swing = (N_ST - 1) * ds + torch.cumsum(gaps[:-1], 0)
    return torch.cat([stance, swing])

# ------------------------------------------------------------------ kinematics
def dyad(P, Q, lp, lq, sigma):
    """
    Point X with |XP| = lp, |XQ| = lq: coupler_angle of leg_opt_torch.py made general.
        phi   = direction of P->Q,  s = |PQ|
        gamma = angle at P = atan2(4*Area, lp^2 + s^2 - lq^2)
        16*Area^2 = 4 lp^2 s^2 - (lp^2 + s^2 - lq^2)^2   (>= 0 iff the dyad closes)
    Clamped so the gradient is finite everywhere; assembly is policed by the residuals.
    """
    ux, uy = Q[..., 0] - P[..., 0], Q[..., 1] - P[..., 1]
    s2 = ux * ux + uy * uy
    k = lp * lp + s2 - lq * lq
    root = torch.sqrt(torch.clamp(4 * lp * lp * s2 - k * k, min=EPS))
    ang = torch.atan2(uy, ux) + sigma * torch.atan2(root, k)
    return torch.stack([P[..., 0] + lp * torch.cos(ang), P[..., 1] + lp * torch.sin(ang)], -1)

def joints(v, t2):
    """Joints at crank angles t2 (n,) in the mechanism frame (crank axle O at the origin)."""
    m, ax, ay, b, c, d, e, f, g, h, i, j, k = v[:13]
    M = torch.stack([m * torch.cos(t2), m * torch.sin(t2)], -1)
    Z = torch.stack([-ax, -ay]).expand_as(M)
    P1 = dyad(M, Z, j, b, SIGMA[0])                  # crank rod j, upper rocker b
    P2 = dyad(Z, P1, d, e, SIGMA[1])                 # rigid triangle b-d-e
    P3 = dyad(M, Z, k, c, SIGMA[2])                  # crank rod k, lower rocker c
    P4 = dyad(P2, P3, f, g, SIGMA[3])                # rod f, foot-triangle side g
    F = dyad(P3, P4, i, h, SIGMA[4])                 # rigid foot triangle g-h-i
    return M, Z, P1, P2, P3, P4, F

def to_body(v, X):
    """Mechanism frame -> body frame: mirror by HAND, then shift to the crank axle (hX, hY)."""
    return torch.stack([v[14] + HAND * X[..., 0], v[15] + X[..., 1]], -1)

def foot(v, dth):
    """Foot in the body frame at crank travel dth (n,), theta2 = theta0 + DIR * dth."""
    return to_body(v, joints(v, v[13] + DIR * dth)[-1])

def mu_acute(lp, lq, s2):
    """Transmission angle (acute) at a dyad's joint, from the law of cosines; branch-free."""
    mm = lp * lp + lq * lq - s2
    root = torch.sqrt(torch.clamp(4 * lp * lp * lq * lq - mm * mm, min=EPS))
    return torch.atan2(root, mm.abs())

# ------------------------------------------------------------------ residuals (SF 4.3) + sloped reject
RES_NAMES = ([f"miss{n}{xy}" for n in range(len(T)) for xy in "XY"]
             + ["mu_P1", "mu_P3", "mu_P4", "asm_P1", "asm_P3", "asm_P4", "tri_bde", "tri_ghi", "Z_up"]
             + [f"ground{int(round(float(t) / D2R))}" for t in t_turn])

def residuals_v(v):
    """r (217,) at the physical design v (27,). Works in any dtype/device (constants follow v)."""
    m, ax, ay, b, c, d, e, f, g, h, i, j, k = v[:13]
    n = len(T)
    t2 = torch.cat([v[13] + DIR * travel(v), t_turn.to(v)])
    M, Z, P1, P2, P3, P4, F = joints(v, t2)
    miss = (to_body(v, F[:n]) - T.to(v)) / math.sqrt(n)                   # 28
    mus, asms = [], []
    for P, Q, lp, lq in ((M, Z, j, b), (M, Z, k, c), (P2, P3, f, g)):      # the three moving dyads
        s2 = ((Q[n:] - P[n:]) ** 2).sum(-1)
        s = torch.sqrt(s2 + EPS)
        mu_min = -TAU_MU * torch.logsumexp(-mu_acute(lp, lq, s2) / TAU_MU, 0)      # soft min over the turn
        mus.append(2 * torch.relu(MU_REQ - mu_min) / D2R)                          # 2 per degree
        gap = torch.maximum(s - (lp + lq), torch.sqrt((lp - lq) ** 2 + EPS) - s)  # > 0: cannot close
        gap_max = TAU_MM * torch.logsumexp(gap / TAU_MM, 0)                        # soft max over the turn
        asms.append(W_REJ * torch.relu(gap_max + MARGIN))                          # slope off the plateau
    tris = []
    for sides in (torch.stack([b, d, e]), torch.stack([g, h, i])):                 # rigid triangles
        l = sides.max()
        tris.append(W_REJ * torch.relu(l - (sides.sum() - l) + MARGIN))
    z_up = W_REJ * torch.relu(Y_PIVOT + MARGIN - (v[15] - ay))                   # frame pivot Z high enough
    # foot must stay above the floor all the way round (C13: dropping clearance put the foot 27 mm
    # through the floor between checks). One hinge per crank sample, like the four-bar's clearance
    # residuals: sum of squares = W^2 * mean squared depth, and piecewise linear, which suits Gauss-Newton
    ground = W_GROUND * torch.relu(G_MARGIN - (v[15] + F[n:, 1])) / math.sqrt(len(t_turn))   # 180
    return torch.cat([miss.reshape(-1), torch.stack(mus + asms + tris + [z_up]), ground])

def residuals(z):
    return residuals_v(unpack(z))

def J_v(v):
    r = residuals_v(v)
    return (r * r).sum()

r_batch = vmap(residuals)
A_batch = vmap(jacfwd(residuals))                # exact 217x27 Jacobian per start (forward mode:
                                                 # 27 inputs, 217 outputs)
J_v_batch = vmap(J_v)

def score(V, device=DEVICE, dtype=torch.float64, chunk=4096):
    """J for a batch of physical designs V (S, 27) numpy -> (S,) numpy. Used by LHS and DE."""
    V = torch.as_tensor(np.asarray(V), dtype=dtype, device=device)
    out = [J_v_batch(V[s:s + chunk]) for s in range(0, len(V), chunk)]
    J = torch.cat(out).double().cpu().numpy()
    return np.nan_to_num(J, nan=1e12, posinf=1e12)

# ------------------------------------------------------------------ batched Levenberg-Marquardt (SF 4.5)
def lm(z, r_fn=None, iters=300, lam=1e-2, tol=1e-10, trace=None):
    """leg_opt_torch.lm made general: any residual function r_fn(z) (default: residuals), and an
    optional list `trace` that collects (z, J) after every step for the C13-style animation."""
    rb = r_batch if r_fn is None else vmap(r_fn)
    Ab = A_batch if r_fn is None else vmap(jacfwd(r_fn))
    N, n = z.shape
    lam = torch.full((N,), lam, device=z.device)
    r = rb(z); J = (r * r).sum(1)
    if trace is not None: trace.append((z.clone(), J.clone()))
    for _ in range(iters):
        A = Ab(z)                                              # (N, m, n)
        g = (A.transpose(1, 2) @ r.unsqueeze(-1))              # A^T r
        H = A.transpose(1, 2) @ A                              # A^T A
        Dg = torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-9))
        dz = -torch.linalg.solve(H + lam[:, None, None] * Dg, g).squeeze(-1)   # Marquardt scaling
        z_new = z + dz
        r_new = rb(z_new); J_new = (r_new * r_new).sum(1)
        ok = torch.isfinite(J_new) & (J_new < J)
        done = ok & ((J - J_new) < tol * (1 + J))
        z = torch.where(ok[:, None], z_new, z)
        r = torch.where(ok[:, None], r_new, r)
        J = torch.where(ok, J_new, J)
        lam = torch.where(ok, lam / 3, lam * 4).clamp(1e-12, 1e12)
        if trace is not None: trace.append((z.clone(), J.clone()))
        if bool((done | (lam >= 1e12)).all()):
            break
    return z, J

# ------------------------------------------------------------------ the holy leg, placed on T
def holy_v(n_phase=36):
    """Jansen's holy numbers with placement and timing fitted to T (LM over theta0, hX, hY, dth_s, q
    from n_phase crank phases), clipped into the DE box. A reference leg and one DE seed."""
    mech = torch.tensor(list(HOLY.values()))
    def r_place(p):
        return residuals_v(torch.cat([mech, p]))
    F = joints(mech, t_turn)[-1]
    P0 = torch.zeros(n_phase, NV - 13)
    P0[:, 0] = torch.linspace(-math.pi, math.pi, n_phase + 1)[:-1]
    P0[:, 1] = T[:, 0].mean() - HAND * F[:, 0].mean()           # centre the loop on T
    P0[:, 2] = T[:, 1].mean() - F[:, 1].mean()
    P0[:, 3] = 40 * D2R
    p, J = lm(P0, r_fn=r_place, iters=100)
    v = torch.cat([mech, p[J.argmin()]])
    v[13] = torch.atan2(torch.sin(v[13]), torch.cos(v[13]))
    return torch.as_tensor(np.clip(v.numpy(), DE_LO, DE_HI))

# ------------------------------------------------------------------ stage 1: LHS + differential evolution
def lhs(n, seed, opt=None):
    """n designs spread evenly over the DE box (scrambled Latin hypercube). opt="random-cd" improves
    the spread further but costs O(n^2): ~5 min for 20k points in 27-D."""
    return qmc.scale(qmc.LatinHypercube(d=NV, optimization=opt, rng=seed).random(n), DE_LO, DE_HI)

def run_de(hand, seed, n_lhs=20000, popsize=15, maxiter=300, strategy="best1bin", holy=None,
           device=DEVICE, dtype=torch.float64, lhs_opt=None, verbose=True):
    """One DE run: LHS screen -> best popsize*27 points (+ the holy leg if given) as the initial
    population -> SciPy DE with the whole population scored in one torch call per generation."""
    set_hand(hand)
    t0 = time.time()
    V = lhs(n_lhs, seed, lhs_opt)
    Jl = score(V, device, dtype)
    S = popsize * NV
    init = V[np.argsort(Jl)[:S]]
    if holy is not None:
        init[-1] = holy
    hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun))
    res = differential_evolution(lambda X: score(X.T, device, dtype), list(zip(DE_LO, DE_HI)),
                                 init=init, vectorized=True, updating="deferred", polish=False,
                                 strategy=strategy, mutation=(0.5, 1.0), recombination=0.7,
                                 maxiter=maxiter, tol=1e-8, rng=seed, callback=cb)
    out = dict(hand=hand, seed=seed, x=res.x, J=float(res.fun), pop=res.population,
               popJ=res.population_energies, hist=hist, lhs_J=Jl, nfev=res.nfev, holy=holy is not None)
    if verbose:
        print(f"  DE hand {hand:+d} seed {seed}: LHS best J {Jl.min():10.3f} -> DE J {res.fun:9.4f} "
              f"({res.nit} generations of {S}, {time.time() - t0:.1f} s)")
    return out

# ------------------------------------------------------------------ stage 2: distinct basins -> batched LM
def unit(V):
    """Physical designs -> the unit DE box (for distances)."""
    return (np.asarray(V) - DE_LO) / (DE_HI - DE_LO)

def box_dist(U, u):
    """RMS distance per coordinate in the unit box; theta0 wraps around."""
    d = np.abs(U - u)
    d[..., 13] = np.minimum(d[..., 13], 1 - d[..., 13])
    return np.sqrt((d ** 2).mean(-1))

def niche(V, J, K=64, eps=0.03):
    """Greedy: best J first, keep a point only if it is eps away from every point already kept."""
    order = np.argsort(J)
    U = unit(V)
    keep = []
    for n in order:
        if not keep or box_dist(U[keep], U[n]).min() > eps:
            keep.append(n)
            if len(keep) == K:
                break
    return np.array(keep)

RF_LO = DE_LO.copy(); RF_HI = DE_HI.copy()          # refine: the box numbers keep their bounds,
RF_LO[13], RF_HI[13] = -np.inf, np.inf               # theta0 is free, and the gap logits get +-2*Q_MAX
RF_LO[17:], RF_HI[17:] = -2 * Q_MAX, 2 * Q_MAX       # (a gap ratio up to e^12; unbounded, one runs off
                                                     # and saturates the softmax: an exactly flat direction)

def refine(v, max_nfev=1500):
    """Final polish of one design: SciPy least_squares (trust-region reflective, C13 Step 6) in the
    physical v with the exact torch Jacobian and the true bounds. Needed because the sigmoid of z
    flattens next to a bound, and batched LM crawls when the best leg sits on one (it often does)."""
    fun = lambda x: residuals_v(torch.as_tensor(x)).numpy()
    jac = lambda x: jacfwd(residuals_v)(torch.as_tensor(x)).numpy()
    x0 = np.clip(v, RF_LO + 1e-9 * (DE_HI - DE_LO), RF_HI - 1e-9 * (DE_HI - DE_LO))
    s = least_squares(fun, x0, jac=jac, bounds=(RF_LO, RF_HI), x_scale="jac", max_nfev=max_nfev,
                      ftol=1e-12, xtol=1e-12, gtol=1e-9)
    return s.x, 2 * s.cost

def polish(runs, K=64, n_refine=24, eps=0.03, device=DEVICE, iters=200, verbose=True):
    """Pool the final DE populations of each hand, pick K distinct starts, LM them all at once,
    refine the n_refine best with least_squares, then merge results that end in the same basin.
    Returns basins sorted by J."""
    basins = []
    for hand in sorted({r["hand"] for r in runs}, reverse=True):
        set_hand(hand)
        rs = [r for r in runs if r["hand"] == hand]
        V = np.vstack([r["pop"] for r in rs])
        J = np.concatenate([r["popJ"] for r in rs])
        src = np.concatenate([[r["seed"]] * len(r["pop"]) for r in rs])
        sel = niche(V, J, K, eps)
        t0 = time.time()
        z, Jz = lm(to_z(V[sel]).to(device), iters=iters)
        Vz = unpack(z).detach().cpu().numpy(); Jz = Jz.cpu().numpy()
        if verbose:
            print(f"  LM hand {hand:+d}: {len(sel)} starts in {time.time() - t0:.1f} s, "
                  f"best J {Jz.min():.4f}")
        t0 = time.time()
        best = np.argsort(Jz)[:n_refine]
        for n in best:
            Vz[n], Jz[n] = refine(Vz[n])
        sel, Vz, Jz = sel[best], Vz[best], Jz[best]
        if verbose:
            print(f"  refine hand {hand:+d}: {len(best)} best LM results in {time.time() - t0:.1f} s, "
                  f"best J {Jz.min():.4f}")
        basins += [dict(hand=hand, v=Vz[n], J=float(Jz[n]), hits=1,
                        seeds={int(src[sel[n]])}, starts=[V[sel[n]]]) for n in range(len(Jz))]
    return merge_basins(basins)

def merge_basins(basins, eps=0.05, jtol=1e-4):
    """Best J first: a result with the same hand, J within jtol and box_dist < eps of a kept basin
    joins it (hits, seeds, starts pooled). Run after polish and again after the saddle escapes,
    which can move a basin onto one already found. Returns basins sorted by J."""
    out = []
    for C in sorted(basins, key=lambda B: B["J"]):
        for B in out:
            if (B["hand"] == C["hand"] and abs(B["J"] - C["J"]) <= jtol * (1 + B["J"])
                    and box_dist(unit(B["v"]), unit(C["v"])) < eps):
                B["hits"] += C["hits"]; B["seeds"] |= C["seeds"]; B["starts"] += C["starts"]
                break
        else:
            out.append(C)
    return out

def summary(v):
    """RMS miss [mm] over the 14 targets, the smallest transmission angle [deg] and the lowest foot
    Y [mm] over the turn (2-deg grid)."""
    vt = torch.as_tensor(v)
    r = residuals_v(vt)
    n = 2 * len(T)
    rms = float(torch.sqrt((r[:n] ** 2).sum()))      # r = miss / sqrt(14), so sum r^2 = mean |miss|^2
    m, ax, ay, b, c, d, e, f, g, h, i, j, k = vt[:13]
    M, Z, P1, P2, P3, P4, F = joints(vt, t_turn)
    mus = [mu_acute(lp, lq, ((Q - P) ** 2).sum(-1)).min() for P, Q, lp, lq in
           ((M, Z, j, b), (M, Z, k, c), (P2, P3, f, g))]
    return rms, float(min(mus)) / D2R, float((vt[15] + F[:, 1]).min())

def basin_table(basins, n=12):
    print(f"  {'#':>2} {'hand':>4} {'J*':>11} {'rms mm':>7} {'mu_min':>7} {'low Y':>7} {'hits':>5}  seeds")
    for q, B in enumerate(basins[:n]):
        set_hand(B["hand"])
        rms, mu, low = summary(B["v"])
        print(f"  {q:2d} {B['hand']:+4d} {B['J']:11.5f} {rms:7.3f} {mu:7.1f} {low:7.2f} {B['hits']:5d}  {sorted(B['seeds'])}")

# ------------------------------------------------------------------ stage 3: Hessian (SF 4.4)
SCALE = torch.as_tensor(DE_HI - DE_LO)            # one unit = the whole DE range of each number

def hessian_report(v, verbose=True):
    """
    Exact gradient and Hessian of J at v, in physical units (the sigmoid of z would flatten the
    curvature near the bounds), made dimensionless by SCALE so eigenvalues compare across mm and rad.
    Numbers sitting on a bound are held out (the test applies to the free subspace).
      strict local minimum : gradient ~ 0 and all eigenvalues > 0
      flat valley          : gradient ~ 0, some eigenvalues ~ 0 (a family of equally good legs)
      saddle               : a negative eigenvalue (not a minimum; restart LM along it)
    A Hessian only certifies a LOCAL minimum; see global_evidence().
    """
    v = torch.as_tensor(v, dtype=torch.float64).detach()
    S = SCALE.to(v)
    r = residuals_v(v); J = float((r * r).sum())
    g = jacrev(J_v)(v) * S
    H = hessian(J_v)(v) * S[:, None] * S[None, :]
    A = jacfwd(residuals_v)(v) * S
    GN = 2 * A.T @ A                                                  # Gauss-Newton part of H
    lo, hi = torch.as_tensor(RF_LO), torch.as_tensor(RF_HI)       # the bounds refine() works in
    width = torch.as_tensor(np.where(np.isfinite(RF_HI - RF_LO), RF_HI - RF_LO, np.inf))
    at_lo, at_hi = (v - lo) < 0.01 * width, (hi - v) < 0.01 * width
    at_bound = [n for n in range(NV) if at_lo[n] or at_hi[n]]
    # KKT: on a bound, J should fall only by leaving the box (gradient pointing out); then the bound,
    # not the fit, is what stops the leg there, and widening it would lower J
    kkt = all((g[n] <= 1e-6 * (1 + J)) if at_hi[n] else (g[n] >= -1e-6 * (1 + J)) for n in at_bound)
    free = [n for n in range(NV) if n not in at_bound]
    Hf = H[free][:, free]
    ev, U = torch.linalg.eigh(Hf)
    lam_max = float(ev.abs().max())
    tol = 1e-8 * lam_max
    n_neg, n_flat = int((ev < -tol).sum()), int((ev.abs() <= tol).sum())
    gnorm = float(g[free].norm())
    converged = gnorm <= 1e-4 * (1 + J)
    cls = "saddle" if n_neg else (f"flat valley ({n_flat} zero directions)" if n_flat else "strict local minimum")
    if at_bound and not kkt:
        converged = False
    if not converged:
        cls += " [gradient not ~0: not converged]"
    vecs = torch.zeros(NV, len(free)); vecs[free] = U                  # eigenvectors in all 27 slots
    soft = []
    for q in range(min(3, len(free))):
        top = torch.argsort(vecs[:, q].abs(), descending=True)[:3]
        soft.append((float(ev[q]), [(NAMES[t], round(float(vecs[t, q]), 2)) for t in top]))
    pen = r[2 * len(T):]
    active = [RES_NAMES[2 * len(T) + n] for n in range(len(pen)) if pen[n] > 1e-9]
    n_gr = sum(nm.startswith("ground") for nm in active)              # 180 hinges: report them as one
    active = [nm for nm in active if not nm.startswith("ground")] + ([f"ground ({n_gr} samples)"] if n_gr else [])
    rep = dict(J=J, grad_norm=gnorm, converged=converged, evals=ev.numpy(), evecs=vecs.numpy(),
               cond=lam_max / max(float(ev.abs().min()), 1e-300), n_neg=n_neg, n_flat=n_flat, cls=cls,
               at_bound=[NAMES[n] for n in at_bound], kkt=kkt, active=active, soft=soft,
               H=H.numpy(), GN=GN.numpy(), gap_GN=float((H - GN).norm() / H.norm()))
    if verbose:
        print(f"  J = {J:.6g} | |grad| = {gnorm:.2e} | eig {float(ev[0]):.3g} .. {lam_max:.3g} "
              f"(cond {rep['cond']:.3g}) | |H - 2A^T A| / |H| = {rep['gap_GN']:.3f}")
        print(f"  -> {cls}")
        if at_bound:
            push = [NAMES[n] for n in at_bound if abs(float(g[n])) > 1e-6 * (1 + J)]
            print("     on a bound (held out):", rep["at_bound"], "| KKT sign", "ok" if kkt else "FAILS",
                  f"| J would still fall past: {push} (widen these to lower J)" if push and kkt else "")
        if active:   print("     active penalties (one-sided curvature there):", active)
        for lam, parts in soft:
            print(f"     soft direction  eig {lam:9.3g}: {parts}")
    return rep

def escape_saddle(v, rep, step=0.02):
    """Restart the refine from v +- step * (most negative eigenvector), keep the better end."""
    d = rep["evecs"][:, 0] * SCALE.numpy() * step
    return min((refine(v + d), refine(v - d)), key=lambda s: s[1])

def global_evidence(basins, runs):
    """What can be said about 'global'. The Hessian cannot say it; these three can."""
    B = basins[0]
    seeds = {(r["hand"], r["seed"]) for r in runs}
    reached = {(B["hand"], s) for s in B["seeds"]}
    print(f"  best J* = {B['J']:.6g}  (hand {B['hand']:+d})")
    if B["J"] < 1e-8:
        print("  J* ~ 0 and J >= 0 everywhere: this IS a global minimum (one of possibly many).")
    print(f"  reached from {len(reached)} of {len(seeds)} independent LHS/DE runs, {B['hits']} LM starts")
    print(f"  no basin found lower; next best J = "
          + (f"{basins[1]['J']:.6g}" if len(basins) > 1 else "none"))

# ------------------------------------------------------------------ the whole pipeline
def synthesize(hands=(+1, -1), seeds=4, n_lhs=20000, popsize=15, maxiter=300, strategy="best1bin",
               K=64, n_refine=24, n_report=5, screen_device=DEVICE, screen_dtype=torch.float64,
               lm_device=DEVICE, lhs_opt=None, verbose=True):
    t0 = time.time()
    if verbose: print("stage 1: Latin hypercube + differential evolution")
    runs = []
    for hand in hands:
        set_hand(hand)
        holy = holy_v().numpy()                    # injected into the first run of each hand only,
        for s in range(seeds):                     # so the other runs stay independent evidence
            runs.append(run_de(hand, 1000 * (hand < 0) + s, n_lhs, popsize, maxiter, strategy,
                               holy if s == 0 else None, screen_device, screen_dtype, lhs_opt, verbose))
    if verbose: print("stage 2: distinct basins -> batched Levenberg-Marquardt -> least_squares")
    basins = polish(runs, K, n_refine, device=lm_device, verbose=verbose)
    if verbose: basin_table(basins)
    if verbose: print("stage 3: Hessian at the best basins")
    for q, B in enumerate(basins[:n_report]):
        set_hand(B["hand"])
        if verbose: print(f" basin {q}:")
        rep = hessian_report(B["v"], verbose)
        if rep["n_neg"]:
            v2, J2 = escape_saddle(B["v"], rep)
            if J2 < B["J"]:
                if verbose: print(f"  saddle escape: J {B['J']:.6g} -> {J2:.6g}")
                B["v"], B["J"] = v2, J2
                rep = hessian_report(v2, verbose)
        B["rep"] = rep
    basins = merge_basins(basins)
    if verbose:
        print("global evidence:")
        global_evidence(basins, runs)
    best = basins[0]
    set_hand(best["hand"])
    trace = []                                     # replay the best basin's first LM start, step by step
    lm(to_z(torch.as_tensor(best["starts"][0])[None]), iters=200, trace=trace)
    keep = [0] + [q for q in range(1, len(trace)) if float(trace[q][1][0]) < float(trace[q - 1][1][0])]
    best["trace_V"] = np.array([unpack(trace[q][0][0]).detach().numpy() for q in keep] + [best["v"]])
    best["trace_J"] = np.array([float(trace[q][1][0]) for q in keep] + [best["J"]])   # last = refined
    if verbose: print(f"done in {time.time() - t0:.0f} s")
    return dict(runs=runs, basins=basins, best=best)

def save(res, path="best_jansen"):
    """best_jansen.pt (torch, like best_leg.pt) and best_jansen.npz (numpy, for verify/viz)."""
    B = res["best"]
    rep = B.get("rep", {})
    torch.save({"v": torch.as_tensor(B["v"]), "J": B["J"], "hand": B["hand"],
                "evals": rep.get("evals"), "cls": rep.get("cls")}, path + ".pt")
    np.savez(path + ".npz", v=B["v"], J=B["J"], hand=B["hand"], trace_V=B["trace_V"],
             trace_J=B["trace_J"], evals=rep.get("evals", np.zeros(0)), evecs=rep.get("evecs", np.zeros(0)))

def top_designs(basins, n=8, eps=0.05):
    """The n best distinct designs: best J first, keep a basin only if it is more than eps (box_dist)
    from every one kept. Points along one flat valley come out of LM/refine as separate 'basins';
    eps = 0 keeps them all (the raw top n)."""
    keep = []
    for B in basins:
        if all(box_dist(unit(K["v"]), unit(B["v"])) > eps for K in keep):
            keep.append(B)
            if len(keep) == n:
                break
    return keep

def basin_ranks(basins, top):
    """Index of each kept design in the full basin list (by identity: the dicts hold arrays)."""
    return [next(q for q, B in enumerate(basins) if B is K) for K in top]

# rod letter -> the two joints it connects (joints() above; O = crank axle, Z = frame pivot)
ROD_JOINTS = dict(m="O-M", j="M-P1", b="Z-P1", d="Z-P2", e="P1-P2", k="M-P3", c="Z-P3",
                  f="P2-P4", g="P3-P4", h="P4-F", i="P3-F")
PARAM_INFO = {**ROD_JOINTS, "ax": "Z = (-ax, -ay) from O", "ay": "Z = (-ax, -ay) from O",
              "theta0": "crank angle at target 0", "hX": "O in body frame", "hY": "O in body frame",
              "dth_s": "crank step per stance target", **{f"q{n}": "swing gap logit" for n in range(1, N_GAP)}}

def save_top(res, out_dir, n=8, eps=0.05):
    """The n best distinct designs (top_designs): top_basins.npz (stacked, rank order, with what each
    number means) and top/basin_XX.{pt,npz}, each in the best_jansen format so verify_jansen.py loads it."""
    top = top_designs(res["basins"], n, eps)
    d = os.path.join(out_dir, "top")
    os.makedirs(d, exist_ok=True)
    for q, B in enumerate(top):
        rep = B.get("rep", {})
        p = os.path.join(d, f"basin_{q:02d}")
        torch.save({"v": torch.as_tensor(B["v"]), "J": B["J"], "hand": B["hand"], "hits": B["hits"],
                    "evals": rep.get("evals"), "cls": rep.get("cls")}, p + ".pt")
        np.savez(p + ".npz", v=B["v"], J=B["J"], hand=B["hand"], hits=B["hits"],
                 evals=rep.get("evals", np.zeros(0)), evecs=rep.get("evecs", np.zeros(0)))
    np.savez(os.path.join(out_dir, "top_basins.npz"), names=np.array(NAMES),
             meaning=np.array([PARAM_INFO[nm] for nm in NAMES]),
             rank=np.array(basin_ranks(res["basins"], top)),
             V=np.stack([np.asarray(B["v"]) for B in top]), J=np.array([B["J"] for B in top]),
             hand=np.array([B["hand"] for B in top]), hits=np.array([B["hits"] for B in top]))
    return top

# ------------------------------------------------------------------ GPU or CPU? measure it
def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

def bench(sizes=(1000, 10000, 100000), Ks=(16, 64, 256)):
    """Throughput per stage and device, so the device choice is measured, not assumed:
      screening = J evaluations / s (LHS + DE), LM = batched iterations / s,
      and the float32 vs float64 rank agreement of J (float32 is enough only if it ranks alike)."""
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print("device:", ", ".join(devs), "| torch", torch.__version__,
          "| CUDA build" if torch.version.cuda else "| CPU-only build")
    set_hand(+1)
    print("screening, J evaluations per second")
    for dev in devs:
        for dt in ((torch.float64, torch.float32) if dev == "cuda" else (torch.float64,)):
            row = []
            for n in sizes:
                V = lhs(n, 0)
                score(V[:min(n, 512)], dev, dt)                        # warm-up
                _sync(dev); t = time.time(); score(V, dev, dt); _sync(dev)
                row.append(f"{n:>7d}: {n / (time.time() - t):10.0f}/s")
            print(f"  {dev:4s} {str(dt)[6:]:8s}", " | ".join(row))
    print("LM, batched iterations per second (float64)")
    for dev in devs:
        row = []
        for K in Ks:
            z = to_z(lhs(K, 1)).to(dev)
            lm(z, iters=1)
            _sync(dev); t = time.time(); lm(z, iters=10, tol=0); _sync(dev)
            row.append(f"K={K:<4d}: {10 / (time.time() - t):7.2f} it/s")
        print(f"  {dev:4s}", " | ".join(row))
    V = lhs(10000, 2)
    dev = devs[-1]
    rho = spearmanr(score(V, dev, torch.float64), score(V, dev, torch.float32)).statistic
    top64 = len(set(np.argsort(score(V, dev, torch.float64))[:64]) & set(np.argsort(score(V, dev, torch.float32))[:64]))
    print(f"float32 vs float64 J on 10k LHS designs ({dev}): Spearman rho = {rho:.6f}, top-64 overlap {top64}/64")

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    pa = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    pa.add_argument("--hands", default="+1,-1", help="frame-pivot side(s): +1 Jansen's, -1 mirrored")
    pa.add_argument("--seeds", type=int, default=4, help="independent LHS/DE runs per hand")
    pa.add_argument("--nlhs", type=int, default=20000)
    pa.add_argument("--popsize", type=int, default=15)
    pa.add_argument("--maxiter", type=int, default=300)
    pa.add_argument("--strategy", default="best1bin")
    pa.add_argument("--k", type=int, default=64, help="LM starts per hand")
    pa.add_argument("--refine", type=int, default=24, help="best LM results per hand finished by least_squares")
    pa.add_argument("--lhs-opt", default=None, help="e.g. random-cd (slow for large --nlhs)")
    pa.add_argument("--screen-device", default=DEVICE)
    pa.add_argument("--screen-dtype", default="float64", choices=["float64", "float32"])
    pa.add_argument("--lm-device", default=DEVICE)
    pa.add_argument("--smoke", action="store_true", help="tiny settings, to check the pipeline")
    pa.add_argument("--bench", action="store_true", help="measure CPU/GPU throughput and exit")
    pa.add_argument("--out", default=".", help="output directory (created if missing)")
    pa.add_argument("--top", type=int, default=8, help="number of best distinct designs whose weights are saved")
    pa.add_argument("--top-eps", type=float, default=0.05,
                    help="min box_dist between saved designs (0 = raw top N, near-duplicates included)")
    pa.add_argument("--no-figs", action="store_true")
    a = pa.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, "best_jansen")
    if a.bench:
        bench(); raise SystemExit
    if a.smoke:
        a.seeds, a.nlhs, a.maxiter, a.k, a.refine = 1, 2000, 30, 16, 6
    res = synthesize(tuple(int(h) for h in a.hands.split(",")), a.seeds, a.nlhs, a.popsize, a.maxiter,
                     a.strategy, a.k, a.refine, screen_device=a.screen_device, screen_dtype=getattr(torch, a.screen_dtype),
                     lm_device=a.lm_device, lhs_opt=a.lhs_opt)
    B = res["best"]
    for nm, x in zip(NAMES, B["v"]):
        print(f"  {nm:7s} {x / (D2R if nm in ('theta0', 'dth_s') else 1):9.3f}")
    save(res, stem)
    torch.save(res, stem + "_run.pt")
    top = save_top(res, a.out, a.top, a.top_eps)
    print(f"saved {stem}.{{pt,npz}}, {stem}_run.pt, and the top {len(top)} distinct designs "
          f"(basins {basin_ranks(res['basins'], top)}) to",
          os.path.join(a.out, "top_basins.npz"), "+", os.path.join(a.out, "top", "basin_XX.{pt,npz}"))
    if not a.no_figs:
        import matplotlib; matplotlib.use("Agg")
        import jansen_viz as JV
        from matplotlib.animation import PillowWriter
        JV.plot_best(B["v"], B["hand"]).savefig(stem + "_best.png", dpi=110)
        JV.plot_basins(res["basins"]).savefig(stem + "_basins.png", dpi=110)
        JV.plot_weights(top).savefig(os.path.join(a.out, "top_weights.png"), dpi=110)
        JV.plot_de_convergence(res["runs"]).savefig(stem + "_de.png", dpi=110)
        if "rep" in B:
            set_hand(B["hand"])
            JV.plot_hessian_slice(B["v"], B["hand"], B["rep"], lambda V: score(V),
                                  B["trace_V"]).savefig(stem + "_hessian.png", dpi=110)
        JV.animate_walk(B["v"], B["hand"]).save(stem + "_walk.gif", writer=PillowWriter(fps=12))
        JV.animate_lm(B["trace_V"], B["trace_J"], B["hand"]).save(stem + "_lm.gif", writer=PillowWriter(fps=4))
        print("figures:", stem + "_{best,basins,de,hessian}.png,", stem + "_{walk,lm}.gif")
