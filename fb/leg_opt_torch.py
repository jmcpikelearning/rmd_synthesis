"""
walking-leg synthesis with PyTorch

Same recipe as SF section 4.3: residual vector r(v), cost J = r^T r, Levenberg-Marquardt
from starts determined by genetic algorithm. What torch adds:
  * exact Jacobians from autograd (no finite-difference nudging, SF 4.4),
  * all starts solved in parallel as one batch (vmap),
  * smooth (sloped) penalties in place of the flat reject plateau (SF 4.6).



Design vector: v = (a, b, c, d, p, delta, theta1, theta0, hX, hY); mm and rad.
"""
import math
import torch
from torch.func import vmap, jacrev

torch.set_default_dtype(torch.float64)          # LM needs float64: A^T A spans ~1e-4..1e3
D2R = math.pi / 180

# ------------------------------------------------------------------ task 
STRIDE = 50.0                                    # mm over half a crank turn
X_START = 0.0                                    # T goes 0 -> -50
dth_st = torch.arange(0, 181, 30) * D2R          # 7 stance crank travels
T = torch.stack([X_START - STRIDE / math.pi * dth_st, torch.zeros(7)], 1)   # (7, 2) targets [X, Y]

dth_sw = torch.arange(185, 360, 5) * D2R         # 35 swing crank travels
h_req = torch.where((dth_sw - math.pi * 1.5).abs() <= 20 * D2R, 15.0, 0.0)  # 15 mm lift near mid-swing, Y>=0 elsewhere
MU_REQ = 40 * D2R
t2_turn = torch.linspace(0, 2 * math.pi, 361)    # full turn for mu_min
SIGMA = +1                                       # +1 = open assembly (= rmd.py branch=+1)
EPS = 1e-12
W_GRASHOF, MARGIN = 10.0, 1.0                    # per mm of violation; keep 1 mm inside the rule

# ------------------------------------------------------------------ bounds -> unconstrained z
#            a    b    c    d    p   hX   hY        (delta, theta1, theta0 are free angles)
LO = torch.tensor([8., 20., 20., 20., 0., -50., 50.])
HI = torch.tensor([60., 150., 150., 150., 200., 50., 150.])

def unpack(z):
    """z (10,) unconstrained -> design numbers. Box limits via a sigmoid, so LM never leaves them."""
    box = LO + (HI - LO) * torch.sigmoid(z[[0, 1, 2, 3, 4, 8, 9]])
    a, b, c, d, p, hX, hY = box
    return a, b, c, d, p, z[5], z[6], z[7], hX, hY      # delta, theta1, theta0 = z[5:8]

def to_z(v):
    """design numbers (..., 10) -> z, inverse of unpack (for seeding from a known leg)."""
    v = torch.as_tensor(v, dtype=torch.float64)
    z = v.clone()
    idx = [0, 1, 2, 3, 4, 8, 9]
    u = ((v[..., idx] - LO) / (HI - LO)).clamp(1e-6, 1 - 1e-6)
    z[..., idx] = torch.log(u / (1 - u))
    return z

# ------------------------------------------------------------------ kinematics
def coupler_angle(a, b, c, d, t2):
    """
    theta3 from triangle A-B-O4 (side s = |AO4|); the same root as the Freudenstein
    quadratic, written so the gradient is finite everywhere:
        phi   = direction of A->O4
        gamma = angle at A, = atan2(4*Area, b^2 + s^2 - c^2)
        16*Area^2 = 4 b^2 s^2 - (b^2 + s^2 - c^2)^2   (>= 0 iff the linkage assembles;
                    this is a^2 b^2 (E^2 - 4DF), see theta3 notes, Step 9)
    """
    ux, uy = d - a * torch.cos(t2), -a * torch.sin(t2)
    s2 = ux * ux + uy * uy
    k = b * b + s2 - c * c
    root = torch.sqrt(torch.clamp(4 * b * b * s2 - k * k, min=EPS))
    return torch.atan2(uy, ux) + SIGMA * torch.atan2(root, k)

def mu_acute(a, b, c, d, t2):
    """Transmission angle (acute) from the law of cosines; needs no theta3/theta4 and is branch-free."""
    s2 = a * a + d * d - 2 * a * d * torch.cos(t2)
    m = b * b + c * c - s2
    root = torch.sqrt(torch.clamp(4 * b * b * c * c - m * m, min=EPS))
    return torch.atan2(root, m.abs())

def foot(z, dth):
    """Foot C(theta2; v) in the body frame, SF 4.2.  theta2 = theta0 - dtheta (motor CW)."""
    a, b, c, d, p, dl, th1, th0, hX, hY = unpack(z)
    t2 = th0 - dth
    t3 = coupler_angle(a, b, c, d, t2)
    x = a * torch.cos(t2) + p * torch.cos(t3 + dl)
    y = a * torch.sin(t2) + p * torch.sin(t3 + dl)
    c1, s1 = torch.cos(th1), torch.sin(th1)
    return torch.stack([hX + c1 * x - s1 * y, hY + s1 * x + c1 * y], -1)

# ------------------------------------------------------------------ residuals (SF 4.3) + sloped reject
def residuals(z):
    a, b, c, d = unpack(z)[:4]
    miss = (foot(z, dth_st) - T) / math.sqrt(7)                      # 14
    clear = torch.relu(h_req - foot(z, dth_sw)[:, 1]) / math.sqrt(35)   # 35
    mu_min = mu_acute(a, b, c, d, t2_turn).min()
    r_mu = 2 * torch.relu(MU_REQ - mu_min) / D2R                     # 2 per degree
    others = torch.stack([b, c, d])
    l = others.max()
    g_short = torch.relu(a - others.min() + MARGIN)                  # crank must be shortest
    g_grash = torch.relu(a + l - (others.sum() - l) + MARGIN)        # s + l < p + q
    return torch.cat([miss.reshape(-1), clear, torch.stack([r_mu, W_GRASHOF * g_short, W_GRASHOF * g_grash])])

r_batch = vmap(residuals)
A_batch = vmap(jacrev(residuals))                # exact 52x10 Jacobian per start

# ------------------------------------------------------------------ batched Levenberg-Marquardt (SF 4.5)
def lm(z, iters=200, lam=1e-2, tol=1e-10):
    N, n = z.shape
    lam = torch.full((N,), lam)
    r = r_batch(z); J = (r * r).sum(1)
    for _ in range(iters):
        A = A_batch(z)                                         # (N, m, n)
        g = (A.transpose(1, 2) @ r.unsqueeze(-1))              # A^T r
        H = A.transpose(1, 2) @ A                              # A^T A
        Dg = torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-9))
        dz = -torch.linalg.solve(H + lam[:, None, None] * Dg, g).squeeze(-1)   # Marquardt scaling
        z_new = z + dz
        r_new = r_batch(z_new); J_new = (r_new * r_new).sum(1)
        ok = torch.isfinite(J_new) & (J_new < J)
        done = ok & ((J - J_new) < tol * (1 + J))
        z = torch.where(ok[:, None], z_new, z)
        r = torch.where(ok[:, None], r_new, r)
        J = torch.where(ok, J_new, J)
        lam = torch.where(ok, lam / 3, lam * 4).clamp(1e-12, 1e12)
        if bool((done | (lam >= 1e12)).all()):
            break
    return z, J

# ------------------------------------------------------------------ many starts (SF 4.6)
def random_starts(N, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.empty(N, 10)
    z[:, [0, 1, 2, 3, 4, 8, 9]] = torch.logit(torch.rand(N, 7, generator=g).clamp(0.02, 0.98))
    z[:, 5:8] = (2 * torch.rand(N, 3, generator=g) - 1) * math.pi
    return z

if __name__ == "__main__":
    import time
    N = 256
    t0 = time.time()
    z, J = lm(random_starts(N))
    print(f"{N} starts in {time.time() - t0:.1f} s")
    order = torch.argsort(J)
    print("best 10 J:", [round(float(j), 3) for j in J[order[:10]]])
    zb = z[order[0]]
    v = torch.stack(unpack(zb)).detach()
    names = "a b c d p delta theta1 theta0 hX hY".split()
    for nm, x in zip(names, v):
        print(f"  {nm:7s} {float(x) / (D2R if nm in ('delta', 'theta1', 'theta0') else 1):9.3f}")
    torch.save({"v": v, "J": J[order[0]]}, "best_leg.pt")
