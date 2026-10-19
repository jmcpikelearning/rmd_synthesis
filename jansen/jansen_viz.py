"""
Pictures of a Jansen leg, in the style of C13_leg_optimizer_v2.ipynb (Steps 8-10).
  plot_task           the 14 targets: 4 timed stance points, 10 untimed swing points
  plot_best           Step 8: leg at mid-stance, foot path at every degree, miss at each target
  animate_walk        the leg through one crank turn, with the three transmission angles
  animate_lm          Step 10: one Levenberg-Marquardt run, leg and J step by step
  plot_hessian_slice  Step 9 at the minimum: J on the plane of the two softest Hessian directions
  plot_basins         the distinct minima found, and how many starts reached each
  plot_de_convergence the LHS screen and the differential-evolution runs
Kinematics come from verify_jansen.py (NumPy), so the pictures do not depend on the torch code.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, Circle
from jansen_opt_torch import T, N_ST, NAMES, MU_REQ, DE_LO, DE_HI
from verify_jansen import pose_np, travel_np, mu_np, DYADS, JOINTS

plt.rcParams["animation.embed_limit"] = 60
BLUE, PINK, ORANGE, PLATE = "#3A8FBF", "#C2185B", "#BF5700", "#CFE3F0"
MU_COLORS = ("#3A8FBF", "#BF5700", "#4C9A2A")
T_C = T.numpy() @ [1, 1j]
TT = np.radians(np.arange(0, 361, 1.0))          # one turn, every degree

class Leg:
    """Artists for one pose: rods as black 'o-' lines, the two rigid triangles as light-blue plates
    (like the C13 coupler), crank circle dashed, ground pivots as grey triangles, foot as a blue dot."""
    RODS = [("O", "M"), ("M", "P1"), ("M", "P3"), ("Z", "P3"), ("P2", "P4")]
    PLATES = [("Z", "P1", "P2"), ("P3", "P4", "F")]

    def __init__(self, ax, alpha=1.0, lw=2.5, ms=4):
        self.plates = [ax.add_patch(Polygon(np.zeros((3, 2)), closed=True, fc=PLATE, ec="k", lw=lw * 0.6,
                                            alpha=alpha, zorder=2)) for _ in self.PLATES]
        self.rods = [ax.plot([], [], "o-", color="k", lw=lw, ms=ms, alpha=alpha, zorder=3)[0] for _ in self.RODS]
        self.crank = ax.add_patch(Circle((0, 0), 1, fill=False, ls="--", lw=0.8, ec="0.5", zorder=1))
        self.piv, = ax.plot([], [], "^", color="0.35", ms=ms + 6, zorder=4)
        self.ft, = ax.plot([], [], "o", color=BLUE, ms=ms + 4, zorder=5)

    def set(self, pose, n=0):
        P = {k: pose[k][n] for k in pose}
        for line, (a, b) in zip(self.rods, self.RODS):
            line.set_data([P[a].real, P[b].real], [P[a].imag, P[b].imag])
        for poly, tri in zip(self.plates, self.PLATES):
            poly.set_xy([[P[t].real, P[t].imag] for t in tri])
        self.crank.center = (P["O"].real, P["O"].imag)
        self.crank.radius = abs(P["M"] - P["O"])
        self.piv.set_data([P["O"].real, P["Z"].real], [P["O"].imag, P["Z"].imag])
        self.ft.set_data([P["F"].real], [P["F"].imag])
        return [*self.rods, *self.plates, self.crank, self.piv, self.ft]

def targets(ax, labels=False):
    ax.axhline(0, color="k", lw=1)
    ax.plot(T_C[:N_ST].real, T_C[:N_ST].imag, "o", color=PINK, ms=7, label="stance targets (timed)")
    ax.plot(T_C[N_ST:].real, T_C[N_ST:].imag, "o", mfc="none", mec=PINK, ms=7, label="swing targets (untimed)")
    if labels:
        for n, t in enumerate(T_C):
            ax.annotate(str(n), (t.real, t.imag), textcoords="offset points", xytext=(5, 4), fontsize=8, color=PINK)

def fit_view(ax, pts, pad=10):
    pts = np.concatenate([np.ravel(p) for p in pts])
    pts = pts[np.isfinite(pts)]
    ax.set_xlim(pts.real.min() - pad, pts.real.max() + pad)
    ax.set_ylim(min(-10, pts.imag.min() - pad), pts.imag.max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

def mid_stance(v):
    return [(N_ST - 1) / 2 * v[16]]

def dyad_mus(pose):
    return [np.degrees(mu_np(pose, X, A, B)) for X, A, B, _, _ in DYADS]

# ------------------------------------------------------------------ static
def plot_task():
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=72)
    targets(ax, labels=True)
    ax.plot(np.r_[T_C, T_C[:1]].real, np.r_[T_C, T_C[:1]].imag, ":", color=PINK, lw=1)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("targets T (jansen_opt.py), traced counter-clockwise")
    return fig

def plot_best(v, hand, title=None):
    """Step 8: the leg at mid-stance, its foot path at every degree, and the miss at each target."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=72, gridspec_kw={"width_ratios": [1.4, 1]})
    turn = pose_np(v, TT, hand)
    Ck = pose_np(v, travel_np(v), hand)["F"]
    targets(ax)
    ax.plot(turn["F"].real, turn["F"].imag, color=BLUE, lw=1.5, label="foot, every degree")
    for c, t in zip(Ck, T_C):
        ax.plot([t.real, c.real], [t.imag, c.imag], color=PINK, lw=0.8)
    ax.plot(Ck.real, Ck.imag, ".", color=BLUE, ms=6)
    Leg(ax).set(pose_np(v, mid_stance(v), hand))
    fit_view(ax, list(turn.values()) + [T_C])
    miss = abs(Ck - T_C)
    mu = min(np.nanmin(m) for m in dyad_mus(turn))
    ax.set_title(title or "rms miss %.2f mm | worst %.2f mm | mu_min %.1f deg" % (np.sqrt(np.mean(miss ** 2)), miss.max(), mu))
    ax.legend(loc="upper left", fontsize=7)
    k = np.arange(len(T_C))
    bx.bar(k[:N_ST], miss[:N_ST], color=PINK, label="stance (timed)")
    bx.bar(k[N_ST:], miss[N_ST:], color="none", edgecolor=PINK, label="swing (untimed)")
    bx.set_xticks(k); bx.set_xlabel("target k"); bx.set_ylabel("miss |C - T| [mm]"); bx.grid(alpha=0.3, axis="y")
    bx.legend(fontsize=8); bx.set_title("miss at each target")
    fig.tight_layout()
    return fig

def plot_hessian_slice(v, hand, rep, J_fn, trace_V=None, n=41):
    """Step 9 at the minimum: log10 J on the plane through v spanned by the two softest Hessian
    eigenvectors (unit = the whole DE range of each number), the LM path projected onto it, and the
    eigenvalue spectrum. A bowl = strict minimum; a long trough = a flat valley (family of legs).
    J_fn(V (S, 27)) -> (S,) must use the same hand (jansen_opt_torch.score after set_hand)."""
    S = DE_HI - DE_LO
    ev, E = rep["evals"], rep["evecs"]
    J0 = rep["J"]
    span = [float(np.clip(np.sqrt(2 * 3 * max(J0, 0.1) / max(abs(l), 1e-12)), 1e-4, 0.3)) for l in ev[:2]]
    a = np.linspace(-span[0], span[0], n); b = np.linspace(-span[1], span[1], n)
    A, B = np.meshgrid(a, b)
    V = v + S * (A.reshape(-1, 1) * E[:, 0] + B.reshape(-1, 1) * E[:, 1])
    Z = np.log10(np.maximum(J_fn(V), 1e-12)).reshape(n, n)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=72, gridspec_kw={"width_ratios": [1.25, 1]})
    ax.contourf(A, B, Z, 24, cmap="Blues_r", alpha=0.6); ax.contour(A, B, Z, 24, colors="#5A7A90", linewidths=0.4)
    if trace_V is not None:
        D = (np.asarray(trace_V) - v) / S
        ax.plot(D @ E[:, 0], D @ E[:, 1], "o-", color=BLUE, lw=2, ms=4, label="Levenberg-Marquardt (projected)")
        ax.legend(loc="upper right", fontsize=8)
    ax.plot(0, 0, "k*", ms=12, mec="w")
    ax.set_xlim(a[0], a[-1]); ax.set_ylim(b[0], b[-1])

    def top(q):
        idx = np.argsort(-abs(E[:, q]))[:2]
        return " ".join(f"{E[t, q]:+.2f} {NAMES[t]}" for t in idx)
    ax.set_xlabel(f"softest direction 1 ({top(0)} ...)"); ax.set_ylabel(f"direction 2 ({top(1)} ...)")
    ax.set_title("log10 J around the minimum; dark = low", fontsize=10)
    lam = np.asarray(ev)
    bx.semilogy(np.arange(len(lam)), np.maximum(abs(lam), 1e-16), "o", color=BLUE)
    neg = lam < 0
    bx.semilogy(np.arange(len(lam))[neg], abs(lam[neg]), "o", color=PINK, label="negative")
    bx.axhline(1e-8 * abs(lam).max(), color="0.5", ls="--", lw=1, label="'zero' threshold")
    bx.set_xlabel("eigenvalue index"); bx.set_ylabel("|eigenvalue| (scaled Hessian)")
    bx.set_title(rep["cls"], fontsize=10); bx.grid(alpha=0.3); bx.legend(fontsize=8)
    fig.tight_layout()
    return fig

def plot_basins(basins, n=8):
    """The distinct minima (best first) as a gallery, plus how many LM starts ended in each."""
    fig = plt.figure(figsize=(13, 5.6), dpi=72)
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1.2])
    for q, B in enumerate(basins[:n]):
        ax = fig.add_subplot(gs[q // 4, q % 4])
        v, hand = B["v"], B["hand"]
        turn = pose_np(v, TT, hand)
        ax.axhline(0, color="k", lw=0.8)
        ax.plot(T_C.real, T_C.imag, "o", mfc="none", mec=PINK, ms=4)
        ax.plot(turn["F"].real, turn["F"].imag, color=BLUE, lw=1.2)
        Leg(ax, lw=1.5, ms=2).set(pose_np(v, mid_stance(v), hand))
        fit_view(ax, list(turn.values()) + [T_C], pad=5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        cls = B.get("rep", {}).get("cls", "").split(" [")[0].replace("strict local minimum", "strict min")
        ax.set_title(f"#{q} J={B['J']:.3g} hand {hand:+d}\n{B['hits']} starts {cls}", fontsize=8)
    bx = fig.add_subplot(gs[:, 4])
    m = min(len(basins), 20)
    bx.barh(np.arange(m), [B["hits"] for B in basins[:m]], color=BLUE)
    bx.set_yticks(np.arange(m)); bx.set_yticklabels([f"#{q} J={B['J']:.3g}" for q, B in enumerate(basins[:m])], fontsize=7)
    bx.invert_yaxis(); bx.set_xlabel("LM starts ending here"); bx.set_title("basins (SF 4.6)", fontsize=10)
    fig.tight_layout()
    return fig

def plot_de_convergence(runs):
    """Left: best J per DE generation for each run. Right: the LHS screen of the first run of each
    hand (log10 J of every sample) with the DE results marked."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=72)
    for r in runs:
        ax.semilogy(r["hist"], ls="--" if r["holy"] else "-", lw=1.2,
                    label=f"hand {r['hand']:+d} seed {r['seed']}" + (" (+holy)" if r["holy"] else ""))
    ax.set_xlabel("generation"); ax.set_ylabel("best J"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title("differential evolution")
    for hand in sorted({r["hand"] for r in runs}, reverse=True):
        r = next(r for r in runs if r["hand"] == hand)
        bx.hist(np.log10(r["lhs_J"]), bins=60, alpha=0.5, label=f"LHS, hand {hand:+d} ({len(r['lhs_J'])} designs)")
    for r in runs:
        bx.axvline(np.log10(max(r["J"], 1e-12)), color="k", lw=0.8)
    bx.set_xlabel("log10 J"); bx.set_ylabel("designs"); bx.legend(fontsize=8)
    bx.set_title("LHS screen; black lines = DE results")
    fig.tight_layout()
    return fig

# ------------------------------------------------------------------ animated
def animate_walk(v, hand, step=5, interval=60):
    """The leg through one crank turn: foot trace building up, transmission angles with a cursor."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=72, gridspec_kw={"width_ratios": [1.4, 1]})
    dd = np.arange(0, 360, step)
    pose = pose_np(v, np.radians(dd), hand)
    turn = pose_np(v, TT, hand)
    targets(ax)
    ax.plot(turn["F"].real, turn["F"].imag, color=BLUE, lw=1, alpha=0.25)
    trail, = ax.plot([], [], color=BLUE, lw=1.8, label="foot path")
    leg = Leg(ax)
    fit_view(ax, list(turn.values()) + [T_C]); ax.legend(loc="upper left", fontsize=7)
    mus = dyad_mus(turn)
    for (X, *_), mu, col in zip(DYADS, mus, MU_COLORS):
        bx.plot(np.degrees(TT), mu, color=col, label=f"mu at {X}")
    bx.axhline(np.degrees(MU_REQ), color=PINK, ls="--", lw=1, label="required")
    tk = np.degrees(travel_np(v))
    bx.axvspan(0, tk[N_ST - 1], color=PINK, alpha=0.08, label="stance")
    cur = bx.axvline(0, color="k", lw=1)
    bx.set_xlim(0, 360); bx.set_ylim(0, 90); bx.set_xlabel("crank travel [deg]"); bx.set_ylabel("transmission angle [deg]")
    bx.grid(alpha=0.3); bx.legend(fontsize=7, loc="lower right")

    def draw(q):
        trail.set_data(pose["F"][:q + 1].real, pose["F"][:q + 1].imag)
        cur.set_xdata([dd[q], dd[q]])
        ax.set_title(f"crank travel {dd[q]:3d} deg")
        return [trail, cur, *leg.set(pose, q)]
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, draw, frames=len(dd), interval=interval)
    plt.close(fig)
    return anim

def animate_lm(V, Js, hand, interval=350):
    """Step 10: one LM run, the leg at mid-stance and its foot path after each step, and J."""
    tt = np.radians(np.arange(0, 361, 3))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=72, gridspec_kw={"width_ratios": [1.4, 1]})
    targets(ax)
    path, = ax.plot([], [], color=BLUE, lw=1.5, label="foot path")
    leg = Leg(ax)
    allp = [T_C]
    for v in V:
        allp += list(pose_np(v, mid_stance(v), hand).values()) + [pose_np(v, tt, hand)["F"]]
    fit_view(ax, allp); ax.legend(loc="upper left", fontsize=7)
    bx.semilogy(Js, color="0.8"); dot, = bx.semilogy([], [], "o", color=BLUE); bx.grid(alpha=0.3)
    bx.set_xlabel("step"); bx.set_ylabel("J"); bx.set_title("J after each step")

    def draw(q):
        v = V[q]
        Cp = pose_np(v, tt, hand)["F"]
        path.set_data(Cp.real, Cp.imag)
        dot.set_data([q], [Js[q]])
        ax.set_title("step %d of %d:  J = %.3g" % (q, len(Js) - 1, Js[q]))
        return [path, dot, *leg.set(pose_np(v, mid_stance(v), hand))]
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, draw, frames=len(Js), interval=interval)
    plt.close(fig)
    return anim
