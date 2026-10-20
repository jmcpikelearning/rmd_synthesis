"""
Pictures of a Jansen leg, in the style of C13_leg_optimizer_v2.ipynb (Steps 8-10).
  plot_task           the 14 targets: 4 timed stance points, 10 untimed swing points
  plot_best           Step 8: leg at mid-stance, foot path at every degree, miss at each target
  animate_walk        the leg through one crank turn, with the three transmission angles
  animate_lm          Step 10: one Levenberg-Marquardt run, leg and J step by step
  plot_hessian_slice  Step 9 at the minimum: J on the plane of the two softest Hessian directions
  plot_basins         the distinct minima found, and how many starts reached each
  plot_weights        the saved designs' numbers, each rod named by the joints it connects
  plot_notation       which letter is which rod, which name is which joint (python jansen_viz.py -> png)
  plot_de_convergence the LHS screen and the differential-evolution runs
Kinematics come from verify_jansen.py (NumPy), so the pictures do not depend on the torch code.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, Circle
from jansen_opt_torch import T, N_ST, NAMES, MU_REQ, DE_LO, DE_HI, MOTION, X_BACK, Y_PIVOT
from verify_jansen import pose_np, travel_np, mu_np, DYADS, JOINTS, LINKS

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

    def __init__(self, ax, alpha=1.0, lw=2.5, ms=4, labels=None, lengths=False, fontsize=8):
        """labels: None, "joints" (O M Z P1..P4 F), "rods" (Jansen's letters, LINKS) or "both";
        lengths: rod labels read 'j=66.8' (measured from the pose, so they follow an LM run)."""
        self.plates = [ax.add_patch(Polygon(np.zeros((3, 2)), closed=True, fc=PLATE, ec="k", lw=lw * 0.6,
                                            alpha=alpha, zorder=2)) for _ in self.PLATES]
        self.rods = [ax.plot([], [], "o-", color="k", lw=lw, ms=ms, alpha=alpha, zorder=3)[0] for _ in self.RODS]
        self.crank = ax.add_patch(Circle((0, 0), 1, fill=False, ls="--", lw=0.8, ec="0.5", zorder=1))
        self.piv, = ax.plot([], [], "^", color="0.35", ms=ms + 6, zorder=4)
        self.ft, = ax.plot([], [], "o", color=BLUE, ms=ms + 4, zorder=5)
        self.lengths = lengths
        self.jl = [ax.annotate(k, (0, 0), textcoords="offset points", xytext=(5, 5), fontsize=fontsize + 1,
                               fontweight="bold", zorder=7, annotation_clip=False) for k in JOINTS] \
            if labels in ("joints", "both") else []
        self.rl = [ax.text(0, 0, nm, fontsize=fontsize, color=ORANGE, ha="center", va="center", zorder=6,
                           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=ORANGE, lw=0.6, alpha=0.9))
                   for _, _, nm in LINKS] if labels in ("rods", "both") else []

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
        for t, k in zip(self.jl, JOINTS):
            t.xy = (P[k].real, P[k].imag)
        for t, (a, b, nm), p in zip(self.rl, LINKS, rod_label_spots(P) if self.rl else []):
            t.set_position((p.real, p.imag))
            t.set_text(f"{nm}={abs(P[a] - P[b]):.1f}" if self.lengths else nm)
        return [*self.rods, *self.plates, self.crank, self.piv, self.ft, *self.jl, *self.rl]

def rod_label_spots(P):
    """Where to put each rod's label (LINKS order): the midpoint, or slid along the rod when another
    label is already close (rods that run nearly parallel, e.g. d and f)."""
    pts = [x for x in P.values() if np.isfinite(x)]
    span = np.ptp([x.real for x in pts]) + np.ptp([x.imag for x in pts]) if pts else 1.0
    placed = []
    for a, b, _ in LINKS:
        cand = [(1 - t) * P[a] + t * P[b] for t in (0.5, 0.3, 0.7, 0.2, 0.8)]
        gap = [min([abs(p - q) for q in placed] + [span]) for p in cand]
        placed.append(next((p for p, g in zip(cand, gap) if g > 0.08 * span), cand[int(np.argmax(np.nan_to_num(gap, nan=-1)))]))
    return placed

def targets(ax, labels=False):
    ax.axhline(0, color="k", lw=1)
    ax.plot(T_C[:N_ST].real, T_C[:N_ST].imag, "o", color=PINK, ms=7, label="stance targets (timed)")
    ax.plot(T_C[N_ST:].real, T_C[N_ST:].imag, "o", mfc="none", mec=PINK, ms=7, label="swing targets (untimed)")
    if labels:
        for n, t in enumerate(T_C):
            ax.annotate(str(n), (t.real, t.imag), textcoords="offset points", xytext=(5, 4), fontsize=8, color=PINK)

def pivot_rules(ax):
    """The pivot-placement rules: crank axle O left of X_BACK, O and Z above Y_PIVOT."""
    ax.axvline(X_BACK, color="0.6", ls="--", lw=0.9, zorder=0, label=f"O.x <= {X_BACK:.0f}")
    ax.axhline(Y_PIVOT, color="0.6", ls="-.", lw=0.9, zorder=0, label=f"O.y, Z.y >= {Y_PIVOT:.1f}")

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
    ax.set_title("targets T (jansen_opt.py), visited " + ("counter-clockwise" if MOTION > 0 else "clockwise"))
    return fig

def plot_best(v, hand, title=None):
    """Step 8: the leg at mid-stance, its foot path at every degree, and the miss at each target."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=72, gridspec_kw={"width_ratios": [1.4, 1]})
    turn = pose_np(v, TT, hand)
    Ck = pose_np(v, travel_np(v), hand)["F"]
    targets(ax)
    pivot_rules(ax)
    ax.plot(turn["F"].real, turn["F"].imag, color=BLUE, lw=1.5, label="foot, every degree")
    for c, t in zip(Ck, T_C):
        ax.plot([t.real, c.real], [t.imag, c.imag], color=PINK, lw=0.8)
    ax.plot(Ck.real, Ck.imag, ".", color=BLUE, ms=6)
    Leg(ax, labels="both", lengths=True).set(pose_np(v, mid_stance(v), hand))
    fit_view(ax, list(turn.values()) + [T_C])
    miss = abs(Ck - T_C)
    mu = min(np.nanmin(m) for m in dyad_mus(turn))
    ax.set_title(title or "rms miss %.2f mm | worst %.2f mm | mu_min %.1f deg | lowest Y %.2f mm"
                 % (np.sqrt(np.mean(miss ** 2)), miss.max(), mu, np.nanmin(turn["F"].imag)), fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=7, frameon=False)
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
        Leg(ax, lw=1.5, ms=2, labels="joints", fontsize=6).set(pose_np(v, mid_stance(v), hand))
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

def frame_dims(ax, P, v, origin=True):
    """Dotted dimension lines: ax, ay (frame pivot Z relative to the crank axle O, mirrored by hand)
    and, with origin, O = (hX, hY) in the body frame (floor at Y = 0)."""
    O, Z = P["O"], P["Z"]
    ax.plot([O.real, Z.real, Z.real], [O.imag, O.imag, Z.imag], ":", color="0.4", lw=1)
    ax.annotate(f"ax={v[1]:.1f}", ((O.real + Z.real) / 2, O.imag), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=8, color="0.3")
    ax.annotate(f"ay={v[2]:.1f}", (Z.real, (O.imag + Z.imag) / 2), textcoords="offset points",
                xytext=(-4, 0), ha="right", va="center", fontsize=8, color="0.3")
    if origin:
        ax.annotate(f"O = (hX, hY) = ({v[14]:.1f}, {v[15]:.1f})", (O.real, O.imag),
                    textcoords="offset points", xytext=(-10, -26), fontsize=8, color="0.3")

# rod letter -> (group, role); groups get one colour each in plot_notation
ROLES = dict(m=("crank", "crank: axle O to crank pin M, turns 360 deg"),
             j=("upper", "crank rod to P1"), b=("upper", "upper rocker, frame pivot Z to P1"),
             d=("upper", "upper triangle Z-P1-P2 (rigid, sides b d e)"), e=("upper", "upper triangle, P1 to P2"),
             k=("lower", "crank rod to P3"), c=("lower", "lower rocker, frame pivot Z to P3"),
             f=("coupler", "rod from the upper triangle (P2) to P4"),
             g=("foot", "foot triangle P3-P4-F (rigid, sides g h i)"), h=("foot", "foot triangle, P4 to foot F"),
             i=("foot", "foot triangle, P3 to foot F"))
GROUP_COLORS = dict(crank="#333333", upper=BLUE, lower="#4C9A2A", coupler=ORANGE, foot=PINK)

def notation_v(hand=+1):
    """Jansen's holy numbers as a design vector: crank axle O at (0, 0) in the body frame, theta0 = 0,
    stance step 20 deg, equal swing gaps (enough to draw the leg; it is not fitted to T)."""
    from jansen_opt_torch import HOLY
    v = np.zeros(len(NAMES))
    for n, nm in enumerate(NAMES[:13]):
        v[n] = HOLY[nm]
    v[16] = np.radians(20)
    return v

def plot_notation(v=None, hand=+1, dth=np.radians(60)):
    """Reference picture of the notation: which letter is which rod, which joint is which point, and
    what the other numbers of the design vector are. Default: Jansen's holy numbers."""
    v = notation_v(hand) if v is None else np.asarray(v, float)
    fig = plt.figure(figsize=(15, 7.5), dpi=72)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0])
    pose = pose_np(v, [dth], hand)
    P = {k: x[0] for k, x in pose.items()}
    turn = pose_np(v, TT, hand)
    ax.plot(turn["F"].real, turn["F"].imag, color=BLUE, lw=1, alpha=0.4, label="foot path F over one turn")
    for tri, col in zip(Leg.PLATES, (GROUP_COLORS["upper"], GROUP_COLORS["foot"])):
        ax.add_patch(Polygon([[P[t].real, P[t].imag] for t in tri], closed=True, fc=col, alpha=0.12, zorder=1))
    ax.add_patch(Circle((P["O"].real, P["O"].imag), abs(P["M"] - P["O"]), fill=False, ls="--", lw=0.8, ec="0.5"))
    for a, b, nm in LINKS:
        ax.plot([P[a].real, P[b].real], [P[a].imag, P[b].imag], "-", color=GROUP_COLORS[ROLES[nm][0]],
                lw=3, solid_capstyle="round", zorder=3)
    for (a, b, nm), p in zip(LINKS, rod_label_spots(P)):
        col = GROUP_COLORS[ROLES[nm][0]]
        ax.text(p.real, p.imag, nm, fontsize=13, fontweight="bold", color=col, ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=col, lw=1.2))
    for k, x in P.items():
        ax.plot(x.real, x.imag, "^" if k in ("O", "Z") else "o", color="0.35" if k in ("O", "Z") else "k",
                ms=12 if k in ("O", "Z") else 6, zorder=4)
        ax.annotate(k, (x.real, x.imag), textcoords="offset points", xytext=(8, 6), fontsize=12,
                    fontweight="bold", zorder=7)
    frame_dims(ax, P, v, origin=False)
    fit_view(ax, list(turn.values()) + [np.array(list(P.values()))])
    ax.set_title("Jansen leg notation (Jansen's holy numbers, hand %+d)" % hand, fontsize=11)
    ax.legend(loc="lower left", fontsize=8)

    bx = fig.add_subplot(gs[1]); bx.axis("off")
    y = 0.98
    bx.text(0, y, "rod   joints     role", family="monospace", fontsize=10, fontweight="bold", va="top")
    for a, b, nm in LINKS:
        y -= 0.05
        bx.text(0, y, f"{nm:<5s} {a + '-' + b:<10s} {ROLES[nm][1]}", family="monospace", fontsize=9.5,
                color=GROUP_COLORS[ROLES[nm][0]], va="top")
    notes = ["",
             "joints:  O crank axle (ground)   Z frame pivot (ground)",
             "         M crank pin   P1..P4 moving joints   F foot",
             "",
             "other numbers in the design vector v (27):",
             "  ax, ay   Z = O - (ax, ay); x mirrored when hand = -1",
             "  hX, hY   crank axle O in the body frame (floor at Y = 0)",
             "  theta0   crank angle at target 0",
             "  dth_s    crank step between the 4 stance targets",
             "  q1..q10  swing-gap logits: share of the rest of the turn",
             "",
             "pivot rules: O.x <= max target X (%.0f),  O.y, Z.y >= max target Y + 20 (%.1f)" % (X_BACK, Y_PIVOT),
             "",
             "hand +1: Z left of the crank (Jansen's drawing); -1: mirrored",
             "lengths in mm, angles in rad (deg in the tables)"]
    for line in notes:
        y -= 0.045
        bx.text(0, y, line, family="monospace", fontsize=9.5, va="top")
    fig.tight_layout()
    return fig

def plot_weights(top, n_cols=17):
    """What the saved numbers mean. Left: design #0 at mid-stance, every joint named and every rod
    labelled 'letter = length'. Right: the first n_cols numbers of each saved design, the column
    header naming the joints a rod joins; colour = where the number sits in its DE box (0 = lower
    bound, 1 = upper), so a number on a bound stands out. q1..q10 (timing logits) are in the files."""
    fig = plt.figure(figsize=(17, 1.6 + 0.42 * max(len(top), 4) + 4), dpi=72)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 2.3])
    ax = fig.add_subplot(gs[0])
    v, hand = np.asarray(top[0]["v"]), top[0]["hand"]
    turn = pose_np(v, TT, hand)
    pose = pose_np(v, mid_stance(v), hand)
    targets(ax)
    pivot_rules(ax)
    ax.plot(turn["F"].real, turn["F"].imag, color=BLUE, lw=1.2, alpha=0.6)
    Leg(ax, labels="both", lengths=True).set(pose)
    frame_dims(ax, {k: x[0] for k, x in pose.items()}, v)
    fit_view(ax, list(turn.values()) + [T_C])
    ax.set_title(f"design #0 (J={top[0]['J']:.3g}, hand {hand:+d}) at mid-stance, lengths in mm", fontsize=10)

    bx = fig.add_subplot(gs[1])
    V = np.stack([np.asarray(B["v"]) for B in top])[:, :n_cols]
    U = (V - DE_LO[:n_cols]) / (DE_HI[:n_cols] - DE_LO[:n_cols])
    bx.imshow(U, cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
    deg = [NAMES.index(nm) for nm in ("theta0", "dth_s")]
    for r in range(V.shape[0]):
        for c in range(n_cols):
            x = np.degrees(V[r, c]) if c in deg else V[r, c]
            bx.text(c, r, f"{x:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(U[r, c] - 0.5) > 0.4 else "black")
    joint = {a_nm: f"{a}-{b}" for a, b, a_nm in LINKS}
    heads = [f"{nm}\n{joint.get(nm, {'ax': 'O->Z x', 'ay': 'O->Z y', 'theta0': 'deg', 'hX': 'O x',
                                     'hY': 'O y', 'dth_s': 'deg'}.get(nm, ''))}" for nm in NAMES[:n_cols]]
    bx.set_xticks(range(n_cols)); bx.set_xticklabels(heads, fontsize=8)
    bx.xaxis.tick_top()
    bx.set_yticks(range(len(top)))
    bx.set_yticklabels([f"#{q} hand {B['hand']:+d}  J={B['J']:.3g}" for q, B in enumerate(top)], fontsize=8)
    bx.set_title("saved designs: rod lengths [mm] (header = joints it connects), O->Z offset, "
                 "crank axle O, timing\ncolour = position in the search box (blue = lower bound, "
                 "red = upper); q1..q10 swing-gap logits are in the files", fontsize=9, pad=36)
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
    leg = Leg(ax, labels="both")
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
    leg = Leg(ax, labels="both", lengths=True)
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

if __name__ == "__main__":
    import os, matplotlib
    matplotlib.use("Agg")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jansen_notation.png")
    plot_notation().savefig(path, dpi=110)
    print("wrote", path)
