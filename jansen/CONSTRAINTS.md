# Constraints in the Jansen-leg optimizer

This is a reference for every rule that `jansen_opt_torch.py` builds into its cost function. For each
rule it gives the math, where it comes from, why it has the form it has, and the command-line flags
that change it. Line references are to `jansen_opt_torch.py`.

Quick start:

```
python jansen_opt_torch.py --help                      # every flag, with its default
python jansen_opt_torch.py --tri-min 25 --eq-qmin 0.85 --bound b=20,90 --mu-req 45
python verify_jansen.py best_jansen.npz                # independent re-check against the saved settings
python plot_path.py best_jansen.npz                    # PNG of the targets and the foot path only
```

---

## 1. One cost function, many residuals

The design vector $v$ has 27 numbers: 13 lengths, the crank phase $\theta_0$, the crank-axle position
$(h_X, h_Y)$, the stance crank step $\Delta\theta_s$ and 10 swing-timing logits $q_1..q_{10}$. Every
requirement is written as a **residual** $r_i(v)$, and the optimizer minimizes

$$
J(v) \;=\; \sum_i r_i(v)^2 \;=\; \lVert r(v)\rVert^2 .
$$

**Why residuals, not hard constraints (SF 4.3).** Levenberg-Marquardt and SciPy's `least_squares`
are Gauss-Newton methods. They need the residual vector and its Jacobian $A = \partial r/\partial v$,
not just $J$, and they build the step from $A^\top A$ and $A^\top r$. A rule that is a residual
needs no special constraint machinery. Torch differentiates it exactly (`jacfwd`), and LM handles it
the same way as the target misses.

**Why sloped hinges (SF 4.6).** A violated rule used to set $J$ to a large constant (a flat "reject
plateau"). That constant has zero gradient, so a start that landed on the plateau could never leave.
Every rule here is instead a hinge,

$$
r = w\,\max(0,\ \text{violation}),
$$

which is zero inside the feasible set and grows linearly with the violation outside it. Outside the
feasible set the gradient therefore always points back toward feasibility.

**Units and weights.** The 28 miss residuals are scaled so that their $\sum r^2$ is the mean squared
target miss in mm² (section 2). One unit of any residual therefore costs as much as 1 mm of RMS miss,
and a weight $w$ reads as "how many mm of miss one unit of violation is worth". For example,
$W_{rej}=10$ means 1 mm of linkage that does not close costs as much as 10 mm of RMS miss.

**Why margins.** At a minimum where a hinge is active, the pull of the fit balances the pull of the
penalty, so a squared hinge always lets a little violation through. The amount is roughly the fit's
gradient divided by $w^2$. Each rule therefore aims a margin $m$ inside the true limit, and the
small leftover violation then lands inside the real rule.

**Soft min / soft max.** Rules that must hold over the whole crank turn are checked on a grid of
$N$ crank angles (`t_turn`, every `--turn-step` deg, so $N = 180$ by default). A hard
$\min_t$ has kinks, and the Hessian test of `hessian_report` needs second derivatives. The code
therefore uses log-sum-exp with a temperature $\tau$:

$$
\operatorname{softmin}_\tau(x_t) = -\tau \log\sum_t e^{-x_t/\tau},\qquad
\min_t x_t - \tau\ln N \;\le\; \operatorname{softmin}_\tau \;\le\; \min_t x_t .
$$

The soft min is never larger than the true min, so the rule is **conservative**, by at most
$\tau\ln N$. The soft max is the mirror image and is never smaller than the true max.

---

## 2. Target miss (the objective)

This term is the objective rather than a rule, but the other weights are measured against it. With
$n = 14$ targets $T_k$ and the foot $F$ in the body frame:

$$
r^{\text{miss}}_{k} = \frac{F\!\left(\theta_0 + \mathrm{DIR}\cdot\Delta\theta_k\right) - T_k}{\sqrt{n}}
\quad(\text{X and Y: } 28 \text{ residuals}),\qquad
\sum_k \lVert r^{\text{miss}}_k\rVert^2 = \frac1n\sum_k \lVert F_k - T_k\rVert^2 .
$$

(lines 168-171; timing in `travel`, line 96)

- **Stance timing.** The four floor targets are reached at $\Delta\theta_k = k\,\Delta\theta_s$
  ($k = 0..3$), which gives equal steps along the ground, i.e. constant ground speed.
- **Swing timing.** The other 10 targets have no fixed timing. The remaining turn
  $2\pi - 3\Delta\theta_s$ is split into 11 gaps by a softmax,
  $\text{gap}_j = (2\pi - 3\Delta\theta_s)\,\mathrm{softmax}(0, q_1, \dots, q_{10})_j$.
  The targets therefore stay in order and the loop closes by construction. The first logit is
  pinned at 0 because softmax is shift-invariant, so a free first logit would add an exactly flat
  direction, i.e. a zero Hessian eigenvalue.

---

## 3. The updatable rules

### 3.1 Transmission angle (`mu_P1`, `mu_P3`, `mu_P4`)

**Where it applies.** It applies at the three moving dyads: P1 (crank rod $j$ and rocker $b$, from
M and Z), P3 ($k$ and $c$, from M and Z), and P4 ($f$ and $g$, from P2 and P3). P2 and F sit on rigid
plates, so they have no transmission angle of their own.

**Math.** At a joint X with rods $l_p$, $l_q$ to points P, Q a distance $s$ apart, the law of
cosines gives

$$
\cos\mu = \frac{l_p^2 + l_q^2 - s^2}{2\,l_p l_q}.
$$

The code (`mu_acute`, line 143) uses the branch-free, always-acute form

$$
\mu = \operatorname{atan2}\!\Big(\sqrt{4l_p^2l_q^2 - (l_p^2+l_q^2-s^2)^2},\ \big|\,l_p^2+l_q^2-s^2\,\big|\Big),
$$

since $4l_p^2l_q^2 - (\cdot)^2 = (2l_pl_q\sin\mu)^2$. The angle takes a soft min over the turn, and
the residual is 2 per degree short:

$$
\mu_{\min} = \operatorname{softmin}_{\tau_\mu}\big(\mu(t)\big),\qquad
r^{\mu} = 2\,\max\!\big(0,\ \mu_{req} - \mu_{\min}\big)\ [\text{deg}].
$$

(lines 176-177)

**Where it comes from.** The transmission angle (Alt) measures how much of a rod's force turns the
next link rather than loading the pivot. As $\mu \to 0$ the joint approaches a toggle: forces blow
up, small length errors give large position errors, and the leg can jam. The usual design guideline
is $\mu_{\min} \gtrsim 40^\circ$ (e.g. Norton, *Design of Machinery*; KF 11.3 and SF 4.9 for this
leg).

**Why this form.** The acute angle is used because 140° is as good as 40°. The atan2 form keeps the
gradient finite everywhere. The soft min keeps the Hessian defined, at a conservative cost of at
most $\tau_\mu\ln N = 0.25^\circ\cdot\ln 180 \approx 1.3^\circ$.

| flag | default | effect |
|---|---|---|
| `--mu-req` | 40 deg | the required minimum; higher gives a stiffer, more robust leg but a worse fit |
| `--tau-mu` | 0.25 deg | soft-min temperature; smaller is closer to the true min but less smooth |
| `--turn-step` | 2 deg | crank sampling of the turn (shared with sections 3.2 and 3.4) |

### 3.2 Assembly / toggle margin (`asm_P1`, `asm_P3`, `asm_P4`)

**Math.** Circles of radius $l_p$ about P and $l_q$ about Q meet only if
$\lvert l_p - l_q\rvert \le s \le l_p + l_q$. The code measures the excess as a signed gap, which
is positive when the dyad cannot close:

$$
g(t) = \max\!\big(s - (l_p+l_q),\ \sqrt{(l_p-l_q)^2+\varepsilon} - s\big),\qquad
g_{\max} = \operatorname{softmax}_{\tau_{mm}}\big(g(t)\big),
$$

$$
r^{asm} = W_{rej}\,\max\!\big(0,\ g_{\max} + m\big).
$$

(lines 178-180)

**Where it comes from.** A leg that fails to assemble at any crank angle cannot run a full turn.
`dyad()` clamps the square root so that its gradient stays finite, which means it returns a point
even where the circles do not meet. This residual is what actually rules such legs out.

**Why this form.** The margin $m$ keeps every dyad at least $m$ mm away from both the
non-assembly and the toggle (fully stretched or folded) positions. The smooth
$\sqrt{(\cdot)^2+\varepsilon}$ replaces $\lvert\cdot\rvert$, and the soft max makes the check
conservative.

| flag | default | effect |
|---|---|---|
| `--w-rej` | 10 per mm | weight (shared with 3.3) |
| `--margin` | 1 mm | distance kept from the limit (shared with 3.3) |
| `--tau-mm` | 0.25 mm | soft-max temperature |

### 3.3 Rigid-triangle closure (`tri_bde`, `tri_ghi`)

**Math.** The plates b-d-e and g-h-i must obey the triangle inequality, which here reads
$l_{\max} < \sum l - l_{\max}$:

$$
r^{tri} = W_{rej}\,\max\!\big(0,\ l_{\max} - (\textstyle\sum l - l_{\max}) + m\big).
$$

(line 184)

**Why.** Without it, three lengths that cannot form a plate could still score well, because the
clamped kinematics would hide the problem. It is a pure existence rule: for that, the plate's shape
does not matter. Shape is handled in 3.5.

Flags: `--w-rej`, `--margin` (as above).

### 3.4 Ground clearance (`ground0` .. `ground358`)

**Math.** There is one hinge per crank sample. The foot's body-frame height is $h_Y + F_y(t)$:

$$
r^{gr}_t = \frac{W_{g}}{\sqrt N}\,\max\!\big(0,\ g_m - (h_Y + F_y(t))\big),\qquad
\sum_t (r^{gr}_t)^2 = W_g^2\cdot\overline{\text{depth}^2}.
$$

(line 191)

**Where it comes from.** The foot must never go below the floor. In C13, checking only the
targets let the foot pass 27 mm through the floor between them.

**Why this form.** One residual per sample gives LM a piecewise-linear set of residuals. That suits
Gauss-Newton better than a single soft min of depth, and it pushes every offending sample. Dividing
by $\sqrt N$ makes $W_g$ a weight on the RMS depth, so the meaning of $W_g$ does not change with
`--turn-step`. The aim is $g_m = 0.05$ mm above the floor so that the leftover from the squared hinge
stays at or above $Y = 0$ (verify reports a lowest Y of about 0.045 mm).

| flag | default | effect |
|---|---|---|
| `--w-ground` | 30 per mm (RMS) | weight |
| `--g-margin` | 0.05 mm | aim height above the floor |
| `--turn-step` | 2 deg | the number of hinges is $N = 360/\text{step}$ |

### 3.5 Triangle shape — penalty for straying from equilateral (`eq_bde`, `eq_ghi`)

**Math.** The shape quality of a triangle with sides $a, b, c$ and area $A$ is

$$
q = \frac{4\sqrt3\,A}{a^2+b^2+c^2},\qquad
16A^2 = (a+b+c)(-a+b+c)(a-b+c)(a+b-c)\ \ (\text{Heron}),
$$

$$
r^{eq} = W_{eq}\,\max\!\big(0,\ q_{\min} - q\big).
$$

(`tri_quality`, line 149; residual line 186)

**Why $q$.** Weitzenböck's inequality $a^2+b^2+c^2 \ge 4\sqrt3\,A$, with equality only for the
equilateral triangle, gives $0 < q \le 1$ with $q = 1$ exactly when the triangle is equilateral.
$q \to 0$ for a sliver, whether one angle closes up or the triangle flattens. $q$ does not depend on
scale, it is smooth in the side lengths, and it needs no angles. It is the standard
triangle-quality measure from finite-element meshing (Bank & Smith).

| triangle | $q$ | smallest angle |
|---|---|---|
| equilateral | 1.000 | 60° |
| right isosceles | 0.866 | 45° |
| Jansen's b-d-e (41.5, 40.1, 55.8) | 0.893 | 45.8° |
| Jansen's g-h-i (36.7, 65.7, 49.0) | 0.763 | 33.5° |
| 30-60-90 | 0.750 | 30° |
| sliver (1, 1, 1.9) | 0.366 | 18.2° |

**Where it comes from.** Manufacturability. A thin plate has two pin holes close to a long edge,
little material between the holes, low bending stiffness, and joint positions that are sensitive to
hole tolerances. Unconstrained, the optimizer drifts toward slivers: in a smoke run with this rule
active, both plates finished exactly on the limit $q = 0.75$.

**Why a dead-band.** Jansen's own foot plate has $q = 0.76$, so requiring equilateral plates would
reject good, buildable legs and trade away fit for nothing. With the defaults, only shapes worse than
30-60-90 cost anything, and the cost then grows linearly. Setting `--eq-qmin 1` turns this into an
always-on pull toward equilateral.

| flag | default | effect |
|---|---|---|
| `--w-eq` | 20 per unit of $q$ | weight; 0 turns it off. $q$ 0.1 short costs as much as 2 mm of RMS miss |
| `--eq-qmin` | 0.75 | the shape allowed at no cost; 1 means always pull toward equilateral |

### 3.6 Minimum triangle side (`len_b`, `len_d`, `len_e`, `len_g`, `len_h`, `len_i`)

**Math.** There is one hinge per plate side $s \in \{b, d, e, g, h, i\}$:

$$
r^{len}_s = W_{len}\,\max\!\big(0,\ L_{\min} + m_L - s\big).
$$

(line 187)

**Where it comes from.** Manufacturability again. Each plate corner carries a pin or bearing, and
two of them need a minimum centre distance for hole diameter, edge distance and clearance. The
default is 20 mm. The q rule bounds the plate's shape and this rule bounds its size, so together
they stop tiny or needle-thin plates.

**Why this form.** It is a hinge like the others. The aim $m_L = 0.5$ mm above $L_{\min}$ absorbs
the leftover from the squared hinge. It is a soft rule, not a bound, so DE and LM may cross it
briefly while searching. If it must be a hard limit, raise the box instead, e.g. `--bound b=20,120`
(section 3.7).

| flag | default | effect |
|---|---|---|
| `--w-len` | 10 per mm | weight; 0 turns it off |
| `--tri-min` | 20 mm | minimum plate side |
| `--len-margin` | 0.5 mm | aim above `--tri-min` |

### 3.7 Design box (hard bounds)

These are not residuals: the parametrization makes them hard limits.

- **LM** works in unconstrained $z$, with $v = lo + (hi - lo)\,\sigma(z)$ (`unpack`, line 81), so it
  can never leave the box.
- **`least_squares`** (`refine`) works in $v$ with the true bounds, because $\sigma$ flattens near a
  bound and LM crawls there.
- **DE and LHS** sample inside the same box. The crank phase $\theta_0$ is free (DE samples
  $\pm\pi$). The gap logits are sampled in $\pm Q_{max}$, refined in $\pm 2Q_{max}$, and are
  unbounded in LM.

| number | low | high | | number | low | high |
|---|---|---|---|---|---|---|
| m (crank) | 5 | 30 | | b..k (10 rods) | 10 | 120 |
| ax | 10 | 80 | | hX | −60 | 120 |
| ay | −30 | 30 | | hY | 30 | 200 |
| dth_s | 10° | 70° | | gap logits | −3 | 3 |

`hessian_report` lists the numbers that sit on a bound and says whether J would still fall past them,
i.e. which bounds to widen.

| flag | default | effect |
|---|---|---|
| `--bound NAME=LO,HI` | table above | repeatable; NAME is one of `m ax ay b c d e f g h i j k hX hY dth_s` (dth_s in deg) |
| `--q-max` | 3 | DE box for the gap logits; the largest gap ratio is $e^{2Q_{max}}$ |

---

## 4. Summary

| residuals (`RES_NAMES`) | count | formula | flags |
|---|---|---|---|
| `miss{k}{X,Y}` | 28 | $(F_k - T_k)/\sqrt{14}$ | (objective) |
| `mu_P1 mu_P3 mu_P4` | 3 | $2\max(0,\mu_{req}-\operatorname{softmin}\mu)$ | `--mu-req --tau-mu --turn-step` |
| `asm_P1 asm_P3 asm_P4` | 3 | $W_{rej}\max(0,\operatorname{softmax} g + m)$ | `--w-rej --margin --tau-mm` |
| `tri_bde tri_ghi` | 2 | $W_{rej}\max(0,2l_{\max}-\sum l+m)$ | `--w-rej --margin` |
| `ground{deg}` | 180 | $W_g\max(0,g_m - Y_t)/\sqrt N$ | `--w-ground --g-margin --turn-step` |
| `eq_bde eq_ghi` | 2 | $W_{eq}\max(0,q_{\min}-q)$ | `--w-eq --eq-qmin` |
| `len_b .. len_i` | 6 | $W_{len}\max(0,L_{\min}+m_L-s)$ | `--w-len --tri-min --len-margin` |
| (box) | — | $lo \le v \le hi$ | `--bound --q-max` |

That is 224 residuals by default. The ground count changes with `--turn-step`.

The settings in force are saved with every result (`cfg` in `best_jansen.pt` and `.npz`). In Python,
`jansen_opt_torch.config()` returns them and `jansen_opt_torch.configure(...)` sets them.

## 5. Fixed in code (not on the command line)

- the target loop `T_LOOP` and its direction `MOTION` (the foot goes round clockwise), and the
  number of timed stance targets `N_ST = 4`;
- the assembly branches `SIGMA`, which pick the classic leg; the hand is chosen with `--hands`;
- the transmission-angle residual's scale of 2 per degree, and the $1/\sqrt n$, $1/\sqrt N$
  normalizations, which define the units in section 1;
- `EPS = 1e-12` in the clamped square roots.

## 6. Checking a result

- `python verify_jansen.py best_jansen.npz` re-derives the leg in NumPy at every 0.1° with no clamps.
  It checks assembly, link lengths, $\mu_{\min}$, toggle margins, misses, lowest Y, and each plate's
  sides, smallest angle and $q$, all against the settings saved with that result.
- The basin table printed during a run has `tri q` and `side` columns.
- The `hessian_report` line "active penalties" names the rules that are binding at the minimum.
  These are the ones to loosen if the fit matters more.
