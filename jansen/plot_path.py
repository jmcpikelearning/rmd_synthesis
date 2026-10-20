"""PNG of a Jansen leg's target points and foot path only (no linkage), with the floor Y = 0.
usage:  python plot_path.py [best_jansen.npz | best_jansen.pt] [--out FILE.png]   (default best_jansen.npz)
        python plot_path.py --holy [--hand -1]                                  (Jansen's own numbers, placed on T)
Kinematics from verify_jansen.py (NumPy), colours from jansen_viz.py, so it matches the other figures."""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verify_jansen import load, pose_np
from jansen_viz import T_C, N_ST, BLUE, PINK

def plot_path(v, hand, step=1.0, title=None):
    """Floor, the 14 targets (stance filled, swing hollow) and the closed foot path at every `step` deg."""
    F = pose_np(v, np.radians(np.arange(0, 360 + step / 2, step)), hand)["F"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(F.real, F.imag, color=BLUE, lw=1.8, label="foot path")
    ax.plot(T_C[:N_ST].real, T_C[:N_ST].imag, "o", color=PINK, ms=7, label="stance targets (timed)")
    ax.plot(T_C[N_ST:].real, T_C[N_ST:].imag, "o", mfc="none", mec=PINK, ms=7, label="swing targets (untimed)")
    ax.set_aspect("equal")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
    ax.legend(fontsize=8, loc="upper left")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("path", nargs="?", default="best_jansen.npz", help="result saved by jansen_opt_torch.py")
    pa.add_argument("--holy", action="store_true", help="plot Jansen's holy numbers placed on T instead")
    pa.add_argument("--hand", type=int, default=+1, choices=[+1, -1], help="frame-pivot side, for --holy")
    pa.add_argument("--out", default=None, help="PNG file (default: <path stem>_path.png, holy_path.png)")
    pa.add_argument("--step", type=float, default=1.0, help="crank step of the foot path [deg]")
    pa.add_argument("--dpi", type=int, default=150)
    a = pa.parse_args()
    if a.holy:
        import jansen_opt_torch as JO
        JO.set_hand(a.hand)
        v, hand, name = JO.holy_v().numpy(), a.hand, "Jansen's holy numbers"
        out = a.out or "holy_path.png"
    else:
        v, hand, _ = load(a.path)
        name = os.path.basename(a.path)
        out = a.out or os.path.splitext(a.path)[0] + "_path.png"
    plot_path(v, hand, a.step, f"{name}, hand {hand:+d}").savefig(out, dpi=a.dpi)
    print("saved", out)
