"""Verify a torch-optimized leg independently with rmd.py at every degree (SF 4.6, Step 8)."""
import sys, numpy as np, torch
from rmd_synthesis.fb.rmd import FourBar
import rmd_synthesis.fb.leg_opt_torch as L
v = torch.load(sys.argv[1] if len(sys.argv) > 1 else "best_leg.pt")["v"].numpy()
a, b, c, d, p, dl, th1, th0, hX, hY = v
fb = FourBar(a, b, c, d)
Ls = sorted([a, b, c, d])
print("Grashof crank-rocker:", Ls[0] == a and Ls[0] + Ls[3] < Ls[1] + Ls[2], f"(s+l={Ls[0]+Ls[3]:.1f}, p+q={Ls[1]+Ls[2]:.1f})")
def body(dth):
    P = fb.coupler_point(th0 - dth, p, dl, branch=L.SIGMA)
    return (hX + 1j * hY) + np.exp(1j * th1) * P
C = body(L.dth_st.numpy()); T = L.T.numpy() @ [1, 1j]
print("stance miss rms / worst [mm]:", np.sqrt(np.mean(abs(C - T) ** 2)).round(2), abs(C - T).max().round(2))
st = body(np.radians(np.arange(0, 181)))
print("stance Y range [mm]:", st.imag.min().round(2), st.imag.max().round(2))
sw = body(np.radians(np.arange(181, 360)))
mid = sw[np.abs(np.arange(181, 360) - 270) <= 20]
print("swing: min Y over whole swing", sw.imag.min().round(2), "| min Y within 20 deg of mid-swing", mid.imag.min().round(2))
print("mu_min over turn [deg]:", np.degrees(fb.min_transmission_angle(L.SIGMA, n=3601)).round(2))
