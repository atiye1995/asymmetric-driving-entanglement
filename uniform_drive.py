#!/usr/bin/env python3
"""Uniform-drive baseline (drive on ALL qubits, beta=0): stationary C_{2,3} from |11 0...0>.
Writes results/table_uniform_drive.csv.  Usage: python3 uniform_drive.py [--outdir results]"""
import argparse, csv, os
import collective_entanglement as ce

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="results"); a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
rows = []
for n in (3, 4, 5):
    for alpha in (0.3, 0.5, 1, 2, 3, 5, 10, 20):
        v, info = ce.steady_state(n, ce.liouvillian(n, alpha, list(range(n))), ce.default_label(n))
        rows.append([n, alpha, ce.C23(v, n), info["null_dim"], info["residual"]])
        print(n, alpha, f"{rows[-1][2]:.6f}", info["null_dim"], flush=True)
with open(os.path.join(a.outdir, "table_uniform_drive.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["n", "alpha", "C_inf_uniform_drive", "null_dim", "residual"]); w.writerows(rows)
# crossing point of the asymmetric drive (n=4) with the field-free value 1/4
import numpy as np
from scipy.optimize import brentq
f = lambda al: ce.C23(ce.steady_state(4, ce.liouvillian(4, al, [0]), "1100")[0], 4) - 0.25
print("n=4 asymmetric drive crosses 1/4 at alpha =", round(brentq(f, 1.0, 2.0, xtol=1e-4), 3))
