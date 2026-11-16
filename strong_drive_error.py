#!/usr/bin/env python3
"""
Accuracy of the strong-drive approximation: relative deviation of the stationary concurrence C_{2,3}(inf)
(beta = 0, initial state |11 0...0>, drive alpha on qubit 1) from its value at alpha = 1000, for several n.
Uses the exact reduced model of symmetric_reduction.py.  Also checks that alpha = 1000 agrees with the
effective (alpha -> infinity) model.

Usage:  python strong_drive_error.py --outdir results [--quick]
Output: table_strong_drive_error.csv (n, alpha, C, rel. deviation in %, rel. deviation * alpha^2), table_strong_drive_limit_check.csv
"""
import argparse, os, time
import numpy as np
import collective_entanglement as ce
import symmetric_reduction as sr


def stat_C(n, alpha):
    v, k, gap, res = sr.steady_dense(sr.full_sym_liouvillian(n, alpha), sr.full_init(n))
    return sr.C23_full_sym(v, n), int(k)


def eff_C(n):
    v, k, gap, res = sr.steady_dense(sr.eff_sym_liouvillian(n), sr.eff_init(n))
    return sr.C23_eff_sym(v, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ref_alpha = 1000.0
    cfg = {4: [5, 10, 20, 50], 8: [5, 10, 20, 50]} if a.quick else \
          {4: [5, 10, 20, 50, 100, 200], 8: [5, 10, 20, 50, 100, 200], 12: [5, 10, 20, 50, 100, 200], 16: [20]}
    rows, lim = [], []
    for n, alphas in cfg.items():
        t0 = time.time()
        cref, _ = stat_C(n, ref_alpha)
        ceff = eff_C(n)
        lim.append([n, f"{cref:.8f}", f"{ceff:.8f}", f"{abs(cref - ceff):.2e}"])
        for al in alphas:
            c, k = stat_C(n, float(al))
            dev = (cref - c) / cref
            rows.append([n, al, f"{c:.8f}", f"{100 * dev:.4f}", f"{dev * al ** 2:.4f}", k])
        print(f"n={n}: C(alpha=1000)={cref:.6f}  C_eff={ceff:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    ce.write_csv(os.path.join(a.outdir, "table_strong_drive_error.csv"),
                 ["n", "alpha", "C_inf", "rel_dev_vs_alpha1000_percent", "rel_dev_times_alpha2", "null_dim"], rows)
    ce.write_csv(os.path.join(a.outdir, "table_strong_drive_limit_check.csv"),
                 ["n", "C_alpha1000", "C_effective_model", "abs_difference"], lim)


if __name__ == "__main__":
    main()
