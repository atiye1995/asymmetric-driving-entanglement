#!/usr/bin/env python3
"""
scaling_symmetric.py
====================
Large-n results for beta = 0 and the initial
state |1 1 0 ... 0>, using the exact permutation-symmetry reduction of
symmetric_reduction.py (state space C^2_1 (x) C^2_2 (x) Dicke_{n-2}).

Produces (in --outdir):
  table_scaling_n3_20.csv      field-free, drive alpha=20 (n<=12), effective alpha->inf (n<=20),
                               and the gain/cost decomposition of the mechanism
  table_alpha_scan_sym.csv     alpha scan for n = 5..12 (full model)
  scaling_fits.json            power-law fits on several n-windows, with uncertainties and
                               degrees of freedom, and local exponents
  scaling_symmetric.json       raw numbers
  fig6_scaling_n20.pdf         C(n), relative change, local exponent, gain vs cost

Mechanism decomposition (effective model, alpha -> infinity)
-------------------------------------------------------------
  C_ff      field-free value
  C_dec     value if qubit 1 simply decouples from the collective bath (undriven qubits
            2..n collective on their own) = 2(n-2)/(n-1)^2    (closed form, tested)
  C_eff     effective model: undriven qubits see qubit 1 as a random-telegraph coherent field
  gain      = C_dec - C_ff    (benefit of removing qubit 1 from the collective channel)
  cost      = C_dec - C_eff   (price of the random field of amplitude 1/2)
  net       = gain - cost = C_eff - C_ff    (> 0 for n >= 4, < 0 for n = 3)
"""
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import symmetric_reduction as sr


def fit_powerlaw(ns, cs):
    """C = a n^-b by least squares on log-log; returns b, its standard error, a, dof."""
    ns, cs = np.asarray(ns, float), np.asarray(cs, float)
    if len(ns) < 3:
        return None
    p, cov = np.polyfit(np.log(ns), np.log(cs), 1, cov=True)
    return dict(b=float(-p[0]), b_err=float(np.sqrt(cov[0, 0])), a=float(np.exp(p[1])),
                dof=int(len(ns) - 2), n_min=int(ns.min()), n_max=int(ns.max()),
                rms_log_residual=float(np.sqrt(np.mean((np.log(cs) - np.polyval(p, np.log(ns))) ** 2))))


def local_exponent(ns, cs):
    ns, cs = np.asarray(ns, float), np.asarray(cs, float)
    return (-(np.diff(np.log(cs)) / np.diff(np.log(ns)))).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--nmax-eff", type=int, default=20)
    ap.add_argument("--nmax-full", type=int, default=12)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    T0 = time.time()
    out = {"field_free": {}, "drive20": {}, "eff_alpha_inf": {}}

    def log(*a):
        print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

    for n in range(3, args.nmax_eff + 1):
        v, ok = sr.steady_evolution(sr.full_sym_liouvillian(n, 0.0), sr.full_init(n), t_max=800.0)
        out["field_free"][n] = dict(C=sr.C23_full_sym(v, n), converged=bool(ok))
        if n <= args.nmax_full:
            vd, k, gap, res = sr.steady_dense(sr.full_sym_liouvillian(n, 20.0), sr.full_init(n))
            out["drive20"][n] = dict(C=sr.C23_full_sym(vd, n), null_dim=int(k), residual=res)
        ve, ke, _, rese = sr.steady_dense(sr.eff_sym_liouvillian(n), sr.eff_init(n))
        out["eff_alpha_inf"][n] = dict(C=sr.C23_eff_sym(ve, n), null_dim=int(ke), residual=rese)
        log(f"n={n} ff={out['field_free'][n]['C']:.6f}({out['field_free'][n]['converged']}) "
            f"d20={out['drive20'].get(n, {}).get('C', float('nan')):.6f} eff={out['eff_alpha_inf'][n]['C']:.6f}")

    # ---- alpha scan with the full model for n = 5..nmax_full ----
    alphas = [1, 2, 5, 10, 20, 50, 100, 300]
    ascan = {}
    for n in range(5, args.nmax_full + 1):
        ascan[n] = {}
        for a in alphas:
            vd, k, gap, res = sr.steady_dense(sr.full_sym_liouvillian(n, float(a)), sr.full_init(n))
            ascan[n][a] = sr.C23_full_sym(vd, n)
        log(f"alpha scan n={n}:", {a: round(c, 5) for a, c in ascan[n].items()})
    out["alpha_scan_sym"] = {str(n): {str(a): c for a, c in d.items()} for n, d in ascan.items()}
    with open(f"{args.outdir}/table_alpha_scan_sym.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "alpha", "C_inf", "C_fieldfree", "change_percent"])
        for n in ascan:
            cff = out["field_free"][n]["C"]
            for a, c in ascan[n].items():
                w.writerow([n, a, c, cff, 100 * (c - cff) / cff])

    # ---- main table with mechanism decomposition ----
    rows = []
    for n in range(3, args.nmax_eff + 1):
        cff = out["field_free"][n]["C"]
        ceff = out["eff_alpha_inf"][n]["C"]
        cd20 = out["drive20"].get(n, {}).get("C", float("nan"))
        cdec = 2 * (n - 2) / (n - 1) ** 2
        rows.append([n, cff, cd20, ceff, 100 * (ceff - cff) / cff, cdec, cdec - cff, cdec - ceff, ceff - cff])
    with open(f"{args.outdir}/table_scaling_n3_20.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "C_fieldfree", "C_drive_alpha20(n<=12)", "C_eff_alpha_inf", "change_percent_eff",
                    "C_decoupled=2(n-2)/(n-1)^2", "gain_from_decoupling", "cost_of_random_field", "net_change"])
        w.writerows(rows)

    # ---- fits: several windows, uncertainties, local exponents ----
    ns_full = [n for n in range(4, args.nmax_full + 1)]
    c_full = [out["drive20"][n]["C"] for n in ns_full]
    ns_eff = list(range(4, args.nmax_eff + 1))
    c_eff = [out["eff_alpha_inf"][n]["C"] for n in ns_eff]
    ns_ff = list(range(4, args.nmax_eff + 1))
    c_ff = [out["field_free"][n]["C"] for n in ns_ff]
    fits = {"drive20_full": {}, "eff_alpha_inf": {}, "field_free": {}}
    for lo, hi in [(4, 6), (4, 8), (4, args.nmax_full), (8, args.nmax_full)]:
        sel = [n for n in ns_full if lo <= n <= hi]
        fits["drive20_full"][f"{lo}-{hi}"] = fit_powerlaw(sel, [out["drive20"][n]["C"] for n in sel])
    for lo, hi in [(4, 8), (4, args.nmax_eff), (10, args.nmax_eff)]:
        sel = [n for n in ns_eff if lo <= n <= hi]
        fits["eff_alpha_inf"][f"{lo}-{hi}"] = fit_powerlaw(sel, [out["eff_alpha_inf"][n]["C"] for n in sel])
        fits["field_free"][f"{lo}-{hi}"] = fit_powerlaw(sel, [out["field_free"][n]["C"] for n in sel])
    fits["local_exponent_drive20_full"] = dict(n_mid=[(a + b) / 2 for a, b in zip(ns_full[:-1], ns_full[1:])],
                                               b_local=local_exponent(ns_full, c_full))
    fits["local_exponent_eff"] = dict(n_mid=[(a + b) / 2 for a, b in zip(ns_eff[:-1], ns_eff[1:])],
                                      b_local=local_exponent(ns_eff, c_eff))
    fits["note"] = ("A pure power law is only an effective description: the local exponent drifts with n. "
                    "Report the fitted window together with b and its standard error; do not quote a 1/n law.")
    json.dump(fits, open(f"{args.outdir}/scaling_fits.json", "w"), indent=2)
    for k, v in fits["drive20_full"].items():
        log(f"fit drive20 n={k}: {v}")
    out["fits"] = fits
    json.dump(out, open(f"{args.outdir}/scaling_symmetric.json", "w"), indent=1)

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10})
        r = np.array(rows, float)
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        ax[0, 0].loglog(r[:, 0], r[:, 1], "o-", label="field-free")
        ax[0, 0].loglog(r[:, 0], r[:, 2], "s-", label=r"drive $\alpha=20$ (full model)")
        ax[0, 0].loglog(r[:, 0], r[:, 3], "^--", label=r"effective, $\alpha\to\infty$")
        ax[0, 0].set_xlabel("n"); ax[0, 0].set_ylabel(r"$C_{2,3}(\infty)$"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3, which="both")
        ax[0, 1].plot(r[:, 0], r[:, 4], "^-")
        ax[0, 1].axhline(0, color="k", lw=.8); ax[0, 1].set_xlabel("n")
        ax[0, 1].set_ylabel(r"change vs field-free [%] (effective, $\alpha\to\infty$)"); ax[0, 1].grid(alpha=.3)
        ax[1, 0].plot(fits["local_exponent_drive20_full"]["n_mid"], fits["local_exponent_drive20_full"]["b_local"], "s-", label=r"drive $\alpha=20$")
        ax[1, 0].plot(fits["local_exponent_eff"]["n_mid"], fits["local_exponent_eff"]["b_local"], "^--", label=r"effective $\alpha\to\infty$")
        ax[1, 0].set_xlabel("n"); ax[1, 0].set_ylabel(r"local exponent $b(n)=-d\ln C/d\ln n$"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
        ax[1, 1].plot(r[:, 0], r[:, 6], "o-", label="gain (decoupling qubit 1)")
        ax[1, 1].plot(r[:, 0], r[:, 7], "s-", label="cost (random field)")
        ax[1, 1].plot(r[:, 0], r[:, 8], "k^-", label="net = gain - cost")
        ax[1, 1].axhline(0, color="k", lw=.8); ax[1, 1].set_xlabel("n"); ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{args.outdir}/fig6_scaling_n20.pdf"); plt.close(fig)
    log("DONE")


if __name__ == "__main__":
    main()
