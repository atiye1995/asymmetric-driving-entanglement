#!/usr/bin/env python3
"""
independent_decay.py
====================
Transient and stationary behaviour with INDEPENDENT (non-collective) decay of every qubit,
    + gamma_ind * sum_i D[sigma^-_i]          (T1 = 1/gamma_ind),
on top of the collective decay (gamma_c = 1) and optional pure dephasing beta. The collective term is
2*gamma_c*D[J_-], so the ratio of non-collective to collective single-qubit emission rate is gamma_ind/(2*gamma_c).

Outputs (in --outdir)
  table_independent_decay.csv            peak, lifetime, sudden death, ratios, time integrals (short and long window)
  table_independent_decay_stationary.csv null-space dimension and signed C_inf with gamma_ind > 0
  fig8_independent_decay.pdf             peak ratio and lifetime ratio vs n
The permutation-invariant construction with gamma_ind is first validated against the full-space code
(n = 3, 4; transient curves); the script aborts on a mismatch above 1e-9.

Usage
  python independent_decay.py --outdir results [--quick] [--nmax N]
"""
import argparse, math, os, sys, time
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collective_entanglement as ce
import extended_results as er

CASES = [  # (beta, gamma_ind, label)
    (1e-3, 0.0, "pure dephasing beta=1e-3 (reference)"),
    (0.0, 1e-3, "gamma_ind=1e-3"),
    (0.0, 1e-2, "gamma_ind=1e-2"),
    (0.0, 1e-1, "gamma_ind=1e-1"),
    (1e-3, 1e-2, "beta=1e-3 + gamma_ind=1e-2"),
]


def window_T(beta, g):
    return max(60.0, 2.5 / (beta + g)) if beta + g > 0 else 400.0


def run_case(label, alpha, beta, g, scale):
    S = er.PISystem(label); red = S.reducer()
    L = S.liouvillian(alpha, beta, g)
    T = window_T(beta, g)
    seg = [(0.05 * scale, 40.0), (0.5 * scale, math.ceil(T / 0.5) * 0.5)]
    if S.dim <= er.DENSE_EXPM_LIMIT:
        return er.transient_pi(S, red, L.toarray(), seg)
    return er.transient_pi_sparse(S, red, L, seg)


def metrics(label, beta, g, halving):
    ts0, c0, r0 = run_case(label, 0.0, beta, g, 1.0)
    tsd, cd, rd = run_case(label, 20.0, beta, g, 1.0)
    p0, tp0, tau0, _ = ce.peak_and_lifetime(ts0, c0)
    pd, tpd, taud, _ = ce.peak_and_lifetime(tsd, cd)
    w = ce.advantage_window(tsd, cd, c0)                               # long window T = 2.5/(beta+g) (~9 lifetimes)
    TI = max(60.0, 0.8 / (beta + g)) if beta + g > 0 else 400.0       # short window = convention of table2_extended.csv
    m = tsd <= TI + 1e-9
    w_short = ce.advantage_window(tsd[m], cd[m], c0[m])
    out = dict(C0=p0, tau0=tau0, Cd=pd, taud=taud, tdeath=ce.death_time(tsd, rd, cd),
               win_end=w["windows"][-1][1] if w["windows"] else float("nan"), integral=w["integral"], integral_short=w_short["integral"],
               C_err=float("nan"), tau_err=float("nan"))
    if halving:
        tsb, cdb, _ = run_case(label, 20.0, beta, g, 0.5)
        pdb, _, taudb, _ = ce.peak_and_lifetime(tsb, cdb)
        out["C_err"] = abs(pd - pdb)
        out["tau_err"] = abs(taud - taudb) if np.isfinite(taud) and np.isfinite(taudb) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nmax", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    T0 = time.time()
    log = lambda *x: print(f"[{time.time()-T0:7.1f}s]", *x, flush=True)

    # ---- validation: PI with gamma_ind vs full space ----
    worst = 0.0
    for n, al, b, g in ((3, 20.0, 1e-3, 1e-2), (4, 20.0, 0.0, 1e-2), (4, 20.0, 1e-3, 3e-2), (4, 0.0, 0.0, 1e-2)):
        lab = ce.default_label(n); S = er.PISystem(lab); red = S.reducer()
        EF = expm(ce.liouvillian(n, al, [0], b, g).toarray() * 0.5); EP = expm(S.liouvillian(al, b, g).toarray() * 0.5)
        vF, vP = ce.init_vec(lab), S.init_vec()
        for _ in range(80):
            vF, vP = EF @ vF, EP @ vP
            worst = max(worst, abs(ce.C23(vF, n) - er.C_pi(S, red, vP)))
    log(f"validation PI vs full space with independent decay: max|dC| = {worst:.2e}")
    assert worst < 1e-9, "validation failed"

    # ---- stationary state with gamma_ind > 0 (unique; signed concurrence) ----
    rows = []
    stat_cases = [(0.0, 1e-3), (0.0, 1e-2), (1e-4, 1e-2), (1e-3, 1e-2)]
    for n in ([3, 4, 5, 6] if a.quick else [3, 4, 5, 6, 7, 8]):
        lab = ce.default_label(n); S = er.PISystem(lab); red = S.reducer()
        todo = [(al, b, g) for al in (0.0, 20.0) for (b, g) in stat_cases]
        if n <= 5:
            todo += [(5.0, 0.0, 1e-2), (100.0, 0.0, 1e-2)]
        for al, b, g in todo:
            v, info = er.stationary_pi(S, S.liouvillian(al, b, g), S.init_vec())
            c = er.C_pi(S, red, v, raw=True)
            rows.append([n, al, b, g, info["null_dim"], c])
            log(f"  stationary n={n} alpha={al} beta={b} gamma_ind={g}: null_dim={info['null_dim']}  signed C_inf={c:+.4e}")
            ce.write_csv(f"{a.outdir}/table_independent_decay_stationary.csv",
                         ["n", "alpha", "beta", "gamma_ind", "null_dim", "signed_C_inf (>0: entangled)"], rows)

    # ---- transient table ----
    ns = [3, 4, 5, 6] if a.quick else [x for x in (3, 4, 5, 6, 8, 10, 12) if x <= a.nmax]
    table, res = [], {}
    for n in ns:
        lab = ce.default_label(n)
        for b, g, name in CASES:
            if n >= 12 and g < 1e-2 and b == 0:      # long windows at large n: skipped (cost)
                continue
            t0 = time.time()
            m = metrics(lab, b, g, halving=(n <= 6))
            res[(n, b, g)] = m
            tr = m["taud"] / m["tau0"] if np.isfinite(m["tau0"]) and np.isfinite(m["taud"]) else float("nan")
            table.append([n, b, g, g / 2.0, m["C0"], m["tau0"], m["Cd"], m["taud"], m["Cd"] / m["C0"], tr,
                          m["tdeath"], m["win_end"], m["integral_short"], m["integral"], m["C_err"], m["tau_err"]])
            log(f"  n={n:2d} beta={b:g} g_ind={g:g}: Cpk {m['C0']:.4f}->{m['Cd']:.4f} ({m['Cd']/m['C0']:.3f})  "
                f"tau {ce.fmt(m['tau0'],1)}->{ce.fmt(m['taud'],1)} ({tr:.3f})  I_short={m['integral_short']:+.2f} I_long={m['integral']:+.2f}  ({time.time()-t0:.0f}s)")
            ce.write_csv(f"{a.outdir}/table_independent_decay.csv",
                         ["n", "beta", "gamma_ind", "gamma_ind/(2*gamma_c)", "Cpeak_nodrive", "tau_nodrive", "Cpeak_drive(alpha=20)",
                          "tau_drive", "Cpeak_ratio", "tau_ratio", "t_death_drive", "advantage_window_end", "integral_dC_dt_short(0.8/(beta+g))", "integral_dC_dt_long(2.5/(beta+g))",
                          "Cpeak_err(grid halving)", "tau_err"], table)

    # ---- figure ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for (b, g, name), mk in zip(CASES, "os^dv"):
        pts = [(r[0], r[8], r[9]) for r in table if r[1] == b and r[2] == g]
        if not pts: continue
        x = [p[0] for p in pts]
        ax[0].plot(x, [p[1] for p in pts], mk + "-", label=name)
        ax[1].plot(x, [p[2] for p in pts], mk + "-", label=name)
    for k, yl in enumerate((r"$C_{\rm peak}^{\rm drive}/C_{\rm peak}^{\rm no\ drive}$", r"$\tau^{\rm drive}/\tau^{\rm no\ drive}$")):
        ax[k].axhline(1, color="k", lw=.8); ax[k].set_xlabel("n"); ax[k].set_ylabel(yl); ax[k].grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig8_independent_decay.pdf"); plt.close(fig)
    log("DONE")


if __name__ == "__main__":
    main()
