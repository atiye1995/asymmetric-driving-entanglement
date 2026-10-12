#!/usr/bin/env python3
"""
peak_map.py
===========
(1) Transient peak concurrence C_peak(alpha, beta) of the pair (2,3), n = 4 (drive on qubit 1, initial state |1100>,
    pure dephasing beta), on a log-log grid, with the ratio to the undriven peak.
(2) Peak ratio versus alpha at beta = 1e-3 for n = 4, 5, 6, 8.
(3) Liouvillian dimension: full space (4^n) versus the permutation-invariant space.
The permutation-invariant peak is first checked against the full space at random (alpha, beta) points.

Outputs (in --outdir)
  table_peak_map_n4.csv, table_peak_vs_alpha.csv, table_dimensions.csv, fig9_peak_map.pdf, fig10_dimension.pdf

Usage
  python peak_map.py --outdir results [--quick]
"""
import argparse, os, sys, time
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collective_entanglement as ce
import extended_results as er


def peak_pi(label, alpha, beta):
    S = er.PISystem(label); red = S.reducer()
    L = S.liouvillian(alpha, beta)
    seg = ce.segments_for(beta, 1.0)
    ts, cs, _ = er.transient_pi(S, red, L.toarray(), seg) if S.dim <= er.DENSE_EXPM_LIMIT else er.transient_pi_sparse(S, red, L, seg)
    i = int(np.argmax(cs))
    return float(cs[i]), float(ts[i])


def peak_full(n, alpha, beta):
    lab = ce.default_label(n)
    seg = ce.segments_for(beta, 1.0)
    ts, cs, _ = ce.transient(n, ce.liouvillian(n, alpha, [0], beta).toarray(), lab, seg)
    return float(np.max(cs))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="results"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:7.1f}s]", *x, flush=True)

    # ---- validation of PI peak against the full space (n=4) ----
    rng = np.random.default_rng(7); worst = 0.0
    for _ in range(4 if a.quick else 8):
        al = 10 ** rng.uniform(0, 2); be = 10 ** rng.uniform(-3.5, -1.5)
        d = abs(peak_pi("1100", al, be)[0] - peak_full(4, al, be)); worst = max(worst, d)
    log(f"validation: PI vs full space, peak concurrence, max|dC| = {worst:.2e}"); assert worst < 1e-9

    # ---- (1) map for n=4 ----
    alphas = np.logspace(0, 2, 13 if a.quick else 33)
    betas = np.logspace(-4, -1.5, 8 if a.quick else 21)
    base = {b: peak_pi("1100", 0.0, b)[0] for b in betas}
    Z = np.zeros((len(betas), len(alphas))); rows = []
    for i, b in enumerate(betas):
        for j, al in enumerate(alphas):
            Z[i, j], tp = peak_pi("1100", al, b)
            rows.append([al, b, Z[i, j], tp, base[b], Z[i, j] / base[b]])
        log(f"  map row beta={b:.2e}: max C_peak={Z[i].max():.4f} at alpha={alphas[Z[i].argmax()]:.2f}")
    ce.write_csv(f"{a.outdir}/table_peak_map_n4.csv", ["alpha", "beta", "Cpeak_drive", "t_peak", "Cpeak_nodrive", "ratio"], rows)

    # ---- (2) peak ratio vs alpha at beta = 1e-3, several n ----
    al2 = np.unique(np.round(np.concatenate([np.linspace(1, 6, 21), [8, 10, 15, 20, 30, 50, 100]]), 4))
    rows2, curves = [], {}
    for n in ([4, 6] if a.quick else [4, 5, 6, 8]):
        lab = ce.default_label(n); c0 = peak_pi(lab, 0.0, 1e-3)[0]
        curves[n] = []
        for al in al2:
            c, tp = peak_pi(lab, al, 1e-3); rows2.append([n, al, c, tp, c0, c / c0]); curves[n].append(c / c0)
        k = int(np.argmax(curves[n]))
        log(f"  n={n}: max peak ratio {curves[n][k]:.3f} at alpha={al2[k]:g}  (alpha=20: {curves[n][list(al2).index(20.0)]:.3f})")
    ce.write_csv(f"{a.outdir}/table_peak_vs_alpha.csv", ["n", "alpha", "Cpeak_drive", "t_peak", "Cpeak_nodrive", "ratio"], rows2)

    # ---- (3) dimensions ----
    ns = list(range(3, 16)); dimrows = []
    for n in ns:
        dimrows.append([n, 4 ** n, er.PISystem(ce.default_label(n)).dim])
    ce.write_csv(f"{a.outdir}/table_dimensions.csv", ["n", "full_space_4^n", "PI_dimension_initial_state_11_0..0"], dimrows)

    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    m = ax[0].pcolormesh(alphas, betas, Z, shading="nearest", cmap="viridis")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); fig.colorbar(m, ax=ax[0], label=r"$C_{2,3}^{\rm peak}$")
    ax[0].set_xlabel(r"$\alpha/\gamma_c$"); ax[0].set_ylabel(r"$\beta/\gamma_c$"); ax[0].set_title(r"Peak concurrence, $n=4$ (drive on qubit 1)")
    R = np.array([[r[5] for r in rows if r[1] == b] for b in betas])
    m2 = ax[1].pcolormesh(alphas, betas, R, shading="nearest", cmap="magma")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); fig.colorbar(m2, ax=ax[1], label=r"$C^{\rm peak}_{\rm drive}/C^{\rm peak}_{\rm no\ drive}$")
    ax[1].set_xlabel(r"$\alpha/\gamma_c$"); ax[1].set_title("Ratio to the undriven peak ($n=4$)")
    for n, c in curves.items():
        ax[2].semilogx(al2, c, "o-", ms=3, label=f"n={n}")
    ax[2].axhline(1, color="k", lw=.8); ax[2].set_xlabel(r"$\alpha/\gamma_c$"); ax[2].set_ylabel("peak ratio drive / no drive")
    ax[2].set_title(r"$\beta=10^{-3}$"); ax[2].legend(); ax[2].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig9_peak_map.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.semilogy([r[0] for r in dimrows], [r[1] for r in dimrows], "o-", color="#d62728", label=r"full space ($4^n$)")
    ax.semilogy([r[0] for r in dimrows], [r[2] for r in dimrows], "s-", color="#1f77b4", label="permutation-invariant")
    ax.set_xlabel("n"); ax.set_ylabel("Liouvillian dimension"); ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig10_dimension.pdf"); plt.close(fig)
    log("DONE")


if __name__ == "__main__":
    main()
