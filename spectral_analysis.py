#!/usr/bin/env python3
"""
Spectral diagnostics of the stationary-state projection  rho_ss = R (L_f^+ R)^{-1} L_f^+ rho(0)   (Eq. (2.3)).

For the driven problem (beta = gamma_ind = 0, drive alpha on qubit 1, initial state |11 0...0>) the script reports,
for the full-space Liouvillian (n = 3..6):
  * dim ker      dimension of the numerical null space (singular values < 1e-9 * sigma_max)
  * N_osc        number of NON-ZERO eigenvalues with |Re lambda| < 1e-8 (persistent oscillatory modes)
  * gap          smallest non-zero decay rate |Re lambda|_min (sets the convergence time ~ 1/gap)
  * kappa        condition number of L_f^+ R
  * Delta_T      || rho(T) - rho_ss ||_1 at T = 40 / gap   (n <= 5; skipped for n = 6 because of the cost)
and, for the permutation-symmetric invariant subspace used for larger n, the null-space dimension.

Usage:  python spectral_analysis.py --outdir results [--quick] [--max-n 6]
Outputs: table_spectral.csv, table_spectral_reduced.csv
"""
import argparse, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.linalg import svd
import collective_entanglement as ce
import symmetric_reduction as sr

TOL_RE = 1e-8      # eigenvalue tolerance for "vanishing real part"
TOL_SV = 1e-9      # relative singular-value tolerance for the null space


def full_space_row(n, alpha=20.0, propagate=True):
    L = ce.liouvillian(n, alpha=alpha, driven=(0,))
    Ld = L.toarray()
    w = np.linalg.eigvals(Ld)
    n_ker_eig = int((np.abs(w) < TOL_RE).sum())
    n_osc = int(((np.abs(w.real) < TOL_RE) & (np.abs(w.imag) > TOL_RE)).sum())
    gap = float(np.min(-w[np.abs(w.real) >= TOL_RE].real))
    U, S, Vh = svd(Ld)
    k = int((S < TOL_SV * S[0]).sum())
    R, Lf = Vh[-k:].conj().T, U[:, -k:]
    M = Lf.conj().T @ R
    kappa = float(np.linalg.cond(M))
    r0 = ce.init_vec(ce.default_label(n))
    dT = float("nan")
    T = 40.0 / gap
    if propagate:
        P = R @ np.linalg.solve(M, Lf.conj().T @ r0)
        vt = sla.expm_multiply(L * T, r0)
        D = ce.to_rho(vt - P, n)
        D = 0.5 * (D + D.conj().T)
        dT = float(np.abs(np.linalg.eigvalsh(D)).sum())
    return dict(n=n, dim_ker=k, dim_ker_eig=n_ker_eig, N_osc=n_osc, gap=gap, kappa=kappa, T=T, Delta_T=dT,
                max_ker_sv=float(S[-k:].max()), next_sv=float(S[-k - 1]))


def reduced_row(n, alpha=20.0):
    Ld = sr.full_sym_liouvillian(n, alpha).toarray()
    S = svd(Ld, compute_uv=False)
    return dict(n=n, dim_ker=int((S < TOL_SV * S[0]).sum()), next_sv=float(S[(S >= TOL_SV * S[0])][-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-n", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    nmax = 4 if a.quick else a.max_n
    rows = []
    for n in range(3, nmax + 1):
        t0 = time.time()
        r = full_space_row(n, propagate=(n <= 5))
        rows.append([n, r["dim_ker"], r["dim_ker_eig"], r["N_osc"], f"{r['gap']:.6e}", f"{r['kappa']:.4f}",
                     f"{r['T']:.1f}", "" if np.isnan(r["Delta_T"]) else f"{r['Delta_T']:.3e}",
                     f"{r['max_ker_sv']:.2e}", f"{r['next_sv']:.3e}"])
        print(f"n={n}: ker={r['dim_ker']} N_osc={r['N_osc']} gap={r['gap']:.4e} kappa={r['kappa']:.2f} "
              f"Delta_T={r['Delta_T']:.2e} ({time.time()-t0:.0f}s)", flush=True)
    ce.write_csv(os.path.join(a.outdir, "table_spectral.csv"),
                 ["n", "dim_ker_svd", "dim_ker_eig", "N_osc", "min_nonzero_decay_rate", "cond(Lf^+R)", "T", "Delta_T_trace_norm",
                  "max_kernel_singular_value", "next_singular_value"], rows)
    rrows = []
    for n in ((3, 4, 5, 6, 8) if a.quick else (3, 4, 5, 6, 8, 12, 16)):
        r = reduced_row(n)
        rrows.append([n, r["dim_ker"], f"{r['next_sv']:.3e}"])
        print(f"reduced n={n}: ker={r['dim_ker']}", flush=True)
    ce.write_csv(os.path.join(a.outdir, "table_spectral_reduced.csv"), ["n", "dim_ker_svd", "next_singular_value"], rrows)


if __name__ == "__main__":
    main()
