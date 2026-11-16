#!/usr/bin/env python3
"""
Robustness of the drive effect against qubit detuning and unequal collective coupling (full space, no symmetry).

  H = alpha sigma^x_1 + (1/2) sum_i delta_i sigma^z_i ,     J = sum_i (1 + eps_i) sigma^-_i ,
  delta_i ~ N(0, sigma_delta^2) (units of gamma_c),  eps_i ~ N(0, sigma_eps^2),   beta = gamma_ind = 0,
  initial state |11 0...0>, alpha = 20.  Every realisation is compared with the undriven system (alpha = 0)
  with the SAME delta_i, eps_i.

Reported per parameter pair (mean +- standard deviation over realisations):
  dim ker            numerical null-space dimension (singular values < 1e-9 * sigma_max)
  C_inf              SIGNED stationary concurrence of the driven system (projection on the null space; < 0: separable)
  peak ratio         max_{t<=10} C_23(t), driven / undriven
  C(20), C(100)      concurrence at t = 20 and 100, driven and undriven (means)

Fixed seed (default 1) makes the run reproducible; the random numbers are drawn in the order of the cases below.

Usage:  python robustness.py --outdir results [--quick] [--n 4] [--nreal 20] [--seed 1] [--cases all|sel]
Output: table_robustness_n<N>.csv
"""
import argparse, os, time
import numpy as np
import scipy.sparse as sp
import collective_entanglement as ce

TOL_SV = 1e-9
TS = np.concatenate([np.linspace(0, 10, 201), np.linspace(10.5, 100, 180), np.linspace(101, 400, 300)])


def liouvillian_disordered(n, alpha, delta=None, eps=None):
    """Collective decay with couplings (1+eps_i), drive alpha*sigma^x_1 and detunings delta_i; gamma_c = 1."""
    d = 2 ** n
    Id = sp.identity(d, format="csr", dtype=complex)
    e = np.zeros(n) if eps is None else eps
    J = sum(((1 + e[i]) * ce.site_op(n, i, ce.SM) for i in range(n)), sp.csr_matrix((d, d), dtype=complex))
    JdJ = (J.getH() @ J).tocsr()
    L = 2 * sp.kron(J.conj(), J, format="csr") - sp.kron(Id, JdJ, format="csr") - sp.kron(JdJ.T, Id, format="csr")
    H = sp.csr_matrix((d, d), dtype=complex)
    if alpha:
        H = H + alpha * ce.site_op(n, 0, ce.SX)
    if delta is not None:
        for i in range(n):
            H = H + 0.5 * delta[i] * ce.site_op(n, i, ce.SZ)
    L = L - 1j * (sp.kron(Id, H, format="csr") - sp.kron(H.T, Id, format="csr"))
    return L.tocsc()


def run_one(n, alpha, delta, eps, v0):
    d = 2 ** n
    Ld = liouvillian_disordered(n, alpha, delta, eps).toarray()
    U, S, Vh = np.linalg.svd(Ld)
    k = int((S < TOL_SV * S[0]).sum())
    R, Lf = Vh[-k:].conj().T, U[:, -k:]
    P = R @ np.linalg.solve(Lf.conj().T @ R, Lf.conj().T @ v0)
    cinf = ce.concurrence_raw(ce.reduced_pair(ce.to_rho(P, n), n, 1, 2))
    w, V = np.linalg.eig(Ld)
    co = np.linalg.solve(V, v0)
    c = np.empty(len(TS))
    for i, t in enumerate(TS):
        v = V @ (np.exp(w * t) * co)
        c[i] = ce.concurrence(ce.reduced_pair(ce.to_rho(v, n), n, 1, 2))
    return k, cinf, c


def at(c, t):
    return c[int(np.argmin(np.abs(TS - t)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--nreal", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cases", choices=["all", "sel"], default="all")
    ap.add_argument("--alpha", type=float, default=20.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    n, NR = a.n, (3 if a.quick else a.nreal)
    allc = [(0.0, 0.0)] + [(s, 0.0) for s in (0.003, 0.01, 0.03, 0.1, 0.3)] + [(0.0, s) for s in (0.02, 0.05, 0.1, 0.2)]
    cases = allc if (a.cases == "all" and not a.quick) else [allc[0], (0.03, 0.0), (0.1, 0.0), (0.0, 0.1)]
    v0 = ce.init_vec(ce.default_label(n))
    rng = np.random.default_rng(a.seed)
    m10 = TS <= 10
    rows = []
    for sd, se in cases:
        t0 = time.time()
        res = []
        for _ in range(1 if (sd, se) == (0.0, 0.0) else NR):
            delta = rng.normal(0, sd, n) if sd > 0 else None
            eps = rng.normal(0, se, n) if se > 0 else None
            k, ci, cd = run_one(n, a.alpha, delta, eps, v0)
            k0, ci0, cu = run_one(n, 0.0, delta, eps, v0)
            res.append([k, ci, cd[m10].max() / cu[m10].max(), at(cd, 20), at(cu, 20), at(cd, 100), at(cu, 100)])
        r = np.array(res)
        m, s = r.mean(0), r.std(0)
        rows.append([n, sd, se, len(res), f"{m[0]:.2f}", f"{m[1]:.4f}", f"{s[1]:.4f}", f"{m[2]:.4f}", f"{s[2]:.4f}",
                     f"{m[3]:.4f}", f"{m[4]:.4f}", f"{m[5]:.4f}", f"{m[6]:.4f}"])
        print(f"n={n} sigma_delta={sd} sigma_eps={se}: dimker={m[0]:.2f} C_inf={m[1]:+.4f} peak_ratio={m[2]:.3f} "
              f"C20={m[3]:.4f}/{m[4]:.4f} C100={m[5]:.4f}/{m[6]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    ce.write_csv(os.path.join(a.outdir, f"table_robustness_n{n}.csv"),
                 ["n", "sigma_delta", "sigma_eps", "n_realisations", "mean_dim_ker", "C_inf_signed_mean", "C_inf_signed_std",
                  "peak_ratio_t<=10_mean", "peak_ratio_std", "C20_drive", "C20_nodrive", "C100_drive", "C100_nodrive"], rows)


if __name__ == "__main__":
    main()
