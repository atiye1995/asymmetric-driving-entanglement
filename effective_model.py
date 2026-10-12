#!/usr/bin/env python3
"""
effective_model.py
==================
Strong-drive (alpha >> gamma_c) effective model for the asymmetric protocol,
derived by a secular (RWA) average in the sigma_x eigenbasis of the driven qubit.

Derivation
---------------------------------------------------
Driven qubit 1: H = alpha*sigma^x_1.  In the basis |±> of sigma^x_1,
    sigma^-_1 = (1/2) sigma^x_1  +  (1/2)(|-><+| - |+><-|),
i.e. a constant part (1/2)sigma^x_1 plus parts that rotate at 2*alpha in the
interaction picture.  Keeping only the non-rotating terms of the collective
dissipator (secular approximation, alpha -> infinity) gives, with J_R the
collective lowering operator of the UNDRIVEN qubits 2..n,

    rho' = 2 D[ J_R + (1/2) sigma^x_1 ] rho + (1/2) D[|+><-|] rho + (1/2) D[|-><+|] rho ,

with D[O]rho = O rho O^+ - (1/2){O^+O, rho}.   Qubit 1 stays in the retained
Hilbert space: it is diagonal in the sigma^x basis (s = ±1) and flips s at rate 1/2.
Writing rho = sum_s |s><s| (x) rho_s, the undriven qubits obey

    d rho_s/dt = 2 D[J_R + s/2] rho_s + (1/2)(rho_{-s} - rho_s)
               = 2 D[J_R] rho_s - i [ H_s, rho_s ] + (1/2)(rho_{-s} - rho_s),
    H_s = -(s/2) * sum_{i>=2} sigma^y_i .

Interpretation: the strongly driven qubit acts on the others as a *classical
random-telegraph coherent drive* of amplitude 1/2 (sign s=±1, switching rate 1/2
each way) superposed on their own collective decay.  The initial state of qubit 1
drops out (pinching gives rho_+ = rho_- = rho_R(0)/2), so C_{2,3}(infinity)
depends only on the initial state of qubits 2..n.

Reduced state of qubits 2..n: rho_R = rho_+ + rho_-.
"""
import argparse, json, time, os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
sys.path.insert(0, os.path.dirname(__file__))
import collective_entanglement as ce

SM, SX, SZ = ce.SM, ce.SX, ce.SZ
SY = sp.csr_matrix(np.array([[0, -1j], [1j, 0]]))


def dissipator_2D(A):
    """Column-stacked superoperator of 2 D[A]."""
    d = A.shape[0]
    Id = sp.identity(d, format="csr", dtype=complex)
    AdA = (A.getH() @ A).tocsr()
    return 2 * sp.kron(A.conj(), A, format="csr") - sp.kron(Id, AdA, format="csr") - sp.kron(AdA.T, Id, format="csr")


def effective_liouvillian(m, lam=1.0, flip=0.5, dephase_beta=0.0):
    """Block Liouvillian for (rho_+, rho_-) of m = n-1 undriven qubits.
    lam  : scale of the shift J_R -> J_R + lam*s/2   (lam=1 is the derived model)
    flip : switching rate between s=+ and s=- (flip=1/2 is the derived value)."""
    JR = sum(ce.site_op(m, i, SM) for i in range(m))
    d = 2 ** m
    Id = sp.identity(d, format="csr", dtype=complex)
    blocks = []
    for s in (+1, -1):
        blocks.append(dissipator_2D(JR + lam * s / 2.0 * sp.identity(d, format="csr", dtype=complex)))
    N = d * d
    I_N = sp.identity(N, format="csr", dtype=complex)
    L = sp.bmat([[blocks[0] - flip * I_N, flip * I_N], [flip * I_N, blocks[1] - flip * I_N]], format="csc")
    if dephase_beta > 0:
        deph = None
        for i in range(m):
            Z = ce.site_op(m, i, SZ)
            t = dephase_beta * (sp.kron(Z, Z, format="csr") - sp.kron(Id, Id, format="csr"))
            deph = t if deph is None else deph + t
        L = L + sp.block_diag([deph, deph], format="csc")
    return L


def eff_init(label_rest):
    """Pinched initial state: rho_+ = rho_- = rho_R(0)/2."""
    m = len(label_rest)
    r = ce.init_vec(label_rest)
    return np.concatenate([0.5 * r, 0.5 * r])


def C_from_eff(v, m):
    N = 4 ** m
    rho_R = ce.to_rho(v[:N] + v[N:], m)
    return ce.concurrence(ce.reduced_pair(rho_R, m, 0, 1))


def eff_steady(m, L, v0, dense_limit=5):
    """Stationary state reached from v0 (projection onto stationary subspace)."""
    if m <= dense_limit:
        R, Lf, k, gap = ce.null_decomposition_dense(L.toarray())
    else:
        R, Lf, k = ce.null_decomposition_sparse(L)
        gap = float("nan")
    v = ce.project(R, Lf, v0)
    return v, dict(null_dim=k, min_nonnull_sv=gap, residual=float(np.linalg.norm(L @ v)))


# --------------------------------------------------------------------------
def check_derivation(n=4, seed=1):
    """Verify numerically that the analytic block form equals the secular
    average (pinching) of the full collective dissipator."""
    rng = np.random.default_rng(seed)
    m = n - 1
    d = 2 ** m
    plus = np.array([1, 1]) / np.sqrt(2); minus = np.array([1, -1]) / np.sqrt(2)
    P = {+1: np.outer(plus, plus), -1: np.outer(minus, minus)}
    X = {}
    for s in (+1, -1):
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        X[s] = A + A.conj().T
    rho = sum(np.kron(P[s], X[s]) for s in (+1, -1))
    Lfull = ce.liouvillian(n)                       # collective decay only (H term drops under pinching)
    drho = ce.to_rho(Lfull @ rho.flatten("F"), n)
    out = {}
    for s, v in ((+1, plus), (-1, minus)):
        K = np.kron(v.reshape(1, 2), np.eye(d))     # <s|_1 (x) 1
        out[s] = K @ drho @ K.conj().T
    Leff = effective_liouvillian(m).toarray()
    vec = np.concatenate([X[+1].flatten("F"), X[-1].flatten("F")])
    pred = Leff @ vec
    N = d * d
    err = max(np.abs(pred[:N].reshape(d, d, order="F") - out[+1]).max(),
              np.abs(pred[N:].reshape(d, d, order="F") - out[-1]).max())
    return float(err)


def main():
    T0 = time.time()
    log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--outdir", default="results"); ap.add_argument("--max-n", type=int, default=6)
    args = ap.parse_args()
    outdir, max_n = args.outdir, args.max_n
    os.makedirs(outdir, exist_ok=True)
    res = {}

    # 1. derivation check
    errs = {n: check_derivation(n) for n in (3, 4, 5)}
    res["derivation_check_max_abs_error"] = errs
    log("derivation check (analytic block form vs pinched full Liouvillian):", errs)

    # 2. agreement with full simulation as alpha grows
    comp = {}
    for n in range(3, 6):
        m = n - 1
        v, info = eff_steady(m, effective_liouvillian(m), eff_init(ce.default_label(n)[1:]))
        c_eff = C_from_eff(v, m)
        full = {}
        for a in (20, 100, 300, 1000):
            vf, _ = ce.steady_state(n, ce.liouvillian(n, a, [0]), ce.default_label(n))
            full[a] = ce.C23(vf, n)
        comp[n] = dict(C_eff=c_eff, C_full=full, null_dim_eff=info["null_dim"])
        log(f"  n={n}: C_eff={c_eff:.6f}; full:", {a: round(c, 6) for a, c in full.items()})
    res["eff_vs_full"] = comp

    # 3. qubit 1 decoupled from the collective channel (lam=0), same initial rest state
    old = {}
    for n in range(3, 6):
        m = n - 1
        v, _ = eff_steady(m, effective_liouvillian(m, lam=0.0), eff_init(ce.default_label(n)[1:]))
        old[n] = C_from_eff(v, m)
    res["decoupled_lam0"] = old
    log("qubit 1 decoupled (no shift, lam=0):", old)

    # 4. what does the physics need?  scan shift amplitude and switching rate
    scan = {}
    for n in (3, 4, 5):
        m = n - 1
        v0 = eff_init(ce.default_label(n)[1:])
        sc = {"lam": {}, "flip": {}}
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            v, _ = eff_steady(m, effective_liouvillian(m, lam=lam), v0)
            sc["lam"][lam] = C_from_eff(v, m)
        for flip in (0.0, 0.1, 0.5, 2.0, 10.0):
            v, _ = eff_steady(m, effective_liouvillian(m, flip=flip), v0)
            sc["flip"][flip] = C_from_eff(v, m)
        scan[n] = sc
        log(f"  n={n} lam-scan:", {k: round(x, 4) for k, x in sc['lam'].items()})
        log(f"  n={n} flip-scan:", {k: round(x, 4) for k, x in sc['flip'].items()})
    res["mechanism_scans"] = scan

    # 5. beyond full-simulation reach: n = 6,7,(8) with the effective model
    big = {}
    for n in range(6, max_n + 1):
        m = n - 1
        t = time.time()
        Leff = effective_liouvillian(m)
        v, info = eff_steady(m, Leff, eff_init(ce.default_label(n)[1:]), dense_limit=4)
        big[n] = dict(C_eff=C_from_eff(v, m), null_dim=info["null_dim"], residual=info["residual"], seconds=time.time() - t)
        log(f"  n={n}: C_eff(alpha->inf)={big[n]['C_eff']:.6f} {big[n]}")
        # field-free reference by evolution
        t = time.time()
        L0 = ce.liouvillian(n)
        v0, i0 = ce.steady_by_evolution(n, L0, ce.default_label(n), t_max=400.0)
        big[n]["C_fieldfree"] = ce.C23(v0, n)
        big[n]["fieldfree_converged"] = i0["converged"]
        big[n]["change_percent"] = 100 * (big[n]["C_eff"] - big[n]["C_fieldfree"]) / big[n]["C_fieldfree"]
        log(f"  n={n}: C_ff={big[n]['C_fieldfree']:.6f} ({time.time()-t:.0f}s) change={big[n]['change_percent']:+.2f}%")
        res["large_n"] = big
        json.dump(res, open(f"{outdir}/effective_model_results.json", "w"), indent=2, default=float)
    res["large_n"] = big
    json.dump(res, open(f"{outdir}/effective_model_results.json", "w"), indent=2, default=float)
    log("DONE")


if __name__ == "__main__":
    main()
