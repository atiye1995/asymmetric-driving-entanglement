#!/usr/bin/env python3
"""
Collective decay of n qubits with a single-qubit drive: full-space solver, tables and figures.

Master equation (gamma_c = 1):
  rho' = -i[H_d, rho] + 2 J rho J^+ - J^+J rho - rho J^+J
         + beta * sum_i D[sz_i] rho + gamma_ind * sum_i D[sigma^-_i] rho
  J = sum_i sigma^-_i,   H_d = alpha * sum_{i in driven} sigma^x_i,   D[O]rho = O rho O^+ - {O^+O, rho}/2
  (D[sz]rho = sz rho sz - rho: coherences decay at the rate 2*beta)

Conventions: |0> ground, |1> excited, sigma^- = |0><1|; qubit 1 is the first tensor factor (leftmost bit of a
label such as '1100'); the driven qubit is qubit 1 and entanglement is measured between qubits 2 and 3;
column-stacking vectorisation, vec(A rho B) = (B^T (x) A) vec(rho).

Methods: transient dynamics by matrix-exponential propagation (scipy.linalg.expm), checked by grid halving and
against an explicit integrator (DOP853). For beta = gamma_ind = 0 the stationary state depends on the initial state;
it is obtained by projecting the initial state on the null space of the Liouvillian.

Usage:
  python collective_entanglement.py --outdir results [--quick] [--max-n N]
"""
import argparse, json, os, time, warnings, csv
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.linalg import expm, svd
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore", category=sla.MatrixRankWarning)

# ----------------------------------------------------------------------------
# 1. Operators and Liouvillian
# ----------------------------------------------------------------------------
SM = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))   # |0><1|
SX = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
SZ = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
I2 = sp.identity(2, format="csr", dtype=complex)


def site_op(n, i, op):
    m = None
    for k in range(n):
        a = op if k == i else I2
        m = a if m is None else sp.kron(m, a, format="csr")
    return m


def liouvillian(n, alpha=0.0, driven=(), beta=0.0, gamma_ind=0.0):
    """Sparse Liouvillian (column-stacking), gamma_c = 1.
    gamma_ind: rate of INDEPENDENT (non-collective) decay, + gamma_ind * sum_i D[sigma^-_i] (T1 = 1/gamma_ind)."""
    d = 2 ** n
    Id = sp.identity(d, format="csr", dtype=complex)
    J = sum(site_op(n, i, SM) for i in range(n))
    Jd = J.getH().tocsr()
    JdJ = (Jd @ J).tocsr()
    L = (2 * sp.kron(J.conj(), J, format="csr")
         - sp.kron(Id, JdJ, format="csr") - sp.kron(JdJ.T, Id, format="csr"))
    if alpha != 0 and len(driven) > 0:
        H = sum(alpha * site_op(n, i, SX) for i in driven)
        L = L - 1j * (sp.kron(Id, H, format="csr") - sp.kron(H.T, Id, format="csr"))
    if beta > 0:
        for i in range(n):
            Z = site_op(n, i, SZ)
            L = L + beta * (sp.kron(Z, Z, format="csr") - sp.kron(Id, Id, format="csr"))
    if gamma_ind > 0:
        for i in range(n):
            A = site_op(n, i, SM)
            AdA = (A.getH() @ A).tocsr()
            L = L + gamma_ind * (sp.kron(A.conj(), A, format="csr")
                                 - 0.5 * sp.kron(Id, AdA, format="csr") - 0.5 * sp.kron(AdA.T, Id, format="csr"))
    return L.tocsc()


def init_vec(label):
    """Initial pure state from a bit label, e.g. '1100' (bit 1 = excited)."""
    n = len(label)
    v = np.zeros(2 ** n, dtype=complex)
    v[int(label, 2)] = 1.0
    return np.outer(v, v).flatten("F")


def default_label(n):
    return "11" + "0" * (n - 2)


def to_rho(vec, n):
    d = 2 ** n
    return vec.reshape(d, d, order="F")


# ----------------------------------------------------------------------------
# 2. Reduced state and concurrence
# ----------------------------------------------------------------------------
_LET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SY = np.array([[0, -1j], [1j, 0]])
_SYSY = np.kron(_SY, _SY)


def reduced_pair(rho, n, i, j):
    t = rho.reshape([2] * (2 * n))
    idx = list(range(2 * n))
    for k in range(n):
        if k not in (i, j):
            idx[k + n] = idx[k]
    inp = "".join(_LET[x] for x in idx)
    out = _LET[i] + _LET[j] + _LET[i + n] + _LET[j + n]
    return np.einsum(inp + "->" + out, t).reshape(4, 4)


def concurrence_raw(r):
    """Signed Wootters quantity lambda1 - lambda2 - lambda3 - lambda4 (no max(0, .)).
    Negative values quantify how far a state is from the entangled region."""
    r = 0.5 * (r + r.conj().T)
    ev = np.linalg.eigvals(r @ _SYSY @ r.conj() @ _SYSY).real
    lam = np.sort(np.sqrt(np.clip(ev, 0, None)))[::-1]
    return float(lam[0] - lam[1] - lam[2] - lam[3])


def concurrence(r):
    return max(0.0, concurrence_raw(r))


def C23(vec, n):
    return concurrence(reduced_pair(to_rho(vec, n), n, 1, 2))


def C23_raw(vec, n):
    return concurrence_raw(reduced_pair(to_rho(vec, n), n, 1, 2))


# ----------------------------------------------------------------------------
# 3. Stationary states
# ----------------------------------------------------------------------------
def null_decomposition_dense(Ld, tol=1e-8):
    """Right/left null vectors of a dense Liouvillian via SVD."""
    U, S, Vh = svd(Ld)
    k = int((S < tol).sum())
    R = Vh[-k:].conj().T
    Lf = U[:, -k:]
    gap = float(S[-k - 1]) if k < len(S) else float("nan")  # smallest non-null singular value
    return R, Lf, k, gap


def project(R, Lf, r0):
    return R @ np.linalg.solve(Lf.conj().T @ R, Lf.conj().T @ r0)


def null_decomposition_sparse(L, sigma=-1e-6, K0=48, tol=1e-7):
    """Right/left null vectors by shift-invert Arnoldi (n >= 6); fixed start vector
    (reproducible).  Multiplicity is cross-checked (right vs left, two shifts)."""
    N = L.shape[0]
    v0 = np.random.default_rng(12345).standard_normal(N) + 0j

    def get(M):
        K = K0
        while True:
            vals, vecs = sla.eigs(M, k=K, sigma=sigma, which="LM", tol=1e-13, v0=v0)
            mask = np.abs(vals) < tol
            if mask.sum() <= K - 6 or K >= N // 4:
                return vecs[:, mask], int(mask.sum())
            K *= 2
    R, kr = get(L)
    Lf, kl = get(L.getH().tocsc())
    if kr != kl:
        raise RuntimeError(f"null-space mismatch right={kr} left={kl}")
    return R, Lf, kr


def steady_state(n, L, label, dense_limit=5, sparse_sigma=-1e-6):
    """Return (rho_vec, info) -- stationary state reached from |label>."""
    r0 = init_vec(label)
    if n <= dense_limit:
        R, Lf, k, gap = null_decomposition_dense(L.toarray())
        v = project(R, Lf, r0)
        info = dict(method="svd-projection", null_dim=k, min_nonnull_sv=gap)
    else:
        R, Lf, k = null_decomposition_sparse(L, sigma=sparse_sigma)
        v = project(R, Lf, r0)
        info = dict(method="arpack-projection", null_dim=k, min_nonnull_sv=float("nan"))
    info["residual"] = float(np.linalg.norm(L @ v))
    rho = to_rho(v, n)
    info["trace"] = float(np.trace(rho).real)
    info["herm_err"] = float(np.abs(rho - rho.conj().T).max())
    info["min_eig"] = float(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).min()) if n <= 6 else float("nan")
    return v, info


def steady_by_evolution(n, L, label, t_chunk=20.0, t_max=2000.0, res_tol=1e-9):
    """Cross-check: evolve with sparse expm_multiply until ||L rho|| < res_tol."""
    v = init_vec(label)
    t = 0.0
    while t < t_max:
        v = sla.expm_multiply(L * t_chunk, v)
        t += t_chunk
        res = np.linalg.norm(L @ v)
        if res < res_tol:
            return v, dict(t=t, residual=float(res), converged=True)
    return v, dict(t=t, residual=float(np.linalg.norm(L @ v)), converged=False)


# ----------------------------------------------------------------------------
# 4. Transient dynamics, peak, lifetime, advantage window
# ----------------------------------------------------------------------------
def transient(n, Ld, label, segments):
    """Propagate with matrix exponentials. segments = [(dt, t_end), ...].
    Returns t, C(t) (clipped at 0) and the signed raw concurrence."""
    v = init_vec(label)
    t, ts, raw = 0.0, [0.0], [C23_raw(v, n)]
    for dt, t_end in segments:
        E = expm(Ld * dt)
        for _ in range(int(round((t_end - t) / dt))):
            v = E @ v
            t += dt
            ts.append(t)
            raw.append(C23_raw(v, n))
    raw = np.array(raw)
    return np.array(ts), np.clip(raw, 0, None), raw


def peak_and_lifetime(ts, cs, c_min=1e-2):
    """C_peak, t_peak, tau_1/e (linear interpolation of the first crossing of
    C_peak/e after the peak).  tau is NaN (undefined) if C_peak < c_min or the
    crossing is not reached within the integration window (censored)."""
    i = int(np.argmax(cs))
    cpk, tpk = float(cs[i]), float(ts[i])
    if cpk < c_min:
        return cpk, tpk, float("nan"), "undefined (C_peak<%g)" % c_min
    thr = cpk / np.e
    after = np.where(cs[i:] < thr)[0]
    if len(after) == 0:
        return cpk, tpk, float("nan"), "censored (no crossing in window)"
    j = i + int(after[0])
    t0, t1, c0, c1 = ts[j - 1], ts[j], cs[j - 1], cs[j]
    tau = float(t0 + (c0 - thr) * (t1 - t0) / (c0 - c1)) if c0 != c1 else float(t1)
    return cpk, tpk, tau, "ok"


def death_time(ts, raw, cs):
    """First time after the peak at which the signed concurrence crosses zero
    (sudden death caused by max(0,.)); NaN if it never does in the window."""
    i = int(np.argmax(cs))
    neg = np.where(raw[i:] <= 0)[0]
    if len(neg) == 0 or cs[i] <= 0:
        return float("nan")
    j = i + int(neg[0])
    if j == 0:
        return float(ts[0])
    t0, t1, r0, r1 = ts[j - 1], ts[j], raw[j - 1], raw[j]
    return float(t0 + r0 * (t1 - t0) / (r0 - r1)) if r0 != r1 else float(t1)


_trapz = getattr(np, "trapezoid", None) or np.trapz


def advantage_window(ts, c_drive, c_nodrive):
    """Intervals on which C_drive(t) > C_nodrive(t) (crossings by linear interpolation).
    Returns dict: windows [(t_on, t_off)], t_off=inf if still true at the end of the grid,
    max advantage and its time, integrated advantage int (C_d - C_nd) dt over the grid."""
    d = c_drive - c_nodrive
    pos = d > 1e-12
    wins, start = [], None
    for k in range(len(ts)):
        if pos[k] and start is None:
            if k == 0:
                start = ts[0]
            else:
                start = ts[k - 1] + (0 - d[k - 1]) * (ts[k] - ts[k - 1]) / (d[k] - d[k - 1])
        if (not pos[k]) and start is not None:
            end = ts[k - 1] + d[k - 1] * (ts[k] - ts[k - 1]) / (d[k - 1] - d[k])
            wins.append((float(start), float(end)))
            start = None
    if start is not None:
        wins.append((float(start), float("inf")))
    k = int(np.argmax(d))
    return dict(windows=wins, max_adv=float(d[k]), t_max_adv=float(ts[k]),
                integral=float(_trapz(d, ts)), t_window_end_grid=float(ts[-1]))


def segments_for(beta, scale=1.0):
    dt1, dt2 = 0.05 * scale, 0.5 * scale
    t_end = 400.0 if beta == 0 else max(60.0, 0.8 / beta)
    return [(dt1, 40.0), (dt2, np.ceil(t_end / dt2) * dt2 if t_end > 40 else 40.0)]


def lifetime_row(n, label, alpha, driven, beta):
    """Peak/lifetime on two sampling grids (dt and dt/2); the propagator itself is exact."""
    out, tr = {}, {}
    Ld = liouvillian(n, alpha, driven, beta).toarray()
    for tag, sc in (("A", 1.0), ("B", 0.5)):
        ts, cs, raw = transient(n, Ld, label, segments_for(beta, sc))
        out[tag] = peak_and_lifetime(ts, cs)
        tr[tag] = (ts, cs, raw)
    cpk, tpk, tau, st = out["A"]
    cpkB, _, tauB, _ = out["B"]
    ts, cs, raw = tr["A"]
    return dict(C_peak=cpk, t_peak=tpk, tau=tau, status=st,
                t_death=death_time(ts, raw, cs),
                C_peak_err=abs(cpk - cpkB),
                tau_err=abs(tau - tauB) if np.isfinite(tau) and np.isfinite(tauB) else float("nan"),
                trace=(ts, cs, raw))


def convergence_table(n=4, label="1100", t_end=60.0):
    """Propagator vs grid halving vs independent explicit integrator (DOP853)."""
    rows = []
    times = np.arange(0.0, t_end + 1e-9, 0.5)          # same sampling times as the expm curves
    for alpha, driven, beta in [(0.0, [], 0.0), (20.0, [0], 0.0), (20.0, [0], 1e-3), (20.0, [0], 1e-2)]:
        Ld = liouvillian(n, alpha, driven, beta).toarray()
        r0 = init_vec(label)
        # (a) expm propagator on the same sampling grid at three step sizes
        curves = {}
        for dt in (0.5, 0.25, 0.125):
            E = expm(Ld * dt)
            v, c = r0.copy(), [C23(r0, n)]
            nsteps = int(round(t_end / dt))
            for k in range(1, nsteps + 1):
                v = E @ v
                if abs(k * dt / 0.5 - round(k * dt / 0.5)) < 1e-9:
                    c.append(C23(v, n))
            curves[dt] = np.array(c)
        # (b) independent integrator
        nn_ = len(r0)

        def f(t, y):
            dy = Ld @ (y[:nn_] + 1j * y[nn_:])
            return np.concatenate([dy.real, dy.imag])
        y0 = np.concatenate([r0.real, r0.imag])
        sol = solve_ivp(f, (0, t_end), y0, method="DOP853", t_eval=times, rtol=1e-11, atol=1e-13)
        vs = sol.y[:len(r0)] + 1j * sol.y[len(r0):]
        c_ivp = np.array([C23(vs[:, k], n) for k in range(vs.shape[1])])
        c_e = curves[0.125]
        rho_end = to_rho(vs[:, -1], n)
        rows.append(dict(alpha=alpha, beta=beta,
                         max_dC_dt0p5_vs_0p25=float(np.abs(curves[0.5] - curves[0.25]).max()),
                         max_dC_dt0p25_vs_0p125=float(np.abs(curves[0.25] - curves[0.125]).max()),
                         max_dC_expm_vs_DOP853=float(np.abs(c_e - c_ivp).max()),
                         trace_err_DOP853=float(abs(np.trace(rho_end) - 1)),
                         min_eig_DOP853=float(np.linalg.eigvalsh(0.5 * (rho_end + rho_end.conj().T)).min())))
    return rows


# ----------------------------------------------------------------------------
# 5. Utilities
# ----------------------------------------------------------------------------
def sym_projector(n):
    """Projector on the permutation-symmetric (Dicke) subspace."""
    d = 2 ** n
    P = np.zeros((d, d))
    for k in range(n + 1):
        v = np.array([1.0 if bin(s).count("1") == k else 0.0 for s in range(d)])
        v /= np.linalg.norm(v)
        P += np.outer(v, v)
    return P


def fmt(x, nd=6):
    return "—" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------------
# 6. Main workflow
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true", help="skip n>=6")
    ap.add_argument("--max-n", type=int, default=6)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out = args.outdir
    os.makedirs(out, exist_ok=True)
    nmax = 5 if args.quick else args.max_n
    T0 = time.time()
    res = {"conventions": "bit1=excited, sigma^-=|0><1|, qubit1=leftmost bit, driven qubit=1, pair=(2,3), "
                          "D[sz]rho=sz rho sz-rho (coherence decay 2*beta, T_phi=1/(2 beta gamma_c)), gamma_c=1"}

    def log(*a):
        print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

    # ---- 6.0 field-free values: validation + provenance ----
    log("Field-free values; validation against Memarzadeh & Mancini (4/9, 1/4)")
    ff, ff_info = {}, {}
    for n in range(3, nmax + 1):
        L = liouvillian(n)
        if n <= 5:
            v, info = steady_state(n, L, default_label(n))
        else:
            v, info = steady_by_evolution(n, L, default_label(n))
        ff[n] = C23(v, n)
        ff_info[n] = info
        log(f"  n={n}: C_ff={ff[n]:.6f}", info)
    assert abs(ff[3] - 4 / 9) < 1e-6 and abs(ff[4] - 0.25) < 1e-6, "validation failed"
    log("  validation PASSED")
    res["field_free"] = {str(k): v for k, v in ff.items()}
    res["field_free_info"] = {str(k): v for k, v in ff_info.items()}
    write_csv(f"{out}/table_fieldfree_provenance.csv", ["n", "C_fieldfree", "source"],
              [[n, ff[n], "analytic, Memarzadeh & Mancini, Eq.(28); reproduced numerically" if n <= 4
                else "numerical (this work): null-space projection / long-time evolution"] for n in ff])

    # ---- 6.1 asymmetric drive: alpha scan (stationary state, beta=0) ----
    alphas = [0.3, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 300]
    scan, scan_info = {}, {}
    log("Alpha scan, asymmetric drive on qubit 1")
    for n in range(3, nmax + 1):
        scan[n] = {}
        al = alphas if n <= 5 else [1, 2, 5, 20, 100]
        for a in al:
            L = liouvillian(n, a, [0])
            v, info = steady_state(n, L, default_label(n))
            scan[n][a] = C23(v, n)
            scan_info[f"{n},{a}"] = info
            log(f"  n={n} alpha={a:>5}: C={scan[n][a]:.6f}  null_dim={info['null_dim']}  res={info['residual']:.1e}")
    res["alpha_scan"] = {str(n): {str(a): c for a, c in d.items()} for n, d in scan.items()}
    res["alpha_scan_info"] = scan_info
    write_csv(f"{out}/table_alpha_scan.csv", ["n", "alpha", "C_inf", "C_fieldfree", "ratio"],
              [[n, a, c, ff[n], c / ff[n]] for n in scan for a, c in scan[n].items()])
    # n=3: non-monotonic behaviour (minimum near alpha ~ 2)
    n3 = scan[3]
    res["n3_min"] = dict(alpha=min(n3, key=n3.get), C=min(n3.values()))

    # ---- 6.2 independent cross-checks of the stationary value ----
    log("Cross-checks (time evolution vs projection; two ARPACK shifts for n=6)")
    xc = {}
    for n in range(3, min(nmax, 5) + 1):
        L = liouvillian(n, 20.0, [0])
        vE, iE = steady_by_evolution(n, L, default_label(n), t_max=6000.0)
        xc[f"n={n}"] = dict(C_evolution=C23(vE, n), C_projection=scan[n][20], t=iE["t"],
                            converged=iE["converged"], residual=iE["residual"])
        log(f"  n={n}: evolution {xc[f'n={n}']['C_evolution']:.6f} vs projection {scan[n][20]:.6f} {iE}")
    if nmax >= 6:
        L = liouvillian(6, 20.0, [0])
        v2, i2 = steady_state(6, L, default_label(6), sparse_sigma=+1.3e-6)
        xc["n=6"] = dict(C_shift_a=scan[6][20], C_shift_b=C23(v2, 6), null_dim=i2["null_dim"])
        log("  n=6 second shift:", xc["n=6"])
    res["crosschecks"] = xc

    # ---- 6.3 stationary concurrence at alpha=20 ----
    tab1 = []
    for n in range(3, nmax + 1):
        cd = scan[n][20]
        tab1.append([n, ff[n], cd, 100 * (cd - ff[n]) / ff[n]])
    write_csv(f"{out}/table1_alpha20.csv", ["n", "C_fieldfree", "C_drive_alpha20", "change_percent"], tab1)
    res["table1"] = tab1
    # power-law fits over larger n (with uncertainties and local exponents): scaling_symmetric.py

    # ---- 6.4 uniqueness of the stationary state for beta > 0 ----
    log("Stationary-state uniqueness and C(inf) for beta>0")
    uniq = []
    for n, a, dr in [(3, 0, []), (4, 0, []), (3, 20.0, [0]), (4, 20.0, [0]), (5, 20.0, [0])]:
        for b in [1e-6, 1e-4, 1e-3, 1e-2]:
            if n == 5 and b not in (1e-4, 1e-2):
                continue
            L = liouvillian(n, a, dr, b)
            v, info = steady_state(n, L, default_label(n))
            row = dict(n=n, alpha=a, beta=b, null_dim=info["null_dim"], C_inf=C23(v, n),
                       C_raw_signed=C23_raw(v, n),
                       min_nonnull_sv=info["min_nonnull_sv"], p_all_ground=float(to_rho(v, n)[0, 0].real))
            uniq.append(row)
            log(f"  n={n} alpha={a:>4} beta={b:.0e}: null_dim={row['null_dim']} C_inf={row['C_inf']:.2e} "
                f"raw={row['C_raw_signed']:+.3e} P(0..0)={row['p_all_ground']:.6f}")
    res["beta_positive_steady"] = uniq
    write_csv(f"{out}/beta_positive_uniqueness.csv", list(uniq[0].keys()), [list(r.values()) for r in uniq])

    # ---- 6.5 lifetime table, advantage window, sudden death (n=4) ----
    log("Lifetime table n=4 (drive alpha=20 on qubit 1), grid-halving errors, advantage windows")
    betas = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    lt, win_rows = [], []
    for b in betas:
        r0_ = lifetime_row(4, "1100", 0.0, [], b)
        rd_ = lifetime_row(4, "1100", 20.0, [0], b)
        lt.append(dict(beta=b, nodrive=r0_, drive=rd_))
        w = advantage_window(rd_["trace"][0], rd_["trace"][1], r0_["trace"][1])
        win_rows.append([b, r0_["t_peak"], rd_["t_peak"], r0_["t_death"], rd_["t_death"],
                         w["windows"][0][0] if w["windows"] else float("nan"),
                         w["windows"][-1][1] if w["windows"] else float("nan"),
                         len(w["windows"]), w["max_adv"], w["t_max_adv"], w["integral"], w["t_window_end_grid"]])
        log(f"  beta={b:.0e}: peak {r0_['C_peak']:.4f}->{rd_['C_peak']:.4f}; tau {fmt(r0_['tau'],1)}->{fmt(rd_['tau'],1)}; "
            f"death {fmt(r0_['t_death'],1)}->{fmt(rd_['t_death'],1)}; windows {w['windows']}")
    for e in lt:                       # traces are not JSON-serialisable
        for k in ("nodrive", "drive"):
            e[k] = {kk: vv for kk, vv in e[k].items() if kk != "trace"}
    res["lifetime_table"] = lt
    write_csv(f"{out}/table_advantage_window.csv",
              ["beta", "t_peak_nodrive", "t_peak_drive", "t_death_nodrive", "t_death_drive",
               "first_window_start", "last_window_end(inf=still>0 at grid end)", "n_windows",
               "max_advantage_dC", "t_of_max_advantage", "integral_dC_dt", "grid_end"], win_rows)
    rows = []
    for e in lt:
        n0, d0 = e["nodrive"], e["drive"]
        b = e["beta"]
        rows.append([b, (1 / (2 * b)) if b > 0 else float("inf"),
                     n0["C_peak"], n0["C_peak_err"], n0["t_peak"], n0["tau"], n0["tau_err"], n0["t_death"],
                     d0["C_peak"], d0["C_peak_err"], d0["t_peak"], d0["tau"], d0["tau_err"], d0["t_death"],
                     d0["C_peak"] / n0["C_peak"], d0["tau"] / n0["tau"] if np.isfinite(n0["tau"]) else float("nan")])
    write_csv(f"{out}/table2_lifetime.csv",
              ["beta", "T_phi(1/gamma_c)=1/(2beta)",
               "Cpeak_nodrive", "err", "t_peak_nodrive", "tau_nodrive", "err", "t_death_nodrive",
               "Cpeak_drive", "err", "t_peak_drive", "tau_drive", "err", "t_death_drive",
               "Cpeak_ratio", "tau_ratio"], rows)
    bp = np.array([e["beta"] for e in lt if e["beta"] > 0])
    fit = {}
    for key in ("nodrive", "drive"):
        taus = np.array([e[key]["tau"] for e in lt if e["beta"] > 0])
        A, B = np.polyfit(1 / bp, taus, 1)
        fit[key] = dict(A=float(A), B=float(B))
    res["tau_fit_A_over_beta_plus_B"] = fit
    log("  tau fit:", fit)

    # ---- 6.6 definitions table for the different n=4 numbers ----
    log("Definitions table for n=4 numbers")
    ref = ff[4]
    cp0 = lt[0]["drive"]["C_peak"]
    cpn0 = lt[0]["nodrive"]["C_peak"]
    c3 = [e for e in lt if e["beta"] == 1e-3][0]
    defs = [
        ["C_inf(alpha=0,beta=0)", "stationary value, no drive; analytic (Memarzadeh & Mancini)", ref, ref, 0.0],
        ["C_inf(alpha=20,beta=0)", "stationary value (projection on null space)", scan[4][20], ref, 100 * (scan[4][20] - ref) / ref],
        ["C_inf(alpha=100,beta=0)", "stationary value", scan[4][100], ref, 100 * (scan[4][100] - ref) / ref],
        ["C_inf(alpha=300,beta=0)", "stationary value (~ alpha->infinity plateau)", scan[4][300], ref, 100 * (scan[4][300] - ref) / ref],
        ["C_peak(alpha=20,beta=0)", "maximum of C(t): transient overshoot above C_inf", cp0, cpn0, 100 * (cp0 - cpn0) / cpn0],
        ["C_peak(alpha=20,beta=1e-3)", "maximum of C(t), compared with C_peak(no drive, same beta)",
         c3["drive"]["C_peak"], c3["nodrive"]["C_peak"], 100 * (c3["drive"]["C_peak"] / c3["nodrive"]["C_peak"] - 1)],
    ]
    write_csv(f"{out}/table_definitions_n4.csv", ["quantity", "definition", "value", "reference_value", "change_percent_vs_reference"], defs)
    res["definitions_n4"] = defs

    # ---- 6.7 initial-state dependence, n=4 ----
    log("Initial-state dependence n=4, alpha=20")
    states = ["1100", "1010", "1001", "0110", "0101", "0011"]
    L0 = liouvillian(4, 20.0, [0])
    R, Lf, k, _ = null_decomposition_dense(L0.toarray())
    isd = []
    for s in states:
        v = project(R, Lf, init_vec(s))
        r = lifetime_row(4, s, 20.0, [0], 1e-3)
        isd.append(dict(state=s, q2q3=s[1:3], q2_neq_q3=(s[1] != s[2]), C_inf_beta0=C23(v, 4),
                        C_peak_beta1e3=r["C_peak"], tau_beta1e3=r["tau"], tau_status=r["status"]))
        log("  ", isd[-1])
    # consistency rules for the initial-state table (checked, not assumed)
    for e in isd:
        assert (e["C_inf_beta0"] > 1e-6) == e["q2_neq_q3"], "rule 'q2 != q3  <=>  C_inf > 0' violated"
    res["initial_state"] = isd
    write_csv(f"{out}/table3_initial_state.csv", list(isd[0].keys()), [list(r.values()) for r in isd])

    # ---- 6.8 leakage out of the symmetric subspace by local dephasing ----
    log("Leakage from the symmetric (Dicke) subspace under local dephasing, n=4, alpha=0")
    P = sym_projector(4)
    v0 = np.zeros(16, complex)
    v0[[int(b, 2) for b in ["1100", "1010", "1001", "0110", "0101", "0011"]]] = 1 / np.sqrt(6)
    rho0 = np.outer(v0, v0).flatten("F")
    leak = []
    for b in [1e-3, 1e-2, 1e-1]:
        Ld = liouvillian(4, 0.0, [], b).toarray()
        for t in [1, 10, 100]:
            r = to_rho(expm(Ld * t) @ rho0, 4)
            leak.append(dict(beta=b, t=t, weight_outside=float(1 - np.trace(P @ r).real)))
    res["symmetric_subspace_leakage"] = leak
    for e in leak:
        log("  ", e)

    # ---- 6.9 numerical convergence / independent integrator ----
    log("Convergence: grid halving and independent integrator (n=4)")
    conv = convergence_table(4)
    res["convergence"] = conv
    write_csv(f"{out}/table_convergence.csv", list(conv[0].keys()), [list(r.values()) for r in conv])
    for r in conv:
        log("  ", r)

    # ---- 6.10 physical units: illustrative grid, NOT literature values ----
    phys = []
    for tau_c in (0.1, 1.0, 10.0):                 # microseconds, 1/gamma_c
        for Tphi in (1.0, 10.0, 100.0, 1000.0):     # microseconds
            beta = tau_c / (2 * Tphi)
            phys.append([tau_c, Tphi, beta, (1e-4 <= beta <= 1e-2), 9 * tau_c])
    write_csv(f"{out}/table_physical_units_illustrative.csv",
              ["1/gamma_c [us]", "T_phi [us]", "beta=(1/gamma_c)/(2 T_phi)", "beta_within_studied_range[1e-4,1e-2]",
               "t_peak_upper_estimate(9/gamma_c) [us]"], phys)

    # ---- 6.11 curves for figure 3 ----
    traces = {}
    for b in [0.0, 1e-3, 1e-2]:
        for tag, a, dr in (("nodrive", 0.0, []), ("drive", 20.0, [0])):
            ts, cs, raw = transient(4, liouvillian(4, a, dr, b).toarray(), "1100", [(0.1, 60.0)])
            traces[f"{b}|{tag}"] = (ts, cs)
    json.dump(res, open(f"{out}/results.json", "w"), indent=2, default=float)
    log("results.json written")

    # ---- Plots ----
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        for n, col, mk in ((3, "#1f77b4", "o"), (4, "#d62728", "s")):
            ax[0].semilogx(alphas, [scan[n][a] for a in alphas], mk + "-", color=col, label=f"n={n}")
            ax[0].axhline(ff[n], ls="--", color=col, alpha=.5)
        ax[0].set_xlabel(r"$\alpha/\gamma_c$"); ax[0].set_ylabel(r"$C_{2,3}(\infty)$ ($\beta=0$)")
        ax[0].set_title("Asymmetric drive on qubit 1 (dashed: field-free)"); ax[0].legend(); ax[0].grid(alpha=.3)
        for n in range(3, nmax + 1):
            xs = sorted(scan[n]); ax[1].semilogx(xs, [scan[n][a] / ff[n] for a in xs], "o-", label=f"n={n}")
        ax[1].axhline(1, color="k", lw=.8); ax[1].set_xlabel(r"$\alpha/\gamma_c$")
        ax[1].set_ylabel(r"$C_{\rm drive}/C_{\rm field-free}$"); ax[1].legend(); ax[1].grid(alpha=.3)
        ax[1].set_title("Relative change")
        fig.tight_layout(); fig.savefig(f"{out}/fig1_asymmetric_drive.pdf"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        nn = list(range(3, nmax + 1))
        ax.plot(nn, [ff[n] for n in nn], "o-", label="field-free")
        ax.plot(nn, [scan[n][20] for n in nn], "s-", label=r"drive $\alpha=20$")
        ax.set_xticks(nn); ax.set_xlabel("n"); ax.set_ylabel(r"$C_{2,3}(\infty)$"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{out}/fig2_vs_n.pdf"); plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
        for j, tag in enumerate(("nodrive", "drive")):
            for b, c in zip([0.0, 1e-3, 1e-2], ["#1f77b4", "#ff7f0e", "#d62728"]):
                ts, cs = traces[f"{b}|{tag}"]
                ax[j].plot(ts, cs, color=c, label=rf"$\beta={b:g}$")
            ax[j].set_xlabel(r"$t\gamma_c$"); ax[j].set_ylabel(r"$C_{2,3}(t)$"); ax[j].grid(alpha=.3); ax[j].legend()
            ax[j].set_title("No drive" if tag == "nodrive" else r"Drive $\alpha=20$ on qubit 1")
        fig.tight_layout(); fig.savefig(f"{out}/fig3_time_traces.pdf"); plt.close(fig)

        # advantage window figure: Delta C(t) = C_drive - C_nodrive
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for b, c in zip([0.0, 1e-3, 1e-2], ["#1f77b4", "#ff7f0e", "#d62728"]):
            ts, cd_ = traces[f"{b}|drive"]; _, cn_ = traces[f"{b}|nodrive"]
            ax.plot(ts, cd_ - cn_, color=c, label=rf"$\beta={b:g}$")
        ax.axhline(0, color="k", lw=.8); ax.set_xlabel(r"$t\gamma_c$")
        ax.set_ylabel(r"$C_{2,3}^{\rm drive}(t)-C_{2,3}^{\rm no\ drive}(t)$"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(f"{out}/fig3b_advantage_window.pdf"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for key, mk, c, lab in (("nodrive", "o", "#1f77b4", "no drive"), ("drive", "s", "#d62728", r"drive $\alpha=20$")):
            ax.loglog(bp, [e[key]["tau"] for e in lt if e["beta"] > 0], mk + "-", color=c, label=lab)
        ax.set_xlabel(r"$\beta/\gamma_c$"); ax.set_ylabel(r"$\tau_{1/e}$ ($1/\gamma_c$)"); ax.legend()
        ax.grid(alpha=.3, which="both"); fig.tight_layout(); fig.savefig(f"{out}/fig4_lifetime.pdf"); plt.close(fig)

        # only defined lifetimes are drawn; undefined ones are marked
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        lab = [f"{e['state']}\n(q2q3={e['q2q3']})" for e in isd]
        ax[0].bar(range(len(isd)), [e["C_inf_beta0"] for e in isd], color="#2ca02c")
        ax[0].set_xticks(range(len(isd))); ax[0].set_xticklabels(lab, fontsize=8)
        ax[0].set_ylabel(r"$C_{2,3}(\infty)$"); ax[0].set_title(r"$\beta=0$, $\alpha=20$ (stationary value)")
        for i, e in enumerate(isd):
            ax[0].text(i, e["C_inf_beta0"] + 0.005, f"{e['C_inf_beta0']:.4f}", ha="center", fontsize=8)
        ax[1].set_xticks(range(len(isd))); ax[1].set_xticklabels(lab, fontsize=8)
        for i, e in enumerate(isd):
            if np.isfinite(e["tau_beta1e3"]):
                ax[1].bar(i, e["tau_beta1e3"], color="#ff7f0e")
                ax[1].text(i, e["tau_beta1e3"] + 3, f"{e['tau_beta1e3']:.0f}", ha="center", fontsize=8)
            else:
                ax[1].text(i, 5, "undefined\n" + r"($C_{\rm peak}<0.01$)", ha="center", fontsize=8, color="gray")
        ax[1].set_ylim(0, 1.15 * max(e["tau_beta1e3"] for e in isd if np.isfinite(e["tau_beta1e3"])))
        ax[1].set_ylabel(r"$\tau_{1/e}$ ($1/\gamma_c$)"); ax[1].set_title(r"$\beta=10^{-3}$, $\alpha=20$ (lifetime)")
        fig.tight_layout(); fig.savefig(f"{out}/fig5_initial_state.pdf"); plt.close(fig)
        log("figures written")
    log("DONE")


if __name__ == "__main__":
    main()
