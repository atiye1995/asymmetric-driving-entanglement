#!/usr/bin/env python3
"""
extended_results.py   (permutation-invariant reduction for any beta, alpha and computational-basis initial state)
================================================================================================================
With the drive on qubit 1 only, the Liouvillian (collective decay, drive, local dephasing and independent decay;
each sum_i D[.] is permutation invariant as a superoperator) commutes with every permutation of qubits 2..n.
A pure initial state is invariant under permutations of qubits carrying the same initial bit, so rho(t) lives in
        Op(q1) (x) Op(q2) (x) Sym^{m0}(C^4) (x) Sym^{m1}(C^4),
where qubits 3..n are split into the initially-0 group (m0) and the initially-1 group (m1) and C^4 = span{|a><b|}.
The dimension is 16*C(m0+3,3)*C(m1+3,3) instead of 4^n. The construction is exact; the permutation-invariant
solver idea follows Shammah et al., Phys. Rev. A 98, 063815 (2018).

The script first validates the construction against the full-space code (stationary and transient, with and
without dephasing) and against symmetric_reduction.py, and aborts on any mismatch.

Outputs (in --outdir)
  table2_extended.csv                peak, lifetime, sudden death, ratios, n = 3..nmax-table2
  table_beta_positive_extended.csv   null-space dimension, C_inf, signed concurrence, beta > 0
  table3_extended.csv                all two-excitation initial states (beta=0 stationary value; rule C_inf>0 <=> q2 != q3)
  table_alpha_beta_scan.csv          peak/lifetime versus alpha with dephasing
  table_pi_validation.csv            validation errors
  fig7_table2_extended.pdf           peak ratio and lifetime ratio vs n

Usage
  python extended_results.py --outdir results [--quick] [--only validate,beta]
"""
import argparse, csv, json, math, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collective_entanglement as ce

SM2 = np.array([[0, 1], [0, 0]], dtype=complex)      # sigma^- = |0><1|
SP2 = SM2.T.copy()
SX2 = np.array([[0, 1], [1, 0]], dtype=complex)
SZ2 = np.array([[1, 0], [0, -1]], dtype=complex)
I2c = np.eye(2, dtype=complex)
I4c = np.eye(4, dtype=complex)


def loc_L(A):            # local superoperator: rho -> A rho   (column stacking, mu = a + 2 b)
    return np.kron(I2c, A)


def loc_R(B):            # local superoperator: rho -> rho B
    return np.kron(B.T, I2c)


# ----------------------------------------------------------------------------
# Symmetric (bosonic) space of m copies of the 4-dim local operator space
# ----------------------------------------------------------------------------
def compositions(m, k=4):
    if k == 1:
        yield (m,)
        return
    for i in range(m + 1):
        for rest in compositions(m - i, k - 1):
            yield (i,) + rest


class Sym:
    def __init__(self, m):
        self.m = m
        self.states = list(compositions(m))
        self.index = {s: i for i, s in enumerate(self.states)}
        self.dim = len(self.states)
        self._E = {}

    def E(self, mu, nu):
        """b^+_mu b_nu in the occupation basis (sparse)."""
        key = (mu, nu)
        if key in self._E:
            return self._E[key]
        rows, cols, vals = [], [], []
        for j, n in enumerate(self.states):
            if n[nu] == 0:
                continue
            if mu == nu:
                rows.append(j); cols.append(j); vals.append(float(n[mu]))
            else:
                t = list(n); t[nu] -= 1; t[mu] += 1
                rows.append(self.index[tuple(t)]); cols.append(j)
                vals.append(math.sqrt(n[nu] * (n[mu] + 1)))
        M = sp.csr_matrix((vals, (rows, cols)), shape=(self.dim, self.dim), dtype=complex)
        self._E[key] = M
        return M

    def onebody(self, A):
        """sum_i A_i  ->  sum_{mu nu} A[mu,nu] b^+_mu b_nu"""
        M = sp.csr_matrix((self.dim, self.dim), dtype=complex)
        for mu in range(4):
            for nu in range(4):
                if A[mu, nu] != 0:
                    M = M + A[mu, nu] * self.E(mu, nu)
        return M

    def marginal_functional(self):
        """4 x dim: coefficient of e_mu on ONE chosen slot after tracing the other m-1 slots."""
        W = np.zeros((4, self.dim))
        m = self.m
        for j, n in enumerate(self.states):
            Nn = math.factorial(m) / np.prod([math.factorial(x) for x in n])
            for mu in range(4):
                if n[mu] == 0:
                    continue
                t = list(n); t[mu] -= 1
                if t[1] != 0 or t[2] != 0:
                    continue
                arr = math.factorial(m - 1) / np.prod([math.factorial(x) for x in t])
                W[mu, j] = arr / math.sqrt(Nn)
        return sp.csr_matrix(W)

    def trace_functional(self):
        """1 x dim: trace over all m slots."""
        T = np.zeros((1, self.dim))
        for j, n in enumerate(self.states):
            if n[1] == 0 and n[2] == 0:
                T[0, j] = math.sqrt(math.factorial(self.m) / np.prod([math.factorial(x) for x in n]))
        return sp.csr_matrix(T)


class PISystem:
    """Space for an initial computational-basis label, e.g. '1100' or '010110'."""

    def __init__(self, label):
        self.label = label
        n = len(label)
        self.n = n
        rest = label[2:]
        self.m0, self.m1 = rest.count("0"), rest.count("1")
        # groups: (name, size, initial local mu, contains qubit 3)
        q3bit = label[2]
        self.groups = [("q1", 1, int(label[0]) * 3, False), ("q2", 1, int(label[1]) * 3, False)]
        if self.m0:
            self.groups.append(("G0", self.m0, 0, q3bit == "0"))
        if self.m1:
            self.groups.append(("G1", self.m1, 3, q3bit == "1"))
        self.sym = [Sym(g[1]) for g in self.groups]
        self.dims = [s.dim for s in self.sym]
        self.dim = int(np.prod(self.dims))

    def _embed(self, k, M):
        left = int(np.prod(self.dims[:k])) if k else 1
        right = int(np.prod(self.dims[k + 1:])) if k + 1 < len(self.dims) else 1
        return sp.kron(sp.kron(sp.identity(left, format="csr"), M, format="csr"),
                       sp.identity(right, format="csr"), format="csr")

    def total(self, A, only=None):
        """sum over all qubits of the local superoperator A (4x4)."""
        out = sp.csr_matrix((self.dim, self.dim), dtype=complex)
        for k, s in enumerate(self.sym):
            if only is not None and k != only:
                continue
            out = out + self._embed(k, s.onebody(A))
        return out.tocsr()

    def liouvillian(self, alpha=0.0, beta=0.0, gamma_ind=0.0):
        LJ, LJd = self.total(loc_L(SM2)), self.total(loc_L(SP2))
        RJ, RJd = self.total(loc_R(SM2)), self.total(loc_R(SP2))
        L = 2 * (LJ @ RJd) - LJd @ LJ - RJ @ RJd
        if alpha != 0:
            L = L - 1j * alpha * (self.total(loc_L(SX2), only=0) - self.total(loc_R(SX2), only=0))
        if beta > 0:
            Dz = loc_L(SZ2) @ loc_R(SZ2) - I4c
            L = L + beta * self.total(Dz)
        if gamma_ind > 0:      # independent decay  gamma_ind * sum_i D[sigma^-_i]  (permutation invariant)
            Dm = (loc_L(SM2) @ loc_R(SP2) - 0.5 * loc_L(SP2 @ SM2) - 0.5 * loc_R(SP2 @ SM2))
            L = L + gamma_ind * self.total(Dm)
        return L.tocsc()

    def init_vec(self):
        v = np.array([1.0 + 0j])
        for g, s in zip(self.groups, self.sym):
            e = np.zeros(s.dim, dtype=complex)
            occ = [0, 0, 0, 0]; occ[g[2]] = g[1]
            e[s.index[tuple(occ)]] = 1.0
            v = np.kron(v, e)
        return v

    def reducer(self):
        """16 x dim sparse matrix: state vector -> (mu2, mu3) components of rho_{q2 q3}."""
        M = sp.identity(1, format="csr")
        for (name, m, mu0, has3), s in zip(self.groups, self.sym):
            if name == "q1":
                F = sp.csr_matrix(np.array([[1.0, 0, 0, 1.0]]))
            elif name == "q2":
                F = sp.identity(4, format="csr")
            elif has3:
                F = s.marginal_functional()
            else:
                F = s.trace_functional()
            M = sp.kron(M, F, format="csr")
        return M.tocsr()


def rho23_from_components(r16):
    """r16[mu2*4+mu3] (mu = a+2b)  ->  4x4 two-qubit density matrix in the basis |a2 a3>."""
    rho = np.zeros((4, 4), dtype=complex)
    for mu2 in range(4):
        a2, b2 = mu2 % 2, mu2 // 2
        for mu3 in range(4):
            a3, b3 = mu3 % 2, mu3 // 2
            rho[2 * a2 + a3, 2 * b2 + b3] = r16[mu2 * 4 + mu3]
    return rho


def C_pi(S, red, v, raw=False):
    r = rho23_from_components(red @ v)
    c = ce.concurrence_raw(r)
    return c if raw else max(0.0, c)


# ----------------------------------------------------------------------------
# Stationary states and transients in the PI space
# ----------------------------------------------------------------------------
def stationary_pi(S, L, v0, dense_limit=2000):
    if S.dim <= dense_limit:
        R, Lf, k, gap = ce.null_decomposition_dense(L.toarray())
    else:
        R, Lf, k = ce.null_decomposition_sparse(L)
        gap = float("nan")
    v = ce.project(R, Lf, v0)
    return v, dict(null_dim=k, min_nonnull_sv=gap, residual=float(np.linalg.norm(L @ v)))


def transient_pi(S, red, Ld, segments):
    v = S.init_vec()
    t, ts, raw = 0.0, [0.0], [C_pi(S, red, v, raw=True)]
    for dt, t_end in segments:
        E = expm(Ld * dt)
        for _ in range(int(round((t_end - t) / dt))):
            v = E @ v
            t += dt
            ts.append(t)
            raw.append(C_pi(S, red, v, raw=True))
    raw = np.array(raw)
    return np.array(ts), np.clip(raw, 0, None), raw


def transient_pi_sparse(S, red, L, segments, chunk=100):
    """Same as transient_pi but with sparse Krylov propagation (scipy expm_multiply on uniform
    sub-grids); used when the dense matrix exponential is too large (dim > DENSE_EXPM_LIMIT)."""
    Ls = L.tocsc()
    v = S.init_vec()
    t, ts, raw = 0.0, [0.0], [C_pi(S, red, v, raw=True)]
    for dt, t_end in segments:
        nsteps = int(round((t_end - t) / dt))
        done = 0
        while done < nsteps:
            k = min(chunk, nsteps - done)
            X = sla.expm_multiply(Ls, v, start=0.0, stop=k * dt, num=k + 1, endpoint=True)
            R = red @ X[1:].T                                  # 16 x k reduced components
            for j in range(k):
                raw.append(ce.concurrence_raw(rho23_from_components(R[:, j])))
                ts.append(t + (j + 1) * dt)
            v = X[-1]
            t += k * dt
            done += k
    raw = np.array(raw)
    return np.array(ts), np.clip(raw, 0, None), raw


DENSE_EXPM_LIMIT = 1500


def _transient(S, red, alpha, beta, scale):
    L = S.liouvillian(alpha, beta)
    seg = ce.segments_for(beta, scale)
    if S.dim <= DENSE_EXPM_LIMIT:
        return transient_pi(S, red, L.toarray(), seg)
    return transient_pi_sparse(S, red, L, seg)


def lifetime_row_pi(label, alpha, beta, halving=True):
    S = PISystem(label)
    red = S.reducer()
    ts, cs, raw = _transient(S, red, alpha, beta, 1.0)
    cpk, tpk, tau, st = ce.peak_and_lifetime(ts, cs)
    row = dict(C_peak=cpk, t_peak=tpk, tau=tau, status=st, t_death=ce.death_time(ts, raw, cs),
               C_peak_err=float("nan"), tau_err=float("nan"), trace=(ts, cs, raw))
    if halving:
        tsB, csB, _ = _transient(S, red, alpha, beta, 0.5)
        cB, _, tB, _ = ce.peak_and_lifetime(tsB, csB)
        row["C_peak_err"] = abs(cpk - cB)
        row["tau_err"] = abs(tau - tB) if np.isfinite(tau) and np.isfinite(tB) else float("nan")
    return row


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate(outdir, log, full_big=True):
    import symmetric_reduction as sr
    rows = []
    # (a) stationary values: full space vs PI, with and without dephasing, several initial states
    cases = [(3, "110", 0.0, 0.0), (4, "1100", 0.0, 0.0), (4, "1100", 20.0, 0.0), (5, "11000", 20.0, 0.0),
             (4, "0101", 20.0, 0.0), (4, "1001", 20.0, 0.0), (5, "10110", 20.0, 0.0), (5, "01100", 0.0, 0.0),
             (4, "1100", 20.0, 1e-3), (5, "11000", 20.0, 1e-2), (4, "0110", 0.0, 1e-2)]
    for n, lab, a, b in cases:
        dr = [0] if a else []
        vf, _ = ce.steady_state(n, ce.liouvillian(n, a, dr, b), lab)
        S = PISystem(lab); red = S.reducer()
        vp, info = stationary_pi(S, S.liouvillian(a, b), S.init_vec())
        cf, cp = ce.C23(vf, n), C_pi(S, red, vp)
        rows.append(["stationary", n, lab, a, b, cf, cp, abs(cf - cp)])
        log(f"  validate stationary n={n} {lab} alpha={a} beta={b}: full={cf:.8f} PI={cp:.8f} dim={S.dim}")
        assert abs(cf - cp) < 1e-6, "PI stationary value differs from full space"
    # (b) transient with dephasing
    for n, lab, a, b in [(4, "1100", 20.0, 1e-2), (5, "11000", 20.0, 1e-2), (4, "0101", 20.0, 1e-2)]:
        dr = [0] if a else []
        S = PISystem(lab); red = S.reducer()
        LdF = ce.liouvillian(n, a, dr, b).toarray()
        LdP = S.liouvillian(a, b).toarray()
        EF, EP = expm(LdF * 0.5), expm(LdP * 0.5)
        vF, vP = ce.init_vec(lab), S.init_vec()
        err = 0.0
        for _ in range(120):
            vF, vP = EF @ vF, EP @ vP
            err = max(err, abs(ce.C23(vF, n) - C_pi(S, red, vP)))
        rows.append(["transient(60)", n, lab, a, b, "", "", err])
        log(f"  validate transient n={n} {lab} beta={b}: max|dC|={err:.2e}")
        assert err < 1e-8
    # (b2) n = 6 transient with dephasing against the full 4^6-dimensional space (sparse propagation)
    S = PISystem("110000"); red = S.reducer()
    LF, LP = ce.liouvillian(6, 20.0, [0], 1e-2), S.liouvillian(20.0, 1e-2)
    vF, vP, err = ce.init_vec("110000"), S.init_vec(), 0.0
    for _ in range(4):
        vF, vP = sla.expm_multiply(LF * 5.0, vF), sla.expm_multiply(LP * 5.0, vP)
        err = max(err, abs(ce.C23(vF, 6) - C_pi(S, red, vP)))
    rows.append(["transient(20), full space n=6", 6, "110000", 20.0, 1e-2, "", "", err])
    log(f"  validate transient n=6 (full 4^6 space) beta=1e-2: max|dC|={err:.2e}")
    assert err < 1e-8
    # (b3) sparse (ARPACK) null-space path vs dense SVD path on the same PI system
    for lab, bt in (("101000", 0.0), ("110000", 1e-2)):
        S = PISystem(lab); red = S.reducer(); L = S.liouvillian(20.0, bt)
        vd, _ = stationary_pi(S, L, S.init_vec(), dense_limit=10 ** 9)
        vs_, _ = stationary_pi(S, L, S.init_vec(), dense_limit=0)
        d = abs(C_pi(S, red, vd) - C_pi(S, red, vs_))
        rows.append(["sparse vs dense null space", 6, lab, 20.0, bt, C_pi(S, red, vd), C_pi(S, red, vs_), d])
        log(f"  validate sparse-vs-dense null space {lab} beta={bt}: |dC|={d:.2e}")
        assert d < 1e-8
    # (b4) direct full-space check for n = 7, 8 (4^7, 4^8 dimensional sparse propagation)
    if full_big:
        for n, dt_, nstep in ((7, 5.0, 3), (8, 5.0, 2)):
            lab = "11" + "0" * (n - 2)
            S = PISystem(lab); red = S.reducer()
            LF, LP = ce.liouvillian(n, 20.0, [0], 1e-2), S.liouvillian(20.0, 1e-2)
            vF, vP, err = ce.init_vec(lab), S.init_vec(), 0.0
            for _ in range(nstep):
                vF, vP = sla.expm_multiply(LF * dt_, vF), sla.expm_multiply(LP * dt_, vP)
                err = max(err, abs(ce.C23(vF, n) - C_pi(S, red, vP)))
            rows.append([f"transient({nstep * dt_:g}), full space n={n}", n, lab, 20.0, 1e-2, "", "", err])
            log(f"  validate transient n={n} (full 4^{n} space) beta=1e-2: max|dC|={err:.2e}")
            assert err < 1e-8
    # (b5) sparse Krylov transient (used for dim > 1500, i.e. n >= 9) vs dense expm on the same system
    global DENSE_EXPM_LIMIT
    S = PISystem("1100000"); red = S.reducer()
    old, DENSE_EXPM_LIMIT = DENSE_EXPM_LIMIT, 10 ** 9
    t_d, c_d, _ = _transient(S, red, 20.0, 1e-2, 1.0)
    DENSE_EXPM_LIMIT = 0
    t_s, c_s, _ = _transient(S, red, 20.0, 1e-2, 1.0)
    DENSE_EXPM_LIMIT = old
    d = float(np.abs(c_d - c_s).max())
    rows.append(["sparse vs dense transient (full curve)", 7, "1100000", 20.0, 1e-2, "", "", d])
    log(f"  validate sparse-vs-dense transient n=7 beta=1e-2: max|dC|={d:.2e}")
    assert d < 1e-8
    # (c) against the beta=0, |11 0..0> reduction up to n = 8
    for n in range(3, 9):
        vs, *_ = sr.steady_dense(sr.full_sym_liouvillian(n, 20.0), sr.full_init(n))
        S = PISystem("11" + "0" * (n - 2)); red = S.reducer()
        vp, _ = stationary_pi(S, S.liouvillian(20.0, 0.0), S.init_vec())
        cs_, cp = sr.C23_full_sym(vs, n), C_pi(S, red, vp)
        rows.append(["vs symmetric_reduction", n, "11" + "0" * (n - 2), 20.0, 0.0, cs_, cp, abs(cs_ - cp)])
        log(f"  validate vs symmetric_reduction n={n}: {cs_:.8f} vs {cp:.8f}")
        assert abs(cs_ - cp) < 1e-6
    ce.write_csv(f"{outdir}/table_pi_validation.csv",
                 ["test", "n", "label", "alpha", "beta", "C_reference", "C_PI", "abs_error"], rows)
    log("  validation PASSED")


# ----------------------------------------------------------------------------
# Production runs
# ----------------------------------------------------------------------------
def run_table2(outdir, nmax, betas, log, halving_nmax=6, nmin=3, big_n=9, big_betas=(1e-3, 1e-2)):
    rows, ratios = [], []
    for n in range(nmin, nmax + 1):
        label = ce.default_label(n)
        for b in (betas if n < big_n else [x for x in betas if x in big_betas]):
            t0 = time.time()
            h = n <= halving_nmax
            r0 = lifetime_row_pi(label, 0.0, b, h)
            rd = lifetime_row_pi(label, 20.0, b, h)
            w = ce.advantage_window(rd["trace"][0], rd["trace"][1], r0["trace"][1])
            rows.append([n, b, 1 / (2 * b) if b > 0 else float("inf"),
                         r0["C_peak"], r0["t_peak"], r0["tau"], r0["t_death"],
                         rd["C_peak"], rd["t_peak"], rd["tau"], rd["t_death"], rd["C_peak_err"], rd["tau_err"],
                         rd["C_peak"] / r0["C_peak"], rd["tau"] / r0["tau"] if np.isfinite(r0["tau"]) else float("nan"),
                         w["windows"][-1][1] if w["windows"] else float("nan"), w["integral"]])
            log(f"  n={n} beta={b:g}: Cpeak {r0['C_peak']:.4f}->{rd['C_peak']:.4f}  tau {ce.fmt(r0['tau'],1)}->{ce.fmt(rd['tau'],1)}  "
                f"ratio_tau={rows[-1][14]:.3f}  window_end={rows[-1][15]:.1f}  ({time.time()-t0:.0f}s)")
            ce.write_csv(f"{outdir}/table2_extended.csv",
                         ["n", "beta", "T_phi=1/(2beta)", "Cpeak_nodrive", "t_peak_nodrive", "tau_nodrive", "t_death_nodrive",
                          "Cpeak_drive(alpha=20)", "t_peak_drive", "tau_drive", "t_death_drive",
                          "Cpeak_err(grid halving; nan if not done)", "tau_err", "Cpeak_ratio", "tau_ratio",
                          "advantage_window_end", "integral_dC_dt"], rows)
    return rows


def run_beta_positive(outdir, nmax, log):
    rows = []
    for n in range(3, nmax + 1):
        S = PISystem(ce.default_label(n)); red = S.reducer(); v0 = S.init_vec()
        for a in (0.0, 20.0):
            for b in (1e-4, 1e-2):
                t0 = time.time()
                L = S.liouvillian(a, b)
                v, info = stationary_pi(S, L, v0)
                rho = rho23_from_components(red @ v)
                rows.append([n, a, b, S.dim, info["null_dim"], max(0.0, ce.concurrence_raw(rho)),
                             ce.concurrence_raw(rho), info["residual"], float(np.trace(rho).real)])
                log(f"  n={n} alpha={a:g} beta={b:g}: dim={S.dim} null_dim={info['null_dim']} "
                    f"C_signed={rows[-1][6]:+.3e} tr={rows[-1][8]:.8f} ({time.time()-t0:.0f}s)")
                ce.write_csv(f"{outdir}/table_beta_positive_extended.csv",
                             ["n", "alpha", "beta", "PI_dim", "null_dim", "C_inf", "C_signed", "residual", "trace_rho23"], rows)
    return rows


def run_table3(outdir, ns, log):
    """All two-excitation initial states.  The result depends only on (q1, q2, q3, number of
    excitations among qubits 4..n): the other labels are permutations of qubits 4..n, for which
    L and the pair (2,3) are invariant.  Each class is solved once (drive and field-free)."""
    import itertools
    cache, rows = {}, []
    for n in ns:
        for exc in itertools.combinations(range(n), 2):
            lab = "".join("1" if i in exc else "0" for i in range(n))
            key = (n, lab[:3], lab[3:].count("1"))
            if key not in cache:
                rep = lab[:3] + "0" * (n - 3 - key[2]) + "1" * key[2]
                S = PISystem(rep); red = S.reducer()
                v, info = stationary_pi(S, S.liouvillian(20.0, 0.0), S.init_vec())
                v0_, _ = stationary_pi(S, S.liouvillian(0.0, 0.0), S.init_vec())   # field-free, same state
                cache[key] = (C_pi(S, red, v), C_pi(S, red, v0_), S.dim, info["null_dim"], rep)
                log(f"  n={n} class q1q2q3={lab[:3]} ones_in_rest={key[2]} (rep {rep}): "
                    f"C_drive={cache[key][0]:.6f} C_ff={cache[key][1]:.6f} dim={S.dim}")
            c, c0, dim, nd, rep = cache[key]
            rows.append([n, lab, lab[1:3], lab[1] != lab[2], c, c0,
                         (100 * (c - c0) / c0) if c0 > 1e-9 else float("nan"), dim, nd])
            assert (c > 1e-6) == (lab[1] != lab[2]), f"rule q2!=q3 <=> C>0 violated for {lab}"
    ce.write_csv(f"{outdir}/table3_extended.csv",
                 ["n", "state(q1..qn)", "q2q3", "q2_neq_q3", "C_inf(alpha=20,beta=0)", "C_inf(field-free, same state)",
                  "change_percent_vs_field_free_same_state", "PI_dim", "null_dim"], rows)
    return rows


def run_alpha_beta(outdir, ns, beta, alphas, log):
    """Peak and lifetime vs drive strength WITH dephasing (is the peak enhancement an artefact of alpha=20?)."""
    rows = []
    for n in ns:
        label = ce.default_label(n)
        r0 = lifetime_row_pi(label, 0.0, beta, False)
        for a in alphas:
            rd = lifetime_row_pi(label, float(a), beta, False)
            rows.append([n, beta, a, r0["C_peak"], rd["C_peak"], rd["C_peak"] / r0["C_peak"],
                         r0["tau"], rd["tau"], rd["tau"] / r0["tau"] if np.isfinite(r0["tau"]) else float("nan")])
            log(f"  n={n} beta={beta:g} alpha={a:g}: Cpeak ratio={rows[-1][5]:.3f}  tau ratio={rows[-1][8]:.3f}")
            ce.write_csv(f"{outdir}/table_alpha_beta_scan.csv",
                         ["n", "beta", "alpha", "Cpeak_nodrive", "Cpeak_drive", "Cpeak_ratio",
                          "tau_nodrive", "tau_drive", "tau_ratio"], rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--nmax-table2", type=int, default=12)
    ap.add_argument("--nmax-beta", type=int, default=14)
    ap.add_argument("--table3-nmax-note", default="", help=argparse.SUPPRESS)
    ap.add_argument("--ab-n", type=int, nargs="+", default=[6, 8], help="n values for the alpha scan with dephasing")
    ap.add_argument("--ab-beta", type=float, default=1e-3)
    ap.add_argument("--nmin-table2", type=int, default=3)
    ap.add_argument("--table3-n", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--betas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--only", default="validate,beta,table3,alphabeta,table2")
    ap.add_argument("--no-full-validation", action="store_true", help="skip the n=7,8 full-space check (~1 min)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.nmax_table2, args.nmax_beta, args.table3_n, args.betas, args.ab_n = 6, 7, [5], [1e-3, 1e-2], [6]
    os.makedirs(args.outdir, exist_ok=True)
    T0 = time.time()

    def log(*a):
        print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

    todo = args.only.split(",")
    if "validate" in todo:
        log("Validation of the permutation-invariant construction")
        validate(args.outdir, log, full_big=not (args.quick or args.no_full_validation))
    if "beta" in todo:
        log("Stationary state for beta>0, n up to", args.nmax_beta)
        run_beta_positive(args.outdir, args.nmax_beta, log)
    if "table3" in todo:
        log("All two-excitation initial states (beta=0, alpha=20)")
        run_table3(args.outdir, args.table3_n, log)
    if "alphabeta" in todo:
        log("alpha scan with dephasing, beta =", args.ab_beta)
        run_alpha_beta(args.outdir, args.ab_n, args.ab_beta, [2, 5, 10, 20, 50, 100], log)
    if "table2" in todo:
        log("Peak and lifetime for n=3..", args.nmax_table2)
        rows = run_table2(args.outdir, args.nmax_table2, args.betas, log, nmin=args.nmin_table2)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
            for b in args.betas:
                sel = [r for r in rows if r[1] == b]
                ax[0].plot([r[0] for r in sel], [r[13] for r in sel], "o-", label=rf"$\beta={b:g}$")
                ax[1].plot([r[0] for r in sel], [r[14] for r in sel], "s-", label=rf"$\beta={b:g}$")
            ax[0].axhline(1, color="k", lw=.8); ax[1].axhline(1, color="k", lw=.8)
            ax[0].set_xlabel("n"); ax[0].set_ylabel(r"$C_{\rm peak}^{\rm drive}/C_{\rm peak}^{\rm no\ drive}$")
            ax[1].set_xlabel("n"); ax[1].set_ylabel(r"$\tau^{\rm drive}/\tau^{\rm no\ drive}$")
            for a_ in ax:
                a_.grid(alpha=.3); a_.legend()
            fig.tight_layout(); fig.savefig(f"{args.outdir}/fig7_table2_extended.pdf"); plt.close(fig)
        except Exception as e:                      # plotting must never lose the numbers
            log("plot skipped:", e)
    log("DONE")


if __name__ == "__main__":
    main()
