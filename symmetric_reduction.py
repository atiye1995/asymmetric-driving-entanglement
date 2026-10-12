#!/usr/bin/env python3
"""
symmetric_reduction.py
======================
Exact reduction to the permutation-symmetric sector of qubits 3..n for beta = gamma_ind = 0 and the initial state
|1>_1 |1>_2 |0...0>_{3..n}. The state stays in C^2_1 (x) C^2_2 (x) Dicke_{n-2} (dimension 4(n-1) instead of 2^n);
the pair (2,3) is obtained by splitting qubit 3 off the Dicke space. Conventions as in collective_entanglement.py.
For beta > 0 or other initial states use extended_results.py.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import collective_entanglement as ce

SM2 = np.array([[0, 1], [0, 0]], dtype=complex)
SX2 = np.array([[0, 1], [1, 0]], dtype=complex)
I2c = np.eye(2, dtype=complex)


def dicke_lowering(k):
    d = k + 1
    J = np.zeros((d, d), dtype=complex)
    for j in range(1, k + 1):
        J[j - 1, j] = np.sqrt(j * (k - j + 1))
    return J


def D2(A):
    """column-stacked superoperator of 2 D[A] (sparse CSR; dense kron is too large for n ~ 20)."""
    A = sp.csr_matrix(A)
    d = A.shape[0]
    Id = sp.identity(d, format="csr", dtype=complex)
    AdA = (A.getH() @ A).tocsr()
    return (2 * sp.kron(A.conj(), A, format="csr") - sp.kron(Id, AdA, format="csr")
            - sp.kron(AdA.T, Id, format="csr")).tocsr()


def full_sym_liouvillian(n, alpha=0.0):
    """Liouvillian on q1 (x) q2 (x) Dicke_{n-2}; collective decay + drive on q1."""
    k = n - 2
    d = k + 1
    Id = np.eye(d, dtype=complex)
    J = np.kron(SM2, np.kron(I2c, Id)) + np.kron(I2c, np.kron(SM2, Id)) + np.kron(I2c, np.kron(I2c, dicke_lowering(k)))
    L = D2(J)
    if alpha:
        H = sp.csr_matrix(alpha * np.kron(SX2, np.kron(I2c, Id)))
        N = H.shape[0]
        IN = sp.identity(N, format="csr", dtype=complex)
        L = L - 1j * (sp.kron(IN, H, format="csr") - sp.kron(H.T, IN, format="csr"))
    return L.tocsc()


def full_init(n):
    d = n - 1
    v = np.zeros(4 * d, dtype=complex)
    v[(1 * 2 + 1) * d + 0] = 1.0                   # q1=1, q2=1, Dicke j=0
    return np.outer(v, v).flatten("F")


def split_isometry(k):
    """W: Dicke_k -> C^2_3 (x) Dicke_{k-1},  shape (2*k, k+1)."""
    W = np.zeros((2, k, k + 1))
    for j in range(k + 1):
        if j <= k - 1:
            W[0, j, j] = np.sqrt((k - j) / k)
        if j >= 1:
            W[1, j - 1, j] = np.sqrt(j / k)
    return W


def C23_full_sym(v, n):
    k = n - 2
    d = k + 1
    rho = v.reshape(4 * d, 4 * d, order="F").reshape(2, 2, d, 2, 2, d)
    W = split_isometry(k)
    tmp = np.einsum("ajx,pqxPQY->pqajPQY", W, rho)
    out = np.einsum("pqajPQY,bkY->pqajPQbk", tmp, W.conj())    # (p,q,a,j ; P,Q,b,k)
    red = np.einsum("pqajpQbj->qaQb", out)                     # trace q1 (p=P) and Dicke_{k-1} (j=k)
    return ce.concurrence(red.reshape(4, 4))


def eff_sym_liouvillian(n, lam=1.0, flip=0.5):
    """Strong-drive effective model on rest = q2 (x) Dicke_{n-2}: blocks rho_+, rho_-."""
    k = n - 2
    d = k + 1
    Id = np.eye(d, dtype=complex)
    JR = np.kron(SM2, Id) + np.kron(I2c, dicke_lowering(k))
    D = JR.shape[0]
    shift = np.eye(D, dtype=complex)
    Lp = D2(JR + lam * 0.5 * shift)
    Lm = D2(JR - lam * 0.5 * shift)
    N = D * D
    I_N = sp.identity(N, format="csr", dtype=complex)
    return sp.bmat([[Lp - flip * I_N, flip * I_N], [flip * I_N, Lm - flip * I_N]], format="csc")


def eff_init(n):
    k = n - 2
    d = k + 1
    D = 2 * d
    v = np.zeros(D, dtype=complex)
    v[1 * d + 0] = 1.0                              # q2=1, Dicke j=0
    r = np.outer(v, v).flatten("F")
    return np.concatenate([0.5 * r, 0.5 * r])


def C23_eff_sym(vec, n):
    k = n - 2
    d = k + 1
    D = 2 * d
    N = D * D
    rho = (vec[:N] + vec[N:]).reshape(D, D, order="F").reshape(2, d, 2, d)
    W = split_isometry(k)
    tmp = np.einsum("ajx,qxQY->qajQY", W, rho)
    out = np.einsum("qajQY,bkY->qajQbk", tmp, W.conj())
    red = np.einsum("qajQbj->qaQb", out)
    return ce.concurrence(red.reshape(4, 4))


def steady_dense(L, v0):
    """Stationary state reached from v0 (projection on the null space; dense SVD, fine up to dim ~ 5000)."""
    L = L.toarray() if sp.issparse(L) else L
    R, Lf, k, gap = ce.null_decomposition_dense(L)
    v = ce.project(R, Lf, v0)
    return v, k, gap, float(np.linalg.norm(L @ v))


def steady_evolution(L, v0, t_chunk=25.0, t_max=1500.0, tol=1e-10):
    Ls = sp.csc_matrix(L)
    v, t = v0.copy(), 0.0
    while t < t_max:
        v = sla.expm_multiply(Ls * t_chunk, v)
        t += t_chunk
        if np.linalg.norm(Ls @ v) < tol:
            return v, True
    return v, False
