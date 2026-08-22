import numpy as np
from scipy.linalg import expm
import json
import time
import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt

# ============================================================================
# 1. OPERATORS AND LINDBLAD SUPEROPERATOR
# ============================================================================

def kron_list(ops):
    r = ops[0]
    for o in ops[1:]:
        r = np.kron(r, o)
    return r

I2 = np.eye(2, dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

def Si(n, i, op):
    l = [I2] * n
    l[i] = op
    return kron_list(l)

def buildL(n, alpha=0.0, drv=[], beta=0.0):
    dim = 2**n
    Id = np.eye(dim, dtype=complex)
    Jm = sum(Si(n, i, sm) for i in range(n))
    Jp = Jm.conj().T
    H = sum(alpha * Si(n, i, sx) for i in drv) if drv else np.zeros((dim, dim), dtype=complex)
    L = (2 * np.kron(Jm.conj(), Jm)
         - np.kron(Id, Jp @ Jm)
         - np.kron((Jp @ Jm).T, Id))
    if alpha != 0:
        L += -1j * (np.kron(Id, H) - np.kron(H.T, Id))
    if beta > 0:
        for i in range(n):
            Sz = Si(n, i, sz)
            L += beta * (np.kron(Sz, Sz) - np.kron(Id, Id))
    return L

def rho_state(n, state_str):
    dim = 2**n
    v = np.zeros(dim, dtype=complex)
    idx = int(state_str, 2)
    v[idx] = 1.0
    return np.outer(v, v)

def concurrence(rho2):
    sy = np.array([[0, -1j], [1j, 0]])
    ss = np.kron(sy, sy)
    R = rho2 @ ss @ rho2.conj() @ ss
    ev = np.sort(np.sqrt(np.abs(np.linalg.eigvals(R).real)))[::-1]
    return max(0.0, ev[0] - ev[1] - ev[2] - ev[3])

def ptrace_einsum(rho, n, i, j):
    """
    Partial trace using einsum (fast, works for n=6).
    """
    rho_tensor = rho.reshape([2] * n + [2] * n)
    alphabet_size = 2 * n
    if alphabet_size > 52:
        raise ValueError(f"Too many qubits ({n}) for einsum mapping. Max n=26.")
    all_indices = [chr(ord('a') + k) for k in range(alphabet_size)]
    output_indices = [all_indices[i], all_indices[j], all_indices[i + n], all_indices[j + n]]
    output_string = "".join(output_indices)
    trace_char_start = ord('x')
    trace_out_mapping = {}
    offset = 0
    for k in range(n):
        if k != i and k != j:
            trace_char = chr(trace_char_start - offset)
            trace_out_mapping[all_indices[k]] = trace_char
            trace_out_mapping[all_indices[k + n]] = trace_char
            offset += 1
    einsum_input_list = []
    for idx_char in all_indices:
        einsum_input_list.append(trace_out_mapping.get(idx_char, idx_char))
    einsum_input_string = "".join(einsum_input_list)
    reduced = np.einsum(f'{einsum_input_string}->{output_string}', rho_tensor)
    return reduced.reshape((4, 4))

def C23_from_rho(rho, n):
    rho_red = ptrace_einsum(rho, n, 1, 2)
    return concurrence(rho_red)

# ============================================================================
# 2. SOLVERS
# ============================================================================

def steady_state_concurrence(n, L, dt=2.0, t_max=400.0, init_state_str=None):
    dim = 2**n
    if init_state_str is None:
        init_state_str = '11' + '0' * (n - 2)
    rho0 = rho_state(n, init_state_str)
    rho_vec = rho0.flatten()
    eLdt = expm(L * dt)
    c_prev = -1.0
    for step in range(int(t_max / dt)):
        rho_vec = eLdt @ rho_vec
        if step % 20 == 0:
            rho = rho_vec.reshape(dim, dim)
            c = C23_from_rho(rho, n)
            if abs(c - c_prev) < 1e-7 and step > 50:
                return c, step * dt, True
            c_prev = c
    rho = rho_vec.reshape(dim, dim)
    return C23_from_rho(rho, n), t_max, False

def compute_lifetime(n, L, dt_fine=0.1, dt_coarse=2.0, t_max=3000.0,
                     init_state_str=None, save_curve=False):
    dim = 2**n
    if init_state_str is None:
        init_state_str = '11' + '0' * (n - 2)
    rho0 = rho_state(n, init_state_str)
    rho_vec = rho0.flatten()
    times = []
    Cs = []

    # Fine grid for peak
    eL_fine = expm(L * dt_fine)
    for step in range(int(30.0 / dt_fine) + 1):
        t = step * dt_fine
        rho = rho_vec.reshape(dim, dim)
        C = C23_from_rho(rho, n)
        times.append(t)
        Cs.append(C)
        rho_vec = eL_fine @ rho_vec

    times = np.array(times)
    Cs = np.array(Cs)
    C_peak = Cs.max()
    t_peak = times[Cs.argmax()]

    # Coarse tail
    eL_coarse = expm(L * dt_coarse)
    t_current = times[-1]
    while t_current < t_max:
        rho_vec = eL_coarse @ rho_vec
        t_current += dt_coarse
        rho = rho_vec.reshape(dim, dim)
        C = C23_from_rho(rho, n)
        times = np.append(times, t_current)
        Cs = np.append(Cs, C)
        if C < C_peak / np.e and t_current > t_peak:
            break

    idx = np.where((times >= t_peak) & (Cs < C_peak / np.e))[0]
    tau_e = times[idx[0]] if len(idx) > 0 else t_max

    if save_curve:
        return C_peak, t_peak, tau_e, times.tolist(), Cs.tolist()
    else:
        return C_peak, t_peak, tau_e, None, None

def lifetime_with_error(n, L, dt_fine_list=[0.1, 0.05], dt_coarse=2.0, t_max=3000.0):
    results = {}
    for dt in dt_fine_list:
        C_peak, _, tau, _, _ = compute_lifetime(n, L, dt_fine=dt, dt_coarse=dt_coarse, t_max=t_max)
        results[dt] = {'C_peak': C_peak, 'tau': tau}
    if len(dt_fine_list) == 2:
        C_peak_err = abs(results[dt_fine_list[0]]['C_peak'] - results[dt_fine_list[1]]['C_peak']) / 2
        tau_err = abs(results[dt_fine_list[0]]['tau'] - results[dt_fine_list[1]]['tau']) / 2
        dt_finer = max(dt_fine_list)
        return {
            'C_peak': results[dt_finer]['C_peak'],
            'C_peak_err': C_peak_err,
            'tau': results[dt_finer]['tau'],
            'tau_err': tau_err
        }
    else:
        return {'C_peak': results[dt_fine_list[0]]['C_peak'], 'C_peak_err': 0.0,
                'tau': results[dt_fine_list[0]]['tau'], 'tau_err': 0.0}

# ============================================================================
# 3. DATA GENERATION
# ============================================================================

print("=" * 80)
print("GENERATING FINAL PAPER DATA (v4)")
print("=" * 80)
start_time = time.time()

data = {}

# ---------- INITIAL STATES ----------
init_states = {
    'default': None,
    '1100': '1100',
    '1010': '1010',
    '1001': '1001',
    '0110': '0110',
    '0101': '0101',
    '0011': '0011'
}

# ---------- Field-free ----------
print("\n[Field-free]")
ff_data = {}
for n in range(3, 7):
    L = buildL(n)
    C, t_conv, ok = steady_state_concurrence(n, L)
    ff_data[str(n)] = float(C)
    print(f"  n={n}: C={C:.6f}  (conv at t={t_conv:.0f})")
data['field_free'] = ff_data

# ---------- Asymmetric alpha=20 ----------
print("\n[Asymmetric alpha=20]")
asym20_data = {}
for n in range(3, 7):
    L = buildL(n, alpha=20.0, drv=[0])
    C, t_conv, ok = steady_state_concurrence(n, L)
    asym20_data[str(n)] = float(C)
    print(f"  n={n}: C={C:.6f}  (conv at t={t_conv:.0f})")
data['asym_alpha20'] = asym20_data

# ---------- Sweep ----------
print("\n[Asymmetric sweep]")
alphas = [0.0, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0]
asym_sweep = {'n=3': {}, 'n=4': {}}
for n in [3, 4]:
    for alpha in alphas:
        drv = [0] if alpha > 0 else []
        L = buildL(n, alpha=alpha, drv=drv)
        C, _, _ = steady_state_concurrence(n, L)
        asym_sweep[f'n={n}'][str(alpha)] = float(C)
    print(f"  n={n}: min={min(asym_sweep[f'n={n}'].values()):.6f}, "
          f"max={max(asym_sweep[f'n={n}'].values()):.6f}")
data['asym_sweep'] = asym_sweep

# ---------- Uniform ----------
print("\n[Uniform drive]")
uniform_data = {}
for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    L = buildL(3, alpha=alpha, drv=[0, 1, 2])
    C, _, _ = steady_state_concurrence(3, L)
    uniform_data[str(alpha)] = float(C)
    print(f"  alpha={alpha:5.1f}: C={C:.6f}")
data['uniform_drive'] = uniform_data

# ---------- Lifetime table ----------
print("\n[Lifetime table]")
betas = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
lifetime_results = {}
for beta in betas:
    if beta == 0:
        L0 = buildL(4)
        Ld = buildL(4, alpha=20.0, drv=[0])
        C0, _, _ = steady_state_concurrence(4, L0)
        Cd, _, _ = steady_state_concurrence(4, Ld)
        lifetime_results[str(beta)] = {
            'C_peak_0': float(C0), 'tau_0': 'inf',
            'C_peak_d': float(Cd), 'tau_d': 'inf',
            'C_peak_0_err': 0.0, 'tau_0_err': 0.0,
            'C_peak_d_err': 0.0, 'tau_d_err': 0.0,
            'ratio': 1.0
        }
        print(f"  beta={beta:.0e}: C0={C0:.4f}, Cd={Cd:.4f}, tau=inf")
    else:
        L0 = buildL(4, beta=beta)
        Ld = buildL(4, alpha=20.0, drv=[0], beta=beta)
        res0 = lifetime_with_error(4, L0, t_max=3000.0)
        resd = lifetime_with_error(4, Ld, t_max=3000.0)
        ratio = resd['tau'] / res0['tau']
        lifetime_results[str(beta)] = {
            'C_peak_0': res0['C_peak'], 'C_peak_0_err': res0['C_peak_err'],
            'tau_0': res0['tau'], 'tau_0_err': res0['tau_err'],
            'C_peak_d': resd['C_peak'], 'C_peak_d_err': resd['C_peak_err'],
            'tau_d': resd['tau'], 'tau_d_err': resd['tau_err'],
            'ratio': ratio
        }
        print(f"  beta={beta:.1e}: tau0={res0['tau']:.0f} ± {res0['tau_err']:.1f}, "
              f"taud={resd['tau']:.0f} ± {resd['tau_err']:.1f}, ratio={ratio:.3f}")
data['lifetime_table'] = lifetime_results

# ---------- Initial-state dependence ----------
print("\n[Initial-state dependence, beta=0]")
init_dep = {}
for label, state_str in init_states.items():
    if label == 'default':
        state_str = '1100'
    L = buildL(4, alpha=20.0, drv=[0])
    C, _, _ = steady_state_concurrence(4, L, init_state_str=state_str)
    init_dep[label] = float(C)
    print(f"  {label} ('{state_str}'): C={C:.6f}")
data['initial_state_dependence'] = init_dep

print("\n[Initial-state dependence, beta=1e-3]")
init_dep_beta = {}
for label, state_str in init_states.items():
    if label == 'default':
        state_str = '1100'
    L = buildL(4, alpha=20.0, drv=[0], beta=1e-3)
    C_peak, _, tau, _, _ = compute_lifetime(4, L, dt_fine=0.1, dt_coarse=2.0, t_max=3000.0,
                                            init_state_str=state_str)
    init_dep_beta[label] = {'C_peak': float(C_peak), 'tau': float(tau)}
    print(f"  {label} ('{state_str}'): C_peak={C_peak:.4f}, tau={tau:.0f}")
data['initial_state_dependence_beta'] = init_dep_beta

# ---------- Time traces for Figure 3 ----------
print("\n[Time traces for Figure 3: n=4, beta=0, 1e-3, 1e-2]")
trace_data = {}
beta_traces = [0.0, 1e-3, 1e-2]
for beta in beta_traces:
    L0 = buildL(4, beta=beta)
    Ld = buildL(4, alpha=20.0, drv=[0], beta=beta)
    _, _, _, t0, C0 = compute_lifetime(4, L0, dt_fine=0.5, dt_coarse=2.0, t_max=200.0,
                                       save_curve=True)
    _, _, _, td, Cd = compute_lifetime(4, Ld, dt_fine=0.5, dt_coarse=2.0, t_max=200.0,
                                       save_curve=True)
    trace_data[str(beta)] = {
        'no_drive': {'t': t0, 'C': C0},
        'with_drive': {'t': td, 'C': Cd}
    }
    print(f"  beta={beta:.0e}: traces saved")
data['time_traces'] = trace_data

# ============================================================================
# 4. SAVE DATA
# ============================================================================
with open('paper_data_final.json', 'w') as f:
    json.dump(data, f, indent=2)
print("\n✅ Data saved to 'paper_data_final.json'")

# ============================================================================
# 5. PLOT FIGURES (with logarithmic scale for Figure 5 right panel)
# ============================================================================
print("\nPlotting figures...")

# ---------- Figure 1: Asymmetric drive ----------
fig1, ax1 = plt.subplots(figsize=(8, 6))
alphas_plot = [float(a) for a in asym_sweep['n=3'].keys()]
c3_plot = list(asym_sweep['n=3'].values())
c4_plot = list(asym_sweep['n=4'].values())
ax1.semilogx(alphas_plot, c3_plot, 'o-', color='#1f77b4', label='n=3', markersize=8)
ax1.semilogx(alphas_plot, c4_plot, 's-', color='#d62728', label='n=4', markersize=8)
ax1.axhline(y=4 / 9, color='#1f77b4', linestyle='--', alpha=0.5)
ax1.axhline(y=1 / 4, color='#d62728', linestyle='--', alpha=0.5)
ax1.set_xlabel(r'$\alpha / \gamma_c$', fontsize=12)
ax1.set_ylabel(r'$C_{2,3}(\infty)$', fontsize=12)
ax1.set_title('Figure 1: Asymmetric drive (qubit 0 only)', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('fig1_asymmetric_drive.pdf', dpi=300)

# ---------- Figure 2: Field-free ----------
fig2, ax2 = plt.subplots(figsize=(8, 6))
n_vals = [int(k) for k in ff_data.keys()]
c_vals = list(ff_data.values())
ax2.plot(n_vals, c_vals, 'o-', color='#2ca02c', markersize=10, linewidth=2)
ax2.set_xlabel('Number of qubits $n$', fontsize=12)
ax2.set_ylabel(r'$C_{2,3}(\infty)$', fontsize=12)
ax2.set_title('Figure 2: Field-free steady-state concurrence', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(n_vals)
fig2.tight_layout()
fig2.savefig('fig2_field_free.pdf', dpi=300)

# ---------- Figure 3: Dynamics ----------
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#1f77b4', '#ff7f0e', '#d62728']
for idx, beta_str in enumerate(['0.0', '0.001', '0.01']):
    d = trace_data[beta_str]
    beta = float(beta_str)
    label = r'$\beta=0$' if beta == 0 else fr'$\beta={beta:.0e}$'
    axes[0].plot(d['no_drive']['t'], d['no_drive']['C'], color=colors[idx], label=label, lw=2)
    axes[1].plot(d['with_drive']['t'], d['with_drive']['C'], color=colors[idx], label=label, lw=2)
axes[0].set_xlabel(r'$t \gamma_c$', fontsize=12)
axes[0].set_ylabel(r'$C_{2,3}(t)$', fontsize=12)
axes[0].set_title('No drive ($\\alpha=0$)', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 60)
axes[1].set_xlabel(r'$t \gamma_c$', fontsize=12)
axes[1].set_ylabel(r'$C_{2,3}(t)$', fontsize=12)
axes[1].set_title('With drive ($\\alpha=20$)', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 60)
fig3.suptitle('Figure 3: Concurrence dynamics for n=4', fontsize=14)
fig3.tight_layout()
fig3.savefig('fig3_time_traces.pdf', dpi=300)

# ---------- Figure 4: Lifetime vs beta ----------
fig4, ax4 = plt.subplots(figsize=(8, 6))
betas_plot = []
tau0_plot = []
taud_plot = []
for beta_str, d in lifetime_results.items():
    beta = float(beta_str)
    if beta > 0:
        betas_plot.append(beta)
        tau0_plot.append(d['tau_0'])
        taud_plot.append(d['tau_d'])
ax4.loglog(betas_plot, tau0_plot, 'o-', label='No drive', color='#1f77b4', markersize=8)
ax4.loglog(betas_plot, taud_plot, 's-', label='With drive', color='#d62728', markersize=8)
beta_fit = np.logspace(-4.5, -1.5, 10)
ax4.loglog(beta_fit, 0.28 / beta_fit, 'k--', alpha=0.5, label=r'$\tau \propto 1/\beta$')
ax4.set_xlabel(r'Dephasing rate $\beta / \gamma_c$', fontsize=12)
ax4.set_ylabel(r'Lifetime $\tau_{1/e}$ ($1/\gamma_c$)', fontsize=12)
ax4.set_title('Figure 4: Entanglement lifetime vs dephasing rate', fontsize=13)
ax4.legend()
ax4.grid(True, alpha=0.3, which='both')
fig4.tight_layout()
fig4.savefig('fig4_lifetime_vs_beta.pdf', dpi=300)

# ---------- Figure 5: Initial-state dependence (with LOG scale for right panel) ----------
states = ['1100', '1010', '1001', '0110', '0101', '0011']
C_ss = [init_dep[st] for st in states]
tau_vals = [init_dep_beta[st]['tau'] for st in states]
C_peak_vals = [init_dep_beta[st]['C_peak'] for st in states]

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: steady-state concurrence (beta=0)
bars1 = ax5a.bar(states, C_ss, color='#2ca02c')
for b, val in zip(bars1, C_ss):
    ax5a.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.4f}",
              ha="center", va="bottom", fontsize=8)
ax5a.set_ylim(0, 0.35)
ax5a.set_xlabel('Initial state', fontsize=12)
ax5a.set_ylabel(r'$C_{2,3}(\infty)$', fontsize=12)
ax5a.set_title(r'$\beta=0,\ \alpha=20$', fontsize=13)
ax5a.grid(True, axis='y', alpha=0.3)

# Right panel: lifetime (beta=1e-3) with LOGARITHMIC SCALE
tau_plot = [t if t > 0 else 0.5 for t in tau_vals]  # avoid log(0) for |0110>
bars2 = ax5b.bar(states, tau_plot, color='#ff7f0e')
for b, val in zip(bars2, tau_vals):
    label = f"{val}" if val > 0 else "~0"
    ax5b.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15, label,
              ha="center", va="bottom", fontsize=8)
ax5b.set_yscale('log')
ax5b.set_ylim(0.3, 5000)
ax5b.set_xlabel('Initial state', fontsize=12)
ax5b.set_ylabel(r'Lifetime $\tau_{1/e}$ (log scale)', fontsize=12)
ax5b.set_title(r'$\beta=10^{-3},\ \alpha=20$', fontsize=13)
ax5b.grid(True, axis='y', alpha=0.3)

fig5.suptitle('Figure 5: Initial-state dependence for n=4', fontsize=14)
fig5.tight_layout()
fig5.savefig('fig5_initial_state_dependence.pdf', dpi=300)

print("\n✅ All figures saved as PDF.")
print("=" * 80)
print(f"Total runtime: {time.time() - start_time:.1f} seconds.")
print("=" * 80)
print("Generated files:")
print("  - paper_data_final.json")
print("  - fig1_asymmetric_drive.pdf")
print("  - fig2_field_free.pdf")
print("  - fig3_time_traces.pdf")
print("  - fig4_lifetime_vs_beta.pdf")
print("  - fig5_initial_state_dependence.pdf")
print("=" * 80)
print("✅ FINAL DATA GENERATION COMPLETE.")
