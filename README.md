# Collective decay with a single-qubit drive

[![DOI](https://zenodo.org/badge/1342584833.svg)](https://doi.org/10.5281/zenodo.24270116)

Lindblad simulations of n qubits decaying collectively into a common zero-temperature bath, with a resonant drive on qubit 1,
optional pure dephasing and independent decay. The entanglement (concurrence) of the undriven pair (2,3) is computed for the
stationary state and for the transient (peak, lifetime, sudden death). Python, MIT licence.

## Install
    pip install -r requirements.txt        # numpy, scipy, matplotlib (Python >= 3.10)

## Run
All results are written to `results/` (CSV/JSON tables and PDF figures).

    python run_all.py              # tests + every script, full run (about 75 min on one core, <= 4 GB RAM; `extended_results.py` takes about 27 min, `robustness.py` about 15 min)
    python run_all.py --quick      # reduced run, about 5 min

In a notebook: `!python run_all.py --quick`.

Step by step (the scripts are independent of each other's outputs; run from the repository folder):

    python tests/test_all.py                                   # 16 tests (~10 s)
    python collective_entanglement.py --outdir results         # full-space solver, n = 3..6
    python uniform_drive.py --outdir results                   # drive on all qubits
    python effective_model.py --outdir results --max-n 6       # strong-drive effective model
    python scaling_symmetric.py --outdir results               # beta = 0, n up to 20
    python extended_results.py --outdir results                # permutation-invariant method, n up to 14
    python independent_decay.py --outdir results               # independent (T1-type) decay
    python peak_map.py --outdir results                        # peak concurrence vs drive strength and dephasing
    python spectral_analysis.py --outdir results               # spectral diagnostics of the projection (Eq. 2.3)
    python strong_drive_error.py --outdir results              # accuracy of the strong-drive approximation vs alpha and n
    python robustness.py --outdir results --n 4                # detuning / unequal coupling, n = 4 (20 realisations)
    python robustness.py --outdir results --n 5 --nreal 3 --cases sel    # n = 5 (3 realisations, indicative)

All scripts accept `--outdir`; `collective_entanglement.py`, `extended_results.py`, `independent_decay.py`, `peak_map.py`, `spectral_analysis.py`, `strong_drive_error.py` and `robustness.py` also accept `--quick`.

## Scripts and outputs
| Script | Content | Main outputs in `results/` |
|---|---|---|
| `collective_entanglement.py` | full-space solver (n = 3..6): stationary values, alpha scan, beta > 0, lifetimes, initial states, convergence | `table_alpha_scan.csv`, `table1_alpha20.csv`, `table2_lifetime.csv`, `table3_initial_state.csv`, `beta_positive_uniqueness.csv`, `table_convergence.csv`, `results.json`, `fig1`-`fig5` |
| `uniform_drive.py` | uniform drive baseline | `table_uniform_drive.csv` |
| `effective_model.py` | effective model for strong driving, comparison with the full model | `effective_model_results.json` |
| `scaling_symmetric.py` | exact reduction for beta = 0, initial state `11 0...0` | `table_scaling_n3_20.csv`, `table_alpha_scan_sym.csv`, `scaling_fits.json`, `fig6_scaling_n20.pdf` |
| `extended_results.py` | permutation-invariant method (any beta, any basis state) | `table2_extended.csv`, `table3_extended.csv`, `table_beta_positive_extended.csv`, `table_alpha_beta_scan.csv`, `table_pi_validation.csv`, `fig7_table2_extended.pdf` |
| `independent_decay.py` | independent decay of each qubit | `table_independent_decay.csv`, `table_independent_decay_stationary.csv`, `fig8_independent_decay.pdf` |
| `peak_map.py` | peak map C_peak(alpha, beta), peak ratio vs alpha, dimensions | `table_peak_map_n4.csv`, `table_peak_vs_alpha.csv`, `table_dimensions.csv`, `fig9_peak_map.pdf`, `fig10_dimension.pdf` |

| `spectral_analysis.py` | null-space dimension, persistent oscillatory modes, smallest decay rate, conditioning and late-time distance for the projection of Eq. (2.3) | `table_spectral.csv`, `table_spectral_reduced.csv` |
| `strong_drive_error.py` | relative deviation of C(inf) from the strong-drive value vs alpha (5..200) and n (4, 8, 12, 16); check against the effective model | `table_strong_drive_error.csv`, `table_strong_drive_limit_check.csv` |
| `robustness.py` | random qubit detunings delta_i and unequal couplings (1+eps_i), full space, fixed seed | `table_robustness_n4.csv`, `table_robustness_n5.csv` |

`symmetric_reduction.py` is a module used by `scaling_symmetric.py`, `strong_drive_error.py` and the tests. `python run_all.py` writes all outputs to `results/`; the folder in the repository holds the reference outputs of a full run.

## Model and conventions
    rho' = -i[H_d, rho] + 2 J rho J^+ - J^+J rho - rho J^+J + beta sum_i D[sz_i] rho + gamma_ind sum_i D[sigma^-_i] rho
    J = sum_i sigma^-_i,   H_d = alpha sigma^x_1,   D[O]rho = O rho O^+ - {O^+O, rho}/2,   gamma_c = 1

* Basis |0> ground, |1> excited, sigma^- = |0><1|; a label such as `1100` lists qubits 1..n (qubit 1 = leftmost bit).
* D[sz]rho = sz rho sz - rho, so coherences decay at the rate 2*beta (T_phi = 1/(2 beta gamma_c)).
* Column-stacking vectorisation. Time evolution by matrix exponentials.
* For beta = gamma_ind = 0 the stationary state depends on the initial state; it is computed by projecting the initial state
  on the null space of the Liouvillian.
* Lifetime tau_1/e: first crossing of C_peak/e after the peak; undefined if C_peak < 0.01 or no crossing occurs in the window.

## Tests
`python tests/test_all.py` (or `python -m pytest -q tests`) runs 16 tests: conventions and column stacking, trace/Hermiticity/positivity,
the dephasing rate, the analytical values 4/9 (n = 3) and 1/4 (n = 4) of Memarzadeh and Mancini, the asymmetric-drive reference values,
the uniform-drive baseline, the permutation-symmetric reduction against the full space, the effective model and its closed-form decoupled
limit, independent decay, dephasing, the transient peak ratio, the spectral diagnostics, the lifting of the degeneracy by detuning/coupling
disorder and the alpha^-2 error scaling of the strong-drive approximation.

## Citation
If you use this code, please cite it as described in `CITATION.cff`.

This repository is archived on Zenodo:

- **Version DOI (v1.0.0):** [10.5281/zenodo.24270117](https://doi.org/10.5281/zenodo.24270117)
- **Concept DOI (all versions):** [10.5281/zenodo.24270116](https://doi.org/10.5281/zenodo.24270116)
