# asymmetric-driving-entanglement
Code for the paper "Asymmetric Driving Enhances Steady-State Entanglement in Collectively Dissipating Qubits".

The code is written in Python 3.8+ and requires the following libraries:

- `numpy`
- `scipy`
- `matplotlib`

You can install them using pip:

pip install numpy scipy matplotlib

# What the Code Does
This code numerically solves the Lindblad master equation for up to 6 qubits subject to:

Collective dissipation into a common zero-temperature reservoir

Coherent driving (uniform or asymmetric)

Local dephasing

It computes the steady-state concurrence (entanglement) between qubits 2 and 3, as well as the entanglement lifetime in the presence of dephasing.

# How to Run
Simply execute the main script:

bash
python generate_all_data_v4_final.py
This will:

Generate all numerical data (saved as paper_data_final.json)

Produce all figures as PDF files (Figures 1–5)

Print summary tables to the terminal

The entire run takes approximately 3–5 minutes on a standard laptop.

# Figures Generated
Figure	Description
Figure 1	Steady-state concurrence vs drive strength for n=3 and n=4
Figure 2	Field-free concurrence vs number of qubits (n=3 to 6)
Figure 3	Time evolution of concurrence for different dephasing rates
Figure 4	Entanglement lifetime vs dephasing rate (log-log scale)
Figure 5	Initial-state dependence (with logarithmic scale for lifetimes)
 
# Data Availability
All data generated or analysed during this study are included in this repository (paper_data_final.json) and its supplementary information files. The code is fully open-source.
