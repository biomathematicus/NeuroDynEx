"""
HHEnsembleTonic.py
Simulation of an uncoupled Hodgkin-Huxley neuron ensemble with Gaussian noise 
and sustained tonic input after a transient stimulus. Produces a population 
average membrane potential trace with confidence interval shading.

By Juan B. Gutiérrez, Professor of Mathematics 
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_NEURONS = 150
NOISE_STD = 1.5
BASELINE_CURRENT = 10.0  # µA/cm² applied after main stimulus

# Hodgkin-Huxley constants
C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.387

# Time settings
T = 100.0
dt = 0.01
time = np.arange(0.0, T + dt, dt)
n_steps = len(time)

# External current function with tonic baseline
def I_ext(t):
    if 10.0 <= t <= 40.0:
        return 10.0
    elif t > 40.0:
        return BASELINE_CURRENT
    else:
        return 0.0

# Rate functions
def alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
def beta_m(V):  return 4.0 * np.exp(-(V + 65.0) / 18.0)
def alpha_h(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def beta_h(V):  return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
def alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
def beta_n(V):  return 0.125 * np.exp(-(V + 65.0) / 80.0)

# Initialization
V = np.full(N_NEURONS, -65.0)
m = alpha_m(V) / (alpha_m(V) + beta_m(V))
h = alpha_h(V) / (alpha_h(V) + beta_h(V))
n = alpha_n(V) / (alpha_n(V) + beta_n(V))
V_traces = np.zeros((N_NEURONS, n_steps))

# Time evolution
for i, t in enumerate(time):
    I_base = I_ext(t)
    I_noise = np.random.normal(0.0, NOISE_STD, size=N_NEURONS)
    I = I_base + I_noise

    gNa = g_Na * m**3 * h
    gK = g_K * n**4
    gL = g_L

    INa = gNa * (V - E_Na)
    IK = gK * (V - E_K)
    IL = gL * (V - E_L)

    dVdt = (I - INa - IK - IL) / C_m
    V += dt * dVdt

    m += dt * (alpha_m(V) * (1.0 - m) - beta_m(V) * m)
    h += dt * (alpha_h(V) * (1.0 - h) - beta_h(V) * h)
    n += dt * (alpha_n(V) * (1.0 - n) - beta_n(V) * n)

    V_traces[:, i] = V

# Compute mean and std across neurons
V_mean = np.mean(V_traces, axis=0)
V_std = np.std(V_traces, axis=0)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(time, V_mean, label='Mean Membrane Potential', color='blue')
plt.fill_between(time, V_mean - V_std, V_mean + V_std, color='blue', alpha=0.3, label='±1 Std Dev')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title(f'Hodgkin-Huxley Ensemble with Noise and Tonic Drive (N={N_NEURONS})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
