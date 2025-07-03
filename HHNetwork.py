import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_NEURONS = 50
NOISE_STD = 1.0
BASELINE_CURRENT = 2.0
E_syn = 0.0  # mV (excitatory synapse)
alpha_syn = 1.0  # rise rate of synaptic gating
beta_syn = 0.2   # decay rate of synaptic gating
theta_spike = 0.0  # mV threshold for synaptic activation

# Hodgkin-Huxley parameters
C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.387

# Time settings
T = 50.0
dt = 0.01
time = np.arange(0.0, T + dt, dt)
n_steps = len(time)

# External current function
def I_ext(t):
    if 10.0 <= t <= 40.0:
        return 10.0
    elif t > 40.0:
        return BASELINE_CURRENT
    else:
        return 0.0

# Gating rate functions
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
s = np.zeros(N_NEURONS)  # synaptic gating variables

V_traces = np.zeros((N_NEURONS, n_steps))

# Random sparse connectivity matrix
W = np.random.rand(N_NEURONS, N_NEURONS) * (np.random.rand(N_NEURONS, N_NEURONS) < 0.1)
np.fill_diagonal(W, 0)

# Simulation
for i, t in enumerate(time):
    I_base = I_ext(t)
    I_noise = np.random.normal(0.0, NOISE_STD, size=N_NEURONS)
    I_ext_total = I_base + I_noise

    # Synaptic gating update
    spike_mask = V > theta_spike
    ds = alpha_syn * (1 - s) * spike_mask - beta_syn * s
    s += dt * ds

    # Synaptic current
    I_syn = np.sum(W * s[None, :], axis=1) * (V - E_syn)

    # Ionic conductances
    gNa = g_Na * m**3 * h
    gK = g_K * n**4
    gL = g_L

    INa = gNa * (V - E_Na)
    IK = gK * (V - E_K)
    IL = gL * (V - E_L)

    # Total current and voltage update
    I_total = I_ext_total - INa - IK - IL - I_syn
    dVdt = I_total / C_m
    V += dt * dVdt

    # Gating updates
    m += dt * (alpha_m(V) * (1.0 - m) - beta_m(V) * m)
    h += dt * (alpha_h(V) * (1.0 - h) - beta_h(V) * h)
    n += dt * (alpha_n(V) * (1.0 - n) - beta_n(V) * n)

    V_traces[:, i] = V

# Compute mean and std
V_mean = np.mean(V_traces, axis=0)
V_std = np.std(V_traces, axis=0)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(time, V_mean, label='Mean Membrane Potential', color='blue')
plt.fill_between(time, V_mean - V_std, V_mean + V_std, color='blue', alpha=0.3, label='±1 Std Dev')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title(f'Hodgkin-Huxley Network with Synaptic Coupling (N = {N_NEURONS})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
