import numpy as np
import matplotlib.pyplot as plt

# Number of neurons in the ensemble
N_NEURONS = 10

# Hodgkin-Huxley parameters
C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.387

# Simulation parameters
T = 50.0
dt = 0.01
time = np.arange(0.0, T + dt, dt)
n_steps = len(time)

# External current
def I_ext(t):
    return 10.0 if 10.0 <= t <= 40.0 else 0.0

# Rate functions
def alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
def beta_m(V):  return 4.0 * np.exp(-(V + 65.0) / 18.0)
def alpha_h(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def beta_h(V):  return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
def alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
def beta_n(V):  return 0.125 * np.exp(-(V + 65.0) / 80.0)

# Initialize state variables
V = np.full(N_NEURONS, -65.0)
m = alpha_m(V) / (alpha_m(V) + beta_m(V))
h = alpha_h(V) / (alpha_h(V) + beta_h(V))
n = alpha_n(V) / (alpha_n(V) + beta_n(V))

V_traces = np.zeros((N_NEURONS, n_steps))

# Simulation loop
for i, t in enumerate(time):
    I = I_ext(t)

    gNa = g_Na * m**3 * h
    gK = g_K * n**4
    gL = g_L

    INa = gNa * (V - E_Na)
    IK = gK * (V - E_K)
    IL = gL * (V - E_L)

    dVdt = (I - INa - IK - IL) / C_m
    V += dt * dVdt

    dm = dt * (alpha_m(V) * (1.0 - m) - beta_m(V) * m)
    dh = dt * (alpha_h(V) * (1.0 - h) - beta_h(V) * h)
    dn = dt * (alpha_n(V) * (1.0 - n) - beta_n(V) * n)

    m += dm
    h += dh
    n += dn

    V_traces[:, i] = V

# Plotting
plt.figure(figsize=(12, 6))
for j in range(N_NEURONS):
    plt.plot(time, V_traces[j], label=f'Neuron {j+1}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title(f'Hodgkin-Huxley Ensemble: {N_NEURONS} Neurons')
plt.grid(True)
plt.tight_layout()
plt.show()
