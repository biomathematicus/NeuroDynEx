"""
HHSingleNeuron.py
Simulation of a single Hodgkin-Huxley neuron under a step current stimulus. 
Plots the membrane potential over time using Euler integration of ion channel 
and voltage dynamics.

By Juan B. Gutiérrez, Professor of Mathematics 
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""

import numpy as np
import matplotlib.pyplot as plt

# Hodgkin-Huxley parameters
C_m = 1.0      # membrane capacitance, in uF/cm^2
g_Na = 120.0   # maximum sodium conductance, in mS/cm^2
g_K = 36.0     # maximum potassium conductance, in mS/cm^2
g_L = 0.3      # leak conductance, in mS/cm^2
E_Na = 50.0    # sodium reversal potential, in mV
E_K = -77.0    # potassium reversal potential, in mV
E_L = -54.387  # leak reversal potential, in mV

# Simulation parameters
T = 50.0       # total time, in ms
dt = 0.01      # time step, in ms
time = np.arange(0.0, T + dt, dt)

# External current
def I_ext(t):
    return 10.0 if 10.0 <= t <= 40.0 else 0.0

# Alpha and beta functions
def alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
def beta_m(V):  return 4.0 * np.exp(-(V + 65.0) / 18.0)
def alpha_h(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def beta_h(V):  return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
def alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
def beta_n(V):  return 0.125 * np.exp(-(V + 65.0) / 80.0)

# Initialization
V = -65.0
m = alpha_m(V) / (alpha_m(V) + beta_m(V))
h = alpha_h(V) / (alpha_h(V) + beta_h(V))
n = alpha_n(V) / (alpha_n(V) + beta_n(V))

V_trace = []
m_trace = []
h_trace = []
n_trace = []

# Time evolution
for t in time:
    I = I_ext(t)

    # Compute conductances
    gNa = g_Na * m**3 * h
    gK = g_K * n**4
    gL = g_L

    # Compute currents
    INa = gNa * (V - E_Na)
    IK = gK * (V - E_K)
    IL = gL * (V - E_L)

    # Update membrane potential
    dVdt = (I - INa - IK - IL) / C_m
    V += dt * dVdt

    # Update gating variables
    dm = dt * (alpha_m(V) * (1.0 - m) - beta_m(V) * m)
    dh = dt * (alpha_h(V) * (1.0 - h) - beta_h(V) * h)
    dn = dt * (alpha_n(V) * (1.0 - n) - beta_n(V) * n)

    m += dm
    h += dh
    n += dn

    V_trace.append(V)
    m_trace.append(m)
    h_trace.append(h)
    n_trace.append(n)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time, V_trace)
plt.title('Hodgkin-Huxley Neuron Membrane Potential')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.grid(True)
plt.tight_layout()
plt.show()
