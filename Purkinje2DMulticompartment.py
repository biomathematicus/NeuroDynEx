"""
Purkinje2DMulticompartment.py
Simulation of a two-dimensional multicompartment Hodgkin-Huxley-type model for a Purkinje neuron, with explicit geometry-based scaling of membrane properties. The model includes classical fast sodium and delayed-rectifier potassium channels, voltage-gated calcium channels, and calcium-activated potassium channels. The dendritic arbor is represented as a two-dimensional grid with distributed stochastic synaptic input and biophysically informed parameters.

By Juan B. Gutierrez, Professor of Mathematics  
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import numpy as np
import matplotlib.pyplot as plt

# Morphological and biophysical dimensions (approximate typical values)
dx = 20e-4      # compartment length (cm, 20 um)
dy = 20e-4      # compartment width (cm)
diam = 15e-4    # compartment diameter (cm, 15 um)
Cm_0 = 1.0      # specific membrane capacitance (uF/cm^2)
Ra_0 = 150      # specific axial resistivity (Ohm*cm)
gNa = 40.0      # maximal Na+ conductance (mS/cm^2), reduced for Purkinje
gK = 12.0       # maximal K+ conductance (mS/cm^2)
gCa = 2.0       # maximal Ca2+ conductance (mS/cm^2)
gKCa = 10.0     # maximal Ca2+-activated K+ conductance (mS/cm^2)
gL = 0.2        # leak conductance (mS/cm^2)
ENa = 60.0      # Na+ reversal potential (mV)
EK = -80.0      # K+ reversal potential (mV)
ECa = 140.0     # Ca2+ reversal (mV)
EL = -65.0      # Leak reversal (mV)

M, N = 8, 8     # grid size (number of dendritic compartments along each axis)
dt = 0.005      # smaller time step (ms)
T = 50          # total time (ms)
steps = int(T/dt)

# Compartment geometry
area = np.pi * diam * dx   # cm^2, membrane area per compartment
Cm = Cm_0 * area           # compartment capacitance (uF)
Ra = Ra_0 * dx / (np.pi*(diam/2)**2) # axial resistance between compartments (Ohms)

# Initialize states
V = -65.0 * np.ones((M, N))   # membrane potential
m = np.zeros((M, N))
h = np.zeros((M, N))
n = np.zeros((M, N))
Ca = np.zeros((M, N))
s = np.zeros((M, N))          # activation for IKCa

# Initial gating variables following steady-state at rest
def minf(V): return 1./(1 + np.exp(-(V+35)/7.8))
def hinf(V): return 1./(1 + np.exp((V+65)/5.8))
def ninf(V): return 1./(1 + np.exp(-(V+34)/10))
m[:,:] = minf(V)
h[:,:] = hinf(V)
n[:,:] = ninf(V)

# Integration of gating variables
def alpha_m(V): return 0.32*(V + 54.) / (1 - np.exp(-(V + 54.)/4))
def beta_m(V):  return 0.28*(V + 27.) / (np.exp((V + 27.)/5) - 1)
def alpha_h(V): return 0.128 * np.exp(-(V+50)/18)
def beta_h(V):  return 4.0 / (1 + np.exp(-(V + 27)/5))
def alpha_n(V): return 0.032*(V + 52.) / (1 - np.exp(-(V + 52.)/5))
def beta_n(V):  return 0.5*np.exp(-(V+57)/40)

def minf_Ca(V): return 1./(1 + np.exp(-(V+20)/6.5))  # Fast, steady-state
def taum_Ca(V): return 1.0                          # ms

def sinf(Ca): return Ca / (Ca + 1)                  # Simple hill function

# External input: distributed, stochastic across dendrites
def I_syn(t, m, n):
    # Distributed background synaptic input to random dendritic compartments
    if np.random.rand() < 0.1: # 10% chance of new synaptic event
        if 5 < t < 30:
            return np.random.normal(10, 3)  # mean 10 uA/cm^2
    return 0.

V_record = np.zeros((steps, M, N))

for step in range(steps):
    t_ms = step * dt
    V_next = V.copy()
    m_next = m.copy()
    h_next = h.copy()
    n_next = n.copy()
    Ca_next = Ca.copy()
    s_next = s.copy()
    
    for i in range(M):
        for j in range(N):
            neighbors = []
            if i > 0:    neighbors.append((i - 1, j))
            if i < M-1:  neighbors.append((i + 1, j))
            if j > 0:    neighbors.append((i, j - 1))
            if j < N-1:  neighbors.append((i, j + 1))
            I_axial = 0.0
            for (ni, nj) in neighbors:
                I_axial += (V[ni, nj] - V[i, j]) / Ra
            I_axial *= area / 1e3   # Convert Ohm to kOhm for uA

            # Fast sodium and potassium (Hodgkin-Huxley)
            m_inf = minf(V[i, j]); h_inf = hinf(V[i, j]); n_inf = ninf(V[i, j])
            m_tau = 0.1; h_tau = 0.5; n_tau = 1.0
            m_next[i, j] += dt * (m_inf - m[i, j]) / m_tau
            h_next[i, j] += dt * (h_inf - h[i, j]) / h_tau
            n_next[i, j] += dt * (n_inf - n[i, j]) / n_tau

            INa = gNa * m[i, j]**3 * h[i, j] * (V[i, j] - ENa) * area
            IK = gK * n[i, j]**4 * (V[i, j] - EK) * area
            IL = gL * (V[i, j] - EL) * area

            # Voltage-gated Ca2+ current (simplified, not full kinetics)
            mCa = minf_Ca(V[i, j])
            ICa = gCa * mCa * (V[i, j] - ECa) * area

            # Intracellular Ca2+ accumulation and removal
            Ca_next[i, j] += -0.002 * ICa * dt - (Ca[i, j]/80.0)*dt

            # Ca2+-activated potassium
            s_inf = sinf(Ca[i, j])
            s_tau = 2.0
            s_next[i, j] += dt * (s_inf - s[i, j]) / s_tau
            IKCa = gKCa * s[i, j] * (V[i, j] - EK) * area

            # Synaptic drive distributed across the dendritic grid
            Isyn = I_syn(t_ms, i, j) * area

            dV = (
                - INa - IK - IL - ICa - IKCa + I_axial + Isyn
            ) / Cm
            V_next[i, j] += dt * dV

    V, m, h, n, Ca, s = V_next, m_next, h_next, n_next, Ca_next, s_next
    V_record[step, :, :] = V

plt.figure(figsize=(12, 7))
for idx in range(N):
    plt.plot(np.arange(steps) * dt, V_record[:, int(M/2), idx], alpha=0.7)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('Purkinje neuron: voltage in the center row of a 2D dendritic grid')
plt.show()
