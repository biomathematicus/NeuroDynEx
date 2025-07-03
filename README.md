# NeuroDynEX

**NeuroDynEX** (Neural Dynamics EXamples) is a collection of concise, simulation-ready [Python](https://www.python.org/) programs focused on electrophysiological modeling using the Hodgkin-Huxley formalism. The examples range from a single isolated neuron to noise-driven ensembles and synaptically coupled networks.

These programs are designed for interactive use in research, instruction, and exploration of the biophysics of excitable cells.

---

## 🧭 Purpose

NeuroDynEX is intended for:
- Instructors presenting core models of neuronal excitability.
- Students learning numerical methods in computational neuroscience.
- Researchers and modelers building intuition about membrane dynamics.

---

## 📁 Contents

### 🔌 Membrane Dynamics and Excitability

- [`HHSingleNeuron.py`](HHSingleNeuron.py): Simulation of a single Hodgkin-Huxley neuron under a square pulse current. Plots membrane potential trajectory.
- [`HHEnsembleBasic.py`](HHEnsembleBasic.py): Simulates an uncoupled ensemble of neurons receiving a common current input. Displays all individual voltage traces.
- [`HHEnsembleNoise.py`](HHEnsembleNoise.py): Adds independent Gaussian noise to the input current of each neuron in the ensemble. Plots population mean and standard deviation band.
- [`HHEnsembleTonic.py`](HHEnsembleTonic.py): Uses a tonic input after a stimulus window to sustain irregular population dynamics under noise.
- [`HHNetwork.py`](HHNetwork.py): Connects the neurons in a sparse network with conductance-based synapses driven by threshold crossing. Implements neurotransmitter-like dynamics and plots the ensemble average voltage.

---

## 📦 Requirements

These programs use only standard scientific libraries:

- `numpy`
- `matplotlib`

Install them using:
```bash
pip install numpy matplotlib
