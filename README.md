# Quantum Reservoir Computing – Semester Project

Hi there,

This repository contains the code developed for a semester project on **Quantum Reservoir Computing (QRC)**, with a focus on time-series forecasting and on understanding how performance depends on the different components of the QRC pipeline.

The project was carried out at **EPFL (Faculty of Quantum Science and Engineering)** during the Autumn Semester 2025.

---

## Repository Structure

The repository is organized around two main Jupyter notebooks, corresponding to two complementary QRC frameworks.

### Notebook A — *Parameterized QRC*

This notebook implements a **parameterized (sequential) Quantum Reservoir Computing** architecture.

---

### Notebook B — *Recurrence-Free QRC (RF-QRC)*

This notebook implements a **recurrence-free QRC (RF-QRC)** framework.

Main features:
- Explicit encoding of a sliding window of past values into the reservoir
- No internal recurrent memory: the quantum system acts as a fixed nonlinear feature map
- Systematic grid search over:
  - reservoir families,
  - measurement bases,
  - sliding window size,
  - number of extra (non-input) qubits
  - ...
- Experiments on:
  - NARMA benchmarks,
  - real-world time series (tomato price dataset),
  - genuinely quantum-generated datasets

---

## Authors

- **Emma Victoria Berenholt**  
- **Alexandra Coralie Golay**

EPFL - Quantum Science and Engineering
