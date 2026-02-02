from __future__ import annotations
import itertools
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit.quantum_info import random_unitary

"""
QRC reservoir construction circuits.

This module contains the six circuit builders for these reservoir families:
- 1) G1: {CNOT, H, X}
- 2) G2: {CNOT, H, S}
- 3) G3: {CNOT, H, T}
- 4) MG: Matchgates
- 5) D : Diagonal circuits with parameter k in {2, 3, N}
- 6) Ising: Random transverse-field Ising evolution

Reference:
https://arxiv.org/pdf/2505.13933
"""

# Optional (only needed for Ising builder using PauliEvolutionGate route)
try:
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import SuzukiTrotter
    _HAS_EVOLUTION = True
except Exception:
    _HAS_EVOLUTION = False

def _rng(seed):
    return np.random.default_rng(seed)

def _pick_two_distinct(rng, n, connectivity="all_to_all", ring=False):
    """Return (i,j) with i!=j. If ring=True, only nearest-neighbor on a ring."""
    if ring:
        i = int(rng.integers(0, n))
        j = (i + 1) % n if rng.random() < 0.5 else (i - 1) % n
        return i, j
    # all-to-all
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n - 1))
    if j >= i:
        j += 1
    return i, j

def _append_single_qubit_gate(qc, gate_name, q):
    if gate_name == "H":
        qc.h(q)
    elif gate_name == "X":
        qc.x(q)
    elif gate_name == "S":
        qc.s(q)
    elif gate_name == "T":
        qc.t(q)
    else:
        raise ValueError(f"Unsupported 1q gate {gate_name}")

def _append_cnot(qc, c, t):
    qc.cx(c, t)

# -------------------------
# 1) G1 family: {CNOT, H, X}
# -------------------------
def build_reservoir_G1(nb_qubits, depth, seed=None, ring_connectivity=False):
    """
    G1 = {CNOT, H, X} (Clifford subgroup)
    depth = number of random gates appended.
    """
    rng = _rng(seed)
    qc = QuantumCircuit(nb_qubits, name="G1_reservoir")
    gate_set = ["CNOT", "H", "X"]

    choices = []
    for _ in range(depth):
        g = gate_set[int(rng.integers(0, len(gate_set)))]
        if g == "CNOT":
            c, t = _pick_two_distinct(rng, nb_qubits, ring=ring_connectivity)
            _append_cnot(qc, c, t)
            choices.append(("CNOT", c, t))
        else:
            q = int(rng.integers(0, nb_qubits))
            _append_single_qubit_gate(qc, g, q)
            choices.append((g, q))

    meta = {"family": "G1", "nb_qubits": nb_qubits, "depth": depth, "seed": seed, "choices": choices}
    return qc, meta


# -------------------------
# 2) G2 family: {CNOT, H, S}
# -------------------------
def build_reservoir_G2(nb_qubits, depth, seed, ring_connectivity= False):
    """
    G2 = {CNOT, H, S} (full Clifford group)
    depth = number of random gates appended.
    """
    rng = _rng(seed)
    qc = QuantumCircuit(nb_qubits, name="G2_reservoir")
    gate_set = ["CNOT", "H", "S"]

    choices = []
    for _ in range(depth):
        g = gate_set[int(rng.integers(0, len(gate_set)))]
        if g == "CNOT":
            c, t = _pick_two_distinct(rng, nb_qubits, ring=ring_connectivity)
            _append_cnot(qc, c, t)
            choices.append(("CNOT", c, t))
        else:
            q = int(rng.integers(0, nb_qubits))
            _append_single_qubit_gate(qc, g, q)
            choices.append((g, q))

    meta = {"family": "G2", "nb_qubits": nb_qubits, "depth": depth, "seed": seed, "choices": choices}
    return qc, meta


# -------------------------
# 3) G3 family: {CNOT, H, T}
# -------------------------
def build_reservoir_G3(nb_qubits, depth, seed, ring_connectivity= False):
    """
    G3 = {CNOT, H, T} (universal)
    depth = number of random gates appended.
    """
    rng = _rng(seed)
    qc = QuantumCircuit(nb_qubits, name="G3_reservoir")
    gate_set = ["CNOT", "H", "T"]

    choices = []
    for _ in range(depth):
        g = gate_set[int(rng.integers(0, len(gate_set)))]
        if g == "CNOT":
            c, t = _pick_two_distinct(rng, nb_qubits, ring=ring_connectivity)
            _append_cnot(qc, c, t)
            choices.append(("CNOT", c, t))
        else:
            q = int(rng.integers(0, nb_qubits))
            _append_single_qubit_gate(qc, g, q)
            choices.append((g, q))

    meta = {"family": "G3", "nb_qubits": nb_qubits, "depth": depth, "seed": seed, "choices": choices}
    return qc, meta


# -------------------------
# 4) Matchgate (MG) family
# -------------------------
def build_reservoir_MG(nb_qubits, depth, seed, ring_connectivity= False):
    """
    Matchgates as in Eq. (6) from https://arxiv.org/pdf/2505.13933: block-structured 4x4 unitaries built from A,B in U(2) with det(A)=det(B).
    We sample:
      - A ~ U(2)
      - B0 ~ SU(2) (via random_unitary then normalize det)
      - B = e^{iθ/2} B0 so det(B)=det(A)=e^{iθ}
    """
    rng = _rng(seed)
    qc = QuantumCircuit(nb_qubits, name="MG_reservoir")

    choices = []
    for _ in range(depth):
        q1, q2 = _pick_two_distinct(rng, nb_qubits, ring=ring_connectivity)
        # Sampling A in U(2)
        A = random_unitary(2, seed=int(rng.integers(0, 2**31 - 1))).data
        detA = np.linalg.det(A)
        theta = np.angle(detA)  # detA ~ e^{i theta}

        # Sampling B0 in U(2) then project to SU(2)
        B0 = random_unitary(2, seed=int(rng.integers(0, 2**31 - 1))).data
        detB0 = np.linalg.det(B0)
        B_su2 = B0 / np.sqrt(detB0)  # now det(B_su2)=1 (up to numerical error)

        # Scaling to match det(A)
        B = np.exp(1j * theta / 2) * B_su2  # det(B)=e^{i theta}=det(A)

        # Building matchgate matrix G(A,B) (Eq. 6)
        a1, a2 = A[0, 0], A[0, 1]
        a3, a4 = A[1, 0], A[1, 1]
        b1, b2 = B[0, 0], B[0, 1]
        b3, b4 = B[1, 0], B[1, 1]

        G = np.array([
            [a1, 0,  0,  a2],
            [0,  b1, b2, 0 ],
            [0,  b3, b4, 0 ],
            [a3, 0,  0,  a4],
        ], dtype=complex)

        qc.unitary(G, [q1, q2], label="MG")
        choices.append(("MG", q1, q2))

    meta = {"family": "MG", "nb_qubits": nb_qubits, "depth": depth, "seed": seed, "choices": choices}
    return qc, meta


# -------------------------
# 5) Diagonal circuits: D2, D3, DN (one method with k)
# -------------------------
def build_reservoir_diagonal(nb_qubits, k, seed=None):
    """
    Diagonal circuits Dk with k in {2,3,N}, with phases uniform in [0,2π) and H^N at beginning and end.
    - D2: apply on all pairs (N choose 2)
    - D3: apply on all triples (N choose 3)
    - DN: apply once on all qubits
    Ordering is random (paper: ordering random; diagonal gates commute).
    """
    if k not in {2, 3, nb_qubits}:
        raise ValueError(f"k must be 2, 3, or nb_qubits (={nb_qubits}). Got k={k}.")

    rng = _rng(seed)
    qc = QuantumCircuit(nb_qubits, name=f"D{k}_reservoir")

    # Hadamards at beginning
    qc.h(range(nb_qubits))

    # Choose subsets
    if k == nb_qubits:
        subsets = [tuple(range(nb_qubits))]
    else:
        subsets = list(itertools.combinations(range(nb_qubits), k))
        rng.shuffle(subsets)

    applied = []
    for qs in subsets:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=2**k)
        gate = Diagonal(np.exp(1j * phases))
        qc.append(gate, list(qs))
        applied.append(tuple(qs))

    # Hadamards at end
    qc.h(range(nb_qubits))

    meta = {"family": f"D{k}", "nb_qubits": nb_qubits, "k": k, "seed": seed, "subsets": applied}
    return qc, meta


# -------------------------
# 6) Ising reservoir (random transverse-field Ising)
# -------------------------
def build_reservoir_ising(nb_qubits,
                        seed=None,
                        Js=1.0,
                        h_over_Js=0.1,
                        time_T=10.0,
                        trotter_steps=1,
                        ring_connectivity=False,
                        ):
    """
    Random transverse-field Ising Hamiltonian:
        H = sum_{i<j} J_ij Z_i Z_j + sum_i h X_i     (Eq. 8 from https://arxiv.org/pdf/2505.13933)
    with J_ij ~ Uniform(-Js/2, Js/2), h = h_over_Js * Js, evolved for time_T.

    Implementation:
      - If PauliEvolutionGate is available: we use Suzuki-Trotter synthesis.
      - Otherwise: we do a simple manual 1st-order trotter using RZZ + RX.
    """
    rng = _rng(seed)
    h = h_over_Js * Js

    # Sample couplings
    if ring_connectivity:
        pairs = [(i, (i+1) % nb_qubits) for i in range(nb_qubits)]
    else:
        pairs = [(i, j) for i in range(nb_qubits) for j in range(i+1, nb_qubits)]
    J = {(i, j): rng.uniform(-Js/2, Js/2) for (i, j) in pairs}

    qc = QuantumCircuit(nb_qubits, name="Ising_reservoir")

    if _HAS_EVOLUTION:
        # Build SparsePauliOp for H
        paulis = []
        coeffs = []
        # ZZ terms
        for (i, j), Jij in J.items():
            label = ["I"] * nb_qubits
            label[i] = "Z"
            label[j] = "Z"
            paulis.append("".join(label))
            coeffs.append(Jij)
        # X terms
        for i in range(nb_qubits):
            label = ["I"] * nb_qubits
            label[i] = "X"
            paulis.append("".join(label))
            coeffs.append(h)

        H = SparsePauliOp(paulis, coeffs=np.array(coeffs, dtype=float))

        synth = SuzukiTrotter(reps=trotter_steps)  # 1st/2nd order depending on qiskit version defaults
        evo = PauliEvolutionGate(H, time=time_T, synthesis=synth)
        qc.append(evo, list(range(nb_qubits)))

    else:
        # Manual 1st-order trotter: exp(-i sum ZZ) exp(-i sum X)
        # ZZ via rzz(2 * Jij * dt) (since rzz(θ)=exp(-i θ/2 Z⊗Z))
        dt = time_T / trotter_steps
        for _ in range(trotter_steps):
            for (i, j), Jij in J.items():
                qc.rzz(2.0 * Jij * dt, i, j)
            for i in range(nb_qubits):
                qc.rx(2.0 * h * dt, i)  # rx(θ)=exp(-i θ/2 X)

    meta = {
        "family": "Ising",
        "nb_qubits": nb_qubits,
        "seed": seed,
        "Js": Js,
        "h_over_Js": h_over_Js,
        "h": h,
        "time_T": time_T,
        "trotter_steps": trotter_steps,
        "ring_connectivity": ring_connectivity,
        "J_ij": J,
        "used_pauli_evolution_gate": _HAS_EVOLUTION,
    }
    return qc, meta
