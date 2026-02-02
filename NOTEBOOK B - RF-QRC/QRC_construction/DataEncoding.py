import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from itertools import combinations

I2 = np.array([[1, 0],
        [0, 1]], dtype=complex)
Pauli_X = np.array([[0, 1],
                [1, 0]], dtype=complex)
Pauli_Y = np.array([[0, -1j],
                [1j,  0]], dtype=complex)
Pauli_Z = np.array([[1,  0],
                [0, -1]], dtype=complex)

# Map characters to 2x2 matrices
SINGLE_PAULIS = {
    "I": I2,
    "X": Pauli_X,
    "Y": Pauli_Y,
    "Z": Pauli_Z,
}

ops = {
    "X":   Pauli("X"),
    "Y":   Pauli("Y"),
    "Z":   Pauli("Z"),
    "XX":  Pauli("XX"),
    "YY":  Pauli("YY"),
    "ZZ":  Pauli("ZZ"),
    "XXX": Pauli("XXX"),
    "YYY": Pauli("YYY"),
    "ZZZ": Pauli("ZZZ"),
}
    
def pauli_expectations_to_rho(exp_dict, n):
    """
    Encoding an IMPERFECT quantum data to an IMPERFECT quantum state

    Parameters:
    exp_dict (dict): A dictionary of Pauli expectation values
    n (int): Number of qubits

    Returns:
    rho (np.ndarray): A density matrix (reconstructed using expectation values)
    """

    def pauli_string_to_matrix(label):
        """
        Parameters:
        label (str): The Pauli string label, e.g. "XXYY"

        Returns:
        A 2^n x 2^n matrix, where n is the number of qubits
        """
        mat = SINGLE_PAULIS[label[0]]

        for ch in label[1:]:
            mat = np.kron(mat, SINGLE_PAULIS[ch])

        return mat
    
    dim = 2**n
    rho = np.zeros((dim, dim), dtype=complex)

    identity_label = "I" * n
    have_identity = identity_label in exp_dict

    if have_identity:
        rho += exp_dict[identity_label] * pauli_string_to_matrix(identity_label)
    else:
        # we assume that <I...I> = 1
        rho += pauli_string_to_matrix(identity_label)

    for label, val in exp_dict.items():
        if label == identity_label:
            continue  # already handled
        P = pauli_string_to_matrix(label)
        rho += val * P

    # Normalisation factor: 1 / 2^n
    rho /= (2**n)
    return rho

def project_to_physical_state(rho, tol=1e-12):
    """
    Project an IMPERFECT quantum state to a PERFECT (physical) quantum state.

    Parameters:
    rho (np.ndarray): The density matrix of the quantum state
    tol (float): The tolerance for determining if the state is maximally mixed

    Returns:
    np.ndarray: The projected density matrix
    """
    # Hermitian symmetrization (optional, but safe)
    rho = 0.5 * (rho + rho.conj().T)

    eigvals, eigvecs = np.linalg.eigh(rho)

    # Clipping negative eigenvalues
    eigvals_clipped = np.maximum(eigvals, 0.0)

    # If all zero, returning maximally mixed state
    if np.sum(eigvals_clipped) < tol:
        dim = rho.shape[0]
        return np.eye(dim, dtype=complex) / dim

    eigvals_norm = eigvals_clipped / np.sum(eigvals_clipped)

    # Projecting the state onto the physical subspace
    rho_proj = (eigvecs @ np.diag(eigvals_norm) @ eigvecs.conj().T)
    return rho_proj

def rho_to_pure_statevector(rho, phase_tol=1e-40):
    """
    Encode a PERFECT (mixed/pure) quantum state to a PURE statevector.

    Parameters:
    rho (np.ndarray): The density matrix of the quantum state
    phase_tol (float): The tolerance for determining if the state is pure

    Returns:
    np.ndarray: The pure statevector
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = np.argmax(eigvals)

    psi = eigvecs[:, idx]

    # Rotate so first non-zero component is real
    nonzero_indices = np.where(np.abs(psi) > phase_tol)[0]
    if nonzero_indices.size > 0:
        k = nonzero_indices[0]
        phase = np.angle(psi[k])
        psi = psi * np.exp(-1j * phase)  # rotate so psi[k] is real >= 0
        # tiny negative due to numerical noise -> abs()
        if psi[k].real < 0:
            psi = -psi

    return psi

def angle_encoding(inputs, min_val=0, max_val=1, Rz=False):
    """
    Encode classical data to a quantum state using angle encoding (most common encoding in qml).
    -> Angle encoding is a quantum encoding technique that maps a classical
        input to a quantum state by applying Ry gates to the qubits
        with the angle proportional to the input values.

    Parmeters:
    inputs (list of floats): The classical data to be encoded.
    min_val (float): The minimum value of the input data.
    max_val (float): The maximum value of the input data.
    Rz (bool): If True, use Rz gates instead of Ry gates.

    Returns:
    QuantumCircuit: The quantum circuit that encodes the input data.
    """
    input_size = len(inputs)
    qc = QuantumCircuit(input_size, name='Angle Encoding')
    for i in range(input_size):
        # Normalizing the input value to the range [0,1]
        y_norm = (inputs[i] - min_val) / (max_val - min_val)
        
        # Clipping the normalized value to ensure it is in the range [0,1]
        y_norm = np.clip(y_norm, 0, 1)

        if Rz:
            # Applying a Rz gate with the angle proportional to the input value
            qc.rz(2.0 * np.arcsin(np.sqrt(y_norm)), i)
        else:
            # Applying a Ry gate with the angle proportional to the input value
            qc.ry(2.0 * np.arcsin(np.sqrt(y_norm)), i)

    return qc

def features(state, nb_qubits, meas_bases, qubits=None):
    """
    Compute Pauli expectation features on selected qubits and on selected bases.

    Parameters:
    state (QuantumState): The quantum state to measure.
    nb_qubits (int): The total number of qubits in the state.
    meas_bases (tuple): A tuple of (singles, pairs, triples) specifying the Pauli bases to measure.
    qubits (list or None): The list of qubits to measure. If None, measure all qubits.

    Returns:
    list: A list of Pauli expectation features (the feature vector).
    """
    singles, pairs, triples = meas_bases

    # Normalizing to tuples (robust)
    if isinstance(singles, str):
        singles = (singles,)
    if isinstance(pairs, str):
        pairs = (pairs,)
    if isinstance(triples, str):
        triples = (triples,)

    if qubits is None:
        qubits = list(range(nb_qubits))
    else:
        qubits = list(qubits)

    feats = []

    # 1-body
    for i in qubits:
        for p in singles:
            op = ops[p]
            val = state.expectation_value(op, qargs=[i]).real
            feats.append(val)

    # 2-body
    for i, j in combinations(qubits, 2):
        for p2 in pairs:
            op2 = ops[p2]
            val2 = state.expectation_value(op2, qargs=[i, j]).real
            feats.append(val2)

    # 3-body on selected triples
    for i, j, k in combinations(qubits, 3):
        for p3 in triples:
            op3 = ops[p3]
            val3 = state.expectation_value(op3, qargs=[i, j, k]).real
            feats.append(val3)

    return np.asarray(feats, float)


def build_pauli_labels(nb_input_qubits, meas_bases, qubits_measured=None):
    """
    Build the list of Pauli string labels corresponding to the order of
    features(...) for given meas_bases = (singles, pairs, triples).

    Parameters:
    nb_input_qubits (int): The number of input qubits.
    meas_bases (tuple): A tuple of (singles, pairs, triples) specifying the Pauli bases to measure.
    qubits_measured (list or None): The list of qubits to measure. If None, measure all qubits.

    Returns:
    list: A list of Pauli string labels
    """
    singles, pairs, triples = meas_bases

    # Normalizing to tuples (robust)
    if isinstance(singles, str):
        singles = (singles,)
    if isinstance(pairs, str):
        pairs = (pairs,)
    if isinstance(triples, str):
        triples = (triples,)

    if qubits_measured is None:
        qubits = list(range(nb_input_qubits))
    else:
        qubits = list(qubits_measured)

    labels = []

    # 1-body
    for i in qubits:
        for p in singles:
            if p not in ("X", "Y", "Z"):
                raise ValueError(f"Unsupported single Pauli '{p}'")
            chars = ["I"] * nb_input_qubits
            chars[i] = p
            labels.append("".join(chars))

    # Helper to expand "XX", "YY", "ZZ" into single-qubit chars
    def expand_pair_label(pair_str):
        if pair_str == "XX":
            return ("X", "X")
        if pair_str == "YY":
            return ("Y", "Y")
        if pair_str == "ZZ":
            return ("Z", "Z")
        raise ValueError(f"Unsupported pair label '{pair_str}'")

    def expand_triple_label(triple_str):
        if triple_str == "XXX":
            return ("X", "X", "X")
        if triple_str == "YYY":
            return ("Y", "Y", "Y")
        if triple_str == "ZZZ":
            return ("Z", "Z", "Z")
        raise ValueError(f"Unsupported triple label '{triple_str}'")

    # 2-body
    for i, j in combinations(qubits, 2):
        for p2 in pairs:
            pa, pb = expand_pair_label(p2)
            chars = ["I"] * nb_input_qubits
            chars[i] = pa
            chars[j] = pb
            labels.append("".join(chars))

    # 3-body
    for i, j, k in combinations(qubits, 3):
        for p3 in triples:
            pa, pb, pc = expand_triple_label(p3)
            chars = ["I"] * nb_input_qubits
            chars[i] = pa
            chars[j] = pb
            chars[k] = pc
            labels.append("".join(chars))

    return labels

def yhat_to_exp_dict(y_hat_vec, labels):
    """
    Convert a 1D y-hat feature vector into a dictionary of expectation values (needed to reconstruct "state_hat").

    Parameters:
    y_hat_vec (numpy.ndarray): y_hat feature vector (1D array).
    labels: (list) labels containing measurement basis choice.

    Returns:
    exp_dict: (dict) A dictionary mapping Pauli string labels to expectation values, extracted from the y_hat_vec.
    """
    # Make sure it's 1D
    y_hat_vec = np.asarray(y_hat_vec, float).ravel()
    if len(y_hat_vec) != len(labels):
        raise ValueError(f"Length mismatch: {len(y_hat_vec)} values, {len(labels)} labels")
    
    exp_dict = dict(zip(labels, y_hat_vec))
    
    return exp_dict

def construct_y_true_from_states(y_states, nb_input_qubits, bases_of_interest, qubits_measured=None):
    """
    Construct the true y-feature matrix from a list of quantum states (needed for training when using quantum dataset).

    Parameters:
    y_states (list): A list of quantum states.
    nb_input_qubits (int): The number of input qubits.
    bases_of_interest (tuple): A tuple of (singles, pairs, triples) specifying the Pauli bases to measure.
    qubits_measured (list or None): The list of qubits to measure. If None, measure all qubits.
    
    Returns:
    numpy.ndarray: A 2D array where each row corresponds to the y-feature vector of a quantum state in y_states.
    """
    if qubits_measured is None:
        qubits_measured = list(range(nb_input_qubits))
    
    Y_true = []
    
    # Iterating over each quantum state
    for psi in y_states:
        # Computuing the true y-feature vector for this state
        y_feat = features(psi, nb_input_qubits, bases_of_interest, qubits_measured)
        Y_true.append(y_feat)
    
    return np.asarray(Y_true, float)
