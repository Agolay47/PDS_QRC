from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from qiskit.quantum_info import Statevector, DensityMatrix, Operator
from qiskit.quantum_info.operators import Pauli

I2 = np.eye(2, dtype=complex)

def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s],
                     [-1j * s, c]], dtype=complex)

def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s],
                     [s,  c]], dtype=complex)

def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)

def kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)

def CRy_I_to_M(phi: float) -> np.ndarray:

    P0 = np.array([[1, 0],
                   [0, 0]], dtype=complex)
    P1 = np.array([[0, 0],
                   [0, 1]], dtype=complex)
    return kron(P0, I2) + kron(P1, Ry(phi))

def U_reservoir(phi: float, theta: float) -> np.ndarray:

    return CRy_I_to_M(phi) @ kron(I2, Rx(theta))

X = Pauli("X")
Y = Pauli("Y")
Z = Pauli("Z")

def larmor_dataset(
    T: int,
    dt: float,
    omega: float,
    axis: tuple[float, float, float] = (1.0, 1.0, 1.0),
    psi0=[1,0],
) -> tuple[list[Statevector], np.ndarray]:
    """
    Simulate 1-qubit Larmor precession and return:
      - states: list of Statevector of length T
      - bloch:  array of shape (T,3), columns [<X>, <Y>, <Z>], each in [-1,1]
    """
    axis_v = np.array(axis, dtype=float)
    axis_v /= np.linalg.norm(axis_v)

    n_dot_sigma = (
        axis_v[0] * X.to_matrix()
        + axis_v[1] * Y.to_matrix()
        + axis_v[2] * Z.to_matrix()
    )

    theta = omega * dt
    U = np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * n_dot_sigma
    Uop = Operator(U)

    psi = Statevector(psi0)
    states: list[Statevector] = []
    bloch = np.zeros((T, 3), dtype=float)

    for t in range(T):
        states.append(psi)
        bloch[t, 0] = psi.expectation_value(X).real
        bloch[t, 1] = psi.expectation_value(Y).real
        bloch[t, 2] = psi.expectation_value(Z).real
        psi = psi.evolve(Uop)

    return states, bloch

def partial_trace_over_input(rho_2q: np.ndarray) -> np.ndarray:

    A = rho_2q[0:2, 0:2]
    D = rho_2q[2:4, 2:4]
    return A + D

def op(label: str) -> Operator:
    return Operator(Pauli(label).to_matrix())

Iop = Operator(I2)

def kron_op(A: Operator, B: Operator) -> Operator:
    return Operator(np.kron(A.data, B.data))

# Features: X0,Y0,Z0, X1,Y1,Z1, and XX,YY,ZZ
FEATURE_OPS: list[Operator] = [
    kron_op(op("X"), Iop), kron_op(op("Y"), Iop), kron_op(op("Z"), Iop),
    kron_op(Iop, op("X")), kron_op(Iop, op("Y")), kron_op(Iop, op("Z")),
]

@dataclass
class QRCConfig:
    scale: float = np.pi / 2
    clip_pred: bool = True

def qrc_step(
    phi: float,
    theta: float,
    u_xyz: np.ndarray,
    rho_prev: DensityMatrix | None,
    cfg: QRCConfig,
) -> tuple[np.ndarray, DensityMatrix]:
    """
    One step of the QRC with "as usual" memory handling:

    1) Memory update:
         - if rho_prev is None: rho_mem = |0><0|
         - else: rho_mem = Tr_input(rho_prev)
    2) Build a fresh input state rho_in from u_xyz=(x,y,z):
         angles (ax,ay,az) = cfg.scale * clip(u_xyz)
         U_in = Rx(ax) Ry(ay) Rz(az)
         rho_in = U_in |0><0| U_in†
    3) Build joint state: rho_joint = rho_in ⊗ rho_mem
    4) Apply reservoir unitary U_reservoir(phi,theta):
         rho_next = U_reservoir rho_joint U_reservoir†
    5) Measure feature expectations and append bias 1.0

    Returns:
      x_feat: feature vector (len=9 + bias=1 => 10)
      rho_next: updated 2-qubit density matrix (input+memory)
    """
    # (1) memory
    if rho_prev is None:
        rho_mem = DensityMatrix.from_label("0").data
    else:
        rho_mem = partial_trace_over_input(rho_prev.data)

    # (2) input from Bloch coordinates
    u_xyz = np.asarray(u_xyz, dtype=float).reshape(3)
    ax, ay, az = cfg.scale * np.clip(u_xyz, -1.0, 1.0)
    U_in = Rx(ax) @ Ry(ay) @ Rz(az)
    rho0 = np.array([[1, 0],
                     [0, 0]], dtype=complex)  # |0><0|
    rho_in = U_in @ rho0 @ U_in.conj().T

    # (3) joint state
    rho_joint = DensityMatrix(np.kron(rho_in, rho_mem))

    # (4) reservoir evolution
    U = Operator(U_reservoir(phi, theta))
    rho_next = rho_joint.evolve(U)

    # (5) features + bias
    x = np.array([rho_next.expectation_value(op_).real for op_ in FEATURE_OPS], dtype=float)
    x = np.concatenate([x, [1.0]])  # bias/intercept
    return x, rho_next

def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def train_qrc(
    phi: float,
    theta: float,
    y_bloch: np.ndarray,
    T_TRAIN: int,
    lam: float,
    cfg: QRCConfig,
) -> tuple[np.ndarray, DensityMatrix, float]:

    rho = None
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []

    for t in range(T_TRAIN):
        x_feat, rho = qrc_step(phi, theta, y_bloch[t], rho, cfg)
        X_list.append(x_feat)
        Y_list.append(y_bloch[t + 1])

    Xmat = np.vstack(X_list)              # (N,d)
    Ymat = np.vstack(Y_list)              # (N,3)
    W = ridge_fit(Xmat, Ymat, lam)
    mse = float(np.mean((Xmat @ W - Ymat) ** 2))
    return W, rho, mse


def grid_search(
    phis: np.ndarray,
    thetas: np.ndarray,
    y_bloch: np.ndarray,
    T_TRAIN: int,
    lam: float,
    cfg: QRCConfig,
) -> tuple[float, float, np.ndarray, float]:

    best_phi = None
    best_theta = None
    best_W = None
    best_mse = np.inf

    for phi in phis:
        for theta in thetas:
            W, _, mse = train_qrc(float(phi), float(theta), y_bloch, T_TRAIN, lam, cfg)
            if mse < best_mse:
                best_mse = mse
                best_phi = float(phi)
                best_theta = float(theta)
                best_W = W

    assert best_phi is not None and best_theta is not None and best_W is not None
    return best_phi, best_theta, best_W, float(best_mse)

def rollout(
    phi: float,
    theta: float,
    W: np.ndarray,
    y_bloch: np.ndarray,
    rho_init: DensityMatrix,
    T_TRAIN: int,
    T_TOTAL: int,
    cfg: QRCConfig,
) -> np.ndarray:

    rho = rho_init
    u = np.array(y_bloch[T_TRAIN], dtype=float).reshape(3)

    preds: list[np.ndarray] = []
    for _t in range(T_TRAIN, T_TOTAL):
        x_feat, rho = qrc_step(phi, theta, u, rho, cfg)
        y_hat = x_feat @ W
        if cfg.clip_pred:
            y_hat = np.clip(y_hat, -1.0, 1.0)
        preds.append(np.asarray(y_hat, dtype=float))
        u = y_hat

    return np.vstack(preds)  # (T_TOTAL - T_TRAIN, 3)
