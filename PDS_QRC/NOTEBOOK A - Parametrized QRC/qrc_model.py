# src/qrc_model.py

import numpy as np


X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Z = np.array([[1,  0],
              [0, -1]], dtype=complex)

Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)

I2 = np.eye(2, dtype=complex)

def encode_input_density(y: float) -> np.ndarray:

    y_clipped = np.clip(y, 0.0, 1.0)
    a = np.sqrt(1.0 - y_clipped)
    b = np.sqrt(y_clipped)
    psi = np.array([[a],
                    [b]], dtype=complex)
    rho_I = psi @ psi.conj().T
    return rho_I

def Ry(phi: float) -> np.ndarray:

    c = np.cos(phi / 2.0)
    s = np.sin(phi / 2.0)
    return np.array([[c, -s],
                     [s,  c]], dtype=complex)

def Rx(theta: float) -> np.ndarray:

    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s],
                     [-1j * s, c]], dtype=complex)


def CRy_I_to_M(phi: float) -> np.ndarray:

    R = Ry(phi)
    zero_block = np.zeros((2, 2), dtype=complex)
    U = np.block([
        [I2,        zero_block],
        [zero_block,    R     ]
    ])
    return U


def partial_trace_over_input(rho_res: np.ndarray) -> np.ndarray:

    A = rho_res[0:2, 0:2]
    D = rho_res[2:4, 2:4]
    return A + D


def exp_I(rho_res: np.ndarray, O: np.ndarray) -> float:

    op = np.kron(O, I2)
    val = np.trace(op @ rho_res)
    return float(np.real_if_close(val))


def exp_M(rho_res: np.ndarray, O: np.ndarray) -> float:

    op = np.kron(I2, O)
    val = np.trace(op @ rho_res)
    return float(np.real_if_close(val))

def qrc_step_features(phi: float,
                      y_input_norm: float,
                      rho_M_pre: np.ndarray,
                      t: int,
                      print_details: bool = False):
    """
    Single QRC step for a normalized input y_input_norm ∈ [0, 1].

    Pipeline:
      1) Encode y_input_norm as ρ_I(t)
      2) Build joint state ρ_IM(t) = ρ_I(t) ⊗ ρ_M_pre
      3) Apply reservoir unitary U(phi)
      4) Update memory ρ_M_pre_next = Tr_input[ρ_res(t)]
      5) Measure ⟨X_I⟩, ⟨Z_I⟩, ⟨X_M⟩, ⟨Z_M⟩ → feature vector x(t)

    """
    # 1) encode input on the input qubit
    rho_I = encode_input_density(y_input_norm)

    # 2) joint state input+memory
    rho_IM = np.kron(rho_I, rho_M_pre)

    # 3) reservoir unitary
    U = CRy_I_to_M(phi)
    Udag = U.conj().T
    rho_res = U @ rho_IM @ Udag

    # 4) updated memory (trace out input qubit)
    rho_M_pre_next = partial_trace_over_input(rho_res)

    # 5) observables (features)
    xI = exp_I(rho_res, X)
    zI = exp_I(rho_res, Z)
    xM = exp_M(rho_res, X)
    zM = exp_M(rho_res, Z)
    x_vec = np.array([xI, zI, xM, zM], dtype=float)

    if print_details:
        print(f"\n=== QRC FEATURES at t = {t} ===")
        print(f"Input y_norm(t) = {y_input_norm}")
        print("x(t) = [<X_I>, <Z_I>, <X_M>, <Z_M>] =")
        print("   ", np.round(x_vec, 8))

    return x_vec, rho_res, rho_M_pre_next


def qrc_step_with_readout(phi: float,
                          W: np.ndarray,
                          y_input_norm: float,
                          rho_M_pre: np.ndarray,
                          t: int,
                          print_details: bool = False):
    
    x_vec, rho_res, rho_M_pre_next = qrc_step_features(
        phi, y_input_norm, rho_M_pre, t, print_details=print_details
    )
    y_hat_next_norm = float(W @ x_vec)
    return y_hat_next_norm, x_vec, rho_res, rho_M_pre_next

def U_reservoir(phi: float, theta: float) -> np.ndarray:
   
    R_mem = Rx(theta)

    zero_block = np.zeros((2, 2), dtype=complex)

    CRy = CRy_I_to_M(phi)

    U_local = np.kron(I2, R_mem)

    return CRy @ U_local

def qrc_step_features_two_angles(phi: float,
                                 theta: float,
                                 y_in_norm: float,
                                 rho_M_pre: np.ndarray,
                                 t: int,
                                 print_details: bool = False):
    """
    QRC step with a two-parameter reservoir U_reservoir(phi, theta).

    Same structure as qrc_step_features, but uses:
        U = U_reservoir(phi, theta)
    instead of CRy_I_to_M(phi).

    Returns:
        x_vec      : feature vector [<X_I>, <Z_I>, <X_M>, <Z_M>]
        rho_res    : 4x4 density matrix after evolution
        rho_M_post : updated memory state
    """
    rho_I = encode_input_density(y_in_norm)
    rho_IM = np.kron(rho_I, rho_M_pre)

    U = U_reservoir(phi, theta)
    rho_res = U @ rho_IM @ U.conj().T

    rho_M_post = partial_trace_over_input(rho_res)

    x_vec = np.array([
        exp_I(rho_res, X),
        exp_I(rho_res, Z),
        exp_M(rho_res, X),
        exp_M(rho_res, Z),
    ], dtype=float)

    if print_details:
        print(f"\n=== QRC FEATURES (two-angles) at t = {t} ===")
        print(f"Input y_norm(t) = {y_in_norm}")
        print("x(t) = [<X_I>, <Z_I>, <X_M>, <Z_M>] =")
        print("   ", np.round(x_vec, 8))

    return x_vec, rho_res, rho_M_post


def qrc_step_with_readout_two_angles(phi: float,
                                     theta: float,
                                     W: np.ndarray,
                                     y_in_norm: float,
                                     rho_M_pre: np.ndarray,
                                     t: int):
  
    x_vec, rho_res, rho_M_post = qrc_step_features_two_angles(
        phi, theta, y_in_norm, rho_M_pre, t
    )
    y_hat_norm = float(W @ x_vec)
    return y_hat_norm, x_vec, rho_res, rho_M_post
