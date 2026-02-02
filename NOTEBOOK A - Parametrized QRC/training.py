# src/training.py

import numpy as np
from qrc_model import *
from normalization import *

def ridge_regression(X: np.ndarray, y: np.ndarray, lam: float):
    """
    Closed-form ridge regression:
        W* = (Xᵀ X + λ I)^(-1) Xᵀ y
    """
    A = X.T @ X + lam * np.eye(X.shape[1])
    b = X.T @ y
    W_star = np.linalg.solve(A, b)
    return W_star


def build_design_matrix(phi, y_norm_series, t_start, t_end, lam=1e-2):
    """
    Build the QRC design matrix X(phi) using teacher forcing,
    and train the optimal ridge readout W*(phi).
    """
    # Initial memory |0><0|
    rho_M = np.array([[1.0, 0.0],
                      [0.0, 0.0]], dtype=complex)

    X_rows = []
    y_tar = []

    for t in range(t_start, t_end):
        y_in_norm     = y_norm_series[t]
        y_target_norm = y_norm_series[t + 1]

        x_vec, _, rho_M = qrc_step_features(
            phi, y_in_norm, rho_M, t, print_details=False
        )

        X_rows.append(x_vec)
        y_tar.append(y_target_norm)

    X = np.vstack(X_rows)
    y_tar = np.array(y_tar)

    # Ridge regression
    W_star = ridge_regression(X, y_tar, lam)

    # Training loss
    y_fit = X @ W_star
    L_train = np.mean((y_fit - y_tar) ** 2)

    return X, y_tar, W_star, L_train


def scan_phi_grid(phis,
                  y_norm_series: np.ndarray,
                  t_start: int,
                  t_end: int,
                  lam: float = 1e-2):
    """
    Scans a grid of φ values and select the best one according
    to the training MSE in normalized space.

    """
    phis = np.asarray(phis, dtype=float)
    losses = []
    W_list = []

    for phi in phis:
        _, _, W_phi, L_phi = build_design_matrix(
            phi=phi,
            y_norm_series=y_norm_series,
            t_start=t_start,
            t_end=t_end,
            lam=lam,
        )
        losses.append(L_phi)
        W_list.append(W_phi)

    losses = np.array(losses)
    idx_best = int(np.argmin(losses))
    phi_star = float(phis[idx_best])
    W_star = W_list[idx_best]

    return phi_star, W_star, losses, W_list

def warmup_memory(phi,
                  W: np.ndarray,
                  y_norm_series: np.ndarray,
                  t_start: int,
                  t_end: int):
    """
    Teacher forcing on the training window to prepare the memory state.

    Applies qrc_step_with_readout for t = t_start,...,t_end-1
    using the true normalized inputs y_norm_series[t].

    """
    rho_M = np.array([[1.0, 0.0],
                      [0.0, 0.0]], dtype=complex)

    for t in range(t_start, t_end):
        y_in_norm = y_norm_series[t]
        _, _, _, rho_M = qrc_step_with_readout(
            phi, W, y_in_norm, rho_M, t, print_details=False
        )

    return rho_M


def autoregressive_rollout(phi,
                           W: np.ndarray,
                           y_norm_series: np.ndarray,
                           t_start_pred: int,
                           t_end_pred: int,
                           rho_M_init: np.ndarray):
    """
    Autoregressive rollout of the QRC from t_start_pred to t_end_pred.

    Starting from an initial memory state rho_M_init and
    an initial input y_norm_series[t_start_pred], we feed the
    model with its own predictions:

        y_in(t+1) = y_hat_norm(t+1)

    """
    rho_M = rho_M_init.copy()

    # first input: true y_norm at t_start_pred
    y_in_norm = float(y_norm_series[t_start_pred])

    t_pred = []
    y_pred_norm = []

    for t in range(t_start_pred, t_end_pred):
        y_hat_next_norm, _, _, rho_M = qrc_step_with_readout(
            phi, W, y_in_norm, rho_M, t, print_details=False
        )

        t_pred.append(t + 1)
        y_pred_norm.append(y_hat_next_norm)

        # autoregressive update
        y_in_norm = y_hat_next_norm

    t_pred = np.array(t_pred)
    y_pred_norm = np.array(y_pred_norm)

    return t_pred, y_pred_norm

def build_design_matrix_two_angles(phi,
                                   theta,
                                   y_norm_series,
                                   t_start,
                                   t_end,
                                   lam: float = 1e-2):
    """
    Builds the design matrix X(phi, theta) using teacher forcing,
    with the two-angle QRC step (phi, theta), and train the ridge
    readout W*(phi, theta).

    """
    # mémoire initiale |0><0|
    rho_M = np.array([[1.0, 0.0],
                      [0.0, 0.0]], dtype=complex)

    X_rows = []
    y_tar = []

    for t in range(t_start, t_end):
        y_in_norm     = y_norm_series[t]
        y_target_norm = y_norm_series[t + 1]

        x_vec, _, rho_M = qrc_step_features_two_angles(
            phi, theta, y_in_norm, rho_M, t, print_details=False
        )

        X_rows.append(x_vec)
        y_tar.append(y_target_norm)

    X = np.vstack(X_rows)
    y_tar = np.array(y_tar)

    # ridge
    W_star = ridge_regression(X, y_tar, lam)

    y_fit = X @ W_star
    L_train = np.mean((y_fit - y_tar) ** 2)

    return X, y_tar, W_star, L_train

def scan_phi_theta_grid(phis,
                        thetas,
                        y_norm_series: np.ndarray,
                        t_start: int,
                        t_end: int,
                        lam: float = 1e-2):
    """
    Grid search over (phi, theta) for the two-angle QRC reservoir.

    """
    phis = np.asarray(phis, dtype=float)
    thetas = np.asarray(thetas, dtype=float)

    L_best = np.inf
    phi_star = None
    theta_star = None
    W_star = None

    for phi in phis:
        for theta in thetas:
            _, _, W, L = build_design_matrix_two_angles(
                phi=phi,
                theta=theta,
                y_norm_series=y_norm_series,
                t_start=t_start,
                t_end=t_end,
                lam=lam,
            )
            if L < L_best:
                L_best = L
                phi_star = float(phi)
                theta_star = float(theta)
                W_star = W

    return phi_star, theta_star, W_star, L_best

def warmup_memory_two_angles(phi,
                             theta,
                             W: np.ndarray,
                             y_norm_series: np.ndarray,
                             t_start: int,
                             t_end: int):
    """
    Teacher forcing on [t_start, t_end) for the two-angle QRC
    to prepare the memory state.
    """
    rho_M = np.array([[1.0, 0.0],
                      [0.0, 0.0]], dtype=complex)

    for t in range(t_start, t_end):
        y_in_norm = y_norm_series[t]
        _, _, _, rho_M = qrc_step_with_readout_two_angles(
            phi, theta, W, y_in_norm, rho_M, t
        )

    return rho_M


def autoregressive_rollout_two_angles(phi,
                                      theta,
                                      W: np.ndarray,
                                      y_norm_series: np.ndarray,
                                      t_start_pred: int,
                                      t_end_pred: int,
                                      rho_M_init: np.ndarray,
                                      clip: bool = True):
    """
    Autoregressive rollout for the two-angle QRC:

        y_in(t+1) = y_hat_norm(t+1)

    Returns:
      - t_pred : times t_start_pred+1 .. t_end_pred
      - y_pred_norm : normalized predictions at those times
    """
    rho_M = rho_M_init.copy()
    y_in_norm = float(y_norm_series[t_start_pred])

    t_pred = []
    y_pred_norm = []

    for t in range(t_start_pred, t_end_pred):
        y_hat_norm, _, _, rho_M = qrc_step_with_readout_two_angles(
            phi, theta, W, y_in_norm, rho_M, t
        )
        if clip:
            y_hat_norm = float(np.clip(y_hat_norm, 0.0, 1.0))

        t_pred.append(t + 1)
        y_pred_norm.append(y_hat_norm)

        y_in_norm = y_hat_norm

    t_pred = np.array(t_pred)
    y_pred_norm = np.array(y_pred_norm)

    return t_pred, y_pred_norm


def scan_margin_phi_theta_grid(margins,
                               phis,
                               thetas,
                               y_all: np.ndarray,
                               T_TRAIN: int,
                               lam: float = 1e-2,
                               clip: bool = False):
    """
    Grid search over (M, phi, theta) where:

      - M is the margin_factor for the 'enlarged' normalization
      - phi, theta are the two reservoir angles.

    For each margin M:
      1) build an enlarged-range Min–Max normalizer on the TRAIN window
      2) normalize the FULL series with it
      3) run a grid search over (phi, theta) using build_design_matrix_two_angles
      4) keep the triple (M, phi, theta, W) that minimizes the training MSE.

    """
    margins = np.asarray(margins, dtype=float)
    phis = np.asarray(phis, dtype=float)
    thetas = np.asarray(thetas, dtype=float)

    L_best = np.inf
    M_star = None
    phi_star = None
    theta_star = None
    W_star = None
    scaler_star = None

    for M in margins:
   
        scaler = make_minmax_normalizer(
            method="enlarged",
            y_all=y_all,
            T_TRAIN=T_TRAIN,
            margin_factor=M,
            clip=clip,  
        )
        y_norm_all = scaler.transform(y_all)

        for phi in phis:
            for theta in thetas:
                _, _, W, L = build_design_matrix_two_angles(
                    phi=phi,
                    theta=theta,
                    y_norm_series=y_norm_all,
                    t_start=0,
                    t_end=T_TRAIN,
                    lam=lam,
                )
                if L < L_best:
                    L_best = L
                    M_star = float(M)
                    phi_star = float(phi)
                    theta_star = float(theta)
                    W_star = W
                    scaler_star = scaler

    return M_star, phi_star, theta_star, W_star, scaler_star, L_best

def evaluate_two_angles_case(
    phi: float,
    theta: float,
    y_norm_all: np.ndarray,
    y_all: np.ndarray,
    scaler: "MinMaxNormalizer",
    T_TRAIN: int,
    T_TOTAL: int,
    lam: float,
    clip: bool = True,
):
    _, _, W, L_train = build_design_matrix_two_angles(
        phi=phi, theta=theta,
        y_norm_series=y_norm_all,
        t_start=0, t_end=T_TRAIN,
        lam=lam,
    )

    rho_M = warmup_memory_two_angles(
        phi=phi, theta=theta, W=W,
        y_norm_series=y_norm_all,
        t_start=0, t_end=T_TRAIN,
    )

    t_pred, y_pred_norm = autoregressive_rollout_two_angles(
        phi=phi, theta=theta, W=W,
        y_norm_series=y_norm_all,
        t_start_pred=T_TRAIN, t_end_pred=T_TOTAL,
        rho_M_init=rho_M,
        clip=clip,
    )

    y_pred_raw = scaler.inverse_transform(y_pred_norm)
    y_true_test = y_all[t_pred]
    mse_raw = float(np.mean((y_pred_raw - y_true_test) ** 2))

    return {
        "phi": float(phi),
        "theta": float(theta),
        "W": W,
        "L_train": float(L_train),
        "t_pred": t_pred,
        "y_pred_norm": y_pred_norm,
        "y_pred_raw": y_pred_raw,
        "mse_raw": mse_raw,
    }


def grid_search_two_angles_losses(
    phis: np.ndarray,
    thetas: np.ndarray,
    y_norm_all: np.ndarray,
    T_TRAIN: int,
    lam: float,
):
    """
    Compute the full training-loss landscape L_grid(theta_i, phi_j).
    Returns:
      - L_grid shape (len(thetas), len(phis))
    """
    phis = np.asarray(phis, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    L_grid = np.zeros((len(thetas), len(phis)), dtype=float)

    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            _, _, _, L = build_design_matrix_two_angles(
                phi=phi, theta=theta,
                y_norm_series=y_norm_all,
                t_start=0, t_end=T_TRAIN,
                lam=lam,
            )
            L_grid[i, j] = float(L)

    return L_grid


def run_baseline_decent_best_two_angles(
    y_norm_all: np.ndarray,
    y_all: np.ndarray,
    scaler: "MinMaxNormalizer",
    T_TRAIN: int,
    T_TOTAL: int,
    lam: float,
    phis: np.ndarray | None = None,
    thetas: np.ndarray | None = None,
    K: int = 30,
    clip: bool = True,
):

    # default grid if not provided
    if phis is None:
        phis = np.linspace(0, 2*np.pi, 40)
    if thetas is None:
        thetas = np.linspace(0, 2*np.pi, 40)

    # 1) baseline
    case_00 = evaluate_two_angles_case(
        phi=0.0, theta=0.0,
        y_norm_all=y_norm_all, y_all=y_all, scaler=scaler,
        T_TRAIN=T_TRAIN, T_TOTAL=T_TOTAL, lam=lam,
        clip=clip,
    )

    # 2) grid losses
    L_grid = grid_search_two_angles_losses(
        phis=phis, thetas=thetas,
        y_norm_all=y_norm_all,
        T_TRAIN=T_TRAIN,
        lam=lam,
    )

    # best
    i_star, j_star = np.unravel_index(np.argmin(L_grid), L_grid.shape)
    phi_star = float(phis[j_star])
    theta_star = float(thetas[i_star])

    case_star = evaluate_two_angles_case(
        phi=phi_star, theta=theta_star,
        y_norm_all=y_norm_all, y_all=y_all, scaler=scaler,
        T_TRAIN=T_TRAIN, T_TOTAL=T_TOTAL, lam=lam,
        clip=clip,
    )

    # 3) decent = K-th best (guard K)
    flat = np.argsort(L_grid, axis=None)
    K = int(K)
    if K < 0:
        K = 0
    if K >= flat.size:
        K = flat.size - 1

    idx_k = flat[K]
    i_k, j_k = np.unravel_index(idx_k, L_grid.shape)
    phi_decent = float(phis[j_k])
    theta_decent = float(thetas[i_k])

    case_decent = evaluate_two_angles_case(
        phi=phi_decent, theta=theta_decent,
        y_norm_all=y_norm_all, y_all=y_all, scaler=scaler,
        T_TRAIN=T_TRAIN, T_TOTAL=T_TOTAL, lam=lam,
        clip=clip,
    )

    return {
        "case_00": case_00,
        "case_decent": case_decent,
        "case_star": case_star,
        "phis": np.asarray(phis, dtype=float),
        "thetas": np.asarray(thetas, dtype=float),
        "L_grid": L_grid,
        "K": K,
        "phi_star": phi_star,
        "theta_star": theta_star,
        "phi_decent": phi_decent,
        "theta_decent": theta_decent,
    }

def pick_kth_best_from_grid(
    L_grid: np.ndarray,
    phis: np.ndarray,
    thetas: np.ndarray,
    K: int,
) -> tuple[float, float]:

    phis = np.asarray(phis, dtype=float)
    thetas = np.asarray(thetas, dtype=float)

    flat = np.argsort(L_grid, axis=None)
    K = int(K)
    if K < 0:
        K = 0
    if K >= flat.size:
        K = flat.size - 1

    idx_k = flat[K]
    i_k, j_k = np.unravel_index(idx_k, L_grid.shape)
    return float(phis[j_k]), float(thetas[i_k])

def run_one_setting_two_angles(
    method: str,
    y_all: np.ndarray,
    T_TRAIN: int,
    T_TOTAL: int,
    phis: np.ndarray,
    thetas: np.ndarray,
    lam: float = 1e-2,
    margin_factor: float = 0.0,
    clip: bool | None = None,
):

    scaler = make_minmax_normalizer(
        method=method,
        y_all=y_all,
        T_TRAIN=T_TRAIN,
        margin_factor=margin_factor,
        clip=clip,
    )
    y_norm_all = scaler.transform(y_all)

    # best angles from training loss
    phi_star, theta_star, W_star, L_train = scan_phi_theta_grid(
        phis=phis,
        thetas=thetas,
        y_norm_series=y_norm_all,
        t_start=0,
        t_end=T_TRAIN,
        lam=lam,
    )

    # warmup memory using true inputs over train window
    rho_M = warmup_memory_two_angles(
        phi=phi_star,
        theta=theta_star,
        W=W_star,
        y_norm_series=y_norm_all,
        t_start=0,
        t_end=T_TRAIN,
    )

    # closed-loop rollout (predictions are for t = T_TRAIN+1 .. T_TOTAL)
    t_pred, y_pred_norm = autoregressive_rollout_two_angles(
        phi=phi_star,
        theta=theta_star,
        W=W_star,
        y_norm_series=y_norm_all,
        t_start_pred=T_TRAIN,
        t_end_pred=T_TOTAL,
        rho_M_init=rho_M,
        clip=True if clip is None else clip,
    )

    y_pred_raw = scaler.inverse_transform(y_pred_norm)
    y_true_test = y_all[t_pred]
    mse_test = float(np.mean((y_pred_raw - y_true_test) ** 2))

    return {
        "method": str(method),
        "margin": float(margin_factor),
        "scaler": scaler,
        "phi": float(phi_star),
        "theta": float(theta_star),
        "W": W_star,
        "L_train": float(L_train),
        "t_pred": t_pred,
        "y_pred_raw": y_pred_raw,
        "mse_test": mse_test,
    }


def scan_margin_M_test(
    M_values: np.ndarray,
    y_all: np.ndarray,
    T_TRAIN: int,
    T_TOTAL: int,
    phis: np.ndarray,
    thetas: np.ndarray,
    lam: float = 1e-2,
    clip: bool = False,
):
    """
    Scan margin M for method='enlarged' and select M* according to MIN test MSE (raw).

    """
    M_values = np.asarray(M_values, dtype=float)

    test_mse = np.zeros_like(M_values, dtype=float)
    train_loss = np.zeros_like(M_values, dtype=float)

    best_idx = 0
    best_mse = np.inf
    best_run = None

    for k, M in enumerate(M_values):
        res = run_one_setting_two_angles(
            method="enlarged",
            y_all=y_all,
            T_TRAIN=T_TRAIN,
            T_TOTAL=T_TOTAL,
            phis=phis,
            thetas=thetas,
            lam=lam,
            margin_factor=float(M),
            clip=clip,
        )
        test_mse[k] = res["mse_test"]
        train_loss[k] = res["L_train"]

        if test_mse[k] < best_mse:
            best_mse = test_mse[k]
            best_idx = k
            best_run = res

    return {
        "M_values": M_values,
        "test_mse": test_mse,
        "train_loss": train_loss,
        "best_idx": int(best_idx),
        "M_best": float(M_values[best_idx]),
        "best_run": best_run,
    }
