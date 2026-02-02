import numpy as np
from qiskit.quantum_info import Statevector, Operator
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Time-series plots of prediction vs ground truth
def plot_predictions(Y_true, Y_hat, T_train, labels):
    Y_true = np.asarray(Y_true, float)
    Y_hat  = np.asarray(Y_hat, float)

    T, d = Y_true.shape

    t_range = np.arange(T_train, T)

    for k, lab in enumerate(labels):
        plt.figure(figsize=(20, 4))
        plt.plot(t_range, Y_true[T_train:, k], "-o", label="True", alpha=0.8)
        plt.plot(t_range, Y_hat[T_train:,  k], "--x", label="Pred", alpha=0.8)

        plt.xlabel("t")
        plt.ylabel(f"⟨{lab}⟩")
        #plt.axvline(x=T_train, color="k", linestyle="--", linewidth=1.5, alpha=0.8, label="train/test split")
        plt.ylim(-1.05, 1.05)
        plt.title(f"Prediction vs True on test region (label: {lab})")
        plt.legend()
        plt.tight_layout()
        plt.show()

def scatter_per_label(Y_true, Y_hat, T_train, labels):
    Y_true = np.asarray(Y_true, float)
    Y_hat  = np.asarray(Y_hat, float)

    T, d = Y_true.shape
    assert Y_hat.shape == (T, d), "Y_true and Y_hat must have same shape"
    assert len(labels) == d, "labels length must match number of features"

    # Only test region
    Y_true_test = Y_true[T_train:]
    Y_hat_test  = Y_hat[T_train:]

    # Make a row of subplots
    fig, axs = plt.subplots(1, d, figsize=(4*d, 4))
    if d == 1:
        axs = [axs]

    for k, lab in enumerate(labels):
        ax = axs[k]
        ax.scatter(Y_true_test[:, k], Y_hat_test[:, k], alpha=0.8)

        # Diagonal line for reference
        min_val = min(Y_true_test[:, k].min(), Y_hat_test[:, k].min())
        max_val = max(Y_true_test[:, k].max(), Y_hat_test[:, k].max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6)

        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Label: {lab}")

    plt.tight_layout()
    plt.show()


def scatter_per_label_fading(Y_true, Y_hat, T_train, labels):
    """
    Scatter plots of Y_true vs Y_hat per label, using only test region t >= T_train.
    Older points = more transparent blue, recent points = more opaque blue.
    """
    Y_true = np.asarray(Y_true, float)
    Y_hat  = np.asarray(Y_hat, float)

    T, d = Y_true.shape
    assert Y_hat.shape == (T, d), "Y_true and Y_hat must have same shape"
    assert len(labels) == d, "labels length must match number of features"

    # Only test region
    Y_true_test = Y_true[T_train:]
    Y_hat_test  = Y_hat[T_train:]
    N_test = Y_true_test.shape[0]

    if N_test == 0:
        raise ValueError("No test points: T_train must be < T")

    # Alpha increases with time: oldest = lowest alpha, newest = highest
    alpha_min = 0.1
    alpha_max = 1.0
    alphas = np.linspace(alpha_min, alpha_max, N_test)

    fig, axs = plt.subplots(1, d, figsize=(4*d, 4))
    if d == 1:
        axs = [axs]

    for k, lab in enumerate(labels):
        ax = axs[k]

        # Build per-point RGBA colors: constant blue, varying alpha
        colors = np.zeros((N_test, 4))
        colors[:, 2] = 1.0          # B channel
        colors[:, 3] = alphas       # alpha

        ax.scatter(Y_true_test[:, k], Y_hat_test[:, k], c=colors)

        # Diagonal reference line
        min_val = min(Y_true_test[:, k].min(), Y_hat_test[:, k].min())
        max_val = max(Y_true_test[:, k].max(), Y_hat_test[:, k].max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6)

        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Label: {lab}")

    # Global legend explaining the fading
    recent_handle = Line2D([0], [0], marker='o', linestyle='None',
                        color=(0, 0, 1, alpha_max),
                        label="Most recent test points (opaque blue)")
    old_handle = Line2D([0], [0], marker='o', linestyle='None',
                        color=(0, 0, 1, alpha_min),
                        label="Older test points (more transparent)")

    #fig.legend(handles=[old_handle, recent_handle],loc="upper right", bbox_to_anchor=(0.5, 0.5))
    plt.tight_layout()
    plt.show()

def plot_error_vs_time(Y_true, Y_hat, T_train):
    Y_true = np.asarray(Y_true, float)
    Y_hat  = np.asarray(Y_hat, float)
    T, d = Y_true.shape
    assert Y_hat.shape == (T, d)

    # Mean |error| over features
    err = np.abs(Y_hat - Y_true)      # (T, d)
    err_mean = err.mean(axis=1)       # (T,)

    # Test region
    t_test   = np.arange(T_train, T)
    err_test = err_mean[T_train:]
    N_test   = len(t_test)
    if N_test == 0:
        raise ValueError("No test points: T_train must be < T")

    plt.figure(figsize=(6, 4))

    # Light grey line to show the curve
    plt.plot(t_test, err_test, "-", color="0.7", alpha=0.7, zorder=1)

    # Scatter with fading alpha
    plt.scatter(t_test, err_test, zorder=2)

    plt.xlabel("t")
    plt.ylabel("Mean |error| over features")
    plt.title("Prediction error vs time (test region)")
    plt.tight_layout()
    plt.show()

def plot_per_feature_error_bars(Y_true, Y_hat, T_train, labels=None):
    Y_true = np.asarray(Y_true, float)
    Y_hat  = np.asarray(Y_hat, float)
    T, d = Y_true.shape
    assert Y_hat.shape == (T, d)

    Y_true_test = Y_true[T_train:]
    Y_hat_test  = Y_hat[T_train:]
    if Y_true_test.shape[0] == 0:
        raise ValueError("No test points: T_train must be < T")

    err = np.abs(Y_hat_test - Y_true_test)  # (N_test, d)
    err_feat_mean = err.mean(axis=0)
    err_feat_std  = err.std(axis=0)

    x = np.arange(d)

    plt.figure(figsize=(6, 4))
    plt.bar(x, err_feat_mean, yerr=err_feat_std, alpha=0.7, capsize=4)

    if labels is not None:
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.xlabel("Pauli label")
    else:
        plt.xlabel("Feature index")

    plt.ylabel("Mean |error| (± std)")
    plt.title("Per-feature error statistics (test region)")
    plt.tight_layout()
    plt.show()

def plot_lorenz63(traj):
    title_prefix="Lorenz-63"
    dt=0.01
    traj = np.asarray(traj, float)
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    t = np.arange(len(traj)) * dt

    # --- 3D plot ---
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=0.8)
    ax.set_title(f"{title_prefix}: 3D trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()

    # --- time series per dimension ---
    plt.figure(figsize=(12, 4))
    plt.plot(t, x, label="x(t)", alpha=0.9)
    plt.plot(t, y, label="y(t)", alpha=0.9)
    plt.plot(t, z, label="z(t)", alpha=0.9)
    plt.title(f"{title_prefix}: components vs time")
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.show()
