# src/plotting.py

import numpy as np
import matplotlib.pyplot as plt


def plot_true_vs_qrc_predictions(
    t_all,
    y_all,
    t_start_train,
    t_end_train,
    t_pred,
    y_pred_raw,
    t_start_pred,
    alpha,
    beta,
):

    t_all = np.asarray(t_all)
    y_all = np.asarray(y_all)
    t_pred = np.asarray(t_pred)
    y_pred_raw = np.asarray(y_pred_raw)

    plt.figure(figsize=(11, 6))

    plt.plot(t_all, y_all, "o-", color="C0", label="True y(t)", linewidth=2)

    train_times = np.arange(t_start_train, t_end_train + 1)
    plt.scatter(
        train_times,
        y_all[train_times],
        color="C2",
        s=80,
        zorder=5,
        label="Training region",
    )

    plt.plot(
        t_pred,
        y_pred_raw,
        "s--",
        color="C1",
        linewidth=2,
        markersize=6,
        label="QRC predictions (AR)",
    )

    plt.axvline(x=t_start_pred + 0.5, color="gray", linestyle="--", linewidth=1.5)
    plt.text(
        t_start_pred + 0.7,
        np.max(y_all) * 0.9,
        "prediction only",
        color="gray",
    )

    plt.title(
        "Linear recurrence (alpha, beta) = "
        f"({alpha:.2f}, {beta:.2f})\nTrue vs QRC autoregressive predictions",
        fontsize=15,
    )
    plt.xlabel("t", fontsize=13)
    plt.ylabel("y(t)", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_three_panel_qrc(
    t_all,
    y_all,
    T_TRAIN: int,
    T_TOTAL: int,
    case_00: dict,
    case_decent: dict,
    case_star: dict,
    title: str = "Autoregressive prediction: baseline vs decent vs optimal reservoir parameters",
):

    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=(14, 10),
        sharex=True,
        sharey=True
    )

    x_min, x_max = 0, T_TOTAL

    panels = [
        ("Baseline reservoir parameters", case_00, dict(linestyle=":", linewidth=2.8, color="gray")),
        ("Decent reservoir parameters", case_decent, dict(linestyle="--", linewidth=2.8, color="#2ca02c")),
        ("Optimal reservoir parameters (grid search)", case_star, dict(linestyle="-", linewidth=3.0, color="#d62728")),
    ]

    for ax, (panel_title, case, style) in zip(axes, panels):
        ax.plot(t_all, y_all, color="black", linewidth=1.5, alpha=0.4, label="True y(t)")
        ax.axvline(T_TRAIN, linestyle="--", color="black", alpha=0.6, label="Train/Test split")

        ax.plot(
            case["t_pred"], case["y_pred_raw"],
            label=rf"MSE={case['mse_raw']:.1e}",
            **style
        )

        ax.set_title(panel_title, fontsize=13)
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(x_min, x_max)

    axes[-1].set_xlabel("t", fontsize=13)
    axes[1].set_ylabel("y(t)", fontsize=13)

    plt.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_loss_landscape_two_angles(
    phis: np.ndarray,
    thetas: np.ndarray,
    L_grid: np.ndarray,
    phi_star: float | None = None,
    theta_star: float | None = None,
    phi_decent: float | None = None,
    theta_decent: float | None = None,
    show_contours: bool = True,
    eps: float = 1e-16,
):
    """
    Plot the training loss landscape L(phi, theta) as:
      - heatmap of log10(L + eps)
      - optional contour plot

    Markers:
      - baseline (0,0)
      - best (phi_star, theta_star) if provided
      - suboptimal (phi_decent, theta_decent) if provided
    """
    phis = np.asarray(phis, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    L_grid = np.asarray(L_grid, dtype=float)

    L_log = np.log10(L_grid + eps)

    # ---------------- Heatmap ----------------
    plt.figure(figsize=(10, 7))
    im = plt.imshow(
        L_log,
        origin="lower",
        aspect="auto",
        extent=[phis[0], phis[-1], thetas[0], thetas[-1]],
    )
    plt.colorbar(im, label=r"$\log_{10}\,\mathcal{L}(\phi,\theta)$ (train, normalized)")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\theta$")
    plt.title(r"Training loss landscape $\mathcal{L}(\phi,\theta)$ over the reservoir parameter grid")

    # baseline
    plt.scatter([0.0], [0.0], s=220, marker="s", c="red", edgecolors="red", linewidths=1, zorder=10,
                label=r"baseline $(0,0)$")

    # best
    if phi_star is not None and theta_star is not None:
        plt.scatter([phi_star], [theta_star], s=260, marker="*", c="red", edgecolors="red", linewidths=1,
                    zorder=11, label=r"best $(\phi^*,\theta^*)$")

    # decent
    if phi_decent is not None and theta_decent is not None:
        plt.scatter([phi_decent], [theta_decent], s=160, marker="D", c="red", edgecolors="red", linewidths=1,
                    zorder=10, label=r"suboptimal")

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    if show_contours:
        plt.figure(figsize=(10, 7))
        cs = plt.contour(phis, thetas, L_log, levels=15)
        plt.clabel(cs, inline=True, fontsize=8)
        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$\theta$")
        plt.title(r"Contour view of $\log_{10}\,\mathcal{L}(\phi,\theta)$")

        plt.scatter([0.0], [0.0], s=180, marker="s", c="red", edgecolors="red", linewidths=1.5, zorder=10,
                    label=r"baseline $(0,0)$")

        if phi_star is not None and theta_star is not None:
            plt.scatter([phi_star], [theta_star], s=320, marker="*", c="red", edgecolors="black", linewidths=1.5,
                        zorder=11, label=r"best $(\phi^*,\theta^*)$")

        if phi_decent is not None and theta_decent is not None:
            plt.scatter([phi_decent], [theta_decent], s=200, marker="D", c="red", edgecolors="black", linewidths=1.5,
                        zorder=10, label=r"suboptimal")

        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

def plot_prediction_case(ax, title, t_all, y_all, T_TRAIN, res, show_legend=True):
    COL_TRUE = "0.55"
    COL_SPLIT = "0.25"
    COL_PRED = "#b73434"

    ax.plot(t_all, y_all, color=COL_TRUE, lw=2.0, label="True y(t)")
    ax.axvline(T_TRAIN, color=COL_SPLIT, lw=1.2, ls="--", alpha=0.9)

    ax.scatter(t_all[:T_TRAIN+1], y_all[:T_TRAIN+1], s=28, color=COL_TRUE, alpha=0.9, zorder=5)

    ax.plot(
        res["t_pred"], res["y_pred_raw"],
        color=COL_PRED, lw=2.4, ls="-",
        label=f"QRC (test, AR) — MSE={res['mse_test']:.2e}"
    )

    ax.set_title(title)
    ax.set_ylabel("y(t)")
    ax.grid(True, alpha=0.22)
    if show_legend:
        ax.legend(loc="upper right", frameon=False)


def plot_three_normalizations(t_all, y_all, T_TRAIN, res_train, res_full, res_bestM, alpha, beta):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"QRC predictions: autoregressive rollout (test)\n"
        f"AR(2): α={alpha}, β={beta}  |  split at t={T_TRAIN}",
        y=0.98
    )

    plot_prediction_case(axes[0], "Train-range normalization", t_all, y_all, T_TRAIN, res_train, True)
    plot_prediction_case(axes[1], "Full-series normalization (non-causal)", t_all, y_all, T_TRAIN, res_full, True)
    plot_prediction_case(
        axes[2],
        f"Enlarged normalization (optimal M ≈ {res_bestM['margin']:.2f})",
        t_all, y_all, T_TRAIN, res_bestM, True
    )

    axes[2].set_xlabel("t")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_enlarged_multiple_M(t_all, y_all, T_TRAIN, results_list, alpha, beta):
    fig, axes = plt.subplots(len(results_list), 1, figsize=(12, 2.6*len(results_list)), sharex=True)
    fig.suptitle(
        "Enlarged normalization: effect of margin M\n"
        f"AR(2): α={alpha}, β={beta}",
        y=0.995
    )

    if len(results_list) == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        M = res["margin"]
        tag = f"M={M:.2f}" if abs(M - round(M)) > 1e-9 else f"M={int(M)}"
        plot_prediction_case(ax, f"Enlarged normalization ({tag})", t_all, y_all, T_TRAIN, res, True)

    axes[-1].set_xlabel("t")
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.show()


def plot_metric_vs_M(M_values, values, ylabel, title, logy=True, marker_x=None, marker_y=None, marker_label=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.2))
    ax.plot(M_values, values, lw=2.2)

    if marker_x is not None and marker_y is not None:
        ax.scatter([marker_x], [marker_y], s=70, zorder=5, label=marker_label)
        ax.axvline(marker_x, ls="--", lw=1.2, alpha=0.9)

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Margin factor M")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    if marker_label is not None:
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def plot_two_normalizations(t_all, y_all, T_TRAIN, res_train, res_bestM, alpha, beta):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.7), sharex=True)
    fig.suptitle(
        f"QRC predictions: autoregressive rollout (test)\n"
        f"AR(2): α={alpha}, β={beta}  |  split at t={T_TRAIN}",
        y=0.98
    )

    plot_prediction_case(axes[0], "Train-range normalization (baseline)", t_all, y_all, T_TRAIN, res_train, True)
    plot_prediction_case(
        axes[1],
        f"Enlarged normalization (optimal M ≈ {res_bestM['margin']:.2f})",
        t_all, y_all, T_TRAIN, res_bestM, True
    )

    axes[1].set_xlabel("t")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
