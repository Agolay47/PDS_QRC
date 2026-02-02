import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression

def linear_regression(X, Y):
    """
    Compute the weight matrix w of the linear regression (without ridge regression).

    Parameters:
    X (numpy.ndarray): feature matrix
    Y (numpy.ndarray): output

    Returns:
    w (numpy.ndarray): weight matrix
    """
    # linear regression without ridge
    w = LinearRegression(fit_intercept=False)
    w.fit(X, Y)
    return w

def LR(y_true, true_coeffs, t_train, t_test, n, title, plotting = True, improved = False):
    """
    Compute the linear regression on the dataset (for f1 and f2 only)

    Parameters:
    y_true (numpy.ndarray): true values
    true_coeffs (list): true coefficients of the function
    t_train (int): training set size
    t_test (int): test set size
    n (int): sliding window size
    title (str): title of the plots
    plotting (bool): whether to plot the results
    improved (bool): whether to use the improved LR model (with cos term)

    Returns:
    y_pred (numpy.ndarray): predicted values
    w (numpy.ndarray): weight matrix
    """

    assert t_train >= n #(>= minimal window size)
    T = t_train + t_test    # total size (of the dataset)

    # Generating the training set
    Y = y_true[:t_train]

    # Generating the feature matrix
    if improved:
        X = np.zeros((t_train, n+2))
    else:
        X = np.zeros((t_train, n+1))

    for i in range(t_train):
        for j in range(n):
            X[i][j] = y_true[i-1-j]
        X[i][n] = 1
        if improved:
            X[i][n+1] = np.cos(y_true[i-1])

    # Train (LR)
    w = linear_regression(X, Y)

    # Predict (LR)
    y_pred = y_true.copy()
    for i in range(t_train+1, T):
        x = [y_pred[i-1-j] for j in range(n)]
        x.append(1)
        if improved:
            x.append(np.cos(y_pred[i-1]))
        y_pred[i] = w.predict([x])[0]

    if plotting:
        # w coeffs
        true_coeffs = np.asarray(true_coeffs)
        learned_coeffs = np.asarray(w.coef_)

        n_true = len(true_coeffs)
        n_learned = len(learned_coeffs)
        if n_learned < n_true:
            learned_coeffs = np.pad(
                learned_coeffs,
                pad_width=(0, n_true - n_learned),
                mode="constant",
                constant_values=0.0
            )

        if improved:
            if len(title) > 2:
                t = title[:2] + " (alpha = " + title[5:] + ", improved LR model)"
            else:
                t = title + " (improved LR model)"
            title += "_IMPROVED"

        else:
            if len(title) > 2:
                t = title[:2] + " (alpha = " + title[5:] + ", simple LR model)"
            else:
                t = title + " (simple LR model)"

        w_plot(true_coeffs, learned_coeffs, t, f"datas/QRC_intuition/{title}_COEFFS_train" + str(t_train) + "_test" + str(t_test) + ".png")
        # training set + test set
        test_plotting(y_true, y_pred, t_train, t, f"datas/QRC_intuition/{title}_TT_train" + str(t_train) + "_test" + str(t_test) + ".png", training_included = True)
        # test set
        test_plotting(y_true, y_pred, t_train, t, f"datas/QRC_intuition/{title}_T_train" + str(t_train) + "_test" + str(t_test) + ".png", training_included = False)
        
    return y_pred, w


def w_plot(true_coeffs, learned_coeffs, title, filename):
        """
        Plot true coefficients vs learned coefficients

        Parameters:
        true_coeffs (numpy.ndarray): true coefficients
        learned_coeffs (numpy.ndarray): learned coefficients
        title (str): title of the plot
        filename (str): filename of the plot

        Returns:
        None
        """
        order = np.arange(len(true_coeffs))[::]
        
        # mask for nonzero learned coefficients
        mask = learned_coeffs != 0.0
        fig = plt.figure(figsize=(8 * 0.75, 4 * 0.75))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2])

        # ---- Scatter plot ----
        ax0 = fig.add_subplot(gs[0])
        ax0.scatter(
            true_coeffs[mask],
            learned_coeffs[mask],
            s=80,
            label="learned"
        )
        lim = max(
            np.max(np.abs(true_coeffs)),
            np.max(np.abs(learned_coeffs))
        ) * 1.1
        ax0.plot([-lim, lim], [-lim, lim], "k--", label="ideal")
        ax0.set_xlabel("True coefficients")
        ax0.set_ylabel("Learned coefficients")
        ax0.set_title("Coefficient recovery")
        ax0.legend()
        ax0.set_aspect("equal", adjustable="box")
        ax0.set_xlim(-lim, lim)
        ax0.set_ylim(-lim, lim)

        # ---- Table plot ----
        ax1 = fig.add_subplot(gs[1])
        ax1.axis("off")
        table_data = []
        for i in order:
            learned_str = "-" if learned_coeffs[i] == 0.0 else f"{learned_coeffs[i]:.3f}"
            table_data.append([f"{true_coeffs[i]:.3f}", learned_str])
        row_labels = [f"$w_{i}$" for i in order]
        col_labels = ["True", "Learned"]
        table = ax1.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center"
        )
        table.scale(1.1, 1.4)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax1.set_title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()


def test_plotting(y_true, y_pred, t_train, title, filename, training_included = True):
    """
    Plot true values vs predicted values

    Parameters:
    y_true (numpy.ndarray): true values
    y_pred (numpy.ndarray): predicted values
    t_train (int): training set size
    title (str): title of the plot
    filename (str): filename of the plot
    training_included (bool): whether to include the training set in the plot

    Returns:
    None
    """
    plt.figure(figsize=(20, 4))
    title = f"Model prediction vs. ground truth on " + str(title) + "."

    if training_included:
        plt.plot(y_true, "-o", label="True", alpha=0.8)
        plt.plot(y_pred, "--x", label="Pred", alpha=0.8)
        title = title + f" Training set included."
    else:
        T = len(y_true)
        train_range = np.arange(t_train, T)
        plt.plot(train_range, y_true[t_train:], "-o", label="True", alpha=0.8)
        plt.plot(train_range, y_pred[t_train:], "--x", label="Pred", alpha=0.8)
        title = title + f" Test set only."

    plt.axvline(x=t_train, color="k", linestyle="--", linewidth=1.5, alpha=0.8, label="train/test split")
    
    plt.xlabel("t")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_param_vs_NRMSE(param_vals, error, filename):
    """
    Plot NRMSE vs parameter values with color bands indicating the quality of the results.

    Parameters:
    param_vals (list or numpy.ndarray): parameter values (e.g., alpha)
    error (list or numpy.ndarray): NRMSE values corresponding to the parameter values
    filename (str): filename to save the plot

    Returns:
    None

    Notes:
    The color bands are as follows:
    - Excellent (0–0.1) : green
    -Good (0.1–0.3) : green
    -Poor (0.7–1) : red
    -Bad (>1) : red-brown
    """

    fig, ax = plt.subplots(figsize=(20, 4))
    ymax = float(max(np.max(error) * 1.10, 1.05))
    ax.set_ylim(0.0, ymax)

    bands = [
        (0.0, 0.1, "#2ecc71", 0.12, "Excellent (0–0.1)"),
        (0.1, 0.3, "#2ecc71", 0.22, "Good (0.1–0.3)"),
        (0.3, 0.7, "#f39c12", 0.18, "Average (0.3–0.7)"),
        (0.7, 1.0, "#e67e22", 0.22, "Poor (0.7–1)"),
        (1.0, ymax, "#e74c3c", 0.15, "Bad (>1)"),
    ]
    for y0, y1, color, a, _ in bands:
        y0c = max(0.0, min(y0, ymax))
        y1c = max(0.0, min(y1, ymax))
        if y1c > y0c:  # only draw if visible
            ax.axhspan(y0c, y1c, color=color, alpha=a, lw=0)

    ax.plot(param_vals, error, "-o", alpha=0.85, label="NRMSE")
    ax.set_xlabel("alpha")
    ax.set_ylabel("NRMSE")
    ax.set_title("NRMSE vs alpha on f2")
    
    # legend patches
    patches = [mpatches.Patch(facecolor=c, alpha=a, label=lab) for _, _, c, a, lab in bands]
    line_proxy = plt.Line2D([0], [0], color="k", marker="o", linestyle="-", alpha=0.85, label="NRMSE")
    ax.legend(handles=[line_proxy] + patches, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    