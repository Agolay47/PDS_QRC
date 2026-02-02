import numpy as np
import time
import matplotlib.pyplot as plt
import re
import random

import QRC_construction.Reservoirs as RS
import QRC_construction.TrainPredict as TP

def NRMSE(y_true, y_pred, eps=1e-12):
    """
    Compute the Normalized Root Mean Square Error (NRMSE) per dimension.
    
    Parameters:
    y_true (np.ndarray): True output values of shape (T, d).
    y_pred (np.ndarray): Predicted output values of shape (T, d).
    eps (float, optional): A small value to avoid division by zero. Defaults to 1e-12.
    
    Returns:
    np.ndarray: An array of NRMSE values for each dimension.
    """
    rmse = np.sqrt(np.mean((y_pred - y_true)**2, axis=0))
    sigma = np.std(y_true, axis=0)
    nrmse_err = rmse / (sigma + eps)
    return nrmse_err                             


# =======================================
# GRID SEARCH LOOP ON CLASSICAL DATASET
# =======================================
def find_optimal_res_sliding_window_classical_dataset(y, T_train, T_test, nb_input_qubits,
                                                    sliding_window_size_range=(1, 3),
                                                    nb_extra_qubits_range=(0, 1),
                                                    seeds=(0, 1),
                                                    # -------- X measurements (features for LR) --------
                                                    x_meas_bases=None,
                                                    x_meas_qubits_range=None, # e.g. (N, N) or (N-2, N)
                                                    # -------- which reservoir families to try --------
                                                    try_res_G=True,
                                                    try_res_MG=True,
                                                    try_res_D=True,
                                                    try_res_ISING=True,
                                                    # -------- reservoir hyperparams --------
                                                    depths_G=(3, 10),
                                                    diagonal_ks=(2, 3),
                                                    ising_times=(10.0,),
                                                    ising_trotter_steps=(1, 2),
                                                    ising_h_over_Js=(0.1,),
                                                    Js=(1.0,),
                                                    
                                                    try_ring_connectivity=False,
                                                    try_Rz=False,
                                                    meas_SZ=True,

                                                    # -------- ranking --------
                                                    top_k=20,
                                                    ):
    """
    Perform a grid search to find the optimal quantum reservoir computing (QRC) for a classical dataset

    Parameters:
    y (np.ndarray): Time series data of shape (T,).
    T_train (int): Size of the training set.
    T_test (int): Size of the test set.
    nb_input_qubits (int): Number of input qubits (dimension of y).
    sliding_window_size_range (tuple): Range of sliding window sizes to try (min, max).
    nb_extra_qubits_range (tuple): Range of extra qubits to try (min, max).
    seeds (tuple): Random seeds for reproducibility.
    x_meas_bases (tuple or None): Measurement bases for input features.
    x_meas_qubits_range (tuple or None): Range of number of qubits to measure for input features (min, max).
    try_res_G (bool): Whether to try G-family reservoirs.
    try_res_MG (bool): Whether to try MG-family reservoirs.
    try_res_D (bool): Whether to try Diagonal reservoirs.
    try_res_ISING (bool): Whether to try Ising reservoirs.
    depths_G (tuple): Depths to try for G-family reservoirs.
    diagonal_ks (tuple): k values to try for Diagonal reservoirs.
    ising_times (tuple): Time T values to try for Ising reservoirs.
    ising_trotter_steps (tuple): Trotter steps to try for Ising reservoirs.
    ising_h_over_Js (tuple): h/J ratios to try for Ising reservoirs.
    Js (tuple): J values to try for Ising reservoirs.
    try_ring_connectivity (bool): Whether to try ring connectivity.
    try_Rz (bool): Whether to try encoding using Rz operator.
    top_k (int): Number of top and bottom configurations to return.

    Returns:
    best (list[dict]): Top k configurations with lowest NRMSE.
    worst (list[dict]): Bottom k configurations with highest NRMSE.
    """

    # --- fixed params ---
    is_dataset_classical = True
    classical_min_val = float(np.min(y))
    classical_max_val = float(np.max(y))

    # --- varying params ranges ---
    n_min, n_max = sliding_window_size_range
    extra_min, extra_max = nb_extra_qubits_range

    # --- x measurement combos default ---
    if x_meas_bases is None:
        singles_xyz = ("X", "Y", "Z")
        pairs_xyz   = ("XX", "YY", "ZZ")

        if meas_SZ:
            meas_combos = [
                ("S:XYZ", (singles_xyz, (), ())),
                ("S+P:XYZ", (singles_xyz, pairs_xyz, ())), #full

                ("S:XZ", (("X","Z"), (), ())),
                ("S+P:XZ", (("X","Z"), ("XX","ZZ"), ())), #reduced

                ("S:Y", (("Y",), (), ())),
                ("S+P:Y", (("Y",), ("YY",), ())),
                ("S:Z", (("Z",), (), ())),
                ("S+P:Z", (("Z",), ("ZZ",), ())), #single axis
            ]
        else:
            meas_combos = [
                ("S:XYZ", (singles_xyz, (), ())),
                ("S+P:XYZ", (singles_xyz, pairs_xyz, ())), #full

                ("S:XZ", (("X","Z"), (), ())),
                ("S+P:XZ", (("X","Z"), ("XX","ZZ"), ())), #reduced

                ("S:Y", (("Y",), (), ())),
                ("S+P:Y", (("Y",), ("YY",), ())), #single axis
            ]

    else:
        meas_combos = x_meas_bases

    # --- which families do we try? ---
    families = []
    if try_res_G:
        families += ["G1", "G2", "G3"]
    if try_res_MG:
        families += ["MG"]
    if try_res_D:
        families += ["D"]
    if try_res_ISING:
        families += ["ISING"]

    if not families:
        raise ValueError("No reservoir family selected. Set at least one try_res_* = True.")

    # --- ring connectivity choices ---
    ring_choices = [False, True] if try_ring_connectivity else [False]

    # --- R encoding choices ---
    R_choices = [False, True] if try_Rz else [False]

    results = []

    # ============================
    # GRID SEARCH LOOP
    # ============================
    for sliding_window_size in range(n_min, n_max + 1):
        for nb_extra_qubits in range(extra_min, extra_max + 1):
            nb_qubits = nb_input_qubits * sliding_window_size + nb_extra_qubits

            # --- number of measured qubits range ---
            if x_meas_qubits_range is None:
                x_meas_qubits_min = nb_qubits
                x_meas_qubits_max = nb_qubits
            else:
                x_meas_qubits_min, x_meas_qubits_max = x_meas_qubits_range
                x_meas_qubits_min = max(1, min(x_meas_qubits_min, nb_qubits))
                x_meas_qubits_max = max(1, min(x_meas_qubits_max, nb_qubits))

            for nb_x_qubits_measured in range(x_meas_qubits_min, x_meas_qubits_max + 1):

                # building list of measured-qubit subsets
                x_qubits_measured_list = []
                if nb_x_qubits_measured == nb_qubits:
                    x_qubits_measured_list.append(list(range(nb_qubits)))
                else:
                    # sampling one subset per seed (reproducible)
                    for s in seeds:
                        rng = random.Random(int(s))
                        subset = rng.sample(range(nb_qubits), nb_x_qubits_measured)
                        x_qubits_measured_list.append(sorted(subset))

                for x_qubits_measured in x_qubits_measured_list:
                    for meas_name, meas_basis_tuple in meas_combos:

                        # diagonal can also include k=nb_qubits (DN)
                        diagonal_ks_effective = tuple(dict.fromkeys(list(diagonal_ks) + [nb_qubits]))

                        for family in families:

                            # param grid per family
                            if family in ["G1", "G2", "G3", "MG"]:
                                family_param_grid = [("depth", d) for d in depths_G]
                            elif family == "D":
                                family_param_grid = [("k", k) for k in diagonal_ks_effective]
                            else:  # ISING
                                family_param_grid = [
                                    ("ising", (time_T, trotter, hratio, js))
                                    for time_T in ising_times
                                    for trotter in ising_trotter_steps
                                    for hratio in ising_h_over_Js
                                    for js in Js
                                ]

                            for param_name, param_val in family_param_grid:
                                for seed in seeds:
                                    for ring_connectivity in ring_choices:
                                        for is_Rz in R_choices:

                                            # --- Building reservoir circuit ---
                                            if family == "G1":
                                                res, meta = RS.build_reservoir_G1(
                                                    nb_qubits=nb_qubits, depth=int(param_val), seed=int(seed),
                                                    ring_connectivity=ring_connectivity
                                                )
                                            elif family == "G2":
                                                res, meta = RS.build_reservoir_G2(
                                                    nb_qubits=nb_qubits, depth=int(param_val), seed=int(seed),
                                                    ring_connectivity=ring_connectivity
                                                )
                                            elif family == "G3":
                                                res, meta = RS.build_reservoir_G3(
                                                    nb_qubits=nb_qubits, depth=int(param_val), seed=int(seed),
                                                    ring_connectivity=ring_connectivity
                                                )
                                            elif family == "MG":
                                                res, meta = RS.build_reservoir_MG(
                                                    nb_qubits=nb_qubits, depth=int(param_val), seed=int(seed),
                                                    ring_connectivity=ring_connectivity
                                                )
                                            elif family == "D":
                                                res, meta = RS.build_reservoir_diagonal(
                                                    nb_qubits=nb_qubits, k=int(param_val), seed=int(seed)
                                                )
                                            else:  # ISING
                                                time_T, trotter, hratio, js = param_val
                                                res, meta = RS.build_reservoir_ising(
                                                    nb_qubits=nb_qubits,
                                                    seed=int(seed),
                                                    Js=float(js),
                                                    h_over_Js=float(hratio),
                                                    time_T=float(time_T),
                                                    trotter_steps=int(trotter),
                                                    ring_connectivity=ring_connectivity
                                                )

                                            # --- Running prediction + timing ---
                                            t0 = time.perf_counter()
                                            y_pred, y_true, W, X_train, Y_train = TP.predict_sliding_window(
                                                y,
                                                T_train,
                                                T_test,
                                                res,
                                                nb_input_qubits,
                                                nb_extra_qubits,
                                                sliding_window_size,
                                                is_dataset_classical=is_dataset_classical,
                                                x_meas_bases=meas_basis_tuple,
                                                x_qubits_measured=x_qubits_measured,
                                                y_meas_bases=None,
                                                y_qubits_measured=None,
                                                Rz=is_Rz,
                                                classical_min_val=classical_min_val,
                                                classical_max_val=classical_max_val
                                            )
                                            elapsed_s = time.perf_counter() - t0

                                            # --- NRMSE on test only ---
                                            yt = y_true[T_train+1:]
                                            yp = y_pred[T_train+1:]
                                            nrmse_val = float(np.mean(NRMSE(yt, yp)))

                                            rec = {
                                                "NRMSE": nrmse_val,
                                                "time_s": float(elapsed_s),
                                                "family": family,
                                                "param_name": param_name,
                                                "param_val": param_val,
                                                "seed": int(seed),
                                                "sliding_window_size": int(sliding_window_size),
                                                "nb_input_qubits": int(nb_input_qubits),
                                                "nb_extra_qubits": int(nb_extra_qubits),
                                                "nb_qubits": int(nb_qubits),
                                                "meas_name": meas_name,
                                                "x_meas_bases": meas_basis_tuple,
                                                "x_qubits_measured": x_qubits_measured,
                                                "ring_connectivity": bool(ring_connectivity),
                                                "Rz": bool(is_Rz),
                                                "meta": meta,
                                                "y_true": y_true,
                                                "y_pred": y_pred
                                            }
                                            results.append(rec)

                                            print(
                                                f"{family} {param_name}={param_val} | seed={seed} | "
                                                f"n={sliding_window_size} | "
                                                f"extra={nb_extra_qubits} | "
                                                f"meas={meas_name} | "
                                                f"Rz={is_Rz} | "
                                                f"time={elapsed_s:.2f}s | NRMSE={nrmse_val:.3f}"
                                            )

    # --- Sort and pick best/worst ---
    results_sorted = sorted(results, key=lambda r: r["NRMSE"])
    best = results_sorted[:top_k]
    worst = results_sorted[-top_k:][::-1]

    return best, worst


# =======================================
# GRID SEARCH LOOP ON QUANTUM DATASET
# =======================================
def find_optimal_res_sliding_window_quantum_dataset(y, T_train, T_test, nb_input_qubits, 
                                                    sliding_window_size_range=(1, 3), 
                                                    nb_extra_qubits_range=(0, 1), seeds=(0, 1),
                                                    # -------- X measurements (features for LR) --------
                                                    x_meas_bases=None,
                                                    x_meas_qubits_range=None,   # e.g. (N, N) or (N-2, N)
                                                    # -------- Y measurements (defines y_true / target) --------
                                                    y_meas_bases=None,
                                                    y_qubits_measured=None, 
                                                    # -------- which reservoir families to try --------
                                                    try_res_G=True,
                                                    try_res_D=True,
                                                    try_res_ISING=True,
                                                    try_res_MG=True,
                                                    # -------- reservoir hyperparams --------
                                                    depths_G=(3, 10),
                                                    diagonal_ks=(2, 3),
                                                    ising_times=(10.0,),
                                                    ising_trotter_steps=(1, 2),
                                                    ising_h_over_Js=(0.1,),
                                                    Js=(1.0,),

                                                    try_ring_connectivity=False,
                                                    # -------- ranking --------
                                                    top_k=20,
                                                ):
    """
    Perform a grid search to find the optimal quantum reservoir computing (QRC) for a quantum dataset.

    Parameters:
    y (np.ndarray): Time series data of shape (T,).
    T_train (int): Size of the training set.
    T_test (int): Size of the test set.
    nb_input_qubits (int): Number of input qubits (dimension of y).
    sliding_window_size_range (tuple): Range of sliding window sizes to try (min, max).
    nb_extra_qubits_range (tuple): Range of extra qubits to try (min, max).
    seeds (tuple): Random seeds for reproducibility.
    x_meas_bases (tuple or None): Measurement bases for input features.
    x_meas_qubits_range (tuple or None): Range of number of qubits to measure for input features (min, max).
    y_meas_bases (tuple or None): Measurement bases for target outputs.
    y_qubits_measured (list or None): List of qubits to measure for target outputs.
    try_res_G (bool): Whether to try G-family reservoirs.
    try_res_MG (bool): Whether to try MG-family reservoirs.
    try_res_D (bool): Whether to try Diagonal reservoirs.
    try_res_ISING (bool): Whether to try Ising reservoirs.
    depths_G (tuple): Depths to try for G-family reservoirs.
    diagonal_ks (tuple): k values to try for Diagonal reservoirs.
    ising_times (tuple): Time T values to try for Ising reservoirs.
    ising_trotter_steps (tuple): Trotter steps to try for Ising reservoirs.
    ising_h_over_Js (tuple): h/J ratios to try for Ising reservoirs.
    Js (tuple): J values to try for Ising reservoirs.
    try_ring_connectivity (bool): Whether to try ring connectivity.
    top_k (int): Number of top and bottom configurations to return.

    Returns:
    best (list[dict]): Top k configurations with lowest NRMSE.
    worst (list[dict]): Bottom k configurations with highest NRMSE.
    """
    # --- fixed param ---
    is_dataset_classical = False
    singles_xyz = ("X", "Y", "Z")
    pairs_xyz   = ("XX", "YY", "ZZ")

    # --- varying params ranges ---
    n_min, n_max = sliding_window_size_range
    extra_min, extra_max = nb_extra_qubits_range

    # --- x measurement combos default ---
    if x_meas_bases is None:
    
        x_meas_combos = [
            ("S:XYZ", (singles_xyz, (), ())),
            ("S+P:XYZ", (singles_xyz, pairs_xyz, ())), #full

            ("S:XZ", (("X","Z"), (), ())),
            ("S+P:XZ", (("X","Z"), ("XX","ZZ"), ())), #reduced

            ("S:Y", (("Y",), (), ())),
            ("S+P:Y", (("Y",), ("YY",), ())), #single
        ]
    else:
        x_meas_combos = x_meas_bases

    # --- y measurement combos default ---
    if y_meas_bases is None:
        y_meas_bases = (singles_xyz, (), ()) # 3-d target (Bloch components)

    if y_qubits_measured is None:
        y_qubits_measured = list(range(nb_input_qubits))

    # --- which families do we try? ---
    families = []
    if try_res_G:
        families += ["G1", "G2", "G3"]
    if try_res_MG:
        families += ["MG"]
    if try_res_D:
        families += ["D"]
    if try_res_ISING:
        families += ["ISING"]

    if not families:
        raise ValueError("No reservoir family selected. Set at least one try_res_* = True.")
    
    # --- ring connectivity choices ---
    ring_choices = [False, True] if try_ring_connectivity else [False]

    results = []

    # ============================
    # GRID SEARCH LOOP
    # ============================
    for sliding_window_size in range(n_min, n_max + 1):
        for nb_extra_qubits in range(extra_min, extra_max + 1):
            nb_qubits = nb_input_qubits * sliding_window_size + nb_extra_qubits

            # --- number of measured qubits range ---
            if x_meas_qubits_range is None:
                x_meas_qubits_min = nb_qubits
                x_meas_qubits_max = nb_qubits
            else:
                x_meas_qubits_min, x_meas_qubits_max = x_meas_qubits_range
                x_meas_qubits_min = max(1, min(x_meas_qubits_min, nb_qubits))
                x_meas_qubits_max = max(1, min(x_meas_qubits_max, nb_qubits))

            for nb_x_qubits_measured in range(x_meas_qubits_min, x_meas_qubits_max + 1):

                # building list of measured-qubit subsets
                x_qubits_measured_list = []
                if nb_x_qubits_measured == nb_qubits:
                    x_qubits_measured_list.append(list(range(nb_qubits)))
                else:
                    # sampling one subset per seed (reproducible)
                    for s in seeds:
                        rng = random.Random(int(s))
                        subset = rng.sample(range(nb_qubits), nb_x_qubits_measured)
                        x_qubits_measured_list.append(sorted(subset))

                for x_qubits_measured in x_qubits_measured_list:
                    for meas_name, meas_basis_tuple in x_meas_combos:

                        # diagonal can also include k=nb_qubits (DN)
                        diagonal_ks_effective = tuple(dict.fromkeys(list(diagonal_ks) + [nb_qubits]))

                        for family in families:
                            # param grid per family
                            if family in ["G1", "G2", "G3", "MG"]:
                                family_param_grid = [("depth", d) for d in depths_G]
                            elif family == "D":
                                family_param_grid = [("k", k) for k in diagonal_ks_effective]
                            else:  # ISING
                                family_param_grid = [
                                    ("ising", (time_T, trotter, hratio, js))
                                    for time_T in ising_times
                                    for trotter in ising_trotter_steps
                                    for hratio in ising_h_over_Js
                                    for js in Js
                                ]

                            for param_name, param_val in family_param_grid:
                                for seed in seeds:
                                    for ring_connectivity in ring_choices:
                                        # --- Building reservoir circuit ---
                                        if family == "G1":
                                            res, meta = RS.build_reservoir_G1(
                                                nb_qubits=nb_qubits, depth=int(param_val), seed=int(seed),
                                                ring_connectivity=ring_connectivity
                                            )
                                        elif family == "G2":
                                            res, meta = RS.build_reservoir_G2(
                                                nb_qubits=nb_qubits, depth=int(param_val), seed=int(seed),
                                                ring_connectivity=ring_connectivity
                                            )
                                        elif family == "G3":
                                            res, meta = RS.build_reservoir_G3(
                                                nb_qubits=nb_qubits, depth=int(param_val), seed=int(seed),
                                                ring_connectivity=ring_connectivity
                                            )
                                        elif family == "MG":
                                            res, meta = RS.build_reservoir_MG(
                                                nb_qubits=nb_qubits, depth=int(param_val), seed=int(seed),
                                                ring_connectivity=ring_connectivity
                                            )
                                        elif family == "D":
                                            res, meta = RS.build_reservoir_diagonal(
                                                nb_qubits=nb_qubits, k=int(param_val), seed=int(seed)
                                            )
                                        else:  # ISING
                                            time_T, trotter, hratio, js = param_val
                                            res, meta = RS.build_reservoir_ising(
                                                nb_qubits=nb_qubits,
                                                seed=int(seed),
                                                Js=float(js),
                                                h_over_Js=float(hratio),
                                                time_T=float(time_T),
                                                trotter_steps=int(trotter),
                                                ring_connectivity=ring_connectivity
                                            )

                                        # --- Running prediction + timing ---
                                        t0 = time.perf_counter()
                                        y_pred, y_true, W, X_train, Y_train = TP.predict_sliding_window(
                                            y,
                                            T_train,
                                            T_test,
                                            res,
                                            nb_input_qubits,
                                            nb_extra_qubits,
                                            sliding_window_size,
                                            is_dataset_classical=is_dataset_classical,
                                            x_meas_bases=meas_basis_tuple,
                                            x_qubits_measured=x_qubits_measured,
                                            y_meas_bases=y_meas_bases,
                                            y_qubits_measured=y_qubits_measured,
                                            # classical_min/max not used here
                                            Rz=False,
                                            classical_min_val=None,
                                            classical_max_val=None,
                                        )
                                        elapsed_s = time.perf_counter() - t0

                                        # --- NRMSE on test only ---
                                        yt = np.asarray(y_true[T_train+1:])
                                        yp = np.asarray(y_pred[T_train+1:])
                                        if yt.ndim == 1:
                                            yt = yt[:, None]; yp = yp[:, None]

                                        nrmse_per_dim = NRMSE(yt, yp)
                                        nrmse_val = float(np.mean(nrmse_per_dim))

                                        rec = {
                                            "NRMSE": nrmse_val,
                                            "time_s": float(elapsed_s),
                                            "family": family,
                                            "param_name": param_name,
                                            "param_val": param_val,
                                            "seed": int(seed),
                                            "sliding_window_size": int(sliding_window_size),
                                            "nb_input_qubits": int(nb_input_qubits),
                                            "nb_extra_qubits": int(nb_extra_qubits),
                                            "nb_qubits": int(nb_qubits),
                                            "meas_name": meas_name,
                                            "x_meas_bases": meas_basis_tuple,
                                            "x_qubits_measured": x_qubits_measured,
                                            "ring_connectivity": bool(ring_connectivity),
                                            "meta": meta,
                                            "y_true": y_true,
                                            "y_pred": y_pred,
                                        }
                                        results.append(rec)

                                        print(
                                            f"{family} {param_name}={param_val} | seed={seed} | "
                                            f"n={sliding_window_size} | "
                                            f"extra={nb_extra_qubits} | "
                                            f"meas={meas_name} | "
                                            f"Rz={False} | "
                                            f"time={elapsed_s:.2f}s | NRMSE={nrmse_val:.3f}"
                                        )

                                    
    # --- Sort and pick best/worst ---
    results_sorted = sorted(results, key=lambda r: r["NRMSE"])
    best = results_sorted[:top_k]
    worst = results_sorted[-top_k:][::-1]

    return best, worst

# =========================================

def _format_params_for_label(p: dict, show_ring=False, show_mQ=False, show_Rz=False):
    """
    Build a compact string with the most relevant params found in `p`.
    Only includes keys if they exist.
    Standard key for phase encoding is now: 'Rz' (bool or None).
    """
    parts = []

    if "family" in p:
        parts.append(str(p["family"]))

    if "param_name" in p and "param_val" in p:
        parts.append(f"{p['param_name']}={p['param_val']}")

    if "sliding_window_size" in p:
        parts.append(f"n={p['sliding_window_size']}")

    if "nb_extra_qubits" in p:
        parts.append(f"extra={p['nb_extra_qubits']}")

    if "meas_name" in p:
        parts.append(f"meas={p['meas_name']}")

    if show_mQ:
        if "mQ" in p:
            parts.append(f"mQ={p['mQ']}")
        elif "logged_mQ" in p:
            parts.append(f"mQ={p['logged_mQ']}")

    if show_ring:
        if "ring" in p:
            parts.append(f"ring={p['ring']}")
        elif "logged_ring" in p:
            parts.append(f"ring={p['logged_ring']}")

    if show_Rz:
        # Support both conventions (new: Rz, legacy: logged_Rz)
        rz_val = None
        if "Rz" in p and p["Rz"] is not None:
            rz_val = p["Rz"]
        elif "logged_Rz" in p and p["logged_Rz"] is not None:
            rz_val = p["logged_Rz"]
        if rz_val is not None:
            parts.append(f"Rz={bool(rz_val)}")

    return ", ".join(parts)


def _meas_name_to_x_meas_bases(meas_name: str):
    """
    Convert a log measurement name into x_meas_bases of the form:
        (singles, pairs, triples)
    where each is a tuple of Pauli strings, e.g.
        (("X","Y","Z"), (), ())
        (("X","Z"), ("XX","ZZ"), ())
    """
    meas_name = meas_name.strip()

    # Normalize old/variant prefixes
    # - S+T:* showed up in your logs; interpret it as "singles only" (same as S:*)
    # - ST:* likewise
    if meas_name.startswith("S+T:"):
        meas_name = "S:" + meas_name.split(":", 1)[1]
    if meas_name.startswith("ST:"):
        meas_name = "S:" + meas_name.split(":", 1)[1]

    # Accept only known families in your code path
    lookup = {
        # singles only
        "S:XYZ": (("X", "Y", "Z"), (), ()),
        "S:XZ":  (("X", "Z"), (), ()),
        "S:Y":   (("Y",), (), ()),
        "S:X":   (("X",), (), ()),
        "S:Z":   (("Z",), (), ()),

        # singles + pairs ("full" and "reduced" in your comment)
        "S+P:XYZ": (("X", "Y", "Z"), ("XX", "YY", "ZZ"), ()),
        "S+P:XZ":  (("X", "Z"), ("XX", "ZZ"), ()),

        # single + pair for Y-only
        "S+P:Y":   (("Y",), ("YY",), ()),
        "S+P:Z": (("Z",), ("ZZ",), ()),

        # (optional) other natural combos, if they appear in logs later:
        "S:XY":    (("X", "Y"), (), ()),
        "S:YZ":    (("Y", "Z"), (), ()),
        "S+P:XY":  (("X", "Y"), ("XX", "YY"), ()),
        "S+P:YZ":  (("Y", "Z"), ("YY", "ZZ"), ()),
    }

    if meas_name in lookup:
        return lookup[meas_name]

    # As a convenience: if you ever logged e.g. "S:YXZ" or "S:ZX" etc,
    # normalize ordering + duplicates for the S:* case.
    if meas_name.startswith("S:"):
        letters = meas_name.split(":", 1)[1].strip()
        bases = []
        for ch in letters:
            if ch in ("X", "Y", "Z") and ch not in bases:
                bases.append(ch)
        if bases:
            return (tuple(bases), (), ())

    raise ValueError(
        f"Unknown meas_name='{meas_name}'. Add it to _meas_name_to_x_meas_bases()."
    )


def _cast_param_val(s: str):
    s = str(s).strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        try:
            if re.fullmatch(r"[\d\.\,\s\(\)\[\]\-eE\+]+", s):
                return eval(s, {"__builtins__": {}})
        except Exception:
            pass
    return s


def _parse_bool(s: str) -> bool:
    s = str(s).strip()
    if s in ("True", "true", "1"):
        return True
    if s in ("False", "false", "0"):
        return False
    raise ValueError(f"Cannot parse bool from '{s}'")


def _parse_log_line_simple(ln: str) -> dict:
    """
    Supports:
      - with Rz:  ... | Rz=True | ... | NRMSE=...
      - no Rz:    ... | ... | NRMSE=...
    """
    raw = ln.strip()
    if not raw:
        return {}

    chunks = [c.strip() for c in raw.split("|")]

    # "<family> <param_name>=<param_val>"
    m = re.match(r"^(?P<family>\S+)\s+(?P<pname>[A-Za-z_]+)=(?P<pval>.+)$", chunks[0])
    if not m:
        raise ValueError(f"Could not parse header chunk:\n{raw}")

    out = {
        "family": m.group("family"),
        "param_name": m.group("pname"),
        "param_val": _cast_param_val(m.group("pval")),
        "raw_line": raw,
        "logged_Rz": None,  # default if not present in line
    }

    for c in chunks[1:]:
        if "=" not in c:
            continue
        k, v = c.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k == "seed":
            out["seed"] = int(v)
        elif k == "n":
            out["sliding_window_size"] = int(v)
        elif k == "extra":
            out["nb_extra_qubits"] = int(v)
        elif k == "meas":
            out["meas_name"] = v
        elif k == "Rz":
            out["logged_Rz"] = _parse_bool(v)
        elif k == "time":
            out["logged_time_s"] = float(v.replace("s", "").strip())
        elif k == "NRMSE":
            out["logged_NRMSE"] = float(v)
        else:
            out[k] = v

    required = ["seed", "sliding_window_size", "nb_extra_qubits", "meas_name", "logged_NRMSE"]
    missing = [r for r in required if r not in out]
    if missing:
        raise ValueError(f"Missing {missing} while parsing:\n{raw}")

    out.setdefault("logged_time_s", None)
    return out


def recompute_best_worst_from_logfile(
    filename: str,
    y: np.ndarray,
    T_train: int,
    T_test: int,
    nb_input_qubits: int,
    k: int = 3,
    *,
    force_ring_connectivity=None,   # logs don't have ring -> default False unless forced
    force_Rz=None,                  # None = use log if present else default below
    default_Rz: bool = False,

    is_dataset_classical: bool = True,
    classical_min_val=None,
    classical_max_val=None,

    # for quantum datasets
    y_meas_bases=None,
    y_qubits_measured=None,

    # ISING defaults if param_val not a tuple
    ising_h_over_Js: float = 0.1,
    ising_Js: float = 1.0,
    ising_trotter_steps: int = 1,
):
    """
    Compatible with BOTH of your print formats (with/without 'Rz=...').

    Notes:
    - Your simplified logs do NOT include mQ or ring:
        * ring is set to False (unless force_ring_connectivity is provided)
        * mQ is set to nb_qubits (measure all)
    """
    if is_dataset_classical:
        if classical_min_val is None:
            classical_min_val = float(np.min(y))
        if classical_max_val is None:
            classical_max_val = float(np.max(y))

    entries = []
    with open(filename, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            entries.append(_parse_log_line_simple(ln))

    if not entries:
        raise ValueError(f"No valid entries found in {filename}")

    entries_sorted = sorted(entries, key=lambda e: e["logged_NRMSE"])
    best_sel = entries_sorted[:k]
    worst_sel = entries_sorted[-k:][::-1]

    def build_reservoir(family: str, param_val, nb_qubits: int, seed: int, ring: bool):
        if family == "G1":
            return RS.build_reservoir_G1(nb_qubits=nb_qubits, depth=int(param_val), seed=seed, ring_connectivity=ring)
        if family == "G2":
            return RS.build_reservoir_G2(nb_qubits=nb_qubits, depth=int(param_val), seed=seed, ring_connectivity=ring)
        if family == "G3":
            return RS.build_reservoir_G3(nb_qubits=nb_qubits, depth=int(param_val), seed=seed, ring_connectivity=ring)
        if family == "MG":
            return RS.build_reservoir_MG(nb_qubits=nb_qubits, depth=int(param_val), seed=seed, ring_connectivity=ring)
        if family == "D":
            return RS.build_reservoir_diagonal(nb_qubits=nb_qubits, k=int(param_val), seed=seed)
        if family == "ISING":
            if isinstance(param_val, (tuple, list)):
                time_T, trotter, hratio, js = param_val
                return RS.build_reservoir_ising(
                    nb_qubits=nb_qubits, seed=seed, Js=float(js),
                    h_over_Js=float(hratio), time_T=float(time_T),
                    trotter_steps=int(trotter), ring_connectivity=ring
                )
            time_T = float(param_val)
            return RS.build_reservoir_ising(
                nb_qubits=nb_qubits, seed=seed, Js=float(ising_Js),
                h_over_Js=float(ising_h_over_Js), time_T=float(time_T),
                trotter_steps=int(ising_trotter_steps), ring_connectivity=ring
            )
        raise ValueError(f"Unknown family '{family}'")

    def recompute_one(entry: dict):
        sw = int(entry["sliding_window_size"])
        extra = int(entry["nb_extra_qubits"])
        nb_qubits = nb_input_qubits * sw + extra

        ring = False if force_ring_connectivity is None else bool(force_ring_connectivity)

        # Determine Rz: forced > logged > default
        if force_Rz is not None:
            Rz_val = bool(force_Rz)
        else:
            Rz_val = bool(entry["logged_Rz"]) if entry.get("logged_Rz", None) is not None else bool(default_Rz)

        # Logs don't have mQ: measure all deterministically
        mQ = nb_qubits
        x_qubits_measured = list(range(nb_qubits))

        y_q = list(range(nb_input_qubits)) if y_qubits_measured is None else list(y_qubits_measured)
        x_meas_bases = _meas_name_to_x_meas_bases(entry["meas_name"])

        res, meta = build_reservoir(entry["family"], entry["param_val"], nb_qubits, int(entry["seed"]), ring)

        y_pred, y_true, W, X_train, Y_train = TP.predict_sliding_window(
            y,
            T_train,
            T_test,
            res,
            nb_input_qubits,
            extra,
            sw,
            is_dataset_classical=is_dataset_classical,
            x_meas_bases=x_meas_bases,
            x_qubits_measured=x_qubits_measured,
            y_meas_bases=y_meas_bases,
            y_qubits_measured=y_q,
            Rz=Rz_val,  # <-- consistent API + key name
            classical_min_val=classical_min_val,
            classical_max_val=classical_max_val,
        )

        yt = y_true[T_train+1:]
        yp = y_pred[T_train+1:]
        nrmse_val = float(np.mean(NRMSE(yt, yp)))

        return {
            "family": entry["family"],
            "param_name": entry["param_name"],
            "param_val": entry["param_val"],
            "seed": int(entry["seed"]),
            "sliding_window_size": sw,
            "nb_input_qubits": nb_input_qubits,
            "nb_extra_qubits": extra,
            "nb_qubits": nb_qubits,
            "meas_name": entry["meas_name"],

            "mQ": mQ,
            "ring": ring,
            "Rz": Rz_val,  # <-- the one canonical key

            "NRMSE": nrmse_val,
            "logged_NRMSE": float(entry["logged_NRMSE"]),
            "raw_line": entry["raw_line"],

            "y_true": y_true,
            "y_pred": y_pred,
            "W": W,
            "X_train": X_train,
            "Y_train": Y_train,
            "meta": meta,
        }

    best = [recompute_one(e) for e in best_sel]
    worst = [recompute_one(e) for e in worst_sel]

    best = sorted(best, key=lambda d: d["NRMSE"])[:k]
    worst = sorted(worst, key=lambda d: d["NRMSE"], reverse=True)[:k]
    return best, worst


def plot_ranked_predictions(
    params,
    T_train,
    title,
    filename=None,
    steps_training_included=None,
    y_lr=None,
    show_ring=False,
    show_mQ=False,
    show_Rz=False,
):
    plt.figure(figsize=(20, 4))

    y_true = np.asarray(params[0]["y_true"]).ravel()

    steps_training_included = 0 if steps_training_included is None else int(steps_training_included)
    t0 = max(0, int(T_train) - steps_training_included)
    t_range = np.arange(t0, len(y_true))

    plt.plot(t_range, y_true[t0:], "--o", lw=1.8, label="True", alpha=0.8)

    for i, p in enumerate(params, start=1):
        y_pred = np.asarray(p["y_pred"]).ravel()
        nrmse1 = float(NRMSE(y_true[T_train+1:], y_pred[T_train+1:]))

        label_core = _format_params_for_label(
            p, show_ring=show_ring, show_mQ=show_mQ, show_Rz=show_Rz
        )

        plt.plot(
            t_range,
            y_pred[t0:],
            "--x",
            lw=1.8,
            label=f"#{i} {label_core} | NRMSE={nrmse1:.3f}",
        )

    if y_lr is not None:
        y_lr = np.asarray(y_lr).ravel()
        nrmse_lr = float(NRMSE(y_true[T_train+1:], y_lr[T_train+1:]))
        plt.plot(t_range, y_lr[t0:], "--x", lw=1.8, label=f"Simple LR | NRMSE={nrmse_lr:.3f}")

    plt.axvline(x=T_train, color="k", linestyle="--", linewidth=1.5, alpha=0.8, label="train/test split")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def plot_ranked_predictions_per_dim(
    params,
    T_train,
    title,
    filename_prefix=None,
    steps_training_included=None,
    y_lr=None,
    show_ring=False,
    show_mQ=False,
    show_Rz=False,
):
    """
    Plot true vs predicted trajectories, one figure per output dimension.
    """

    # --- extract y_true ---
    y_true = np.asarray(params[0]["y_true"])
    if y_true.ndim == 1:
        y_true = y_true[:, None]  # (T, 1)

    T, d = y_true.shape

    steps_training_included = 0 if steps_training_included is None else int(steps_training_included)
    t0 = max(0, int(T_train) - steps_training_included)
    t_range = np.arange(t0, T)

    # --- optional LR baseline ---
    if y_lr is not None:
        y_lr = np.asarray(y_lr)
        if y_lr.ndim == 1:
            y_lr = y_lr[:, None]

    # ============================
    # ONE FIGURE PER DIMENSION
    # ============================
    for dim in range(d):
        plt.figure(figsize=(20, 4))

        # True
        plt.plot(
            t_range,
            y_true[t0:, dim],
            "--o",
            lw=1.8,
            alpha=0.8,
            label="True",
        )

        # QRC predictions
        for i, p in enumerate(params, start=1):
            y_pred = np.asarray(p["y_pred"])
            if y_pred.ndim == 1:
                y_pred = y_pred[:, None]

            nrmse_dim = float(
                NRMSE(
                    y_true[T_train+1:, dim],
                    y_pred[T_train+1:, dim],
                )
            )

            label_core = _format_params_for_label(
                p, show_ring=show_ring, show_mQ=show_mQ, show_Rz=show_Rz
            )

            plt.plot(
                t_range,
                y_pred[t0:, dim],
                "--x",
                lw=1.8,
                label=f"#{i} {label_core} | NRMSE={nrmse_dim:.3f}",
            )

        # Linear regression baseline
        if y_lr is not None:
            nrmse_lr = float(
                NRMSE(
                    y_true[T_train+1:, dim],
                    y_lr[T_train+1:, dim],
                )
            )
            plt.plot(
                t_range,
                y_lr[t0:, dim],
                "--x",
                lw=1.8,
                label=f"Simple LR | NRMSE={nrmse_lr:.3f}",
            )

        # cosmetics
        plt.axvline(
            x=T_train,
            color="k",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="train/test split",
        )
        plt.xlabel("t")
        plt.ylabel(f"y (dim {dim})")
        plt.title(f"{title} — dimension {dim}")
        plt.legend()
        plt.tight_layout()

        if filename_prefix is not None:
            plt.savefig(f"{filename_prefix}_dim{dim}.pdf")

        plt.show()
