from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import ast

# ----------------------------
# Parsing normalized log lines
# ----------------------------

def parse_normalized_log_line(ln: str) -> dict:
    """
    Robust parser for normalized log line, including ISING tuples with spaces:
      ISING ising=(10.0, 1, 0.1, 1.0) | seed=0 | n=4 | extra=1 | meas=S+P:Y | Rz=False | time=0.91s | NRMSE=2.159
    """
    parts = [p.strip() for p in ln.strip().split("|")]
    if len(parts) < 8:
        raise ValueError(f"Bad line: {ln}")

    # head is everything before first "|"
    head_str = parts[0]  # e.g. "ISING ising=(10.0, 1, 0.1, 1.0)"
    if not head_str:
        raise ValueError(f"Empty head in line: {ln}")

    # family = first token; param_expr = rest (may contain spaces)
    first_space = head_str.find(" ")
    if first_space == -1:
        raise ValueError(f"Bad head (no param expr): {head_str}")

    family = head_str[:first_space].strip().upper()
    param_expr = head_str[first_space + 1 :].strip()  # e.g. "ising=(10.0, 1, 0.1, 1.0)"

    if "=" not in param_expr:
        raise ValueError(f"Bad param expr in head: {head_str}")

    param_name, param_val_raw = param_expr.split("=", 1)
    param_name = param_name.strip()
    param_val_raw = param_val_raw.strip()

    # parse other fields key=value
    kv = {}
    for p in parts[1:]:
        if "=" not in p:
            raise ValueError(f"Bad field '{p}' in line: {ln}")
        k, v = p.split("=", 1)
        kv[k.strip()] = v.strip()

    seed = int(kv["seed"])
    n = int(kv["n"])
    extra = int(kv["extra"])
    meas = kv["meas"]
    Rz = kv["Rz"].lower() == "true"

    t = kv["time"]
    if t.endswith("s"):
        t = t[:-1]
    time_val = float(t)

    nrmse = float(kv["NRMSE"])

    # Cast param_val smartly
    if param_name == "ising":
        # parse tuple like "(10.0, 1, 0.1, 1.0)"
        try:
            param_val = ast.literal_eval(param_val_raw)
        except Exception:
            param_val = param_val_raw
    else:
        # int -> float -> raw
        try:
            param_val = int(param_val_raw)
        except Exception:
            try:
                param_val = float(param_val_raw)
            except Exception:
                param_val = param_val_raw

    return dict(
        family=family,
        param_name=param_name,
        param_val=param_val,
        param_val_raw=param_val_raw,
        seed=seed,
        n=n,
        extra=extra,
        meas=meas,
        Rz=Rz,
        time=time_val,
        NRMSE=nrmse,
        raw_line=ln.strip(),
    )

def load_entries(filename: str) -> list[dict]:
    out = []
    with open(filename, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(parse_normalized_log_line(ln))
            except Exception:
                continue
    return out


def _apply_filters(entries: list[dict], filters: dict | None) -> list[dict]:
    if not filters:
        return entries
    out = []
    for e in entries:
        ok = True
        for k, v in filters.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                if e.get(k) not in v:
                    ok = False
                    break
            else:
                if e.get(k) != v:
                    ok = False
                    break
        if ok:
            out.append(e)
    return out


def table_classification_with_witness(
    filename: str,
    param: str,
    tri: str | None = None,   # "best" | "avg" | "worst" | None
    k: int | None = None,
    *,
    filters: dict | None = None,
) -> list | tuple[list, list, list]:
    """
    Rank values of `param` by tri-aggregate NRMSE.
    Also returns the "witness" entry:
      - tri="best": the entry achieving min NRMSE in that group
      - tri="worst": the entry achieving max NRMSE in that group
      - tri="avg": mean NRMSE + best_witness + worst_witness
    """
    entries = _apply_filters(load_entries(filename), filters)

    if not entries:
        raise ValueError("No entries after filtering.")

    # group by param value
    groups = defaultdict(list)
    for e in entries:
        groups[e[param]].append(e)

    def summarize_group(e_list: list[dict], mode: str) -> dict:
        nrmse_vals = np.array([e["NRMSE"] for e in e_list], dtype=float)
        i_best = int(np.argmin(nrmse_vals))
        i_worst = int(np.argmax(nrmse_vals))

        best_e = e_list[i_best]
        worst_e = e_list[i_worst]
        mean_val = float(np.mean(nrmse_vals))

        if mode == "best":
            score = float(best_e["NRMSE"])
            witness = best_e
            return score, witness
        if mode == "worst":
            score = float(worst_e["NRMSE"])
            witness = worst_e
            return score, witness
        if mode == "avg":
            # witness not unique -> return mean plus best/worst witnesses
            score = mean_val
            witness = {"best_witness": best_e, "worst_witness": worst_e}
            return score, witness

        raise ValueError("tri must be 'best','avg','worst' or None")

    def make_rows(mode: str) -> list[dict]:
        rows = []
        for val, e_list in groups.items():
            score, witness = summarize_group(e_list, mode)

            row = {
                "param": param,
                "value": val,
                "tri": mode,
                "NRMSE": float(score),
                "count": len(e_list),
                "witness": witness,
            }
            rows.append(row)

        # rank by NRMSE (lower is better)
        rows.sort(key=lambda r: r["NRMSE"])

        # apply k logic: keep k best + k worst
        if k is None:
            return rows
        if 2 * k >= len(rows):
            return rows
        return rows[:k] + rows[-k:]

    if tri is None:
        return make_rows("best"), make_rows("avg"), make_rows("worst")

    return make_rows(tri)



def graph_classification(
    filenames,
    x_param: str,
    tri: str | None = None,
    *,
    metric: str = "NRMSE",
    labels=None,
    filters=None,
    restrict_param_name: str | None = None,
    families_only=None,
    figsize=(8, 4),
):
    if isinstance(filenames, str):
        filenames = [filenames]

    if labels is None:
        labels = [f"file{i+1}" for i in range(len(filenames))]
    if len(labels) != len(filenames):
        raise ValueError("labels must match number of filenames")

    filters = filters or {}

    fig, ax = plt.subplots(figsize=figsize)

    for file, lab in zip(filenames, labels):
        entries = load_entries(file)
        entries = _apply_filters(entries, filters)

        if families_only is not None:
            entries = [e for e in entries if e["family"] in set(families_only)]

        if restrict_param_name is not None:
            entries = [e for e in entries if e["param_name"] == restrict_param_name]

        if not entries:
            continue

        # group values
        groups = defaultdict(list)
        for e in entries:
            groups[e[x_param]].append(e[metric])

        # sort x values
        x_vals = list(groups.keys())
        try:
            x_vals_sorted = sorted(x_vals, key=float)
        except Exception:
            x_vals_sorted = sorted(x_vals, key=str)

        means, mins, maxs, stds = [], [], [], []
        for x in x_vals_sorted:
            vals = np.asarray(groups[x], dtype=float)
            means.append(np.mean(vals))
            mins.append(np.min(vals))
            maxs.append(np.max(vals))
            stds.append(np.std(vals))

        # ---- plotting (CRITICAL PART) ----

        # Plot mean curve FIRST to get its color
        (mean_line,) = ax.plot(
            x_vals_sorted,
            means,
            marker="o",
            label=lab,
        )
        color = mean_line.get_color()

        # std error bars in SAME color
        ax.errorbar(
            x_vals_sorted,
            means,
            yerr=stds,
            fmt="none",
            ecolor=color,
            capsize=3,
            alpha=0.8,
        )

        if tri is None:
            # min / max markers in SAME color
            ax.scatter(x_vals_sorted, mins, marker="x", color=color)
            ax.scatter(x_vals_sorted, maxs, marker="x", color=color)
        else:
            if tri == "best":
                ax.plot(x_vals_sorted, mins, marker="o", color=color)
            elif tri == "worst":
                ax.plot(x_vals_sorted, maxs, marker="o", color=color)
            elif tri == "avg":
                pass
            else:
                raise ValueError("tri must be 'best', 'avg', 'worst', or None")

    ax.set_xlabel(x_param)
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)

    # ---- legend OUTSIDE on the right ----
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 1])  # leave space for legend

    return fig, ax

import numpy as np
from collections import defaultdict

def _is_single_axis_meas(meas: str, axis: str) -> bool:
    """
    axis in {"Y","Z"}
    Accepts meas strings like:
      "S:Y", "S+P:Y"
      "S:Z", "S+P:Z"
    Rejects mixed axes like "S:XYZ", "S:XZ", "S+P:XZ"
    """
    meas = meas.strip()

    # Only accept known prefixes in your logs
    if not (meas.startswith("S:") or meas.startswith("S+P:")):
        return False

    rhs = meas.split(":", 1)[1].strip()  # e.g. "Y", "XYZ", "XZ"

    # single-axis means exactly "Y" or exactly "Z"
    return rhs == axis


def correlation_encoding_measurement(
    filenames,
    *,
    filters: dict | None = None,
    encoding_names=("Rz", "Ry"),  # purely labeling: Rz=True vs Rz=False
    return_raw=False,
):
    """
    Compute and rank the 4 averages:
      (Rz, Z-only), (Rz, Y-only), (Ry, Z-only), (Ry, Y-only)

    Interpretation:
      - Rz encoding <=> entry["Rz"] == True
      - Ry encoding <=> entry["Rz"] == False

    Measurement basis (single-axis only):
      - Z-only: meas is S:Z or S+P:Z
      - Y-only: meas is S:Y or S+P:Y

    Returns
    -------
    ranked : list[dict]
      sorted by mean NRMSE ascending
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    filters = filters or {}

    # Load and filter all entries
    all_entries = []
    for fn in filenames:
        es = load_entries(fn)          # uses your project loader
        es = _apply_filters(es, filters)
        all_entries.extend(es)

    if not all_entries:
        raise ValueError("No entries after filtering.")

    groups = defaultdict(list)

    for e in all_entries:
        enc = encoding_names[0] if e["Rz"] else encoding_names[1]  # Rz vs Ry label

        meas = e["meas"]
        if _is_single_axis_meas(meas, "Z"):
            meas_axis = "Z"
        elif _is_single_axis_meas(meas, "Y"):
            meas_axis = "Y"
        else:
            continue  # ignore non single-axis measurements

        groups[(enc, meas_axis)].append(float(e["NRMSE"]))

    # Ensure all four exist (even if empty)
    combos = [
        (encoding_names[0], "Z"),
        (encoding_names[0], "Y"),
        (encoding_names[1], "Z"),
        (encoding_names[1], "Y"),
    ]

    rows = []
    for key in combos:
        vals = np.asarray(groups.get(key, []), dtype=float)
        if vals.size == 0:
            rows.append({
                "encoding": key[0],
                "meas_axis": key[1],
                "count": 0,
                "mean_NRMSE": float("nan"),
                "std_NRMSE": float("nan"),
                "best_NRMSE": float("nan"),
                "worst_NRMSE": float("nan"),
            })
        else:
            rows.append({
                "encoding": key[0],
                "meas_axis": key[1],
                "count": int(vals.size),
                "mean_NRMSE": float(np.mean(vals)),
                "std_NRMSE": float(np.std(vals)),
                "best_NRMSE": float(np.min(vals)),
                "worst_NRMSE": float(np.max(vals)),
            })

    # rank: put NaNs at the end
    def sort_key(r):
        m = r["mean_NRMSE"]
        return (np.isnan(m), m)

    ranked = sorted(rows, key=sort_key)

    if return_raw:
        return ranked, groups

    return ranked

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_metric_vs_depth_G1G2G3(
    filenames,
    *,
    metric="NRMSE",              # "NRMSE" or "time"
    labels=None,
    filters=None,                # dict applied to ALL files
    figsize=(14, 4),
    show_minmax=False,           # optional: show min/max markers per depth
    show_std=False,              # optional: show std error bars around the mean/topk-mean
    avg_mode="mean",             # "mean" or "topk"
    top_k=10,                    # only used if avg_mode="topk"
):
    """
    Plot avg(metric) vs depth for families G1/G2/G3, side-by-side (3 panels).
    Each file keeps the same color across panels.

    avg_mode:
      - "mean": mean of all values at each depth
      - "topk": mean of the top_k best (lowest) values at each depth

    Returns
    -------
    fig, axes
    """
    if metric not in ("NRMSE", "time"):
        raise ValueError("metric must be 'NRMSE' or 'time'")
    if avg_mode not in ("mean", "topk"):
        raise ValueError("avg_mode must be 'mean' or 'topk'")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive int")

    if isinstance(filenames, str):
        filenames = [filenames]

    if labels is None:
        labels = [f"file{i+1}" for i in range(len(filenames))]
    if len(labels) != len(filenames):
        raise ValueError("labels must match number of filenames")

    filters = filters or {}

    families = ["G1", "G2", "G3"]
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    # Matplotlib default color cycle, one per file
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None) or [None]

    def agg(vals):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return np.nan
        if avg_mode == "mean":
            return float(np.mean(vals))
        # avg_mode == "topk": mean of k smallest (best) values
        k = min(top_k, vals.size)
        # partition is O(n) and avoids full sort
        best_k = np.partition(vals, k - 1)[:k]
        return float(np.mean(best_k))

    for idx, (file, lab) in enumerate(zip(filenames, labels)):
        entries = load_entries(file)
        entries = _apply_filters(entries, filters)

        # Keep only entries that actually correspond to a depth sweep for G1/G2/G3
        entries = [
            e for e in entries
            if e.get("family") in families and e.get("param_name") == "depth"
        ]
        if not entries:
            continue

        color = color_cycle[idx % len(color_cycle)]

        # by_fam[fam][depth] -> list of metric values
        by_fam = {fam: defaultdict(list) for fam in families}
        for e in entries:
            try:
                depth = int(e["param_val"])
            except Exception:
                continue
            by_fam[e["family"]][depth].append(float(e[metric]))

        # Plot each family in its panel
        for ax, fam in zip(axes, families):
            depths = sorted(by_fam[fam].keys())
            if not depths:
                continue

            y_center = [agg(by_fam[fam][d]) for d in depths]

            # main curve + markers
            ax.plot(depths, y_center, marker="o", label=lab, color=color)

            if show_std:
                # std computed over all vals (not just top-k subset)
                stds = [float(np.std(np.asarray(by_fam[fam][d], dtype=float))) for d in depths]
                ax.errorbar(depths, y_center, yerr=stds, fmt="none", capsize=3, ecolor=color, alpha=0.8)

            if show_minmax:
                mins = [float(np.min(np.asarray(by_fam[fam][d], dtype=float))) for d in depths]
                maxs = [float(np.max(np.asarray(by_fam[fam][d], dtype=float))) for d in depths]
                ax.scatter(depths, mins, marker="x", color=color, alpha=0.9)
                ax.scatter(depths, maxs, marker="x", color=color, alpha=0.9)

    # Cosmetics
    for ax, fam in zip(axes, families):
        ax.set_title(fam)
        ax.set_xlabel("depth")
        ax.grid(alpha=0.3)

    ylabel = metric if avg_mode == "mean" else f"{metric} (mean of best {top_k})"
    axes[0].set_ylabel(ylabel)

    # Legend outside (collect from all axes, avoid duplicates)
    handles, leg_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in leg_labels:
                handles.append(hh)
                leg_labels.append(ll)

    if handles:
        fig.tight_layout(rect=[0, 0, 0.86, 1])
    else:
        fig.tight_layout()

    return fig, axes

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_metric_vs_depth_G1G2G3_pooled(
    filenames,
    *,
    metric="NRMSE",              # "NRMSE" or "time"
    filters=None,
    figsize=(14, 4),
    show_minmax=False,
    show_std=False,
    avg_mode="mean",             # "mean" or "topk"
    top_k=10,
    curve_label=None,            # legend label
    title=None,                  # optional override
):
    """
    Plot ONE pooled curve (across all files) per family (G1/G2/G3), side-by-side.

    For each family and depth:
      pool all metric values across *all* files (after filters),
      aggregate by avg_mode:
        - "mean": mean of all pooled values
        - "topk": mean of the top_k smallest (best) pooled values

    Returns
    -------
    fig, axes
    """
    if metric not in ("NRMSE", "time"):
        raise ValueError("metric must be 'NRMSE' or 'time'")
    if avg_mode not in ("mean", "topk"):
        raise ValueError("avg_mode must be 'mean' or 'topk'")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive int")

    if isinstance(filenames, str):
        filenames = [filenames]

    filters = filters or {}
    families = ["G1", "G2", "G3"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    def agg(vals):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return np.nan
        if avg_mode == "mean":
            return float(np.mean(vals))
        k = min(top_k, vals.size)
        best_k = np.partition(vals, k - 1)[:k]
        return float(np.mean(best_k))

    # pooled[fam][depth] -> list of metric values pooled over all files
    pooled = {fam: defaultdict(list) for fam in families}

    # Collect all values across all files
    for file in filenames:
        entries = load_entries(file)
        entries = _apply_filters(entries, filters)

        # only depth sweeps for G1/G2/G3
        entries = [e for e in entries if e.get("family") in families and e.get("param_name") == "depth"]

        for e in entries:
            try:
                depth = int(e["param_val"])
            except Exception:
                continue
            pooled[e["family"]][depth].append(float(e[metric]))

    # Plot one curve per panel
    for ax, fam in zip(axes, families):
        depths = sorted(pooled[fam].keys())
        if not depths:
            ax.set_title(f"{fam} (no data)")
            ax.set_xlabel("depth")
            ax.grid(alpha=0.3)
            continue

        y_center = [agg(pooled[fam][d]) for d in depths]

        (line,) = ax.plot(
            depths,
            y_center,
            marker="o",
            label=(curve_label or "pooled"),
        )
        color = line.get_color()

        if show_std:
            stds = [float(np.std(np.asarray(pooled[fam][d], dtype=float))) for d in depths]
            ax.errorbar(depths, y_center, yerr=stds, fmt="none", capsize=3, ecolor=color, alpha=0.8)

        if show_minmax:
            mins = [float(np.min(np.asarray(pooled[fam][d], dtype=float))) for d in depths]
            maxs = [float(np.max(np.asarray(pooled[fam][d], dtype=float))) for d in depths]
            ax.scatter(depths, mins, marker="x", color=color, alpha=0.9)
            ax.scatter(depths, maxs, marker="x", color=color, alpha=0.9)

        ax.set_title(fam)
        ax.set_xlabel("depth")
        ax.grid(alpha=0.3)

    # Axis label
    ylabel = metric if avg_mode == "mean" else f"{metric} (mean of best {top_k})"
    axes[0].set_ylabel(ylabel)

    # --------- TITLE (THIS IS THE IMPORTANT BIT) ----------
    if title is None:
        metric_title = "NRMSE" if metric == "NRMSE" else "Time complexity"
        title = f"{metric_title} vs depth for G1, G2 and G3 reservoirs"
    fig.suptitle(title, fontsize=14)

    # Legend outside (single pooled curve label)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    # Leave room for both suptitle (top) and legend (right)
    if handles:
        #fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.tight_layout(rect=[0, 0, 0.86, 0.92])  # <--- top < 1 so suptitle is visible
    else:
        fig.tight_layout(rect=[0, 0, 1.00, 0.92])  # <--- top < 1 so suptitle is visible

    return fig, axes


def _apply_filters(entries: list[dict], filters: dict) -> list[dict]:
    """
    filters: dict(field -> value or list/tuple/set of values)
    Example: {"family": "MG", "n": 2, "Rz": False}
    """
    if not filters:
        return entries

    out = []
    for e in entries:
        ok = True
        for k, v in filters.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                if e.get(k) not in v:
                    ok = False
                    break
            else:
                if e.get(k) != v:
                    ok = False
                    break
        if ok:
            out.append(e)
    return out


def aggregate_by_key(entries: list[dict], group_key: str, metric: str = "NRMSE") -> dict:
    """
    Returns dict: group_value -> np.array(metric values)
    """
    groups = defaultdict(list)
    for e in entries:
        groups[e[group_key]].append(e[metric])
    return {k: np.asarray(v, dtype=float) for k, v in groups.items()}


def summary_stats(values: np.ndarray) -> dict:
    """
    values: 1D array
    """
    return dict(
        count=int(values.size),
        best=float(np.min(values)),
        worst=float(np.max(values)),
        avg=float(np.mean(values)),
        var=float(np.var(values)),
        std=float(np.std(values)),
    )

def table_classification(
    filename: str,
    param: str,
    tri: str | None = None,
    k: int | None = None,
    *,
    metric: str = "NRMSE",
    filters: dict | None = None,
) -> list | tuple[list, list, list]:
    """
    Return ranked parameter values by tri-aggregate of metric.

    If tri is None, returns (best_list, avg_list, worst_list).
    Each list contains param values ordered from best -> worst.

    k:
      - if None: return all
      - else: return at most 2k values (k best + k worst) if possible.
        If 2k >= total unique values: return all.
    """
    entries = load_entries(filename)
    entries = _apply_filters(entries, filters or {})

    if not entries:
        raise ValueError("No entries after filtering.")

    groups = aggregate_by_key(entries, param, metric=metric)

    def _rank(tri_mode: str) -> list:
        if tri_mode not in ("best", "avg", "worst"):
            raise ValueError("tri must be 'best', 'avg', 'worst' or None")

        scored = []
        for pval, vals in groups.items():
            if tri_mode == "best":
                score = float(np.min(vals))
            elif tri_mode == "worst":
                score = float(np.max(vals))
            else:
                score = float(np.mean(vals))
            scored.append((pval, score))

        scored.sort(key=lambda x: x[1])  # lower metric is better

        ordered_vals = [p for p, _ in scored]
        if k is None:
            return ordered_vals

        total = len(ordered_vals)
        if 2 * k >= total:
            return ordered_vals

        return ordered_vals[:k] + ordered_vals[-k:]

    if tri is None:
        return _rank("best"), _rank("avg"), _rank("worst")

    return _rank(tri)
