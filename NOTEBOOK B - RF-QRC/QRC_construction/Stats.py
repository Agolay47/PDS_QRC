import re
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import numpy as np

import numpy as np
from collections import defaultdict


def summarize_nrmse_by(
    filename,
    group_by,                 # e.g. "seed", "param_val", "family", "meas", "n", ...
    *,
    # filters (set None to ignore)
    family=None,
    param_name=None,          # e.g. "depth", "k", "ising"
    param_val=None,           # e.g. 3, 10, (10.0,1,0.1,1.0)  (string match if tuple)
    seed=None,
    n=None,
    extra=None,
    meas=None,
    Rz=None,
    sort_by="mean",           # "mean", "best", "worst", "var"
    ascending=True,
):
    """
    Parse normalized log file, filter entries, then group by *any* field and summarize NRMSE.

    group_by can be one of:
      "family", "param_name", "param_val", "seed", "n", "extra", "meas", "Rz"
    """
    allowed = {"family", "param_name", "param_val", "seed", "n", "extra", "meas", "Rz"}
    if group_by not in allowed:
        raise ValueError(f"group_by must be one of {sorted(allowed)}. Got {group_by}")

    entries = []
    with open(filename, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = _parse_normalized_log_line(ln)
            except Exception:
                continue

            # filters
            if family is not None and e["family"] != family:
                continue
            if param_name is not None and e["param_name"] != param_name:
                continue
            if seed is not None and e["seed"] != seed:
                continue
            if n is not None and e["n"] != n:
                continue
            if extra is not None and e["extra"] != extra:
                continue
            if meas is not None and e["meas"] != meas:
                continue
            if Rz is not None and e["Rz"] != bool(Rz):
                continue

            # param_val filter: match against parsed param_val OR raw string
            if param_val is not None:
                if e["param_val"] != param_val and e["param_val_raw"] != str(param_val):
                    continue

            entries.append(e)

    if not entries:
        raise ValueError("No entries matched your filters.")

    groups = defaultdict(list)
    for e in entries:
        key = e[group_by]
        groups[key].append(e["NRMSE"])

    rows = []
    for key, vals in groups.items():
        vals = np.asarray(vals, dtype=float)
        rows.append({
            group_by: key,
            "count": int(vals.size),
            "worst_NRMSE": float(np.max(vals)),
            "best_NRMSE": float(np.min(vals)),
            "mean_NRMSE": float(np.mean(vals)),
            "var_NRMSE": float(np.var(vals)),   # use ddof=1 if you want sample variance
        })

    key_map = {"mean": "mean_NRMSE", "best": "best_NRMSE", "worst": "worst_NRMSE", "var": "var_NRMSE"}
    if sort_by not in key_map:
        raise ValueError(f"sort_by must be one of {list(key_map.keys())}")

    rows.sort(key=lambda r: r[key_map[sort_by]], reverse=not ascending)
    return rows


def _parse_normalized_log_line(ln: str) -> dict:
    """
    Parse normalized log line of the form:
      MG depth=3 | seed=1 | n=2 | extra=0 | meas=S:XYZ | Rz=False | time=0.72s | NRMSE=0.890
    Returns a dict with keys:
      family, param_name, param_val(str), seed(int), n(int), extra(int), meas(str), Rz(bool), time(float), NRMSE(float)
    """
    parts = [p.strip() for p in ln.strip().split("|")]
    if len(parts) < 8:
        raise ValueError(f"Bad line (expected >=8 fields): {ln}")

    # First field: "MG depth=3"
    head = parts[0].split()
    if len(head) != 2:
        raise ValueError(f"Bad head '{parts[0]}' in line: {ln}")
    family = head[0].strip()
    param_name, param_val_raw = head[1].split("=", 1)
    param_name = param_name.strip()
    param_val_raw = param_val_raw.strip()

    # Other fields: key=value
    def get_kv(field):
        if "=" not in field:
            raise ValueError(f"Bad field '{field}' in line: {ln}")
        k, v = field.split("=", 1)
        return k.strip(), v.strip()

    kv = dict(get_kv(p) for p in parts[1:])

    # Cast types
    seed = int(kv["seed"])
    n = int(kv["n"])
    extra = int(kv["extra"])
    meas = kv["meas"]
    Rz = kv["Rz"].lower() == "true"

    # time like "0.72s"
    time_s = kv["time"].strip()
    if time_s.endswith("s"):
        time_s = time_s[:-1]
    time_val = float(time_s)

    nrmse = float(kv["NRMSE"])

    # Keep param_val both as string and numeric if possible
    try:
        param_val_num = int(param_val_raw)
    except Exception:
        try:
            param_val_num = float(param_val_raw)
        except Exception:
            param_val_num = None

    return {
        "family": family,
        "param_name": param_name,
        "param_val_raw": param_val_raw,
        "param_val": param_val_num if param_val_num is not None else param_val_raw,
        "seed": seed,
        "n": n,
        "extra": extra,
        "meas": meas,
        "Rz": Rz,
        "time": time_val,
        "NRMSE": nrmse,
        "raw_line": ln.strip(),
    }


def summarize_nrmse_by_param(
    filename,
    group_param_name=None,
    *,
    # filters (set to None to not filter on that field)
    family=None,
    seed=None,
    n=None,
    extra=None,
    meas=None,
    Rz=None,
    # how to order rows
    sort_by="mean",  # "mean" or "best" or "worst" or "var"
    ascending=True,
):
    """
    Read a normalized log file, filter entries, then group by a parameter value,
    and return a summary table.

    Parameters
    ----------
    group_param_name : str or None
        Which parameter to group by (e.g. "depth", "k", "ising").
        If None: inferred from the first matching line, and must be consistent.
    family, seed, n, extra, meas, Rz : optional filters
    sort_by : one of {"mean","best","worst","var"}
    ascending : bool

    Returns
    -------
    rows : list[dict]
        Each dict has:
          param_name, param_val, best, worst, mean, var, count
    """
    # Read and parse
    entries = []
    with open(filename, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = _parse_normalized_log_line(ln)
            except Exception:
                continue

            # Apply filters
            if family is not None and e["family"] != family:
                continue
            if seed is not None and e["seed"] != seed:
                continue
            if n is not None and e["n"] != n:
                continue
            if extra is not None and e["extra"] != extra:
                continue
            if meas is not None and e["meas"] != meas:
                continue
            if Rz is not None and e["Rz"] != bool(Rz):
                continue

            entries.append(e)

    if not entries:
        raise ValueError("No entries matched your filters.")

    # Infer param name if not given
    if group_param_name is None:
        group_param_name = entries[0]["param_name"]

    # Keep only entries with that param_name (important if file contains multiple families/params)
    entries = [e for e in entries if e["param_name"] == group_param_name]
    if not entries:
        raise ValueError(f"No entries found with param_name='{group_param_name}' after filtering.")

    # Group by param_val
    groups = defaultdict(list)
    for e in entries:
        groups[e["param_val"]].append(e["NRMSE"])

    rows = []
    for pval, vals in groups.items():
        vals = np.asarray(vals, dtype=float)
        rows.append({
            "param_name": group_param_name,
            "param_val": pval,
            "worst_NRMSE": float(np.max(vals)),
            "best_NRMSE": float(np.min(vals)),
            "mean_NRMSE": float(np.mean(vals)),
            "var_NRMSE": float(np.var(vals)),  # population variance
            "count": int(vals.size),
        })

    key_map = {
        "mean": "mean_NRMSE",
        "best": "best_NRMSE",
        "worst": "worst_NRMSE",
        "var": "var_NRMSE",
    }
    if sort_by not in key_map:
        raise ValueError(f"sort_by must be one of {list(key_map.keys())}")

    rows.sort(key=lambda r: r[key_map[sort_by]], reverse=not ascending)

    return rows


def parse_results_file(file_path: str) -> pd.DataFrame:
    """
    Parses lines like (phase optional):

      G3 depth=10 | seed=2 | n=3 | extra=0 (N=3) | meas=S+P+T:Y | mQ=3 | ring=False | phase=False | time=1.16s | NRMSE=1.515
      G3 depth=10 | seed=2 | n=3 | extra=0 (N=3) | meas=S+P+T:Y | mQ=3 | ring=False | time=1.16s | NRMSE=1.515

    Conventions:
      n     -> sliding_window_size
      extra -> nb_extra_qubits
      N     -> nb_qubits
      mQ    -> number of measured qubits (len(x_qubits_measured))
      ring  -> ring_connectivity
      phase -> is_phase_encoded (optional in log line)
    """
    with open(file_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    rows = []

    pat = re.compile(
        r"""
        ^\s*
        (?P<family>[A-Za-z0-9]+)
        \s+
        (?P<param_name>[A-Za-z_]+)=(?P<param_val>[^|]+?)
        \s*\|\s*
        seed=(?P<seed>\d+)
        \s*\|\s*
        n=(?P<sw>\d+)
        \s*\|\s*
        extra=(?P<extra>\d+)\s*\(N=(?P<N>\d+)\)
        \s*\|\s*
        meas=(?P<meas>[^|]+?)
        \s*\|\s*
        mQ=(?P<mQ>\d+)
        \s*\|\s*
        ring=(?P<ring>True|False)
        (?:\s*\|\s*phase=(?P<phase>True|False))?
        \s*\|\s*
        time=(?P<time>[\d.]+)s
        \s*\|\s*
        NRMSE=(?P<nrmse>[-+]?[\d.]+(?:[eE][-+]?\d+)?)
        \s*$
        """,
        re.VERBOSE,
    )

    def cast_param(v: str):
        v = v.strip()
        # try int
        try:
            return int(v)
        except ValueError:
            pass
        # try float
        try:
            return float(v)
        except ValueError:
            pass
        # keep as string (e.g., "(10.0, 2, 0.1, 1.0)")
        return v

    def cast_bool(v):
        if v is None:
            return None
        return True if v == "True" else False

    for ln in lines:
        m = pat.match(ln)
        if not m:
            raise ValueError(f"Could not parse line:\n{ln}")

        d = m.groupdict()

        rows.append({
            "family": d["family"],
            "param_name": d["param_name"],
            "param_val": cast_param(d["param_val"]),
            "seed": int(d["seed"]),

            # KEY SEMANTICS
            "sliding_window_size": int(d["sw"]),
            "nb_extra_qubits": int(d["extra"]),
            "nb_qubits": int(d["N"]),

            "meas_name": d["meas"].strip(),
            "mQ": int(d["mQ"]),
            "ring": cast_bool(d["ring"]),
            "phase": cast_bool(d["phase"]),   # None if not logged

            "time_s": float(d["time"]),
            "NRMSE": float(d["nrmse"]),
            "raw_line": ln,
        })

    return pd.DataFrame(rows)


def print_topk_from_df(
    df: pd.DataFrame,
    k: int = 3,
    best: bool = True,
    title: str = ""
):
    if df.empty:
        print("Empty DataFrame.")
        return

    title = title or (f"Top {k} BEST configs" if best else f"Top {k} WORST configs")
    print(f"\n{title}")
    print("-" * len(title))

    sub = df.sort_values("NRMSE", ascending=best).head(k)

    # some files may not have 'phase' column populated (None)
    has_phase = "phase" in df.columns

    for i, r in enumerate(sub.itertuples(index=False), start=1):
        phase_str = f"\n  Phase             : {r.phase}" if has_phase else ""
        print(f"\n#{i}")
        print(
            f"  Family            : {r.family}\n"
            f"  Param             : {r.param_name} = {r.param_val}\n"
            f"  Seed              : {r.seed}\n"
            f"  Sliding window n  : {r.sliding_window_size}\n"
            f"  Extra qubits      : {r.nb_extra_qubits}\n"
            f"  Total qubits N    : {r.nb_qubits}\n"
            f"  Measurement       : {r.meas_name}\n"
            f"  Measured qubits   : {r.mQ}\n"
            f"  Ring connectivity : {r.ring}"
            f"{phase_str}\n"
            f"  NRMSE             : {r.NRMSE:.4f}\n"
            f"  Runtime           : {r.time_s:.2f} s"
        )


def summarize_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Returns a summary DataFrame with:
      count, best_nrmse, avg_nrmse, std_nrmse, median_nrmse, best_config_line

    Works with any grouping column present in df, e.g.:
      'family', 'meas_name', 'sliding_window_size', 'nb_extra_qubits',
      'nb_qubits', 'mQ', 'ring', 'phase', 'seed', 'param_name', 'param_val'
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in DataFrame. Available: {list(df.columns)}")

    dfg = df.dropna(subset=[col]).copy()
    if dfg.empty:
        return pd.DataFrame()

    grp = dfg.groupby(col, dropna=True)

    idx_best = grp["NRMSE"].idxmin()
    best_rows = dfg.loc[idx_best, [col, "NRMSE", "raw_line"]].set_index(col)

    out = grp["NRMSE"].agg(
        count="count",
        best_nrmse="min",
        avg_nrmse="mean",
        std_nrmse="std",
        median_nrmse="median",
    )

    out = out.join(best_rows.rename(columns={"NRMSE": "best_nrmse_check"}), how="left")
    out = out.drop(columns=["best_nrmse_check"])
    out = out.rename(columns={"raw_line": "best_config_line"})
    out = out.sort_values("best_nrmse", ascending=True)

    return out
