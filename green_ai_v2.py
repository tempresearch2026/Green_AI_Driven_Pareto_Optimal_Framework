# -*- coding: utf-8 -*-
"""
Green AI-oriented HPO benchmark for battery SOH estimation  (v2, single overnight run)
=====================================================================================

What this script does (in order, fully automatic):
  1. PREFLIGHT  : loads every dataset, validates columns/units, builds features,
                  prints a summary. Stops within seconds if anything is wrong.
  2. SMOKE TEST : runs the WHOLE pipeline end-to-end with a tiny budget (2 seeds,
                  14 evaluations) in OUTPUT_DIR/_smoke, including the statistics.
                  Stops if anything fails, and prints an estimated runtime.
  3. FULL RUN   : 2 protocols x cells x 3 models x 7 strategies x up to 20 seeds,
                  100 objective evaluations per HPO run, parallel workers.
                  Every finished task is saved immediately (resume-safe).
                  Seeds are processed in order; at the deadline (MAX_HOURS) no new
                  seed is started, so whatever finished is a complete design.
  4. ANALYSIS   : all tables/figures for the paper are generated ONLY from the saved
                  raw results, with automatic sanity checks.

Protocols
  CHRONO : within-cell chronological split (first 85% -> train, last 15% -> test).
           HPO objective = mean RMSE of 2 forward-chaining folds inside the first 85%
           (train 0-55% / val 55-70%, train 0-70% / val 70-85%).
  LOBO   : leave-one-battery-out across the NASA cells (train on 2 cells, test on the
           ENTIRE life of the held-out cell). HPO objective = mean RMSE of inner
           leave-one-cell-out folds over the 2 training cells.

Strategies: Default (no HPO, library defaults), RS, BO (GP-EI), PSO, GA, WOA, GWO.
All HPO strategies search the SAME unit hypercube [0,1]^d (identical encoding,
identical decoding, identical log-scaling) with exactly N_EVALS evaluations.

Usage
  python green_ai_v2.py                  # preflight + smoke + full run + analysis
  python green_ai_v2.py --aggregate-only # re-run only the analysis on saved results
  python green_ai_v2.py --skip-smoke     # (resume) skip the smoke test
  python green_ai_v2.py --workers 4 --hours 9
Re-running the same command resumes: finished tasks are never recomputed.
"""

import os

# Single-threaded numerical libraries: parallelism is done at task level, and
# CPU-time measurements stay meaningful.
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import sys
import json
import time
import math
import pickle
import hashlib
import argparse
import platform
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# =============================================================================
# 0. CONFIGURATION  (edit here only)
# =============================================================================
NASA_DIR = r"C:\Users\LENOVO\Desktop\Transfer Learning Project\Veriler\cleaned_dataset\organized_dataset\25 degree"
LFP_PATH = r"C:\Users\LENOVO\Desktop\LiFePO4 Veri Setleri\LiFePO4_CH17\duzenlenmis_2018-04-12_batch8_CH17.csv"

DATASETS = {
    "B0005": {"path": os.path.join(NASA_DIR, "B0005_discharge_soh_standardized.csv"),
              "group": "NASA", "chemistry": "Li-ion 18650 (NASA PCoE)", "required": True},
    "B0006": {"path": os.path.join(NASA_DIR, "B0006_discharge_soh_standardized.csv"),
              "group": "NASA", "chemistry": "Li-ion 18650 (NASA PCoE)", "required": True},
    "B0018": {"path": os.path.join(NASA_DIR, "B0018_discharge_soh_standardized.csv"),
              "group": "NASA", "chemistry": "Li-ion 18650 (NASA PCoE)", "required": True},
    "LFP_CH17": {"path": LFP_PATH,
                 "group": "LFP", "chemistry": "LiFePO4/graphite 18650 (2018-04-12 batch, CH17)",
                 "required": True},
}
LOBO_GROUP = "NASA"            # cross-cell protocol is run inside this group (needs >= 3 cells)

OUTPUT_DIR = os.path.join(NASA_DIR, "green_ai_v2_runs")

SEEDS = [11, 23, 37, 41, 53, 67, 79, 83, 97, 101,
         113, 127, 139, 149, 163, 173, 181, 193, 211, 223]
N_EVALS = 100                  # objective evaluations per HPO run (all strategies)
POP_SIZE = 10                  # population size of PSO / GA / WOA / GWO
BO_N_INIT = 10                 # initial Latin-hypercube points of BO
N_WORKERS = 6                  # parallel worker processes (i5-12450HX: 8 cores / 12 threads)
MAX_HOURS = 10.0               # no new seed is started after this many hours of full run
MIN_SEEDS_FOR_ANALYSIS = 5

SMOKE_SEEDS = [11, 23]
SMOKE_N_EVALS = 14

STRATEGIES = ["Default", "RS", "BO", "PSO", "GA", "WOA", "GWO"]
HPO_STRATEGIES = [s for s in STRATEGIES if s != "Default"]
MODELS = ["RandomForest", "XGBoost", "LightGBM"]

FAIL_PENALTY = 10.0            # objective value for a failed fit (SOH RMSE is ~0.01-0.1)

# Search spaces. kind: int | float | logfloat. Library defaults lie inside each space,
# so every optimizer can in principle recover the default configuration.
SEARCH_SPACES = {
    "RandomForest": [
        ("n_estimators", 50, 500, "int"),
        ("max_depth", 2, 30, "int"),
        ("min_samples_split", 2, 20, "int"),
        ("min_samples_leaf", 1, 10, "int"),
        ("max_features", 0.2, 1.0, "float"),
    ],
    "XGBoost": [
        ("n_estimators", 50, 500, "int"),
        ("max_depth", 2, 10, "int"),
        ("learning_rate", 0.01, 0.3, "logfloat"),
        ("subsample", 0.5, 1.0, "float"),
        ("colsample_bytree", 0.5, 1.0, "float"),
        ("min_child_weight", 0.5, 20.0, "logfloat"),
        ("gamma", 1e-8, 1e-2, "logfloat"),        # SOH is a fraction: loss reductions are tiny
        ("reg_alpha", 1e-8, 1.0, "logfloat"),
        ("reg_lambda", 1e-3, 10.0, "logfloat"),
    ],
    "LightGBM": [
        ("n_estimators", 50, 500, "int"),
        ("learning_rate", 0.01, 0.3, "logfloat"),
        ("num_leaves", 4, 64, "int"),
        ("min_child_samples", 2, 30, "int"),
        ("subsample", 0.5, 1.0, "float"),
        ("colsample_bytree", 0.5, 1.0, "float"),
        ("reg_alpha", 1e-8, 1.0, "logfloat"),
        ("reg_lambda", 1e-8, 10.0, "logfloat"),
    ],
}

FEATURES = [
    # current-cycle statistics (15)
    "terminal_voltage_mean", "terminal_voltage_std", "terminal_voltage_min", "terminal_voltage_max",
    "terminal_current_mean", "terminal_current_std", "terminal_current_min", "terminal_current_max",
    "temperature_mean", "temperature_std", "temperature_min", "temperature_max",
    "voltage_drop", "temp_rise", "current_stability",
    # rolling window over the 3 PREVIOUS cycles (6)
    "terminal_voltage_mean_roll3_mean", "terminal_voltage_mean_roll3_std",
    "terminal_voltage_std_roll3_mean", "terminal_voltage_std_roll3_std",
    "temperature_mean_roll3_mean", "temperature_mean_roll3_std",
    # drift relative to the first cycle of the same cell (2)
    "voltage_shift_from_first", "temperature_shift_from_first",
]

COLUMN_ALIASES = {
    "cycle": ["cycle", "cycle_number", "cycle_index", "cycle_no", "cycle_num", "cyclenumber", "cycle_id"],
    "terminal_voltage": ["terminal_voltage", "voltage", "voltage_measured", "v", "voltage_v", "voltage(v)", "volt"],
    "terminal_current": ["terminal_current", "current", "current_measured", "i", "current_a", "current(a)"],
    "temperature": ["temperature", "temperature_measured", "temp", "t", "temperature_c", "temperature(c)"],
    "SOH": ["soh", "state_of_health"],
}

SOH_OUTLIER_WINDOW = 5         # centred rolling median window (cycles) for target glitch removal
SOH_OUTLIER_THRESHOLD = 0.05   # absolute SOH (fraction) deviation from the rolling median

# =============================================================================
# 1. LOGGING
# =============================================================================
_LOG_FILE = None


def log(msg=""):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    if _LOG_FILE is not None:
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def set_log_file(path):
    global _LOG_FILE
    _LOG_FILE = path


def atomic_write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def clean_float(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None


# =============================================================================
# 2. DATA LOADING AND FEATURE CONSTRUCTION
# =============================================================================
def _norm_col(c):
    return str(c).strip().lower().replace(" ", "_")


def resolve_columns(columns):
    normed = {_norm_col(c): c for c in columns}
    mapping, missing = {}, []
    for target, aliases in COLUMN_ALIASES.items():
        found = None
        for a in aliases:
            if a in normed:
                found = normed[a]
                break
        if found is None:
            missing.append(target)
        else:
            mapping[found] = target
    return mapping, missing


def load_raw_cell(name, spec):
    path = spec["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name}: file not found: {path}")
    header = pd.read_csv(path, nrows=0)
    mapping, missing = resolve_columns(header.columns)
    if missing:
        raise ValueError(f"{name}: required columns not found: {missing}. "
                         f"Columns in file: {list(header.columns)}")
    raw = pd.read_csv(path, usecols=list(mapping.keys()))
    raw = raw.rename(columns=mapping)
    for c in ["cycle", "terminal_voltage", "terminal_current", "temperature", "SOH"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    n0 = len(raw)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    info = {"raw_rows": int(n0), "rows_after_nan_drop": int(len(raw))}

    soh_med = float(raw["SOH"].median())
    if soh_med > 1.5:                     # stored as percentage -> convert to fraction
        raw["SOH"] = raw["SOH"] / 100.0
        info["soh_unit_conversion"] = "percent -> fraction"
    else:
        info["soh_unit_conversion"] = "none (already fraction)"
    smin, smax = float(raw["SOH"].min()), float(raw["SOH"].max())
    if smin < 0.0 or smax > 1.5:
        raise ValueError(f"{name}: SOH range [{smin:.4f}, {smax:.4f}] is not a plausible fraction. "
                         f"Is the SOH column standardized (z-score)?")
    vmed = float(raw["terminal_voltage"].median())
    info["voltage_median"] = vmed
    info["voltage_looks_physical"] = bool(1.5 <= vmed <= 4.6)
    return raw, info


def build_cell_features(raw):
    g = raw.groupby("cycle", sort=True)
    df = g.agg(
        terminal_voltage_mean=("terminal_voltage", "mean"),
        terminal_voltage_std=("terminal_voltage", "std"),
        terminal_voltage_min=("terminal_voltage", "min"),
        terminal_voltage_max=("terminal_voltage", "max"),
        terminal_current_mean=("terminal_current", "mean"),
        terminal_current_std=("terminal_current", "std"),
        terminal_current_min=("terminal_current", "min"),
        terminal_current_max=("terminal_current", "max"),
        temperature_mean=("temperature", "mean"),
        temperature_std=("temperature", "std"),
        temperature_min=("temperature", "min"),
        temperature_max=("temperature", "max"),
        SOH=("SOH", "first"),
        n_samples=("terminal_voltage", "size"),
    ).reset_index()
    n_cycles_raw = len(df)
    df = df[df["n_samples"] >= 5].reset_index(drop=True)
    n_short = n_cycles_raw - len(df)

    # target glitch removal (measurement artefacts), identical rule for every cell
    med = df["SOH"].rolling(SOH_OUTLIER_WINDOW, center=True, min_periods=1).median()
    keep = (df["SOH"] - med).abs() <= SOH_OUTLIER_THRESHOLD
    n_outliers = int((~keep).sum())
    df = df[keep].reset_index(drop=True)

    df["voltage_drop"] = df["terminal_voltage_max"] - df["terminal_voltage_min"]
    df["temp_rise"] = df["temperature_max"] - df["temperature_min"]
    df["current_stability"] = df["terminal_current_std"] / (df["terminal_current_mean"].abs() + 1e-8)

    for col in ["terminal_voltage_mean", "terminal_voltage_std", "temperature_mean"]:
        prev = df[col].shift(1)                      # strictly previous cycles only
        rm = prev.rolling(3, min_periods=1).mean()
        rs = prev.rolling(3, min_periods=1).std()
        df[f"{col}_roll3_mean"] = rm.fillna(df[col])
        df[f"{col}_roll3_std"] = rs.fillna(0.0)

    df["voltage_shift_from_first"] = df["terminal_voltage_mean"].iloc[0] - df["terminal_voltage_mean"]
    df["temperature_shift_from_first"] = df["temperature_mean"] - df["temperature_mean"].iloc[0]

    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna(subset=FEATURES + ["SOH"]).reset_index(drop=True)
    df["cycle_pos"] = np.arange(len(df))
    stats = {"cycles_raw": int(n_cycles_raw), "cycles_dropped_short": int(n_short),
             "cycles_dropped_soh_glitch": n_outliers, "cycles_dropped_nan": int(n_before - len(df)),
             "cycles_used": int(len(df))}
    return df, stats


def preflight(datasets):
    cells, report = {}, {}
    for name, spec in datasets.items():
        try:
            raw, info = load_raw_cell(name, spec)
            feat, stats = build_cell_features(raw)
            if len(feat) < 30:
                raise ValueError(f"{name}: only {len(feat)} usable cycles (>= 30 required)")
            info.update(stats)
            info["soh_first"] = float(feat["SOH"].iloc[0])
            info["soh_last"] = float(feat["SOH"].iloc[-1])
            info["soh_min"] = float(feat["SOH"].min())
            info["soh_max"] = float(feat["SOH"].max())
            info["group"] = spec["group"]
            info["chemistry"] = spec["chemistry"]
            cells[name] = feat
            report[name] = info
            log(f"  [OK] {name:9s} ({spec['group']}): cycles used={info['cycles_used']}, "
                f"SOH {info['soh_first']:.3f} -> {info['soh_last']:.3f}, "
                f"dropped short/glitch/nan={stats['cycles_dropped_short']}/"
                f"{stats['cycles_dropped_soh_glitch']}/{stats['cycles_dropped_nan']}, "
                f"SOH unit: {info['soh_unit_conversion']}")
            if not info["voltage_looks_physical"]:
                log(f"  [WARNING] {name}: median voltage = {info['voltage_median']:.3f} "
                    f"does not look like volts; inputs may be standardized.")
        except Exception as e:
            if spec.get("required", True):
                raise
            log(f"  [SKIPPED] {name}: {e}")
    return cells, report


# =============================================================================
# 3. EVALUATION UNITS (protocols)
# =============================================================================
def build_units(cells, datasets):
    units = []
    for name, df in cells.items():
        n = len(df)
        a, b, c = int(round(0.55 * n)), int(round(0.70 * n)), int(round(0.85 * n))
        units.append({
            "protocol": "CHRONO", "unit": name, "group": datasets[name]["group"],
            "X": df[FEATURES].to_numpy(dtype=float), "y": df["SOH"].to_numpy(dtype=float),
            "folds": [(np.arange(0, a), np.arange(a, b)), (np.arange(0, b), np.arange(b, c))],
            "train_idx": np.arange(0, c), "test_idx": np.arange(c, n),
            "row_cell": np.array([name] * n), "row_pos": df["cycle_pos"].to_numpy(),
        })
    lobo_cells = [n for n in cells if datasets[n]["group"] == LOBO_GROUP]
    if len(lobo_cells) >= 3:
        for held in lobo_cells:
            order = [c for c in lobo_cells if c != held] + [held]
            Xs, ys, rc, rp, idx, start = [], [], [], [], {}, 0
            for cname in order:
                df = cells[cname]
                Xs.append(df[FEATURES].to_numpy(dtype=float))
                ys.append(df["SOH"].to_numpy(dtype=float))
                rc.append(np.array([cname] * len(df)))
                rp.append(df["cycle_pos"].to_numpy())
                idx[cname] = np.arange(start, start + len(df))
                start += len(df)
            tr_cells = order[:-1]
            folds = []
            for j in tr_cells:
                others = np.concatenate([idx[c] for c in tr_cells if c != j])
                folds.append((others, idx[j]))
            units.append({
                "protocol": "LOBO", "unit": f"test_{held}", "group": LOBO_GROUP,
                "X": np.vstack(Xs), "y": np.concatenate(ys), "folds": folds,
                "train_idx": np.concatenate([idx[c] for c in tr_cells]), "test_idx": idx[held],
                "row_cell": np.concatenate(rc), "row_pos": np.concatenate(rp),
            })
    return units


def units_manifest(units):
    return [{"protocol": u["protocol"], "unit": u["unit"], "group": u["group"],
             "n_train": int(len(u["train_idx"])), "n_test": int(len(u["test_idx"])),
             "folds": [[int(len(tr)), int(len(va))] for tr, va in u["folds"]]} for u in units]


# =============================================================================
# 4. MODELS, SEARCH SPACE DECODING, METRICS
# =============================================================================
def make_model(model_name, seed, params=None):
    params = dict(params or {})
    if model_name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=seed, n_jobs=1, **params)
    if model_name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=1, verbosity=0, **params)
    if model_name == "LightGBM":
        from lightgbm import LGBMRegressor
        if "subsample" in params:
            params["subsample_freq"] = 1      # otherwise LightGBM ignores subsample
        return LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1, **params)
    raise ValueError(model_name)


def decode(space, u):
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    out = {}
    for (name, lo, hi, kind), ui in zip(space, u):
        if kind == "logfloat":
            v = math.exp(math.log(lo) + ui * (math.log(hi) - math.log(lo)))
            out[name] = float(min(max(v, lo), hi))
        elif kind == "int":
            v = lo + ui * (hi - lo)
            out[name] = int(min(max(int(round(v)), lo), hi))
        else:
            v = lo + ui * (hi - lo)
            out[name] = float(min(max(v, lo), hi))
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def regression_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "test_rmse": float(np.sqrt(np.mean(err ** 2))),
        "test_mae": float(np.mean(np.abs(err))),
        "test_mape_pct": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-8)) * 100.0),
        "test_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "test_max_abs_err": float(np.max(np.abs(err))),
        "test_bias": float(np.mean(err)),
    }


def count_nodes(model, model_name):
    try:
        if model_name == "RandomForest":
            return int(sum(e.tree_.node_count for e in model.estimators_))
        if model_name == "XGBoost":
            dump = model.get_booster().get_dump()
            return int(sum(len([ln for ln in t.split("\n") if ln.strip()]) for t in dump))
        if model_name == "LightGBM":
            info = model.booster_.dump_model()["tree_info"]
            return int(sum(2 * t["num_leaves"] - 1 for t in info))
    except Exception:
        return None
    return None


# =============================================================================
# 5. OBJECTIVE WITH EXACT BUDGET
# =============================================================================
class BudgetExhausted(Exception):
    pass


class Evaluator:
    def __init__(self, model_name, unit, n_evals, model_seed):
        self.model_name = model_name
        self.space = SEARCH_SPACES[model_name]
        self.d = len(self.space)
        self.unit = unit
        self.n_evals = n_evals
        self.model_seed = model_seed
        self.n = 0
        self.n_fits = 0
        self.n_failed = 0
        self.best_f = float("inf")
        self.best_u = None
        self.best_params = None
        self.history = []          # best-so-far objective after each evaluation

    def objective_params(self, params):
        X, y = self.unit["X"], self.unit["y"]
        scores = []
        for tr, va in self.unit["folds"]:
            m = make_model(self.model_name, self.model_seed, params)
            m.fit(X[tr], y[tr])
            self.n_fits += 1
            scores.append(rmse(y[va], m.predict(X[va])))
        f = float(np.mean(scores))
        return f if math.isfinite(f) else FAIL_PENALTY

    def __call__(self, u):
        if self.n >= self.n_evals:
            raise BudgetExhausted()
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        params = decode(self.space, u)
        try:
            f = self.objective_params(params)
        except Exception:
            self.n_failed += 1
            f = FAIL_PENALTY
        self.n += 1
        if f < self.best_f:
            self.best_f, self.best_u, self.best_params = f, u.copy(), params
        self.history.append(self.best_f)
        return f

    @property
    def frac(self):
        return min(1.0, self.n / float(self.n_evals))


# =============================================================================
# 6. OPTIMIZERS (all minimise on [0,1]^d and run until the budget is exhausted)
# =============================================================================
def opt_rs(ev, rng):
    while True:
        ev(rng.random(ev.d))


def opt_pso(ev, rng, n=POP_SIZE, c1=1.5, c2=1.5, vmax=0.2):
    d = ev.d
    pos = rng.random((n, d))
    vel = rng.uniform(-vmax / 2, vmax / 2, (n, d))
    pbest, pcost = pos.copy(), np.full(n, np.inf)
    g, gc = pos[0].copy(), np.inf
    for i in range(n):
        c = ev(pos[i])
        pcost[i] = c
        if c < gc:
            g, gc = pos[i].copy(), c
    while True:
        for i in range(n):
            w = 0.9 - 0.5 * ev.frac                       # linearly decreasing inertia 0.9 -> 0.4
            r1, r2 = rng.random(d), rng.random(d)
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) + c2 * r2 * (g - pos[i])
            vel[i] = np.clip(vel[i], -vmax, vmax)
            pos[i] = np.clip(pos[i] + vel[i], 0.0, 1.0)
            c = ev(pos[i])
            if c < pcost[i]:
                pbest[i], pcost[i] = pos[i].copy(), c
                if c < gc:
                    g, gc = pos[i].copy(), c


def _sbx(p1, p2, rng, eta=15.0):
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u) + 1e-12)) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def _poly_mut(x, rng, eta=20.0):
    x = x.copy()
    pm = 1.0 / len(x)
    for j in range(len(x)):
        if rng.random() < pm:
            u = rng.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            x[j] = x[j] + delta
    return np.clip(x, 0, 1)


def opt_ga(ev, rng, n=POP_SIZE, pc=0.9):
    d = ev.d
    pop = rng.random((n, d))
    fit = np.full(n, np.inf)
    for i in range(n):
        fit[i] = ev(pop[i])

    def tournament():
        idx = rng.choice(n, size=3, replace=False)
        return pop[idx[np.argmin(fit[idx])]]

    while True:
        e = int(np.argmin(fit))
        elite, elite_f = pop[e].copy(), fit[e]
        children = []
        while len(children) < n - 1:
            p1, p2 = tournament(), tournament()
            if rng.random() < pc:
                c1, c2 = _sbx(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            children.append(_poly_mut(c1, rng))
            children.append(_poly_mut(c2, rng))
        children = children[:n - 1]
        child_f = []
        for ch in children:
            child_f.append(ev(ch))
        pop = np.vstack([elite[None, :]] + [c[None, :] for c in children])
        fit = np.array([elite_f] + child_f)


def opt_woa(ev, rng, n=POP_SIZE, b=1.0):
    d = ev.d
    pos = rng.random((n, d))
    fit = np.full(n, np.inf)
    best, best_f = pos[0].copy(), np.inf
    for i in range(n):
        fit[i] = ev(pos[i])
        if fit[i] < best_f:
            best, best_f = pos[i].copy(), fit[i]
    while True:
        for i in range(n):
            a = 2.0 * (1.0 - ev.frac)
            r1, r2 = rng.random(), rng.random()
            A, C = 2 * a * r1 - a, 2 * r2
            p, l = rng.random(), rng.uniform(-1, 1)
            if p < 0.5:
                if abs(A) < 1:
                    new = best - A * np.abs(C * best - pos[i])
                else:
                    j = int(rng.integers(n))
                    new = pos[j] - A * np.abs(C * pos[j] - pos[i])
            else:
                new = np.abs(best - pos[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + best
            pos[i] = np.clip(new, 0, 1)
            fit[i] = ev(pos[i])
            if fit[i] < best_f:
                best, best_f = pos[i].copy(), fit[i]


def opt_gwo(ev, rng, n=POP_SIZE):
    d = ev.d
    pos = rng.random((n, d))
    leaders = []                                   # best-so-far archive of (f, x), size 3

    def update_leaders(f, x):
        leaders.append((f, x.copy()))
        leaders.sort(key=lambda t: t[0])
        del leaders[3:]

    for i in range(n):
        update_leaders(ev(pos[i]), pos[i])
    while True:
        for i in range(n):
            a = 2.0 * (1.0 - ev.frac)
            acc = np.zeros(d)
            for _, L in leaders:
                r1, r2 = rng.random(d), rng.random(d)
                A, C = 2 * a * r1 - a, 2 * r2
                acc += L - A * np.abs(C * L - pos[i])
            pos[i] = np.clip(acc / len(leaders), 0, 1)
            update_leaders(ev(pos[i]), pos[i])


def opt_bo(ev, rng, n_init=BO_N_INIT, n_cand=2000, n_local=500, xi=0.01):
    """Gaussian-process Bayesian optimisation with Expected Improvement (scikit-learn GP,
    Matern-5/2 ARD kernel, log-transformed objective, LHS initial design)."""
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    d = ev.d
    n_init = int(min(n_init, max(2, ev.n_evals // 2)))
    try:
        from scipy.stats import qmc
        X0 = qmc.LatinHypercube(d=d, seed=int(rng.integers(2 ** 31 - 1))).random(n_init)
    except Exception:
        X0 = rng.random((n_init, d))
    Xs, Ys = [], []
    for x in X0:
        Ys.append(ev(x))
        Xs.append(x)
    while True:
        Xa = np.array(Xs)
        Ya = np.log(np.maximum(np.array(Ys), 1e-12))
        x_next = None
        try:
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * Matern(length_scale=np.full(d, 0.5), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                      + WhiteKernel(1e-4, (1e-8, 1e-1)))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
                                          random_state=int(rng.integers(2 ** 31 - 1)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(Xa, Ya)
                best_x = Xa[int(np.argmin(Ya))]
                cand = np.vstack([rng.random((n_cand, d)),
                                  np.clip(best_x + rng.normal(0, 0.05, (n_local, d)), 0, 1)])
                mu, sd = gp.predict(cand, return_std=True)
            sd = np.maximum(sd, 1e-12)
            imp = Ya.min() - mu - xi
            z = imp / sd
            ei = imp * norm.cdf(z) + sd * norm.pdf(z)
            x_next = cand[int(np.argmax(ei))]
        except Exception:
            x_next = rng.random(d)
        Ys.append(ev(x_next))
        Xs.append(x_next)


OPTIMIZERS = {"RS": opt_rs, "BO": opt_bo, "PSO": opt_pso, "GA": opt_ga, "WOA": opt_woa, "GWO": opt_gwo}


# =============================================================================
# 7. ONE TASK  = (protocol, unit, model, strategy, seed)
# =============================================================================
def config_seed(seed, protocol, unit, model, strategy):
    txt = f"{seed}|{protocol}|{unit}|{model}|{strategy}"
    return int(hashlib.sha256(txt.encode()).hexdigest()[:8], 16)


def task_path(out_dir, protocol, unit, model, strategy, seed):
    return os.path.join(out_dir, "raw", protocol, unit, model, f"{strategy}_seed{seed:03d}.json")


def _time_inference(model, X):
    x1 = X[:1]
    for _ in range(5):
        model.predict(x1)
    single = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict(x1)
        single.append((time.perf_counter() - t0) * 1000.0)
    batch = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.predict(X)
        batch.append((time.perf_counter() - t0) * 1e6 / max(1, len(X)))
    return float(np.median(single)), float(np.median(batch))


def run_task(unit, model_name, strategy, seed, n_evals):
    warnings.filterwarnings("ignore")
    X, y = unit["X"], unit["y"]
    tr, te = unit["train_idx"], unit["test_idx"]
    cseed = config_seed(seed, unit["protocol"], unit["unit"], model_name, strategy)
    rng = np.random.default_rng(cseed)
    ev = Evaluator(model_name, unit, n_evals, model_seed=seed)

    cpu0, wall0 = time.process_time(), time.perf_counter()
    if strategy == "Default":
        best_params = {}
        val_obj = ev.objective_params({})        # diagnostic only; not an HPO cost
        hpo_cpu = hpo_wall = 0.0
        n_used, n_fits, n_failed, history = 0, 0, 0, []
    else:
        try:
            OPTIMIZERS[strategy](ev, rng)
        except BudgetExhausted:
            pass
        hpo_cpu = time.process_time() - cpu0
        hpo_wall = time.perf_counter() - wall0
        if ev.n != n_evals:
            raise RuntimeError(f"budget mismatch: {ev.n} != {n_evals}")
        best_params = ev.best_params if ev.best_params is not None else {}
        val_obj = ev.best_f
        n_used, n_fits, n_failed, history = ev.n, ev.n_fits, ev.n_failed, ev.history

    model = make_model(model_name, seed, best_params)
    c0, w0 = time.process_time(), time.perf_counter()
    model.fit(X[tr], y[tr])
    train_cpu, train_wall = time.process_time() - c0, time.perf_counter() - w0
    y_pred = model.predict(X[te])
    met = regression_metrics(y[te], y_pred)
    inf_single_ms, inf_batch_us = _time_inference(model, X[te])
    size_kb = len(pickle.dumps(model, protocol=4)) / 1024.0
    ytr_min, ytr_max = float(np.min(y[tr])), float(np.max(y[tr]))

    res = {
        "status": "ok", "protocol": unit["protocol"], "unit": unit["unit"], "group": unit["group"],
        "model": model_name, "strategy": strategy, "seed": int(seed), "config_seed": int(cseed),
        "n_evals": int(n_used), "n_fits": int(n_fits), "n_failed_evals": int(n_failed),
        "val_objective": clean_float(val_obj),
        "hpo_cpu_s": float(hpo_cpu), "hpo_wall_s": float(hpo_wall),
        "train_cpu_s": float(train_cpu), "train_wall_s": float(train_wall),
        "infer_single_ms": inf_single_ms, "infer_batch_us_per_sample": inf_batch_us,
        "model_size_kb": float(size_kb), "n_nodes": count_nodes(model, model_name),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "y_train_min": ytr_min, "y_train_max": ytr_max,
        "frac_test_below_train_min": float(np.mean(y[te] < ytr_min)),
        "pred_min": float(np.min(y_pred)),
        "best_params": best_params,
        "convergence": [clean_float(h) for h in history],
        "y_test": [float(v) for v in y[te]], "y_pred": [float(v) for v in y_pred],
        "test_cell": [str(c) for c in unit["row_cell"][te]],
        "test_pos": [int(p) for p in unit["row_pos"][te]],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    res.update(met)
    return res


def run_task_safe(unit, model_name, strategy, seed, n_evals, out_dir, deadline):
    path = task_path(out_dir, unit["protocol"], unit["unit"], model_name, strategy, seed)
    if os.path.exists(path):
        return "exists"
    if deadline is not None and time.time() > deadline:
        return "deadline"
    try:
        res = run_task(unit, model_name, strategy, seed, n_evals)
        atomic_write_json(res, path)
        return "ok"
    except Exception as e:
        err = {"status": "error", "protocol": unit["protocol"], "unit": unit["unit"],
               "model": model_name, "strategy": strategy, "seed": int(seed),
               "error": repr(e), "traceback": traceback.format_exc()}
        epath = os.path.join(out_dir, "errors",
                             f"{unit['protocol']}_{unit['unit']}_{model_name}_{strategy}_seed{seed:03d}.json")
        atomic_write_json(err, epath)
        return "error"


# expected relative cost (longest first -> fewer stragglers at the end of a seed)
def _task_cost_key(unit, model_name, strategy):
    s = {"BO": 3.0, "Default": 0.05}.get(strategy, 1.0)
    m = {"RandomForest": 2.0, "XGBoost": 1.0, "LightGBM": 1.0}[model_name]
    return -(s * m * len(unit["train_idx"]) * len(unit["folds"]))


def run_tasks(units, seeds, n_evals, out_dir, workers, deadline, label):
    from joblib import Parallel, delayed
    total_err = 0
    t_start = time.time()
    for si, seed in enumerate(seeds):
        if deadline is not None and time.time() > deadline:
            log(f"[{label}] deadline reached: seed {seed} and later seeds are not started.")
            break
        tasks = [(u, m, s) for u in units for m in MODELS for s in STRATEGIES]
        tasks.sort(key=lambda t: _task_cost_key(*t))
        todo = [t for t in tasks
                if not os.path.exists(task_path(out_dir, t[0]["protocol"], t[0]["unit"], t[1], t[2], seed))]
        if not todo:
            log(f"[{label}] seed {seed}: already complete ({len(tasks)} tasks).")
            continue
        t0 = time.time()
        if workers > 1:
            status = Parallel(n_jobs=workers, backend="loky", batch_size=1, pre_dispatch="2*n_jobs")(
                delayed(run_task_safe)(u, m, s, seed, n_evals, out_dir, deadline) for (u, m, s) in todo)
        else:
            status = [run_task_safe(u, m, s, seed, n_evals, out_dir, deadline) for (u, m, s) in todo]
        n_err = sum(1 for st in status if st == "error")
        n_dl = sum(1 for st in status if st == "deadline")
        total_err += n_err
        el = time.time() - t0
        done = si + 1
        per_seed = (time.time() - t_start) / done
        log(f"[{label}] seed {seed} ({done}/{len(seeds)}) finished in {el / 60:.1f} min | "
            f"errors={n_err} deadline-skipped={n_dl} | est. remaining "
            f"{per_seed * (len(seeds) - done) / 3600:.2f} h")
        if n_err:
            log(f"[{label}] WARNING: {n_err} task(s) failed; see {os.path.join(out_dir, 'errors')}")
    return total_err


# =============================================================================
# 8. ANALYSIS
# =============================================================================
def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([])
    order = np.argsort(p)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj_sorted
    return out


def rank_biserial(ref, other):
    """>0 means `other` has LOWER RMSE than `ref` (other is better)."""
    from scipy.stats import rankdata
    d = np.asarray(ref) - np.asarray(other)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    wp, wn = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wn) / (wp + wn))


def boot_ci_mean(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float(x.mean()), float(x.mean())) if len(x) else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pareto_mask(P):
    P = np.asarray(P, dtype=float)
    n = len(P)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(P[j] <= P[i]) and np.any(P[j] < P[i]):
                mask[i] = False
                break
    return mask


def _studentized_range_tools():
    from scipy.stats import studentized_range
    for df in (np.inf, 1e5):
        try:
            v = float(studentized_range.ppf(0.95, 3, df))
            if math.isfinite(v):
                return studentized_range, df
        except Exception:
            continue
    return studentized_range, 1e5


def load_results(out_dir):
    rows = []
    raw_dir = os.path.join(out_dir, "raw")
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            if fn.endswith(".json"):
                with open(os.path.join(root, fn), "r", encoding="utf-8") as f:
                    rows.append(json.load(f))
    return rows


def analyse(out_dir, manifest, seeds, min_seeds=MIN_SEEDS_FOR_ANALYSIS, make_figures=True):
    from scipy.stats import friedmanchisquare, wilcoxon, rankdata, spearmanr
    an_dir = os.path.join(out_dir, "analysis")
    os.makedirs(an_dir, exist_ok=True)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})

    rows = load_results(out_dir)
    if not rows:
        raise RuntimeError("no results found")
    full = pd.DataFrame(rows)
    full = full[full["status"] == "ok"]

    expected = [(m["protocol"], m["unit"], mo, s) for m in manifest for mo in MODELS for s in STRATEGIES]
    have = set(zip(full["protocol"], full["unit"], full["model"], full["strategy"], full["seed"]))
    complete_seeds = [sd for sd in seeds if all((p, u, mo, s, sd) in have for (p, u, mo, s) in expected)]
    log(f"[analysis] complete seeds: {len(complete_seeds)} -> {complete_seeds}")
    if len(complete_seeds) < min_seeds:
        raise RuntimeError(f"only {len(complete_seeds)} complete seeds (< {min_seeds})")
    df = full[full["seed"].isin(complete_seeds)].copy()
    N = len(complete_seeds)
    check("complete block design (every config has every seed)",
          bool(df.groupby(["protocol", "unit", "model", "strategy"]).size().eq(N).all()
               and df.groupby(["protocol", "unit", "model", "strategy"]).ngroups == len(expected)),
          f"N={N}, configs={len(expected)}")
    hpo_df = df[df["strategy"] != "Default"]
    check("exact HPO budget in every run", bool((hpo_df["n_evals"] == hpo_df["n_evals"].max()).all()),
          f"n_evals={sorted(hpo_df['n_evals'].unique().tolist())}")

    long_cols = ["protocol", "unit", "group", "model", "strategy", "seed", "test_rmse", "test_mae",
                 "test_mape_pct", "test_r2", "test_max_abs_err", "test_bias", "val_objective",
                 "hpo_cpu_s", "hpo_wall_s", "train_cpu_s", "train_wall_s", "infer_single_ms",
                 "infer_batch_us_per_sample", "model_size_kb", "n_nodes", "n_evals", "n_fits",
                 "n_failed_evals", "n_train", "n_test", "frac_test_below_train_min", "y_train_min", "pred_min"]
    df[long_cols].to_csv(os.path.join(an_dir, "all_runs_long.csv"), index=False)

    # ---- per-config summary ----
    keys = ["protocol", "unit", "group", "model", "strategy"]
    summ = []
    for k, g in df.groupby(keys):
        lo, hi = boot_ci_mean(g["test_rmse"].values)
        rec = dict(zip(keys, k))
        rec.update({
            "n_runs": len(g),
            "rmse_mean": g["test_rmse"].mean(), "rmse_std": g["test_rmse"].std(ddof=1),
            "rmse_median": g["test_rmse"].median(), "rmse_ci95_lo": lo, "rmse_ci95_hi": hi,
            "rmse_pct_points_mean": 100 * g["test_rmse"].mean(),
            "mae_mean": g["test_mae"].mean(), "mape_pct_mean": g["test_mape_pct"].mean(),
            "r2_mean": g["test_r2"].mean(), "bias_mean": g["test_bias"].mean(),
            "val_objective_mean": g["val_objective"].mean(),
            "hpo_cpu_s_mean": g["hpo_cpu_s"].mean(), "hpo_wall_s_mean": g["hpo_wall_s"].mean(),
            "train_cpu_s_mean": g["train_cpu_s"].mean(),
            "infer_single_ms_median": g["infer_single_ms"].median(),
            "infer_batch_us_median": g["infer_batch_us_per_sample"].median(),
            "model_size_kb_mean": g["model_size_kb"].mean(),
            "n_nodes_mean": pd.to_numeric(g["n_nodes"], errors="coerce").mean(),
        })
        summ.append(rec)
    summ = pd.DataFrame(summ)
    base = summ[summ["strategy"] == "Default"][["protocol", "unit", "model", "rmse_mean"]].rename(
        columns={"rmse_mean": "default_rmse_mean"})
    summ = summ.merge(base, on=["protocol", "unit", "model"], how="left")
    summ["rmse_change_vs_default_pct"] = 100 * (summ["rmse_mean"] - summ["default_rmse_mean"]) / summ["default_rmse_mean"]
    # win rate vs Default (paired by seed)
    wr = []
    for (p, u, mo), g in df.groupby(["protocol", "unit", "model"]):
        piv = g.pivot(index="seed", columns="strategy", values="test_rmse")
        for s in HPO_STRATEGIES:
            wr.append({"protocol": p, "unit": u, "model": mo, "strategy": s,
                       "win_rate_vs_default": float((piv[s] < piv["Default"]).mean())})
    summ = summ.merge(pd.DataFrame(wr), on=["protocol", "unit", "model", "strategy"], how="left")
    summ.to_csv(os.path.join(an_dir, "summary_by_config.csv"), index=False)

    # ---- Friedman + Nemenyi + Wilcoxon per (protocol, unit, model) ----
    sr, sr_df = _studentized_range_tools()
    k = len(STRATEGIES)
    q_crit = float(sr.ppf(0.95, k, sr_df)) / math.sqrt(2)
    CD = q_crit * math.sqrt(k * (k + 1) / (6.0 * N))
    fr_rows, nem_rows, wil_rows = [], [], []
    for (p, u, mo), g in df.groupby(["protocol", "unit", "model"]):
        piv = g.pivot(index="seed", columns="strategy", values="test_rmse")[STRATEGIES]
        R = np.apply_along_axis(rankdata, 1, piv.values)
        mean_ranks = R.mean(axis=0)
        chi2_plain = 12 * N / (k * (k + 1)) * (np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4.0)
        try:
            chi2, pval = friedmanchisquare(*[piv[s].values for s in STRATEGIES])
        except Exception:
            chi2, pval = np.nan, np.nan
        tag = f"{p}/{u}/{mo}"
        check(f"Friedman rank sum = k(k+1)/2 [{tag}]", abs(mean_ranks.sum() - k * (k + 1) / 2) < 1e-9,
              f"sum={mean_ranks.sum():.4f}")
        check(f"Friedman chi2 <= N(k-1) [{tag}]", chi2_plain <= N * (k - 1) + 1e-9,
              f"chi2={chi2_plain:.3f}, max={N * (k - 1)}")
        rec = {"protocol": p, "unit": u, "model": mo, "N": N, "k": k,
               "chi2_tie_corrected": clean_float(chi2), "chi2_no_tie_correction": float(chi2_plain),
               "p_value": clean_float(pval), "nemenyi_CD": CD}
        for s, r in zip(STRATEGIES, mean_ranks):
            rec[f"mean_rank_{s}"] = float(r)
        fr_rows.append(rec)
        se = math.sqrt(k * (k + 1) / (12.0 * N))
        for i in range(k):
            for j in range(i + 1, k):
                diff = abs(mean_ranks[i] - mean_ranks[j])
                pn = float(sr.sf(diff / se, k, sr_df))
                better = STRATEGIES[i] if mean_ranks[i] < mean_ranks[j] else STRATEGIES[j]
                nem_rows.append({"protocol": p, "unit": u, "model": mo, "A": STRATEGIES[i], "B": STRATEGIES[j],
                                 "rank_A": float(mean_ranks[i]), "rank_B": float(mean_ranks[j]),
                                 "rank_diff": float(diff), "CD": CD, "p_nemenyi": pn,
                                 "significant": bool(diff > CD), "better": better})
                check(f"Nemenyi p<0.05 <=> diff>CD [{tag} {STRATEGIES[i]}-{STRATEGIES[j]}]",
                      (pn < 0.05) == (diff > CD) or abs(diff - CD) < 1e-6, f"diff={diff:.3f}, p={pn:.4f}")
        for ref in ["Default", "RS"]:
            fam = []
            for s in HPO_STRATEGIES:
                if s == ref:
                    continue
                x, yv = piv[ref].values, piv[s].values
                try:
                    if np.allclose(x, yv):
                        raise ValueError("identical samples")
                    W, pw = wilcoxon(x, yv)
                except Exception:
                    W, pw = np.nan, 1.0
                fam.append({"protocol": p, "unit": u, "model": mo, "reference": ref, "strategy": s,
                            "n_pairs": N, "W": clean_float(W), "p_value": float(pw),
                            "median_diff_ref_minus_strategy": float(np.median(x - yv)),
                            "rank_biserial_strategy_better": rank_biserial(x, yv),
                            "strategy_mean_rmse": float(yv.mean()), "reference_mean_rmse": float(x.mean())})
            adj = holm([r["p_value"] for r in fam])
            for r, a in zip(fam, adj):
                r["p_holm_within_family"] = float(a)
            wil_rows.extend(fam)
    fr = pd.DataFrame(fr_rows)
    nem = pd.DataFrame(nem_rows)
    wil = pd.DataFrame(wil_rows)
    wil["p_holm_global"] = holm(wil["p_value"].values)
    fr.to_csv(os.path.join(an_dir, "friedman.csv"), index=False)
    nem.to_csv(os.path.join(an_dir, "nemenyi.csv"), index=False)
    wil.to_csv(os.path.join(an_dir, "wilcoxon_holm.csv"), index=False)

    # ---- Pareto + Green AI per scope ----
    scopes = []
    for (p, u), _ in summ.groupby(["protocol", "unit"]):
        scopes.append(("unit", p, u, summ[(summ["protocol"] == p) & (summ["unit"] == u)]))
    for (p, gname), _ in summ.groupby(["protocol", "group"]):
        sub = summ[(summ["protocol"] == p) & (summ["group"] == gname)]
        scopes.append(("group", p, gname, sub))
    weights = {"equal": (1 / 3, 1 / 3, 1 / 3), "accuracy": (0.5, 0.25, 0.25),
               "efficiency": (0.25, 0.5, 0.25), "storage": (0.25, 0.25, 0.5)}
    par_rows, green_rows = [], []
    for level, p, name, sub in scopes:
        agg = sub.groupby(["model", "strategy"]).agg(
            rmse=("rmse_mean", "mean"), hpo_cpu_s=("hpo_cpu_s_mean", "mean"),
            size_kb=("model_size_kb_mean", "mean"), mape_pct=("mape_pct_mean", "mean"),
            infer_ms=("infer_single_ms_median", "mean")).reset_index()
        P = agg[["rmse", "hpo_cpu_s", "size_kb"]].values
        mask = pareto_mask(P)
        tag = f"{level}:{p}/{name}"
        for ci, obj in enumerate(["rmse", "hpo_cpu_s", "size_kb"]):
            best = P[:, ci] == P[:, ci].min()
            check(f"Pareto front contains the best {obj} [{tag}]", bool(mask[best].any()))
        for i, r in agg.iterrows():
            par_rows.append({"level": level, "protocol": p, "scope": name, "model": r["model"],
                             "strategy": r["strategy"], "rmse_mean": r["rmse"], "hpo_cpu_s_mean": r["hpo_cpu_s"],
                             "model_size_kb_mean": r["size_kb"], "pareto_optimal": bool(mask[i])})
        norm = {}
        for col in ["rmse", "hpo_cpu_s", "size_kb"]:
            lo, hi = agg[col].min(), agg[col].max()
            norm[col] = (agg[col] - lo) / (hi - lo) if hi > lo else agg[col] * 0.0
        scores = {}
        for wname, (wr_, wh, ws) in weights.items():
            scores[wname] = 1.0 - (wr_ * norm["rmse"] + wh * norm["hpo_cpu_s"] + ws * norm["size_kb"])
        for i, r in agg.iterrows():
            rec = {"level": level, "protocol": p, "scope": name, "model": r["model"], "strategy": r["strategy"],
                   "rmse_mean": r["rmse"], "hpo_cpu_s_mean": r["hpo_cpu_s"], "model_size_kb_mean": r["size_kb"],
                   "pareto_optimal": bool(mask[i])}
            for wname in weights:
                rec[f"greenai_{wname}"] = float(scores[wname].iloc[i])
            green_rows.append(rec)
        for wname in ["accuracy", "efficiency", "storage"]:
            rho = spearmanr(scores["equal"], scores[wname]).correlation
            green_rows_meta = {"level": level, "protocol": p, "scope": name, "model": "__spearman__",
                               "strategy": f"equal_vs_{wname}", f"greenai_{wname}": clean_float(rho)}
            green_rows.append(green_rows_meta)
    par = pd.DataFrame(par_rows)
    green = pd.DataFrame(green_rows)
    par.to_csv(os.path.join(an_dir, "pareto.csv"), index=False)
    green.to_csv(os.path.join(an_dir, "greenai_scores.csv"), index=False)
    gsub = green[green["model"] != "__spearman__"]
    opt_level = gsub.groupby(["level", "protocol", "scope", "strategy"])["greenai_equal"].mean().reset_index()
    opt_level = opt_level.rename(columns={"greenai_equal": "greenai_equal_mean_over_models"})
    opt_level.to_csv(os.path.join(an_dir, "greenai_strategy_level.csv"), index=False)
    # consistency: strategy-level value equals mean of model-level values
    chk = gsub.groupby(["level", "protocol", "scope", "strategy"])["greenai_equal"].mean().values
    check("strategy-level Green AI = mean of model-level scores",
          bool(np.allclose(chk, opt_level["greenai_equal_mean_over_models"].values)))

    # ---- diagnostics: extrapolation and validation->test gap ----
    ext = df.groupby(["protocol", "unit", "model", "strategy"]).agg(
        frac_test_below_train_min=("frac_test_below_train_min", "mean"),
        y_train_min=("y_train_min", "mean"), pred_min_mean=("pred_min", "mean"),
        bias_mean=("test_bias", "mean"), rmse_mean=("test_rmse", "mean"),
        val_objective_mean=("val_objective", "mean")).reset_index()
    ext["test_over_val_ratio"] = ext["rmse_mean"] / ext["val_objective_mean"]
    ext.to_csv(os.path.join(an_dir, "diagnostics_extrapolation_overfitting.csv"), index=False)
    bp = []
    for _, r in df[df["strategy"] != "Default"].iterrows():
        rec = {"protocol": r["protocol"], "unit": r["unit"], "model": r["model"],
               "strategy": r["strategy"], "seed": r["seed"]}
        rec.update(r["best_params"] if isinstance(r["best_params"], dict) else {})
        bp.append(rec)
    pd.DataFrame(bp).to_csv(os.path.join(an_dir, "best_hyperparameters_long.csv"), index=False)

    # convergence (median best-so-far validation objective)
    conv = []
    for (p, u, mo, s), g in df[df["strategy"] != "Default"].groupby(["protocol", "unit", "model", "strategy"]):
        H = np.array([np.array(h, dtype=float) for h in g["convergence"]])
        med = np.nanmedian(H, axis=0)
        for e, v in enumerate(med, start=1):
            conv.append({"protocol": p, "unit": u, "model": mo, "strategy": s, "eval": e, "median_best_val": v})
    conv = pd.DataFrame(conv)
    conv.to_csv(os.path.join(an_dir, "convergence_median.csv"), index=False)

    # ---- markdown tables for the manuscript ----
    _write_paper_tables(an_dir, summ, fr, wil, green, par, opt_level, N, CD)

    # ---- figures ----
    if make_figures:
        _make_figures(an_dir, df, summ, par, conv)

    pd.DataFrame(checks).to_csv(os.path.join(an_dir, "sanity_checks.csv"), index=False)
    n_fail = sum(1 for c in checks if c["result"] == "FAIL")
    log(f"[analysis] sanity checks: {len(checks) - n_fail} PASS, {n_fail} FAIL "
        f"(see analysis/sanity_checks.csv)")
    atomic_write_json({"complete_seeds": complete_seeds, "N": N, "nemenyi_CD": CD,
                       "n_checks": len(checks), "n_failed_checks": n_fail,
                       "generated_at": datetime.now().isoformat(timespec="seconds")},
                      os.path.join(an_dir, "analysis_meta.json"))
    return n_fail


def _fmt(v, nd=4):
    try:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "-"
        return f"{v:.{nd}f}"
    except Exception:
        return str(v)


def _md_table(df_, cols, nd=4):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df_.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], nd) if isinstance(r[c], (float, np.floating)) else str(r[c])
                                       for c in cols) + " |")
    return "\n".join(lines)


def _write_paper_tables(an_dir, summ, fr, wil, green, par, opt_level, N, CD):
    out = [f"# Auto-generated tables (N = {N} complete seeds, Nemenyi CD = {CD:.3f})",
           "All numbers below are computed from analysis/*.csv; SOH is a fraction, "
           "RMSE x 100 = percentage points.\n"]
    for (p, u), g in summ.groupby(["protocol", "unit"]):
        g = g.copy()
        g["RMSE (mean+-sd)"] = g.apply(lambda r: f"{r['rmse_mean']:.5f} +- {r['rmse_std']:.5f}", axis=1)
        g["CI95"] = g.apply(lambda r: f"[{r['rmse_ci95_lo']:.5f}, {r['rmse_ci95_hi']:.5f}]", axis=1)
        g = g.sort_values(["model", "rmse_mean"])
        out.append(f"## Accuracy and cost - {p} / {u}\n")
        out.append(_md_table(g, ["model", "strategy", "RMSE (mean+-sd)", "CI95", "mape_pct_mean",
                                 "rmse_change_vs_default_pct", "win_rate_vs_default", "val_objective_mean",
                                 "hpo_cpu_s_mean", "model_size_kb_mean"]))
        out.append("")
    out.append("## Friedman tests (mean ranks; lower = better)\n")
    out.append(_md_table(fr, ["protocol", "unit", "model", "chi2_tie_corrected", "p_value"]
                         + [f"mean_rank_{s}" for s in STRATEGIES], nd=3))
    out.append("\n## Wilcoxon signed-rank vs Default (Holm within family)\n")
    w = wil[wil["reference"] == "Default"]
    out.append(_md_table(w, ["protocol", "unit", "model", "strategy", "strategy_mean_rmse",
                             "reference_mean_rmse", "p_value", "p_holm_within_family",
                             "rank_biserial_strategy_better"]))
    out.append("\n## Pareto front and Green AI score (group level)\n")
    g = green[(green["level"] == "group") & (green["model"] != "__spearman__")].sort_values(
        ["protocol", "scope", "greenai_equal"], ascending=[True, True, False])
    out.append(_md_table(g, ["protocol", "scope", "model", "strategy", "rmse_mean", "hpo_cpu_s_mean",
                             "model_size_kb_mean", "pareto_optimal", "greenai_equal", "greenai_accuracy",
                             "greenai_efficiency", "greenai_storage"]))
    out.append("\n## Green AI at strategy level (mean of the three model-level scores)\n")
    out.append(_md_table(opt_level[opt_level["level"] == "group"].sort_values(
        ["protocol", "scope", "greenai_equal_mean_over_models"], ascending=[True, True, False]),
        ["protocol", "scope", "strategy", "greenai_equal_mean_over_models"]))
    with open(os.path.join(an_dir, "paper_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def _make_figures(an_dir, df, summ, par, conv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(an_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    def safe(fn, name):
        try:
            fn()
        except Exception as e:
            log(f"[analysis] figure '{name}' skipped: {e}")
        finally:
            plt.close("all")

    def boxplots():
        for (p, u), g in df.groupby(["protocol", "unit"]):
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
            for ax, mo in zip(axes, MODELS):
                gg = g[g["model"] == mo]
                data = [gg[gg["strategy"] == s]["test_rmse"].values for s in STRATEGIES]
                ax.boxplot(data, showmeans=True)
                ax.set_xticks(range(1, len(STRATEGIES) + 1))
                ax.set_xticklabels(STRATEGIES)
                ax.set_title(mo)
                ax.set_ylabel("Test RMSE (SOH fraction)")
                ax.tick_params(axis="x", rotation=45)
            fig.suptitle(f"{p} - {u}")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"box_{p}_{u}.png"), dpi=200)
            plt.close(fig)

    def convergence():
        for (p, u), g in conv.groupby(["protocol", "unit"]):
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
            for ax, mo in zip(axes, MODELS):
                for s in HPO_STRATEGIES:
                    gg = g[(g["model"] == mo) & (g["strategy"] == s)]
                    ax.plot(gg["eval"], gg["median_best_val"], label=s)
                ax.set_title(mo)
                ax.set_xlabel("Objective evaluations")
                ax.set_ylabel("Best validation RMSE (median)")
            axes[-1].legend(fontsize=8)
            fig.suptitle(f"Convergence - {p} - {u}")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"convergence_{p}_{u}.png"), dpi=200)
            plt.close(fig)

    def pareto():
        sub = par[par["level"] == "group"]
        for (p, sc), g in sub.groupby(["protocol", "scope"]):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for mo, mk in zip(MODELS, ["o", "s", "^"]):
                gg = g[g["model"] == mo]
                ax.scatter(gg["hpo_cpu_s_mean"] + 1e-2, gg["rmse_mean"], marker=mk, s=40, label=mo,
                           edgecolors=["k" if po else "none" for po in gg["pareto_optimal"]])
                for _, r in gg.iterrows():
                    ax.annotate(r["strategy"], (r["hpo_cpu_s_mean"] + 1e-2, r["rmse_mean"]), fontsize=7)
            ax.set_xscale("log")
            ax.set_xlabel("HPO CPU time (s, log; Default plotted at 0.01)")
            ax.set_ylabel("Mean test RMSE")
            ax.set_title(f"Accuracy-cost trade-off: {p} / {sc} (outlined = Pareto-optimal)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"pareto_{p}_{sc}.png"), dpi=200)
            plt.close(fig)

    def predictions():
        first_seed = sorted(df["seed"].unique())[0]
        for (p, u), g in df[df["seed"] == first_seed].groupby(["protocol", "unit"]):
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
            for ax, mo in zip(axes, MODELS):
                gg = g[g["model"] == mo]
                d0 = gg[gg["strategy"] == "Default"].iloc[0]
                ax.plot(d0["test_pos"], d0["y_test"], "k-", lw=2, label="Measured")
                ax.plot(d0["test_pos"], d0["y_pred"], "--", label="Default")
                best_s = summ[(summ["protocol"] == p) & (summ["unit"] == u) & (summ["model"] == mo)
                              & (summ["strategy"] != "Default")].sort_values("rmse_mean").iloc[0]["strategy"]
                db = gg[gg["strategy"] == best_s].iloc[0]
                ax.plot(db["test_pos"], db["y_pred"], ":", label=f"{best_s} (best HPO)")
                ax.axhline(d0["y_train_min"], color="grey", lw=0.8, label="min SOH in training")
                ax.set_title(mo)
                ax.set_xlabel("Cycle index")
                ax.set_ylabel("SOH")
            axes[-1].legend(fontsize=7)
            fig.suptitle(f"Test predictions (seed {first_seed}) - {p} - {u}")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"pred_{p}_{u}.png"), dpi=200)
            plt.close(fig)

    def val_vs_test():
        fig, ax = plt.subplots(figsize=(6, 5))
        for s in STRATEGIES:
            g = df[df["strategy"] == s]
            ax.scatter(g["val_objective"], g["test_rmse"], s=6, alpha=0.4, label=s)
        lim = [0, max(df["val_objective"].max(), df["test_rmse"].max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("Validation objective (RMSE)")
        ax.set_ylabel("Test RMSE")
        ax.legend(fontsize=7, markerscale=3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "val_vs_test.png"), dpi=200)
        plt.close(fig)

    for fn, nm in [(boxplots, "boxplots"), (convergence, "convergence"), (pareto, "pareto"),
                   (predictions, "predictions"), (val_vs_test, "val_vs_test")]:
        safe(fn, nm)


# =============================================================================
# 9. ENVIRONMENT RECORD + RUNTIME ESTIMATE
# =============================================================================
def environment_record():
    rec = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
           "cpu_count_logical": os.cpu_count(), "started_at": datetime.now().isoformat(timespec="seconds")}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "joblib", "matplotlib"]:
        try:
            rec[mod] = __import__(mod).__version__
        except Exception as e:
            rec[mod] = f"unavailable: {e}"
    try:
        import psutil
        rec["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass
    return rec


def estimate_runtime(smoke_dir, units, n_evals, n_seeds, workers):
    rows = [r for r in load_results(smoke_dir) if r.get("status") == "ok" and r["strategy"] != "Default"]
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["cpu_per_eval"] = d["hpo_cpu_s"] / d["n_evals"].clip(lower=1)
    per = d.groupby(["protocol", "unit", "model", "strategy"])["cpu_per_eval"].mean()
    per_seed = float(per.sum()) * n_evals
    efficiency = 0.75                                       # hybrid P/E cores, stragglers
    hours = per_seed * n_seeds / (workers * efficiency) / 3600.0
    return hours, per_seed / (workers * efficiency) / 60.0


# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--hours", type=float, default=MAX_HOURS)
    ap.add_argument("--output", type=str, default=OUTPUT_DIR)
    ap.add_argument("--datasets-json", type=str, default=None, help="override DATASETS (testing)")
    args = ap.parse_args()

    datasets = DATASETS
    if args.datasets_json:
        with open(args.datasets_json, "r", encoding="utf-8") as f:
            datasets = json.load(f)
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    set_log_file(os.path.join(out_dir, "run_log.txt"))
    log("=" * 78)
    log("Green AI HPO benchmark v2")
    log(f"output: {out_dir}")

    if args.aggregate_only:
        with open(os.path.join(out_dir, "units_manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        analyse(out_dir, manifest, SEEDS)
        return 0

    # ---- 1. preflight ----
    log("STEP 1/4  preflight: loading and validating datasets")
    try:
        import xgboost, lightgbm, sklearn, scipy, joblib  # noqa: F401
    except Exception as e:
        log(f"[FATAL] missing package: {e}")
        return 2
    try:
        cells, report = preflight(datasets)
    except Exception as e:
        log(f"[FATAL] preflight failed: {e}")
        log("Nothing has been computed. Please send this message.")
        return 2
    units = build_units(cells, datasets)
    manifest = units_manifest(units)
    for m in manifest:
        log(f"  unit {m['protocol']:6s} {m['unit']:12s} train={m['n_train']:5d} test={m['n_test']:5d} "
            f"HPO folds (train,val)={m['folds']}")
    atomic_write_json(report, os.path.join(out_dir, "dataset_report.json"))
    atomic_write_json(manifest, os.path.join(out_dir, "units_manifest.json"))
    atomic_write_json(environment_record(), os.path.join(out_dir, "environment.json"))
    atomic_write_json({"seeds": SEEDS, "n_evals": N_EVALS, "pop_size": POP_SIZE, "bo_n_init": BO_N_INIT,
                       "strategies": STRATEGIES, "models": MODELS, "features": FEATURES,
                       "search_spaces": SEARCH_SPACES, "soh_outlier_rule": [SOH_OUTLIER_WINDOW,
                                                                             SOH_OUTLIER_THRESHOLD]},
                      os.path.join(out_dir, "experiment_config.json"))
    pd.concat([df.assign(cell=name) for name, df in cells.items()]).to_csv(
        os.path.join(out_dir, "prepared_features.csv"), index=False)

    # ---- 2. smoke test ----
    smoke_dir = os.path.join(out_dir, "_smoke")
    if not args.skip_smoke:
        log(f"STEP 2/4  smoke test (seeds={SMOKE_SEEDS}, n_evals={SMOKE_N_EVALS}) - a few minutes")
        t0 = time.time()
        n_err = run_tasks(units, SMOKE_SEEDS, SMOKE_N_EVALS, smoke_dir, args.workers, None, "smoke")
        if n_err:
            log(f"[FATAL] smoke test produced {n_err} error(s); see {os.path.join(smoke_dir, 'errors')}")
            return 3
        try:
            n_fail = analyse(smoke_dir, manifest, SMOKE_SEEDS, min_seeds=2, make_figures=True)
        except Exception as e:
            log(f"[FATAL] smoke analysis failed: {e}\n{traceback.format_exc()}")
            return 3
        if n_fail:
            log(f"[WARNING] smoke analysis: {n_fail} sanity check(s) failed (small N may cause this).")
        log(f"smoke test OK in {(time.time() - t0) / 60:.1f} min")
        est = estimate_runtime(smoke_dir, units, N_EVALS, len(SEEDS), args.workers)
        if est:
            hours, min_per_seed = est
            fit = min(len(SEEDS), int(args.hours * 60 // max(min_per_seed, 1e-9)))
            log(f"estimated: ~{min_per_seed:.0f} min per seed, ~{hours:.1f} h for {len(SEEDS)} seeds "
                f"with {args.workers} workers. Within the {args.hours:.1f} h limit about {fit} seeds "
                f"will complete (>= {MIN_SEEDS_FOR_ANALYSIS} needed).")
        if args.smoke_only:
            return 0
    else:
        log("STEP 2/4  smoke test skipped")

    # ---- 3. full run ----
    log(f"STEP 3/4  full run: {len(units)} units x {len(MODELS)} models x {len(STRATEGIES)} strategies "
        f"x {len(SEEDS)} seeds, n_evals={N_EVALS}, workers={args.workers}, limit={args.hours} h")
    deadline = time.time() + args.hours * 3600.0
    run_tasks(units, SEEDS, N_EVALS, out_dir, args.workers, deadline, "full")

    # ---- 4. analysis ----
    log("STEP 4/4  analysis")
    try:
        analyse(out_dir, manifest, SEEDS)
    except Exception as e:
        log(f"[ERROR] analysis failed: {e}\n{traceback.format_exc()}")
        log("Raw results are safe. Fix and run:  python green_ai_v2.py --aggregate-only")
        return 4
    log("DONE. Results: analysis/paper_tables.md, analysis/*.csv, analysis/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())