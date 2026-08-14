import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os
import time
import psutil
import json
import joblib
import random
import hashlib
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import wilcoxon, norm, bootstrap, rankdata
import seaborn as sns
from skopt import Optimizer
from skopt.space import Real, Integer
from deap import base, creator, tools, algorithms
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import threading
import queue

# Uyarıları filtreleme - sadece bilinen gereksiz uyarıları filtrele
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Sabit seed'ler (20 run)
FIXED_SEEDS = [11, 23, 37, 41, 53, 67, 79, 83, 97, 101,
               113, 127, 139, 149, 163, 173, 181, 193, 211, 223]


# =========================
# RANDOM SEED SETTER
# =========================
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def make_config_seed(base_seed, battery, model, optimizer):
    """Deterministic seed per configuration."""
    text = f"{base_seed}|{battery}|{model}|{optimizer}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return int(digest[:8], 16)


# =========================
# 1. FEATURE EXTRACTION
# =========================
def extract_cycle_features_fast(df):
    df = df.copy()
    grouped = df.groupby('cycle').agg({
        'terminal_voltage': ['mean', 'std', 'min', 'max'],
        'terminal_current': ['mean', 'std', 'min', 'max'],
        'temperature': ['mean', 'std', 'min', 'max'],
        'SOH': 'first'
    })
    grouped.columns = ['_'.join(col).rstrip('_') for col in grouped.columns]
    grouped = grouped.reset_index(drop=True)
    grouped['cycle_id'] = grouped.index
    grouped['voltage_drop'] = grouped['terminal_voltage_max'] - grouped['terminal_voltage_min']
    grouped['temp_rise'] = grouped['temperature_max'] - grouped['temperature_min']
    grouped['current_stability'] = grouped['terminal_current_std'] / (np.abs(grouped['terminal_current_mean']) + 1e-8)
    if 'SOH_first' in grouped.columns:
        grouped.rename(columns={'SOH_first': 'SOH'}, inplace=True)
    return grouped


def add_rolling_features_correct(df, window=3):
    df = df.copy()
    key_cols = ['terminal_voltage_mean', 'terminal_voltage_std', 'temperature_mean']
    for col in key_cols:
        if col in df.columns:
            shifted = df[col].shift(1)
            df[f'{col}_rolling_mean'] = shifted.rolling(window, min_periods=1).mean()
            df[f'{col}_rolling_std'] = shifted.rolling(window, min_periods=1).std()
            df[f'{col}_rolling_mean'].fillna(df[col], inplace=True)
            df[f'{col}_rolling_std'].fillna(0, inplace=True)
    return df


def add_degradation_features_correct(df):
    df = df.copy()
    first_voltage = df['terminal_voltage_mean'].iloc[0]
    first_temp = df['temperature_mean'].iloc[0]
    df['rel_voltage_drop'] = (first_voltage - df['terminal_voltage_mean']) / (abs(first_voltage) + 1e-8)
    df['rel_temp_rise'] = (df['temperature_mean'] - first_temp) / (abs(first_temp) + 1e-8)
    return df


# =========================
# 2. CHRONOLOGICAL CYCLE-LEVEL SPLIT
# =========================
def split_chronologically(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    batteries = df['battery_id'].unique()
    train_list, val_list, test_list = [], [], []

    for bat in batteries:
        bat_df = df[df['battery_id'] == bat].sort_values('cycle_id')
        n = len(bat_df)
        if n < 10:
            raise ValueError(f"Battery {bat} has only {n} cycles; minimum 10 required.")

        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        train_list.append(bat_df.iloc[:train_end])
        val_list.append(bat_df.iloc[train_end:val_end])
        test_list.append(bat_df.iloc[val_end:])

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    return train_df, val_df, test_df


# =========================
# 3. HELPER FUNCTIONS
# =========================
def comprehensive_evaluation(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {'R2': -np.inf, 'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
    return {
        'R2': r2_score(yt, yp),
        'RMSE': np.sqrt(mean_squared_error(yt, yp)),
        'MAE': mean_absolute_error(yt, yp),
        'MAPE': np.mean(np.abs((yt - yp) / (yt + 1e-8))) * 100
    }


def get_models(seed=42, n_jobs=1):
    return {
        'RandomForest': RandomForestRegressor(random_state=seed, n_jobs=n_jobs),
        'XGBoost': XGBRegressor(random_state=seed, n_jobs=n_jobs, verbosity=0),
        'LightGBM': LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1, subsample_freq=1)
    }


def measure_inference_latency(model, X_sample, n_warmup=10, n_measure=100):
    try:
        for _ in range(n_warmup):
            model.predict(X_sample[:1])

        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            model.predict(X_sample[:1])
            times.append((time.perf_counter() - t0) * 1000)

        return float(np.median(times))
    except Exception:
        return float('nan')


def measure_model_size(model, temp_prefix="temp_model"):
    temp_path = f"{temp_prefix}_{random.randint(0, 999999)}.pkl"
    joblib.dump(model, temp_path)
    size_mb = os.path.getsize(temp_path) / 1024 / 1024
    os.remove(temp_path)
    return float(size_mb)


# =========================
# 4. RAM MONITOR
# =========================
class RAMMonitor:
    def __init__(self):
        self._stop = False
        self._queue = queue.Queue()
        self._thread = None
        self.start_rss = None

    def start(self):
        self.start_rss = psutil.Process().memory_info().rss
        self._stop = False
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            peak = self._queue.get_nowait()
        except queue.Empty:
            peak = psutil.Process().memory_info().rss

        return {
            'absolute_peak_mb': peak / 1024 / 1024,
            'incremental_peak_mb': max(0.0, (peak - self.start_rss) / 1024 / 1024)
        }

    def _monitor(self, interval=0.05):
        proc = psutil.Process()
        peak = proc.memory_info().rss
        while not self._stop:
            mem = proc.memory_info().rss
            if mem > peak:
                peak = mem
            time.sleep(interval)
        self._queue.put(peak)


# =========================
# 5. OPTIMIZERS CLASS
# =========================
class Optimizers:
    def __init__(self, model, X_train, y_train, X_val, y_val, n_evals=100, seed=42):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_evals = n_evals
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.py_random = random.Random(seed)
        self.eval_count = 0
        self.best_rmse = float('inf')
        self.best_params = None
        self.convergence_history = []  # (eval_count, best_rmse)
        self._setup_spaces()

    def _setup_spaces(self):
        """Model family detection using isinstance."""
        if isinstance(self.model, RandomForestRegressor):
            self.model_family = 'RandomForest'
            self.dim = 5
            self.bounds = np.array([
                [50, 500],  # n_estimators
                [3, 30],  # max_depth
                [2, 20],  # min_samples_split
                [1, 10],  # min_samples_leaf
                [0.2, 1.0]  # max_features
            ], dtype=float)
            self.log_params = [False, False, False, False, False]
            self.search_spaces = [
                Integer(50, 500),
                Integer(3, 30),
                Integer(2, 20),
                Integer(1, 10),
                Real(0.2, 1.0)
            ]

        elif isinstance(self.model, XGBRegressor):
            self.model_family = 'XGBoost'
            self.dim = 9
            self.bounds = np.array([
                [50, 500],  # n_estimators
                [3, 15],  # max_depth
                [0.01, 0.3],  # learning_rate
                [0.5, 1.0],  # subsample
                [0.5, 1.0],  # colsample_bytree
                [0, 1],  # min_child_weight
                [0, 1],  # gamma
                [0, 1],  # reg_alpha
                [0, 1]  # reg_lambda
            ], dtype=float)
            self.log_params = [False, False, True, False, False, False, False, False, False]
            self.search_spaces = [
                Integer(50, 500),
                Integer(3, 15),
                Real(0.01, 0.3, 'log-uniform'),
                Real(0.5, 1.0),
                Real(0.5, 1.0),
                Real(0, 1),
                Real(0, 1),
                Real(0, 1),
                Real(0, 1)
            ]

        elif isinstance(self.model, LGBMRegressor):
            self.model_family = 'LightGBM'
            self.dim = 9
            self.bounds = np.array([
                [50, 500],  # n_estimators
                [3, 15],  # max_depth
                [0.01, 0.3],  # learning_rate
                [0.5, 1.0],  # subsample
                [0.5, 1.0],  # colsample_bytree
                [5, 50],  # min_child_samples
                [0, 1],  # reg_alpha
                [0, 1],  # reg_lambda
                [2, 50]  # num_leaves
            ], dtype=float)
            self.log_params = [False, False, True, False, False, False, False, False, False]
            self.search_spaces = [
                Integer(50, 500),
                Integer(3, 15),
                Real(0.01, 0.3, 'log-uniform'),
                Real(0.5, 1.0),
                Real(0.5, 1.0),
                Integer(5, 50),
                Real(0, 1),
                Real(0, 1),
                Integer(2, 50)
            ]

        else:
            raise TypeError(f"Unsupported model type: {type(self.model).__name__}")

        # Metaheuristic search space (log-space only for learning_rate)
        self.search_lb = self.bounds[:, 0].copy()
        self.search_ub = self.bounds[:, 1].copy()
        for i, is_log in enumerate(self.log_params):
            if is_log:
                if self.bounds[i, 0] > 0:
                    self.search_lb[i] = np.log10(self.bounds[i, 0])
                else:
                    self.search_lb[i] = np.log10(self.bounds[i, 0] + 1e-10)
                self.search_ub[i] = np.log10(self.bounds[i, 1])

        # For PSO velocity scaling
        self.vmax = 0.2 * (self.search_ub - self.search_lb)

        # For GA - convert to list for DEAP compatibility
        self.search_lb_list = self.search_lb.tolist()
        self.search_ub_list = self.search_ub.tolist()

    def _decode_meta(self, arr):
        """Decode metaheuristic coordinates to actual parameter values."""
        decoded = np.asarray(arr, dtype=float).copy()
        for i, is_log in enumerate(self.log_params):
            if is_log:
                decoded[i] = 10 ** decoded[i]
        return decoded

    def _create_params_dict(self, arr):
        arr = np.asarray(arr, dtype=float)

        if self.model_family == 'RandomForest':
            return {
                'n_estimators': int(np.clip(arr[0], 50, 500)),
                'max_depth': int(np.clip(arr[1], 3, 30)),
                'min_samples_split': int(np.clip(arr[2], 2, 20)),
                'min_samples_leaf': int(np.clip(arr[3], 1, 10)),
                'max_features': float(np.clip(arr[4], 0.2, 1.0))
            }
        elif self.model_family == 'XGBoost':
            return {
                'n_estimators': int(np.clip(arr[0], 50, 500)),
                'max_depth': int(np.clip(arr[1], 3, 15)),
                'learning_rate': float(np.clip(arr[2], 0.01, 0.3)),
                'subsample': float(np.clip(arr[3], 0.5, 1.0)),
                'colsample_bytree': float(np.clip(arr[4], 0.5, 1.0)),
                'min_child_weight': float(np.clip(arr[5], 0, 1)),
                'gamma': float(np.clip(arr[6], 0, 1)),
                'reg_alpha': float(np.clip(arr[7], 0, 1)),
                'reg_lambda': float(np.clip(arr[8], 0, 1))
            }
        elif self.model_family == 'LightGBM':
            return {
                'n_estimators': int(np.clip(arr[0], 50, 500)),
                'max_depth': int(np.clip(arr[1], 3, 15)),
                'learning_rate': float(np.clip(arr[2], 0.01, 0.3)),
                'subsample': float(np.clip(arr[3], 0.5, 1.0)),
                'colsample_bytree': float(np.clip(arr[4], 0.5, 1.0)),
                'min_child_samples': int(np.clip(arr[5], 5, 50)),
                'reg_alpha': float(np.clip(arr[6], 0, 1)),
                'reg_lambda': float(np.clip(arr[7], 0, 1)),
                'num_leaves': int(np.clip(arr[8], 2, 50))
            }
        else:
            raise RuntimeError(f"Unknown model family: {self.model_family}")

    def objective_function(self, params_dict):
        if self.eval_count >= self.n_evals:
            return float('inf')
        try:
            model_clone = clone(self.model)
            model_clone.set_params(**params_dict)
            model_clone.fit(self.X_train, self.y_train)
            y_pred = model_clone.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
            self.eval_count += 1
            if not np.isfinite(rmse):
                return float('inf')
            return float(rmse)
        except Exception:
            self.eval_count += 1
            return float('inf')

    def _evaluate_candidate(self, arr, is_meta=True):
        """Evaluate a candidate point."""
        if is_meta:
            decoded = self._decode_meta(arr)
            params = self._create_params_dict(decoded)
        else:
            params = self._create_params_dict(arr)

        rmse = self.objective_function(params)
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_params = params
        self.convergence_history.append(
            (self.eval_count, self.best_rmse if np.isfinite(self.best_rmse) else float('inf')))
        return rmse

    # ---------- PSO (doğru linearly decreasing inertia) ----------
    def pso_optimize(self):
        n_particles = max(2, int(np.sqrt(self.n_evals)))
        n_iter = max(1, self.n_evals // n_particles)
        lb, ub = self.search_lb, self.search_ub

        pos = self.rng.uniform(lb, ub, (n_particles, self.dim))
        vel = self.rng.uniform(-self.vmax, self.vmax, (n_particles, self.dim))

        pbest_pos = pos.copy()
        pbest_cost = np.array([self._evaluate_candidate(p) for p in pos])
        gbest_idx = np.argmin(pbest_cost)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_cost = pbest_cost[gbest_idx]

        c1, c2 = 1.5, 1.5
        w_start, w_end = 0.9, 0.5

        for t in range(n_iter):
            if n_iter > 1:
                w = w_start - (w_start - w_end) * t / (n_iter - 1)
            else:
                w = w_start

            for i in range(n_particles):
                if self.eval_count >= self.n_evals:
                    break
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)

                vel[i] = w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (gbest_pos - pos[i])
                vel[i] = np.clip(vel[i], -self.vmax, self.vmax)
                pos[i] = np.clip(pos[i] + vel[i], lb, ub)

                cost = self._evaluate_candidate(pos[i])
                if cost < pbest_cost[i]:
                    pbest_cost[i] = cost
                    pbest_pos[i] = pos[i].copy()
                    if cost < gbest_cost:
                        gbest_cost = cost
                        gbest_pos = pos[i].copy()

            if self.eval_count >= self.n_evals:
                break

        return self.best_params, self.best_rmse

    # ---------- WOA ----------
    def woa_optimize(self):
        n_whales = max(2, int(np.sqrt(self.n_evals)))
        n_iter = max(1, self.n_evals // n_whales)
        lb, ub = self.search_lb, self.search_ub
        pos = self.rng.uniform(lb, ub, (n_whales, self.dim))
        fit = np.array([self._evaluate_candidate(p) for p in pos])
        best_idx = np.argmin(fit)
        best_pos = pos[best_idx].copy()
        best_fit = fit[best_idx]
        for t in range(n_iter):
            a = 2 - t * (2 / n_iter)
            for i in range(n_whales):
                if self.eval_count >= self.n_evals:
                    break
                r1, r2 = self.rng.random(2)
                A = 2 * a * r1 - a
                C = 2 * r2
                p = self.rng.random()
                l = self.rng.uniform(-1, 1)
                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * best_pos - pos[i])
                        new_pos = best_pos - A * D
                    else:
                        rand_idx = self.rng.integers(0, n_whales)
                        D = abs(C * pos[rand_idx] - pos[i])
                        new_pos = pos[rand_idx] - A * D
                else:
                    D = abs(best_pos - pos[i])
                    new_pos = D * np.exp(l) * np.cos(2 * np.pi * l) + best_pos
                new_pos = np.clip(new_pos, lb, ub)
                pos[i] = new_pos.copy()
                fit[i] = self._evaluate_candidate(new_pos)
                if fit[i] < best_fit:
                    best_fit = fit[i]
                    best_pos = new_pos.copy()
            if self.eval_count >= self.n_evals:
                break
        return self.best_params, self.best_rmse

    # ---------- GA ----------
    def ga_optimize(self):
        n_pop = max(2, int(np.sqrt(self.n_evals)))
        n_gen = max(1, self.n_evals // n_pop)

        # DEAP uses Python's random module
        random.seed(self.seed)

        if 'creator' in globals():
            if hasattr(creator, "FitnessMin"): del creator.FitnessMin
            if hasattr(creator, "Individual"): del creator.Individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        for i in range(self.dim):
            lb_i, ub_i = self.search_lb_list[i], self.search_ub_list[i]
            if isinstance(lb_i, (int, np.integer)):
                toolbox.register(f"attr{i}", random.randint, int(lb_i), int(ub_i))
            else:
                toolbox.register(f"attr{i}", random.uniform, lb_i, ub_i)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         [getattr(toolbox, f"attr{i}") for i in range(self.dim)], n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(ind):
            rmse = self._evaluate_candidate(np.array(ind))
            return (rmse if np.isfinite(rmse) else 1.0,)

        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        # DEAP mutPolynomialBounded için list kullan (NumPy array değil!)
        toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=self.search_lb_list,
            up=self.search_ub_list,
            eta=20,
            indpb=0.2
        )
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)

        pop = toolbox.population(n=n_pop)
        fits = list(map(toolbox.evaluate, pop))
        for fit, ind in zip(fits, pop):
            ind.fitness.values = fit

        for gen in range(n_gen):
            if self.eval_count >= self.n_evals:
                break
            remaining_budget = self.n_evals - self.eval_count
            if remaining_budget <= 0:
                break

            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
            if len(offspring) > remaining_budget:
                offspring = offspring[:remaining_budget]
            fits = list(map(toolbox.evaluate, offspring))
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            pop = toolbox.select(offspring, k=n_pop)
            if self.eval_count >= self.n_evals:
                break
        return self.best_params, self.best_rmse

    # ---------- GWO ----------
    def gwo_optimize(self):
        n_wolves = max(3, int(np.sqrt(self.n_evals)))
        n_iter = max(1, self.n_evals // n_wolves)
        lb, ub = self.search_lb, self.search_ub
        pos = self.rng.uniform(lb, ub, (n_wolves, self.dim))
        fit = np.array([self._evaluate_candidate(p) for p in pos])
        idx = np.argsort(fit)
        alpha_pos = pos[idx[0]].copy()
        alpha_fit = fit[idx[0]]
        beta_pos = pos[idx[1]].copy() if n_wolves > 1 else pos[0].copy()
        beta_fit = fit[idx[1]] if n_wolves > 1 else fit[0]
        delta_pos = pos[idx[2]].copy() if n_wolves > 2 else pos[0].copy()
        delta_fit = fit[idx[2]] if n_wolves > 2 else fit[0]
        for t in range(n_iter):
            a = 2 - t * (2 / n_iter)
            for i in range(n_wolves):
                if self.eval_count >= self.n_evals:
                    break
                for j in range(self.dim):
                    r1, r2 = self.rng.random(2)
                    A1, C1 = 2 * a * r1 - a, 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - pos[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    r1, r2 = self.rng.random(2)
                    A2, C2 = 2 * a * r1 - a, 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - pos[i, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    r1, r2 = self.rng.random(2)
                    A3, C3 = 2 * a * r1 - a, 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - pos[i, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    pos[i, j] = (X1 + X2 + X3) / 3
                pos[i] = np.clip(pos[i], lb, ub)
                fit[i] = self._evaluate_candidate(pos[i])

            idx = np.argsort(fit)
            alpha_pos = pos[idx[0]].copy()
            alpha_fit = fit[idx[0]]
            beta_pos = pos[idx[1]].copy() if n_wolves > 1 else pos[0].copy()
            beta_fit = fit[idx[1]] if n_wolves > 1 else fit[0]
            delta_pos = pos[idx[2]].copy() if n_wolves > 2 else pos[0].copy()
            delta_fit = fit[idx[2]] if n_wolves > 2 else fit[0]

            if self.eval_count >= self.n_evals:
                break
        return self.best_params, self.best_rmse

    # ---------- Bayesian ----------
    def bayesian_optimize(self):
        opt = Optimizer(
            self.search_spaces,
            base_estimator="GP",
            n_initial_points=10,
            acq_func="EI",
            random_state=self.seed
        )
        for _ in range(self.n_evals):
            if self.eval_count >= self.n_evals:
                break
            suggested = opt.ask()
            rmse = self._evaluate_candidate(np.asarray(suggested, dtype=float), is_meta=False)
            safe_rmse = rmse if np.isfinite(rmse) else 1e6
            opt.tell(suggested, safe_rmse)
        return self.best_params, self.best_rmse

    # ---------- Random Search ----------
    def random_search(self):
        for _ in range(self.n_evals):
            if self.eval_count >= self.n_evals:
                break
            arr = self.rng.uniform(self.search_lb, self.search_ub, self.dim)
            self._evaluate_candidate(arr)
        return self.best_params, self.best_rmse


# =========================
# 6. FULL FACTORIAL RUN PER BATTERY
# =========================
def run_battery_full_factorial(battery_name, train_df, val_df, test_df, feature_cols,
                               models, optimizers_list, n_evals, base_seed):
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df['SOH'].values
    y_val = val_df['SOH'].values
    y_test = test_df['SOH'].values
    test_cycle_ids = test_df['cycle_id'].tolist()

    results = {'base': {}, 'optimized': {}}

    # ----- 1. Base models -----
    for model_name, base_model in models.items():
        model_clone = clone(base_model)
        model_clone.fit(X_train, y_train)
        y_pred_val = model_clone.predict(X_val)
        val_metrics = comprehensive_evaluation(y_val, y_pred_val)

        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
        base_final = clone(base_model)

        train_monitor = RAMMonitor()
        train_monitor.start()
        train_start = time.perf_counter()
        try:
            base_final.fit(X_full, y_full)
        finally:
            train_time = time.perf_counter() - train_start
            train_ram = train_monitor.stop()

        y_pred_test = base_final.predict(X_test)
        test_metrics = comprehensive_evaluation(y_test, y_pred_test)

        results['base'][model_name] = {
            'val_RMSE': val_metrics['RMSE'],
            'val_MAE': val_metrics['MAE'],
            'val_R2': val_metrics['R2'],
            'test_RMSE': test_metrics['RMSE'],
            'test_MAE': test_metrics['MAE'],
            'test_R2': test_metrics['R2'],
            'test_MAPE': test_metrics['MAPE'],
            'training_time': float(train_time),
            'inference_ms_per_sample': measure_inference_latency(base_final, X_test),
            'model_size_mb': measure_model_size(base_final, f"temp_base_{battery_name}_{model_name}"),
            'train_peak_ram_abs_mb': train_ram['absolute_peak_mb'],
            'train_peak_ram_delta_mb': train_ram['incremental_peak_mb'],
            'y_test': y_test.tolist(),
            'y_pred_test': y_pred_test.tolist(),
            'test_cycle_ids': test_cycle_ids
        }

    # ----- 2. Optimized models -----
    rng = random.Random(base_seed + 999)
    opt_order = optimizers_list.copy()
    rng.shuffle(opt_order)

    for model_name, base_model in models.items():
        results['optimized'][model_name] = {}
        for opt_name in opt_order:
            # Independent seed per configuration
            config_seed = make_config_seed(base_seed, battery_name, model_name, opt_name)

            opt = Optimizers(base_model, X_train, y_train, X_val, y_val,
                             n_evals=n_evals, seed=config_seed)

            hpo_monitor = RAMMonitor()
            hpo_monitor.start()
            hpo_start = time.perf_counter()

            try:
                if opt_name == 'PSO':
                    best_params, best_val_rmse = opt.pso_optimize()
                elif opt_name == 'WOA':
                    best_params, best_val_rmse = opt.woa_optimize()
                elif opt_name == 'GA':
                    best_params, best_val_rmse = opt.ga_optimize()
                elif opt_name == 'GWO':
                    best_params, best_val_rmse = opt.gwo_optimize()
                elif opt_name == 'Bayesian':
                    best_params, best_val_rmse = opt.bayesian_optimize()
                else:
                    best_params, best_val_rmse = opt.random_search()
            finally:
                hpo_time = time.perf_counter() - hpo_start
                hpo_ram = hpo_monitor.stop()

            # Budget check - RAISE ERROR if mismatch
            if opt.eval_count != n_evals:
                raise RuntimeError(
                    f"{battery_name} | {model_name} | {opt_name}: "
                    f"expected {n_evals} evaluations, got {opt.eval_count}"
                )

            # Final model
            X_full = np.vstack([X_train, X_val])
            y_full = np.hstack([y_train, y_val])
            final_model = clone(base_model)
            final_model.set_params(**best_params)

            train_monitor = RAMMonitor()
            train_monitor.start()
            train_start = time.perf_counter()
            try:
                final_model.fit(X_full, y_full)
            finally:
                train_time = time.perf_counter() - train_start
                train_ram = train_monitor.stop()

            y_pred_test = final_model.predict(X_test)
            test_metrics = comprehensive_evaluation(y_test, y_pred_test)
            inference_ms = measure_inference_latency(final_model, X_test)
            model_size_mb = measure_model_size(final_model, f"temp_opt_{battery_name}_{model_name}_{opt_name}")

            results['optimized'][model_name][opt_name] = {
                'best_val_RMSE': float(best_val_rmse),
                'test_RMSE': float(test_metrics['RMSE']),
                'test_MAE': float(test_metrics['MAE']),
                'test_R2': float(test_metrics['R2']),
                'test_MAPE': float(test_metrics['MAPE']),
                'hpo_time': float(hpo_time),
                'hpo_peak_ram_abs_mb': hpo_ram['absolute_peak_mb'],
                'hpo_peak_ram_delta_mb': hpo_ram['incremental_peak_mb'],
                'training_time': float(train_time),
                'train_peak_ram_abs_mb': train_ram['absolute_peak_mb'],
                'train_peak_ram_delta_mb': train_ram['incremental_peak_mb'],
                'inference_ms_per_sample': inference_ms,
                'model_size_mb': float(model_size_mb),
                'n_evals': opt.eval_count,
                'convergence_history': [[e, float(r)] for e, r in opt.convergence_history],
                'best_params': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                for k, v in best_params.items()},
                'y_test': y_test.tolist(),
                'y_pred_test': y_pred_test.tolist(),
                'test_cycle_ids': test_cycle_ids
            }

    return results


# =========================
# 7. MAIN EXPERIMENT RUN
# =========================
def run_experiment(files, run_id, n_evals=100, output_dir=None, n_jobs=1, experiment_id=None):
    seed = FIXED_SEEDS[run_id - 1]
    set_seeds(seed)

    print(f"\n{'=' * 80}\nRUN {run_id} (seed={seed}, n_evals={n_evals})\n{'=' * 80}")

    run_dir = os.path.join(output_dir, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    # Load data
    all_dfs = []
    for name, path in files.items():
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df_feat = extract_cycle_features_fast(df)
        df_feat['battery_id'] = name
        all_dfs.append(df_feat)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values(['battery_id', 'cycle_id'])
    full_df = full_df.groupby('battery_id', group_keys=False).apply(add_rolling_features_correct)
    full_df = full_df.groupby('battery_id', group_keys=False).apply(add_degradation_features_correct)
    full_df = full_df.replace([np.inf, -np.inf], np.nan).dropna()

    exclude = ['SOH', 'battery_id', 'cycle_id']
    feature_cols = [c for c in full_df.columns if c not in exclude]

    models = get_models(seed=seed, n_jobs=n_jobs)
    optimizers = ['PSO', 'WOA', 'GA', 'GWO', 'Bayesian', 'RandomSearch']

    all_results = {}

    for bat in full_df['battery_id'].unique():
        print(f"\n--- Battery: {bat} ---")
        bat_df = full_df[full_df['battery_id'] == bat].copy()
        train_df, val_df, test_df = split_chronologically(bat_df, 0.7, 0.15, 0.15)
        for df in [train_df, val_df, test_df]:
            df.dropna(subset=feature_cols, inplace=True)

        bat_results = run_battery_full_factorial(
            bat, train_df, val_df, test_df, feature_cols,
            models, optimizers, n_evals, seed
        )
        all_results[bat] = bat_results

    # Save results with robust JSON sanitizer
    run_summary = {
        'experiment_id': experiment_id,
        'run_id': run_id,
        'seed': seed,
        'n_evals': n_evals,
        'battery_results': all_results
    }

    def convert(obj):
        """Robust JSON sanitizer for NumPy and pandas types."""
        # Handle non-finite floats
        if isinstance(obj, (float, np.floating)):
            val = float(obj)
            return val if np.isfinite(val) else None

        if isinstance(obj, (int, np.integer)):
            return int(obj)

        if isinstance(obj, np.ndarray):
            return [convert(x) for x in obj.tolist()]

        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]

        if isinstance(obj, (datetime, pd.Timestamp)):
            return str(obj)

        return obj

    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(convert(run_summary), f, indent=2)

    print(f"✅ Run {run_id} completed.")
    return run_summary


# =========================
# 8. AGGREGATION
# =========================
def holm_correction(p_values):
    """Corrected Holm-Bonferroni correction."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.minimum(
        (m - np.arange(m)) * sorted_p,
        1.0
    )
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def compute_rank_biserial(x, y):
    """Compute matched-pairs rank-biserial correlation."""
    diff = np.array(y) - np.array(x)
    nonzero = diff != 0
    if not np.any(nonzero):
        return 0.0
    diff = diff[nonzero]
    ranks = rankdata(np.abs(diff))
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    if w_pos + w_neg == 0:
        return 0.0
    return (w_pos - w_neg) / (w_pos + w_neg)


def aggregate_runs(output_dir, num_runs, expected_n_evals, strict=False, expected_experiment_id=None):
    print("\n" + "=" * 80)
    print(f"AGGREGATING RESULTS FROM {num_runs} RUNS (n_evals={expected_n_evals})")
    if strict:
        print("  STRICT MODE: All runs and configurations must be complete")
    if expected_experiment_id:
        print(f"  Experiment ID: {expected_experiment_id}")
    print("=" * 80)

    all_data = []
    successful_runs = 0

    for run_id in range(1, num_runs + 1):
        run_dir = os.path.join(output_dir, f"run_{run_id:02d}")
        result_file = os.path.join(run_dir, 'results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                # Check experiment_id
                if expected_experiment_id and data.get('experiment_id') != expected_experiment_id:
                    print(
                        f"⚠️ Run {run_id}: experiment_id={data.get('experiment_id')} != {expected_experiment_id}, skipping.")
                    continue
                if data.get('n_evals') != expected_n_evals:
                    print(f"⚠️ Run {run_id}: n_evals={data.get('n_evals')} != {expected_n_evals}, skipping.")
                    continue
                all_data.append(data)
                successful_runs += 1
        else:
            print(f"⚠️ Run {run_id} results not found, skipping.")

    if strict and successful_runs != num_runs:
        raise RuntimeError(
            f"Strict mode: expected {num_runs} runs, got {successful_runs}"
        )

    if successful_runs == 0:
        print("❌ No successful runs found.")
        return None, None

    print(f"✅ {successful_runs}/{num_runs} runs available.")

    # Build DataFrame
    rows = []
    for run in all_data:
        run_id = run['run_id']
        seed = run['seed']
        for battery, bat_res in run['battery_results'].items():
            # Baseline
            for model_name, base_metrics in bat_res['base'].items():
                rows.append({
                    'run_id': run_id,
                    'seed': seed,
                    'battery': battery,
                    'model': model_name,
                    'optimizer': 'Baseline',
                    'test_RMSE': base_metrics.get('test_RMSE', float('inf')),
                    'test_MAE': base_metrics.get('test_MAE', float('inf')),
                    'test_R2': base_metrics.get('test_R2', -float('inf')),
                    'test_MAPE': base_metrics.get('test_MAPE', float('inf')),
                    'training_time': base_metrics.get('training_time', 0),
                    'inference_ms': base_metrics.get('inference_ms_per_sample', float('nan')),
                    'model_size_mb': base_metrics.get('model_size_mb', 0),
                    'train_peak_ram_delta_mb': base_metrics.get('train_peak_ram_delta_mb', 0),
                    'hpo_time': 0,
                    'hpo_peak_ram_delta_mb': 0,
                    'n_evals': 0
                })

            # Optimized
            for model_name, opt_res in bat_res['optimized'].items():
                for opt_name, metrics in opt_res.items():
                    # Check per-configuration n_evals
                    if metrics.get('n_evals') != expected_n_evals:
                        print(f"⚠️ {battery}|{model_name}|{opt_name}: "
                              f"n_evals={metrics.get('n_evals')} != {expected_n_evals}, skipping")
                        continue

                    rmse = metrics.get('test_RMSE')
                    if rmse is None or not np.isfinite(float(rmse)):
                        continue
                    rows.append({
                        'run_id': run_id,
                        'seed': seed,
                        'battery': battery,
                        'model': model_name,
                        'optimizer': opt_name,
                        'test_RMSE': rmse,
                        'test_MAE': metrics.get('test_MAE', float('inf')),
                        'test_R2': metrics.get('test_R2', -float('inf')),
                        'test_MAPE': metrics.get('test_MAPE', float('inf')),
                        'training_time': metrics.get('training_time', 0),
                        'inference_ms': metrics.get('inference_ms_per_sample', float('nan')),
                        'model_size_mb': metrics.get('model_size_mb', 0),
                        'train_peak_ram_delta_mb': metrics.get('train_peak_ram_delta_mb', 0),
                        'hpo_time': metrics.get('hpo_time', 0),
                        'hpo_peak_ram_delta_mb': metrics.get('hpo_peak_ram_delta_mb', 0),
                        'n_evals': metrics.get('n_evals', 0)
                    })

    df = pd.DataFrame(rows)

    # ---- 1. Aggregated summary ----
    group_cols = ['battery', 'model', 'optimizer']
    summary = df.groupby(group_cols).agg({
        'test_RMSE': ['mean', 'std', 'count'],
        'test_MAE': ['mean', 'std'],
        'test_R2': ['mean', 'std'],
        'test_MAPE': ['mean', 'std'],
        'hpo_time': ['mean', 'std'],
        'training_time': ['mean', 'std'],
        'inference_ms': ['mean', 'std'],
        'model_size_mb': ['mean', 'std'],
        'hpo_peak_ram_delta_mb': ['mean', 'std'],
        'train_peak_ram_delta_mb': ['mean', 'std']
    }).reset_index()

    # Fix column names
    summary.columns = ['_'.join([str(x) for x in col if x]).strip('_')
                       for col in summary.columns.values]

    # Check that all configurations have expected number of runs
    if strict:
        optimized = summary[summary['optimizer'] != 'Baseline']
        bad_counts = optimized[optimized['test_RMSE_count'] != num_runs]
        if len(bad_counts) > 0:
            print(f"⚠️ Configurations with incomplete runs:")
            for _, row in bad_counts.iterrows():
                print(f"   {row['battery']} | {row['model']} | {row['optimizer']}: {row['test_RMSE_count']}/{num_runs}")
            raise RuntimeError(
                f"Strict mode: {len(bad_counts)} configurations missing runs"
            )

    # ---- 2. Bootstrap 95% CI (DÜZELTİLDİ) ----
    ci_data = []
    for (battery, model, optimizer), group in df.groupby(['battery', 'model', 'optimizer']):
        rmse_vals = group['test_RMSE'].dropna()
        if len(rmse_vals) >= 2:
            try:
                res = bootstrap(
                    (rmse_vals.to_numpy(),),
                    np.mean,
                    confidence_level=0.95,
                    n_resamples=5000,
                    method='BCa',
                    random_state=42
                )
                ci_low, ci_high = res.confidence_interval
            except Exception:
                # Fallback: bootstrap percentile CI for the mean
                vals = rmse_vals.to_numpy(dtype=float)
                rng_ci = np.random.default_rng(42)
                boot_means = np.array([
                    np.mean(rng_ci.choice(vals, size=len(vals), replace=True))
                    for _ in range(5000)
                ])
                ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        elif len(rmse_vals) == 1:
            val = rmse_vals.iloc[0]
            ci_low, ci_high = val, val
        else:
            ci_low, ci_high = np.nan, np.nan

        ci_data.append({
            'battery': battery,
            'model': model,
            'optimizer': optimizer,
            'RMSE_CI_lower': float(ci_low),
            'RMSE_CI_upper': float(ci_high)
        })

    ci_df = pd.DataFrame(ci_data)
    summary = summary.merge(ci_df, on=['battery', 'model', 'optimizer'], how='left')

    # ---- 3. RMSE Improvement over Baseline ----
    baseline_rmse = summary[summary['optimizer'] == 'Baseline'][['battery', 'model', 'test_RMSE_mean']]
    baseline_rmse.rename(columns={'test_RMSE_mean': 'baseline_RMSE_mean'}, inplace=True)
    summary = summary.merge(baseline_rmse, on=['battery', 'model'], how='left')
    summary['RMSE_Improvement_pct'] = 100 * (summary['baseline_RMSE_mean'] - summary['test_RMSE_mean']) / summary[
        'baseline_RMSE_mean']

    # ---- 4. Battery-level Pareto ----
    for battery in summary['battery'].unique():
        mask = (summary['battery'] == battery) & (summary['optimizer'] != 'Baseline')
        bat_data = summary[mask].copy()
        if len(bat_data) > 0:
            points = bat_data[['test_RMSE_mean', 'hpo_time_mean', 'model_size_mb_mean']].values
            is_pareto = np.ones(len(points), dtype=bool)
            for i in range(len(points)):
                for j in range(len(points)):
                    if i != j:
                        if (points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and
                                points[j, 2] <= points[i, 2] and
                                (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1] or points[j, 2] < points[
                                    i, 2])):
                            is_pareto[i] = False
                            break
            summary.loc[mask, 'Pareto_Optimal_Battery'] = is_pareto
        else:
            summary.loc[mask, 'Pareto_Optimal_Battery'] = False

    # ---- 5. Overall Pareto ----
    agg_overall = summary[summary['optimizer'] != 'Baseline'].groupby(['model', 'optimizer']).agg({
        'test_RMSE_mean': 'mean',
        'hpo_time_mean': 'mean',
        'model_size_mb_mean': 'mean'
    }).reset_index()

    if len(agg_overall) > 0:
        points = agg_overall[['test_RMSE_mean', 'hpo_time_mean', 'model_size_mb_mean']].values
        is_pareto_overall = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    if (points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and
                            points[j, 2] <= points[i, 2] and
                            (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1] or points[j, 2] < points[
                                i, 2])):
                        is_pareto_overall[i] = False
                        break
        agg_overall['Pareto_Optimal_Overall'] = is_pareto_overall
        summary = summary.merge(
            agg_overall[['model', 'optimizer', 'Pareto_Optimal_Overall']],
            on=['model', 'optimizer'],
            how='left'
        )
    else:
        summary['Pareto_Optimal_Overall'] = False

    # ---- 6. Green AI Score ----
    green_agg = summary[summary['optimizer'] != 'Baseline'].groupby(['model', 'optimizer']).agg({
        'test_RMSE_mean': 'mean',
        'hpo_time_mean': 'mean',
        'model_size_mb_mean': 'mean'
    }).reset_index()

    if len(green_agg) > 0:
        for col in ['test_RMSE_mean', 'hpo_time_mean', 'model_size_mb_mean']:
            min_val = green_agg[col].min()
            max_val = green_agg[col].max()
            if max_val > min_val:
                green_agg[f'{col}_norm'] = (green_agg[col] - min_val) / (max_val - min_val)
            else:
                green_agg[f'{col}_norm'] = 0.0

        green_agg['GreenAI_Score'] = 1 - (
                green_agg['test_RMSE_mean_norm'] +
                green_agg['hpo_time_mean_norm'] +
                green_agg['model_size_mb_mean_norm']
        ) / 3

        summary = summary.merge(
            green_agg[['model', 'optimizer', 'GreenAI_Score']],
            on=['model', 'optimizer'],
            how='left'
        )
    else:
        summary['GreenAI_Score'] = np.nan

    # ---- 7. Statistical tests ----
    stats_results = []
    for battery in df['battery'].unique():
        for model in df['model'].unique():
            rs_df = df[(df['battery'] == battery) &
                       (df['model'] == model) &
                       (df['optimizer'] == 'RandomSearch')][['run_id', 'test_RMSE']].dropna()
            if len(rs_df) < 2:
                continue

            for opt in ['PSO', 'WOA', 'GA', 'GWO', 'Bayesian']:
                opt_df = df[(df['battery'] == battery) &
                            (df['model'] == model) &
                            (df['optimizer'] == opt)][['run_id', 'test_RMSE']].dropna()
                if len(opt_df) < 2:
                    continue

                merged = rs_df.merge(opt_df, on='run_id', suffixes=('_rs', '_opt'))
                if len(merged) < 2:
                    continue

                rs_vals = merged['test_RMSE_rs'].values
                opt_vals = merged['test_RMSE_opt'].values

                try:
                    stat, p = wilcoxon(rs_vals, opt_vals)
                    es = compute_rank_biserial(rs_vals, opt_vals)

                    stats_results.append({
                        'battery': battery,
                        'model': model,
                        'optimizer': opt,
                        'comparison': f'{opt}_vs_RandomSearch',
                        'n_pairs': len(merged),
                        'wilcoxon_stat': float(stat),
                        'p_value': float(p),
                        'effect_size': float(es),
                        'significant_raw': bool(p < 0.05),
                        'rs_mean': float(np.mean(rs_vals)),
                        'opt_mean': float(np.mean(opt_vals))
                    })
                except Exception as e:
                    print(f"⚠️ Wilcoxon failed for {battery}-{model}-{opt}: {e}")

    stats_df = pd.DataFrame(stats_results) if stats_results else pd.DataFrame()

    # ---- 8. Holm correction ----
    if len(stats_df) > 0:
        stats_df['holm_p_value'] = holm_correction(stats_df['p_value'].values)
        stats_df['significant_holm'] = stats_df['holm_p_value'] < 0.05

    # ---- 9. Save results ----
    summary.to_csv(os.path.join(output_dir, 'aggregated_results.csv'), index=False)
    if len(stats_df) > 0:
        stats_df.to_csv(os.path.join(output_dir, 'statistical_tests.csv'), index=False)

    expected_configs = 3 * 3 * 6
    actual_configs = len(summary[summary['optimizer'] != 'Baseline'])

    # Calculate total optimized data points
    optimized_summary = summary[summary['optimizer'] != 'Baseline']
    total_optimized_points = int(optimized_summary['test_RMSE_count'].sum()) if len(optimized_summary) > 0 else 0

    print(f"\n✅ Aggregation complete.")
    print(f"   - aggregated_results.csv: {len(summary)} configurations")
    print(f"   - statistical_tests.csv: {len(stats_df)} comparisons")
    print(f"   - Optimized configurations: {actual_configs}/{expected_configs}")
    print(f"   - Total optimized data points: {total_optimized_points}")
    print(f"   - Expected: {expected_configs} × {num_runs} = {expected_configs * num_runs}")

    return summary, stats_df


# =========================
# 9. MAIN - RESUME MODE ONLY
# =========================
if __name__ == "__main__":
    base_path = r"C:\Users\LENOVO\Desktop\Transfer Learning Project\Veriler\cleaned_dataset\organized_dataset\25 degree"
    files = {
        "B0005": os.path.join(base_path, "B0005_discharge_soh_standardized.csv"),
        "B0006": os.path.join(base_path, "B0006_discharge_soh_standardized.csv"),
        "B0018": os.path.join(base_path, "B0018_discharge_soh_standardized.csv")
    }

    N_JOBS = 1
    EXPERIMENT_ID = "FINAL_V1_20260811"

    FULL_OUTPUT_DIR = os.path.join(base_path, "green_ai_final_runs")
    FULL_NUM_RUNS = 20
    FULL_N_EVALS = 100

    print(f"\n{'='*80}")
    print(f"RESUME MODE: Checking existing runs...")
    print(f"  Output: {FULL_OUTPUT_DIR}")
    print(f"{'='*80}\n")

    # Tamamlanmış run'ları kontrol et
    completed_runs = []
    for run_id in range(1, FULL_NUM_RUNS + 1):
        run_dir = os.path.join(FULL_OUTPUT_DIR, f"run_{run_id:02d}")
        result_file = os.path.join(run_dir, 'results.json')
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if data.get('experiment_id') == EXPERIMENT_ID and data.get('n_evals') == FULL_N_EVALS:
                        completed_runs.append(run_id)
            except:
                pass

    print(f"✅ Runs already completed: {completed_runs}")
    print(f"   ({len(completed_runs)}/{FULL_NUM_RUNS})")

    # Eksik run'ları bul
    missing_runs = [r for r in range(1, FULL_NUM_RUNS + 1) if r not in completed_runs]

    if missing_runs:
        print(f"🔄 Runs to resume: {missing_runs}")

        for run_id in missing_runs:
            try:
                print(f"\n--- Resuming run {run_id} ---")
                run_experiment(files, run_id, n_evals=FULL_N_EVALS,
                               output_dir=FULL_OUTPUT_DIR, n_jobs=N_JOBS,
                               experiment_id=EXPERIMENT_ID)
            except Exception as e:
                print(f"❌ Run {run_id} failed: {e}")
                import traceback
                traceback.print_exc()
                print("Stopping due to error.")
                break
        else:
            # Tüm run'lar tamamlandı
            print("\n" + "="*80)
            print("ALL RUNS COMPLETED! Running final aggregation...")
            print("="*80)

            try:
                summary, stats = aggregate_runs(FULL_OUTPUT_DIR, FULL_NUM_RUNS, FULL_N_EVALS,
                                                strict=True, expected_experiment_id=EXPERIMENT_ID)

                if summary is not None:
                    print("\n" + "="*80)
                    print("🎉 FULL EXPERIMENT COMPLETED SUCCESSFULLY!")
                    print("="*80)

                    optimized = summary[summary['optimizer'] != 'Baseline']
                    total_points = int(optimized['test_RMSE_count'].sum()) if len(optimized) > 0 else 0

                    print(f"\n📊 Summary of final results:")
                    print(f"   Optimized configurations: {len(optimized)}")
                    print(f"   Total optimized data points: {total_points}")
                    print(f"   Expected: 54 × 20 = 1080")
                    print(f"   Statistical comparisons: {len(stats) if stats is not None else 0}")

                    best_overall = optimized.sort_values('test_RMSE_mean').head(5)
                    print("\n🏆 Top 5 configurations (by RMSE):")
                    for _, row in best_overall.iterrows():
                        print(f"   {row['battery']} | {row['model']} | {row['optimizer']}: "
                              f"RMSE={row['test_RMSE_mean']:.5f} ± {row['test_RMSE_std']:.5f} "
                              f"GreenAI={row.get('GreenAI_Score', np.nan):.4f}")

                    print(f"\n📁 Final results saved to: {FULL_OUTPUT_DIR}")
                    print(f"   - aggregated_results.csv")
                    print(f"   - statistical_tests.csv")
            except Exception as e:
                print(f"\n❌ Aggregation failed: {e}")
    else:
        print("✅ All runs already completed!")
        try:
            summary, stats = aggregate_runs(FULL_OUTPUT_DIR, FULL_NUM_RUNS, FULL_N_EVALS,
                                            strict=True, expected_experiment_id=EXPERIMENT_ID)
            print("\n✅ Aggregation completed successfully!")
        except Exception as e:
            print(f"\n❌ Aggregation failed: {e}")