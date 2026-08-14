


import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, rankdata


# ============================================================
# 0. USER CONFIGURATION
# ============================================================

RUNS_DIR = r"C:\Users\LENOVO\Desktop\Transfer Learning Project\Veriler\cleaned_dataset\organized_dataset\25 degree\green_ai_final_runs"

# Existing experiment
EXPECTED_RUNS = 20

# Main experiment uses 100 evaluations
EXPECTED_N_EVALS = 100

# Optimizer order used in the FINAL experiment
OPTIMIZER_ORDER = [
    "RandomSearch",
    "PSO",
    "WOA",
    "GA",
    "GWO",
    "Bayesian"
]

# Model order used in the FINAL experiment
MODEL_ORDER = [
    "RandomForest",
    "XGBoost",
    "LightGBM"
]

# Only optimized results are used for optimization comparisons
OPTIMIZED_ONLY = True

# Output directory
OUTPUT_DIR = os.path.join(
    RUNS_DIR,
    "stability_visualizations_FINAL"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. MATPLOTLIB SETTINGS
# ============================================================

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    """
    Convert values safely to float.
    JSON sanitizer in the main code converts non-finite values
    to None, so None must be handled.
    """
    if value is None:
        return np.nan

    try:
        value = float(value)

        if np.isfinite(value):
            return value

        return np.nan

    except Exception:
        return np.nan


def save_figure(filename):
    """
    Save publication-quality PNG.
    """
    path = os.path.join(OUTPUT_DIR, filename)

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close()

    print(f"   ✅ {filename}")


def get_nested_metric(metrics, key):
    """
    Safely extract a metric.
    """
    if not isinstance(metrics, dict):
        return np.nan

    return safe_float(metrics.get(key))


def holm_correction(p_values):
    """
    Holm-Bonferroni correction.
    """
    p = np.asarray(p_values, dtype=float)

    if len(p) == 0:
        return np.array([])

    m = len(p)

    order = np.argsort(p)

    sorted_p = p[order]

    adjusted_sorted = np.minimum(
        (m - np.arange(m)) * sorted_p,
        1.0
    )

    adjusted_sorted = np.maximum.accumulate(
        adjusted_sorted
    )

    adjusted = np.empty(m)

    adjusted[order] = adjusted_sorted

    return adjusted


def rank_biserial(x, y):
    """
    Matched-pairs rank-biserial correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diff = y - x

    diff = diff[np.isfinite(diff)]

    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    ranks = rankdata(np.abs(diff))

    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()

    denominator = positive + negative

    if denominator == 0:
        return 0.0

    return float(
        (positive - negative) / denominator
    )


# ============================================================
# 3. VERIFY RUN DIRECTORY
# ============================================================

print("\n" + "=" * 80)
print("EXISTING RESULTS VISUALIZATION")
print("=" * 80)

print("\n📁 Results directory:")
print(RUNS_DIR)

if not os.path.exists(RUNS_DIR):
    raise FileNotFoundError(
        f"\n❌ Results directory not found:\n{RUNS_DIR}"
    )


# ============================================================
# 4. FIND EXISTING RUNS
# ============================================================

print("\n" + "-" * 80)
print("STEP 1 - CHECKING EXISTING RUNS")
print("-" * 80)

run_directories = []

for run_id in range(1, EXPECTED_RUNS + 1):

    run_dir = os.path.join(
        RUNS_DIR,
        f"run_{run_id:02d}"
    )

    result_file = os.path.join(
        run_dir,
        "results.json"
    )

    if os.path.exists(result_file):

        run_directories.append(
            (run_id, result_file)
        )

        print(
            f"   ✅ Run {run_id:02d}: results.json"
        )

    else:

        print(
            f"   ⚠️ Run {run_id:02d}: NOT FOUND"
        )


print(
    f"\n📊 Existing runs found: "
    f"{len(run_directories)}/{EXPECTED_RUNS}"
)

if len(run_directories) == 0:
    raise RuntimeError(
        "❌ No results.json files were found."
    )


# ============================================================
# 5. LOAD JSON RESULTS
# ============================================================

print("\n" + "-" * 80)
print("STEP 2 - READING EXISTING JSON RESULTS")
print("-" * 80)

loaded_runs = []

for run_id, result_file in run_directories:

    try:

        with open(
            result_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        loaded_runs.append(
            {
                "run_id": run_id,
                "data": data,
                "path": result_file
            }
        )

        print(
            f"   ✅ Run {run_id:02d} loaded"
        )

    except Exception as e:

        print(
            f"   ❌ Run {run_id:02d} failed: {e}"
        )


if not loaded_runs:
    raise RuntimeError(
        "❌ JSON files could not be loaded."
    )


# ============================================================
# 6. CHECK EXPERIMENT METADATA
# ============================================================

print("\n" + "-" * 80)
print("STEP 3 - CHECKING EXPERIMENT METADATA")
print("-" * 80)

experiment_ids = []
n_evals_values = []

for item in loaded_runs:

    data = item["data"]

    experiment_ids.append(
        data.get("experiment_id")
    )

    n_evals_values.append(
        data.get("n_evals")
    )


unique_experiments = sorted(
    set(
        x for x in experiment_ids
        if x is not None
    )
)

unique_n_evals = sorted(
    set(
        x for x in n_evals_values
        if x is not None
    )
)

print(
    f"   Experiment IDs: {unique_experiments}"
)

print(
    f"   n_evals values: {unique_n_evals}"
)

if EXPECTED_N_EVALS not in unique_n_evals:

    print(
        f"   ⚠️ Expected n_evals={EXPECTED_N_EVALS}"
        f" but detected {unique_n_evals}"
    )

else:

    print(
        f"   ✅ n_evals={EXPECTED_N_EVALS} confirmed"
    )


# ============================================================
# 7. EXTRACT NEW DATA STRUCTURE
# ============================================================
#
# FINAL structure:
#
# run
# └── battery_results
#     ├── B0005
#     │   ├── base
#     │   │   ├── RandomForest
#     │   │   ├── XGBoost
#     │   │   └── LightGBM
#     │   │
#     │   └── optimized
#     │       ├── RandomForest
#     │       │   ├── PSO
#     │       │   ├── WOA
#     │       │   ├── GA
#     │       │   ├── GWO
#     │       │   ├── Bayesian
#     │       │   └── RandomSearch
#     │       ├── XGBoost
#     │       └── LightGBM
#
# ============================================================

print("\n" + "-" * 80)
print("STEP 4 - EXTRACTING RESULTS")
print("-" * 80)

records = []

base_records = []

convergence_records = []


for item in loaded_runs:

    run_id = item["run_id"]

    data = item["data"]

    battery_results = data.get(
        "battery_results",
        {}
    )

    if not battery_results:

        print(
            f"   ⚠️ Run {run_id:02d}: "
            f"'battery_results' missing"
        )

        continue


    for battery, battery_data in battery_results.items():

        if not isinstance(
            battery_data,
            dict
        ):
            continue


        # ----------------------------------------------------
        # BASE MODELS
        # ----------------------------------------------------

        base_section = battery_data.get(
            "base",
            {}
        )

        for model, metrics in base_section.items():

            if not isinstance(metrics, dict):
                continue

            base_records.append(
                {
                    "run_id": run_id,
                    "battery": battery,
                    "model": model,
                    "optimizer": "Baseline",
                    "test_RMSE": get_nested_metric(
                        metrics,
                        "test_RMSE"
                    ),
                    "test_MAE": get_nested_metric(
                        metrics,
                        "test_MAE"
                    ),
                    "test_R2": get_nested_metric(
                        metrics,
                        "test_R2"
                    ),
                    "test_MAPE": get_nested_metric(
                        metrics,
                        "test_MAPE"
                    ),
                    "training_time": get_nested_metric(
                        metrics,
                        "training_time"
                    ),
                    "hpo_time": np.nan,
                    "model_size_mb": get_nested_metric(
                        metrics,
                        "model_size_mb"
                    ),
                    "inference_ms_per_sample":
                        get_nested_metric(
                            metrics,
                            "inference_ms_per_sample"
                        ),
                    "train_peak_ram_abs_mb":
                        get_nested_metric(
                            metrics,
                            "train_peak_ram_abs_mb"
                        ),
                    "train_peak_ram_delta_mb":
                        get_nested_metric(
                            metrics,
                            "train_peak_ram_delta_mb"
                        ),
                    "hpo_peak_ram_abs_mb": np.nan,
                    "hpo_peak_ram_delta_mb": np.nan
                }
            )


        # ----------------------------------------------------
        # OPTIMIZED MODELS
        # ----------------------------------------------------

        optimized_section = battery_data.get(
            "optimized",
            {}
        )

        for model, optimizer_data in optimized_section.items():

            if not isinstance(
                optimizer_data,
                dict
            ):
                continue


            for optimizer, metrics in optimizer_data.items():

                if not isinstance(metrics, dict):
                    continue


                record = {
                    "run_id": run_id,
                    "battery": battery,
                    "model": model,
                    "optimizer": optimizer,

                    "test_RMSE": get_nested_metric(
                        metrics,
                        "test_RMSE"
                    ),

                    "test_MAE": get_nested_metric(
                        metrics,
                        "test_MAE"
                    ),

                    "test_R2": get_nested_metric(
                        metrics,
                        "test_R2"
                    ),

                    "test_MAPE": get_nested_metric(
                        metrics,
                        "test_MAPE"
                    ),

                    "training_time": get_nested_metric(
                        metrics,
                        "training_time"
                    ),

                    "hpo_time": get_nested_metric(
                        metrics,
                        "hpo_time"
                    ),

                    "model_size_mb": get_nested_metric(
                        metrics,
                        "model_size_mb"
                    ),

                    "inference_ms_per_sample":
                        get_nested_metric(
                            metrics,
                            "inference_ms_per_sample"
                        ),

                    "train_peak_ram_abs_mb":
                        get_nested_metric(
                            metrics,
                            "train_peak_ram_abs_mb"
                        ),

                    "train_peak_ram_delta_mb":
                        get_nested_metric(
                            metrics,
                            "train_peak_ram_delta_mb"
                        ),

                    "hpo_peak_ram_abs_mb":
                        get_nested_metric(
                            metrics,
                            "hpo_peak_ram_abs_mb"
                        ),

                    "hpo_peak_ram_delta_mb":
                        get_nested_metric(
                            metrics,
                            "hpo_peak_ram_delta_mb"
                        )
                }


                records.append(record)


                # ------------------------------------------------
                # CONVERGENCE HISTORY
                # ------------------------------------------------

                history = metrics.get(
                    "convergence_history",
                    []
                )

                if isinstance(
                    history,
                    list
                ):

                    for point in history:

                        if (
                            isinstance(point, list)
                            and len(point) >= 2
                        ):

                            convergence_records.append(
                                {
                                    "run_id": run_id,
                                    "battery": battery,
                                    "model": model,
                                    "optimizer": optimizer,
                                    "evaluation": safe_float(
                                        point[0]
                                    ),
                                    "best_RMSE": safe_float(
                                        point[1]
                                    )
                                }
                            )


# Convert to DataFrames

df = pd.DataFrame(records)

base_df = pd.DataFrame(base_records)

convergence_df = pd.DataFrame(
    convergence_records
)


if df.empty:

    raise RuntimeError(
        "❌ No optimized results were extracted."
    )


# ============================================================
# 8. BASIC DATA CLEANING
# ============================================================

df = df[
    df["test_RMSE"].notna()
].copy()

base_df = base_df[
    base_df["test_RMSE"].notna()
].copy()


print(
    f"\n   Optimized records: {len(df)}"
)

print(
    f"   Baseline records:  {len(base_df)}"
)

print(
    f"   Batteries:         "
    f"{sorted(df['battery'].unique())}"
)

print(
    f"   Models:             "
    f"{sorted(df['model'].unique())}"
)

print(
    f"   Optimizers:         "
    f"{sorted(df['optimizer'].unique())}"
)


# ============================================================
# 9. ACTUAL OPTIMIZER ORDER
# ============================================================

optimizers = [
    x for x in OPTIMIZER_ORDER
    if x in df["optimizer"].unique()
]

models = [
    x for x in MODEL_ORDER
    if x in df["model"].unique()
]

batteries = sorted(
    df["battery"].unique()
)


print(
    "\n   Optimizer order:"
)

print(
    "   " + " → ".join(optimizers)
)


# ============================================================
# 10. SUMMARY STATISTICS
# ============================================================

summary = (
    df
    .groupby(
        ["battery", "model", "optimizer"],
        as_index=False
    )
    .agg(
        test_RMSE_mean=("test_RMSE", "mean"),
        test_RMSE_std=("test_RMSE", "std"),
        test_RMSE_min=("test_RMSE", "min"),
        test_RMSE_max=("test_RMSE", "max"),

        test_MAE_mean=("test_MAE", "mean"),
        test_R2_mean=("test_R2", "mean"),
        test_MAPE_mean=("test_MAPE", "mean"),

        hpo_time_mean=("hpo_time", "mean"),
        hpo_time_std=("hpo_time", "std"),

        training_time_mean=("training_time", "mean"),
        training_time_std=("training_time", "std"),

        model_size_mb_mean=("model_size_mb", "mean"),
        model_size_mb_std=("model_size_mb", "std"),

        inference_ms_mean=(
            "inference_ms_per_sample",
            "mean"
        ),

        hpo_ram_mean=(
            "hpo_peak_ram_delta_mb",
            "mean"
        ),

        train_ram_mean=(
            "train_peak_ram_delta_mb",
            "mean"
        ),

        n_runs=("run_id", "nunique")
    )
)


# ============================================================
# 11. CONFIDENCE INTERVALS
# ============================================================

ci_records = []

for keys, group in df.groupby(
    ["battery", "model", "optimizer"]
):

    values = group["test_RMSE"].dropna().values

    if len(values) >= 2:

        mean_val = np.mean(values)

        std_val = np.std(
            values,
            ddof=1
        )

        se = std_val / np.sqrt(
            len(values)
        )

        ci_low = (
            mean_val -
            1.96 * se
        )

        ci_high = (
            mean_val +
            1.96 * se
        )

    elif len(values) == 1:

        ci_low = values[0]
        ci_high = values[0]

    else:

        ci_low = np.nan
        ci_high = np.nan


    ci_records.append(
        {
            "battery": keys[0],
            "model": keys[1],
            "optimizer": keys[2],
            "RMSE_CI_lower": ci_low,
            "RMSE_CI_upper": ci_high
        }
    )


ci_df = pd.DataFrame(
    ci_records
)

summary = summary.merge(
    ci_df,
    on=[
        "battery",
        "model",
        "optimizer"
    ],
    how="left"
)


# ============================================================
# 12. BEST RMSE PER RUN
# ============================================================

best_per_run = (
    df
    .groupby("run_id")["test_RMSE"]
    .min()
    .sort_index()
)

best_rmse_per_run = (
    best_per_run.values
)


# ============================================================
# 13. FIGURE 1
#     RMSE BOXPLOT
# ============================================================

print("\n" + "-" * 80)
print("STEP 5 - GENERATING FIGURES")
print("-" * 80)

print("\n📊 Figure 1: RMSE Boxplot")

plt.figure(
    figsize=(12, 7)
)

box_data = [
    df.loc[
        df["optimizer"] == opt,
        "test_RMSE"
    ].dropna().values
    for opt in optimizers
]

box_data = [
    x for x in box_data
    if len(x) > 0
]

box_labels = [
    opt
    for opt in optimizers
    if len(
        df.loc[
            df["optimizer"] == opt,
            "test_RMSE"
        ].dropna()
    ) > 0
]

bp = plt.boxplot(
    box_data,
    labels=box_labels,
    patch_artist=True,
    showmeans=True,
    meanline=True
)

for patch in bp["boxes"]:
    patch.set_alpha(0.65)

plt.ylabel(
    "Test RMSE",
    fontsize=12,
    fontweight="bold"
)

plt.xlabel(
    "Optimization Algorithm",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Test RMSE Distribution Across 20 Independent Runs",
    fontsize=14,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.25
)

save_figure(
    "1_rmse_boxplot.png"
)


# ============================================================
# 14. FIGURE 2
#     BEST RMSE PER RUN
# ============================================================

print("\n📊 Figure 2: Best RMSE per Run")

if len(best_rmse_per_run) > 0:

    plt.figure(
        figsize=(14, 7)
    )

    run_numbers = list(
        best_per_run.index
    )

    plt.plot(
        run_numbers,
        best_rmse_per_run,
        "o-",
        linewidth=2,
        markersize=7,
        label="Best RMSE per Run"
    )

    if len(best_rmse_per_run) > 1:

        mean_rmse = np.mean(
            best_rmse_per_run
        )

        std_rmse = np.std(
            best_rmse_per_run,
            ddof=1
        )

        ci_low = (
            mean_rmse -
            1.96 * std_rmse /
            np.sqrt(
                len(best_rmse_per_run)
            )
        )

        ci_high = (
            mean_rmse +
            1.96 * std_rmse /
            np.sqrt(
                len(best_rmse_per_run)
            )
        )

        plt.axhline(
            mean_rmse,
            linestyle="--",
            linewidth=2,
            label=(
                f"Mean RMSE = "
                f"{mean_rmse:.5f}"
            )
        )

        plt.fill_between(
            run_numbers,
            ci_low,
            ci_high,
            alpha=0.18,
            label=(
                f"95% CI "
                f"[{ci_low:.5f}, "
                f"{ci_high:.5f}]"
            )
        )

    plt.xlabel(
        "Run Number",
        fontsize=12,
        fontweight="bold"
    )

    plt.ylabel(
        "Best Test RMSE",
        fontsize=12,
        fontweight="bold"
    )

    plt.title(
        "Best Test RMSE Across Independent Runs",
        fontsize=14,
        fontweight="bold"
    )

    plt.xticks(
        run_numbers
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.legend(
        fontsize=10
    )

    save_figure(
        "2_best_rmse_per_run.png"
    )


# ============================================================
# 15. FIGURE 3
#     VIOLIN PLOT
# ============================================================

print("\n📊 Figure 3: Optimizer Violin Plot")

plt.figure(
    figsize=(12, 7)
)

violin_data = [
    df.loc[
        df["optimizer"] == opt,
        "test_RMSE"
    ].dropna().values
    for opt in optimizers
]

valid_labels = [
    opt
    for opt, values in zip(
        optimizers,
        violin_data
    )
    if len(values) > 0
]

valid_data = [
    values
    for values in violin_data
    if len(values) > 0
]

if valid_data:

    vp = plt.violinplot(
        valid_data,
        positions=np.arange(
            1,
            len(valid_data) + 1
        ),
        showmeans=True,
        showmedians=True
    )

    for body in vp["bodies"]:
        body.set_alpha(0.6)

    plt.xticks(
        np.arange(
            1,
            len(valid_data) + 1
        ),
        valid_labels
    )

    plt.ylabel(
        "Test RMSE",
        fontsize=12,
        fontweight="bold"
    )

    plt.xlabel(
        "Optimization Algorithm",
        fontsize=12,
        fontweight="bold"
    )

    plt.title(
        "Test RMSE Distribution by Optimizer",
        fontsize=14,
        fontweight="bold"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    save_figure(
        "3_optimizer_violin_plot.png"
    )


# ============================================================
# 16. FIGURE 4
#     TRAINING TIME BOXPLOT
# ============================================================

print("\n📊 Figure 4: Training Time")

time_data = [
    df.loc[
        df["optimizer"] == opt,
        "training_time"
    ].dropna().values
    for opt in optimizers
]

time_labels = [
    opt
    for opt, values in zip(
        optimizers,
        time_data
    )
    if len(values) > 0
]

time_data = [
    values
    for values in time_data
    if len(values) > 0
]

if time_data:

    plt.figure(
        figsize=(12, 7)
    )

    bp_time = plt.boxplot(
        time_data,
        labels=time_labels,
        patch_artist=True,
        showmeans=True
    )

    for patch in bp_time["boxes"]:
        patch.set_alpha(0.65)

    plt.ylabel(
        "Training Time (s)",
        fontsize=12,
        fontweight="bold"
    )

    plt.xlabel(
        "Optimization Algorithm",
        fontsize=12,
        fontweight="bold"
    )

    plt.title(
        "Training Time Distribution Across Independent Runs",
        fontsize=14,
        fontweight="bold"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    save_figure(
        "4_training_time_boxplot.png"
    )


# ============================================================
# 17. FIGURE 5
#     RMSE VS TRAINING TIME
# ============================================================

print("\n📊 Figure 5: RMSE vs Training Time")

plt.figure(
    figsize=(12, 8)
)

for optimizer in optimizers:

    subset = df[
        df["optimizer"] == optimizer
    ]

    if subset.empty:
        continue

    x = subset[
        "training_time"
    ].mean()

    y = subset[
        "test_RMSE"
    ].mean()

    x_std = subset[
        "training_time"
    ].std()

    y_std = subset[
        "test_RMSE"
    ].std()

    if not np.isfinite(x):
        continue

    if not np.isfinite(y):
        continue

    if not np.isfinite(x_std):
        x_std = 0

    if not np.isfinite(y_std):
        y_std = 0

    plt.errorbar(
        x,
        y,
        xerr=x_std,
        yerr=y_std,
        fmt="o",
        markersize=11,
        capsize=5,
        linewidth=1.5,
        label=optimizer
    )

plt.xlabel(
    "Training Time (s)",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Test RMSE",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Test RMSE vs Training Time (Mean ± SD)",
    fontsize=14,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.25
)

plt.legend(
    fontsize=10
)

save_figure(
    "5_rmse_vs_time.png"
)


# ============================================================
# 18. FIGURE 6
#     RMSE HEATMAP
# ============================================================

print("\n📊 Figure 6: RMSE Heatmap")

heatmap_df = (
    df
    .pivot_table(
        index="run_id",
        columns="optimizer",
        values="test_RMSE",
        aggfunc="mean"
    )
)

heatmap_df = heatmap_df.reindex(
    columns=optimizers
)

plt.figure(
    figsize=(14, 8)
)

plt.imshow(
    heatmap_df.values,
    aspect="auto"
)

plt.colorbar(
    label="Test RMSE"
)

plt.xticks(
    np.arange(
        len(heatmap_df.columns)
    ),
    heatmap_df.columns
)

plt.yticks(
    np.arange(
        len(heatmap_df.index)
    ),
    [
        f"Run {x}"
        for x in heatmap_df.index
    ]
)

plt.xlabel(
    "Optimization Algorithm",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Run",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Test RMSE Heatmap Across Runs and Optimizers",
    fontsize=14,
    fontweight="bold"
)

# Numeric annotations
for i in range(
    heatmap_df.shape[0]
):

    for j in range(
        heatmap_df.shape[1]
    ):

        value = heatmap_df.iloc[i, j]

        if np.isfinite(value):

            plt.text(
                j,
                i,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=7
            )

save_figure(
    "6_heatmap.png"
)


# ============================================================
# 19. GREEN AI SCORE
# ============================================================
#
# Compatible with the FINAL aggregation logic:
#
# GreenAI Score =
# 1 - mean(
#   normalized RMSE,
#   normalized HPO time,
#   normalized model size
# )
#
# This follows the main code's Green AI calculation.
# ============================================================

green_base = (
    df
    .groupby(
        ["model", "optimizer"],
        as_index=False
    )
    .agg(
        test_RMSE_mean=(
            "test_RMSE",
            "mean"
        ),
        hpo_time_mean=(
            "hpo_time",
            "mean"
        ),
        model_size_mb_mean=(
            "model_size_mb",
            "mean"
        )
    )
)

green_base = green_base[
    green_base["optimizer"] != "Baseline"
].copy()


for column in [
    "test_RMSE_mean",
    "hpo_time_mean",
    "model_size_mb_mean"
]:

    min_val = green_base[
        column
    ].min()

    max_val = green_base[
        column
    ].max()

    if (
        np.isfinite(min_val)
        and
        np.isfinite(max_val)
        and
        max_val > min_val
    ):

        green_base[
            column + "_norm"
        ] = (
            green_base[column] -
            min_val
        ) / (
            max_val -
            min_val
        )

    else:

        green_base[
            column + "_norm"
        ] = 0.0


green_base[
    "GreenAI_Score"
] = 1 - (

    green_base[
        "test_RMSE_mean_norm"
    ]

    +

    green_base[
        "hpo_time_mean_norm"
    ]

    +

    green_base[
        "model_size_mb_mean_norm"
    ]

) / 3


# Overall optimizer score
green_optimizer = (
    green_base
    .groupby(
        "optimizer",
        as_index=False
    )
    .agg(
        GreenAI_Score=(
            "GreenAI_Score",
            "mean"
        )
    )
    .sort_values(
        "GreenAI_Score",
        ascending=False
    )
)


# ============================================================
# 20. FIGURE 7
#     GREEN AI SCORE
# ============================================================

print("\n📊 Figure 7: Green AI Score")

plt.figure(
    figsize=(12, 7)
)

plot_green = green_optimizer.copy()

plt.bar(
    plot_green["optimizer"],
    plot_green["GreenAI_Score"]
)

plt.ylabel(
    "Green AI Score",
    fontsize=12,
    fontweight="bold"
)

plt.xlabel(
    "Optimization Algorithm",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Green AI Score Across Optimization Algorithms",
    fontsize=14,
    fontweight="bold"
)

plt.grid(
    axis="y",
    alpha=0.25
)

for i, value in enumerate(
    plot_green["GreenAI_Score"]
):

    plt.text(
        i,
        value,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

save_figure(
    "7_green_ai_score.png"
)


# ============================================================
# 21. PARETO ANALYSIS
# ============================================================
#
# Objectives:
#   1. Test RMSE      ↓
#   2. HPO time       ↓
#   3. Model size     ↓
#
# A point is Pareto-optimal if no other point is equal/better
# in all three objectives and strictly better in at least one.
# ============================================================

pareto_df = green_base.copy()

pareto_flags = []

points = pareto_df[
    [
        "test_RMSE_mean",
        "hpo_time_mean",
        "model_size_mb_mean"
    ]
].values

for i in range(
    len(points)
):

    dominated = False

    for j in range(
        len(points)
    ):

        if i == j:
            continue

        p = points[i]
        q = points[j]

        if (
            q[0] <= p[0]
            and
            q[1] <= p[1]
            and
            q[2] <= p[2]
            and
            (
                q[0] < p[0]
                or
                q[1] < p[1]
                or
                q[2] < p[2]
            )
        ):

            dominated = True
            break

    pareto_flags.append(
        not dominated
    )


pareto_df[
    "Pareto_Optimal"
] = pareto_flags


# ============================================================
# 22. FIGURE 8
#     PARETO FRONT
# ============================================================

print("\n📊 Figure 8: Pareto Front")

plt.figure(
    figsize=(12, 8)
)

for optimizer in sorted(
    pareto_df["optimizer"].unique()
):

    subset = pareto_df[
        pareto_df["optimizer"] == optimizer
    ]

    plt.scatter(
        subset["hpo_time_mean"],
        subset["test_RMSE_mean"],
        s=90,
        alpha=0.75,
        label=optimizer
    )


# Highlight Pareto solutions
pareto_only = pareto_df[
    pareto_df["Pareto_Optimal"]
]

if not pareto_only.empty:

    plt.scatter(
        pareto_only["hpo_time_mean"],
        pareto_only["test_RMSE_mean"],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Pareto-optimal"
    )

    for _, row in pareto_only.iterrows():

        plt.annotate(
            f"{row['model']}-{row['optimizer']}",
            (
                row["hpo_time_mean"],
                row["test_RMSE_mean"]
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8
        )


plt.xlabel(
    "Mean HPO Time (s)",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Mean Test RMSE",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Pareto Analysis: Accuracy vs Computational Cost",
    fontsize=14,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.25
)

plt.legend(
    fontsize=9
)

save_figure(
    "8_pareto_front.png"
)


# ============================================================
# 23. FIGURE 9
#     RMSE VS HPO TIME
# ============================================================

print("\n📊 Figure 9: RMSE vs HPO Time")

plt.figure(
    figsize=(12, 8)
)

for optimizer in optimizers:

    subset = green_base[
        green_base["optimizer"] == optimizer
    ]

    if subset.empty:
        continue

    plt.scatter(
        subset["hpo_time_mean"],
        subset["test_RMSE_mean"],
        s=100,
        alpha=0.8,
        label=optimizer
    )


plt.xlabel(
    "Mean HPO Time (s)",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Mean Test RMSE",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Accuracy–Computation Trade-off",
    fontsize=14,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.25
)

plt.legend(
    fontsize=9
)

save_figure(
    "9_rmse_vs_hpo_time.png"
)


# ============================================================
# 24. FIGURE 10
#     RMSE VS MODEL SIZE
# ============================================================

print("\n📊 Figure 10: RMSE vs Model Size")

plt.figure(
    figsize=(12, 8)
)

for optimizer in optimizers:

    subset = green_base[
        green_base["optimizer"] == optimizer
    ]

    if subset.empty:
        continue

    plt.scatter(
        subset["model_size_mb_mean"],
        subset["test_RMSE_mean"],
        s=100,
        alpha=0.8,
        label=optimizer
    )


plt.xlabel(
    "Mean Model Size (MB)",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Mean Test RMSE",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Accuracy vs Model Footprint",
    fontsize=14,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.25
)

plt.legend(
    fontsize=9
)

save_figure(
    "10_rmse_vs_model_size.png"
)


# ============================================================
# 25. FIGURE 11
#     RESOURCE FOOTPRINT
# ============================================================

print("\n📊 Figure 11: Resource Footprint")

resource_summary = (
    df
    .groupby("optimizer", as_index=False)
    .agg(
        hpo_time=("hpo_time", "mean"),
        model_size=("model_size_mb", "mean"),
        hpo_ram=("hpo_peak_ram_delta_mb", "mean")
    )
)

resource_summary = resource_summary[
    resource_summary["optimizer"].isin(
        optimizers
    )
]

# Normalize each resource
for col in [
    "hpo_time",
    "model_size",
    "hpo_ram"
]:

    min_val = resource_summary[
        col
    ].min()

    max_val = resource_summary[
        col
    ].max()

    if (
        np.isfinite(min_val)
        and
        np.isfinite(max_val)
        and
        max_val > min_val
    ):

        resource_summary[
            col + "_norm"
        ] = (
            resource_summary[col] -
            min_val
        ) / (
            max_val -
            min_val
        )

    else:

        resource_summary[
            col + "_norm"
        ] = 0.0


plt.figure(
    figsize=(14, 8)
)

x = np.arange(
    len(resource_summary)
)

width = 0.25

plt.bar(
    x - width,
    resource_summary[
        "hpo_time_norm"
    ],
    width,
    label="HPO Time"
)

plt.bar(
    x,
    resource_summary[
        "model_size_norm"
    ],
    width,
    label="Model Size"
)

plt.bar(
    x + width,
    resource_summary[
        "hpo_ram_norm"
    ],
    width,
    label="HPO RAM"
)

plt.xticks(
    x,
    resource_summary[
        "optimizer"
    ]
)

plt.ylabel(
    "Normalized Resource Consumption",
    fontsize=12,
    fontweight="bold"
)

plt.xlabel(
    "Optimization Algorithm",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Normalized Computational Resource Footprint",
    fontsize=14,
    fontweight="bold"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend()

save_figure(
    "11_resource_footprint.png"
)


# ============================================================
# 26. FIGURE 12
#     CONVERGENCE CURVES
# ============================================================

print("\n📊 Figure 12: Convergence Curves")

if not convergence_df.empty:

    plt.figure(
        figsize=(14, 8)
    )

    # Average convergence across runs,
    # batteries and models
    convergence_summary = (
        convergence_df
        .groupby(
            ["optimizer", "evaluation"],
            as_index=False
        )
        .agg(
            best_RMSE=(
                "best_RMSE",
                "mean"
            )
        )
    )

    for optimizer in optimizers:

        subset = convergence_summary[
            convergence_summary["optimizer"]
            == optimizer
        ]

        if subset.empty:
            continue

        subset = subset.sort_values(
            "evaluation"
        )

        plt.plot(
            subset["evaluation"],
            subset["best_RMSE"],
            linewidth=2,
            label=optimizer
        )

    plt.xlabel(
        "Function Evaluations",
        fontsize=12,
        fontweight="bold"
    )

    plt.ylabel(
        "Best Validation RMSE",
        fontsize=12,
        fontweight="bold"
    )

    plt.title(
        "Average Optimization Convergence",
        fontsize=14,
        fontweight="bold"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.legend(
        fontsize=9
    )

    save_figure(
        "12_convergence_curves.png"
    )

else:

    print(
        "   ⚠️ convergence_history not found."
    )


# ============================================================
# 27. STATISTICAL TESTS
# ============================================================

print("\n" + "-" * 80)
print("STEP 6 - STATISTICAL TESTS")
print("-" * 80)

stats_records = []


# ------------------------------------------------------------
# Paired Wilcoxon:
# RandomSearch vs each optimizer
# within each battery/model combination.
# ------------------------------------------------------------

for battery in batteries:

    for model in models:

        rs = df[
            (df["battery"] == battery)
            &
            (df["model"] == model)
            &
            (df["optimizer"] == "RandomSearch")
        ][
            ["run_id", "test_RMSE"]
        ].dropna()


        if len(rs) < 2:
            continue


        for optimizer in [
            "PSO",
            "WOA",
            "GA",
            "GWO",
            "Bayesian"
        ]:

            opt = df[
                (df["battery"] == battery)
                &
                (df["model"] == model)
                &
                (df["optimizer"] == optimizer)
            ][
                ["run_id", "test_RMSE"]
            ].dropna()


            if len(opt) < 2:
                continue


            merged = rs.merge(
                opt,
                on="run_id",
                suffixes=(
                    "_random",
                    "_optimizer"
                )
            )


            if len(merged) < 2:
                continue


            x = merged[
                "test_RMSE_random"
            ].values

            y = merged[
                "test_RMSE_optimizer"
            ].values


            try:

                statistic, p_value = wilcoxon(
                    x,
                    y
                )

                effect = rank_biserial(
                    x,
                    y
                )

                stats_records.append(
                    {
                        "battery": battery,
                        "model": model,
                        "optimizer": optimizer,
                        "comparison":
                            f"{optimizer}_vs_RandomSearch",
                        "n_pairs": len(merged),
                        "wilcoxon_statistic":
                            float(statistic),
                        "p_value":
                            float(p_value),
                        "rank_biserial":
                            float(effect),
                        "significant_raw":
                            bool(
                                p_value < 0.05
                            )
                    }
                )

            except Exception as e:

                print(
                    f"   ⚠️ Wilcoxon failed: "
                    f"{battery} | "
                    f"{model} | "
                    f"{optimizer} | {e}"
                )


stats_df = pd.DataFrame(
    stats_records
)


if not stats_df.empty:

    stats_df[
        "holm_p_value"
    ] = holm_correction(
        stats_df["p_value"].values
    )

    stats_df[
        "significant_holm"
    ] = (
        stats_df["holm_p_value"]
        < 0.05
    )

    print(
        f"   ✅ Statistical comparisons: "
        f"{len(stats_df)}"
    )

else:

    print(
        "   ⚠️ No statistical comparison could be calculated."
    )


# ============================================================
# 28. RMSE IMPROVEMENT OVER RANDOM SEARCH
# ============================================================

random_baseline = (
    df[
        df["optimizer"] == "RandomSearch"
    ]
    .groupby(
        ["battery", "model"],
        as_index=False
    )
    .agg(
        RandomSearch_RMSE=(
            "test_RMSE",
            "mean"
        )
    )
)

summary = summary.merge(
    random_baseline,
    on=[
        "battery",
        "model"
    ],
    how="left"
)

summary[
    "RMSE_Improvement_vs_RandomSearch_pct"
] = (
    100
    *
    (
        summary["RandomSearch_RMSE"]
        -
        summary["test_RMSE_mean"]
    )
    /
    summary["RandomSearch_RMSE"]
)


# ============================================================
# 29. OVERALL PARETO FLAGS
# ============================================================

summary = summary.merge(
    pareto_df[
        [
            "model",
            "optimizer",
            "Pareto_Optimal"
        ]
    ],
    on=[
        "model",
        "optimizer"
    ],
    how="left"
)


# ============================================================
# 30. SAVE CSV FILES
# ============================================================

print("\n" + "-" * 80)
print("STEP 7 - SAVING CSV FILES")
print("-" * 80)

raw_csv = os.path.join(
    OUTPUT_DIR,
    "all_optimized_results_long_format.csv"
)

df.to_csv(
    raw_csv,
    index=False
)

print(
    f"   ✅ {os.path.basename(raw_csv)}"
)


base_csv = os.path.join(
    OUTPUT_DIR,
    "baseline_results.csv"
)

base_df.to_csv(
    base_csv,
    index=False
)

print(
    f"   ✅ {os.path.basename(base_csv)}"
)


summary_csv = os.path.join(
    OUTPUT_DIR,
    "final_visualization_summary.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)

print(
    f"   ✅ {os.path.basename(summary_csv)}"
)


green_csv = os.path.join(
    OUTPUT_DIR,
    "green_ai_scores.csv"
)

green_base.to_csv(
    green_csv,
    index=False
)

print(
    f"   ✅ {os.path.basename(green_csv)}"
)


pareto_csv = os.path.join(
    OUTPUT_DIR,
    "pareto_analysis.csv"
)

pareto_df.to_csv(
    pareto_csv,
    index=False
)

print(
    f"   ✅ {os.path.basename(pareto_csv)}"
)


if not stats_df.empty:

    stats_csv = os.path.join(
        OUTPUT_DIR,
        "statistical_tests.csv"
    )

    stats_df.to_csv(
        stats_csv,
        index=False
    )

    print(
        f"   ✅ {os.path.basename(stats_csv)}"
    )


# ============================================================
# 31. EXCEL WORKBOOK
# ============================================================

print("\n" + "-" * 80)
print("STEP 8 - CREATING EXCEL WORKBOOK")
print("-" * 80)

excel_path = os.path.join(
    OUTPUT_DIR,
    "SCI_Makale_Tablolari_FINAL.xlsx"
)


with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:


    # --------------------------------------------------------
    # Sheet 1
    # --------------------------------------------------------

    best_table = pd.DataFrame(
        {
            "Run": list(
                best_per_run.index
            ),
            "Best_RMSE": list(
                best_per_run.values
            )
        }
    )

    best_table.to_excel(
        writer,
        sheet_name="01_Best_RMSE_Per_Run",
        index=False
    )


    # --------------------------------------------------------
    # Sheet 2
    # --------------------------------------------------------

    optimizer_stats = (
        df
        .groupby(
            "optimizer",
            as_index=False
        )
        .agg(
            Mean_RMSE=(
                "test_RMSE",
                "mean"
            ),
            SD_RMSE=(
                "test_RMSE",
                "std"
            ),
            Min_RMSE=(
                "test_RMSE",
                "min"
            ),
            Max_RMSE=(
                "test_RMSE",
                "max"
            ),
            Mean_MAE=(
                "test_MAE",
                "mean"
            ),
            Mean_R2=(
                "test_R2",
                "mean"
            ),
            Mean_MAPE=(
                "test_MAPE",
                "mean"
            ),
            Mean_HPO_Time_s=(
                "hpo_time",
                "mean"
            ),
            Mean_Training_Time_s=(
                "training_time",
                "mean"
            ),
            Mean_Model_Size_MB=(
                "model_size_mb",
                "mean"
            ),
            Mean_Inference_ms=(
                "inference_ms_per_sample",
                "mean"
            )
        )
    )

    optimizer_stats.to_excel(
        writer,
        sheet_name="02_Optimizer_Statistics",
        index=False
    )


    # --------------------------------------------------------
    # Sheet 3
    # --------------------------------------------------------

    run_optimizer_table = (
        df
        .groupby(
            ["run_id", "optimizer"],
            as_index=False
        )
        .agg(
            Mean_RMSE=(
                "test_RMSE",
                "mean"
            ),
            Mean_MAE=(
                "test_MAE",
                "mean"
            ),
            Mean_R2=(
                "test_R2",
                "mean"
            ),
            Mean_HPO_Time_s=(
                "hpo_time",
                "mean"
            ),
            Mean_Training_Time_s=(
                "training_time",
                "mean"
            ),
            Mean_Model_Size_MB=(
                "model_size_mb",
                "mean"
            )
        )
    )

    run_optimizer_table.to_excel(
        writer,
        sheet_name="03_Run_Optimizer",
        index=False
    )


    # --------------------------------------------------------
    # Sheet 4
    # --------------------------------------------------------

    battery_table = (
        df
        .groupby(
            ["battery", "optimizer"],
            as_index=False
        )
        .agg(
            Mean_RMSE=(
                "test_RMSE",
                "mean"
            ),
            SD_RMSE=(
                "test_RMSE",
                "std"
            ),
            Mean_MAE=(
                "test_MAE",
                "mean"
            ),
            Mean_R2=(
                "test_R2",
                "mean"
            ),
            Mean_HPO_Time_s=(
                "hpo_time",
                "mean"
            ),
            Mean_Model_Size_MB=(
                "model_size_mb",
                "mean"
            )
        )
    )

    battery_table.to_excel(
        writer,
        sheet_name="04_Battery_Performance",
        index=False
    )


    # --------------------------------------------------------
    # Sheet 5
    # --------------------------------------------------------

    stability_metrics = pd.DataFrame(
        {
            "Metric": [
                "Number of Runs",
                "Mean Best RMSE",
                "SD Best RMSE",
                "Minimum Best RMSE",
                "Maximum Best RMSE",
                "95% CI Lower",
                "95% CI Upper",
                "Relative SD (%)"
            ],

            "Value": [
                len(best_rmse_per_run),

                np.mean(
                    best_rmse_per_run
                ),

                np.std(
                    best_rmse_per_run,
                    ddof=1
                ) if len(
                    best_rmse_per_run
                ) > 1 else 0,

                np.min(
                    best_rmse_per_run
                ),

                np.max(
                    best_rmse_per_run
                ),

                (
                    np.mean(
                        best_rmse_per_run
                    )
                    -
                    1.96 *
                    np.std(
                        best_rmse_per_run,
                        ddof=1
                    )
                    /
                    np.sqrt(
                        len(
                            best_rmse_per_run
                        )
                    )
                ) if len(
                    best_rmse_per_run
                ) > 1 else np.nan,

                (
                    np.mean(
                        best_rmse_per_run
                    )
                    +
                    1.96 *
                    np.std(
                        best_rmse_per_run,
                        ddof=1
                    )
                    /
                    np.sqrt(
                        len(
                            best_rmse_per_run
                        )
                    )
                ) if len(
                    best_rmse_per_run
                ) > 1 else np.nan,

                (
                    100
                    *
                    np.std(
                        best_rmse_per_run,
                        ddof=1
                    )
                    /
                    np.mean(
                        best_rmse_per_run
                    )
                ) if len(
                    best_rmse_per_run
                ) > 1 else np.nan
            ]
        }
    )

    stability_metrics.to_excel(
        writer,
        sheet_name="05_Stability",
        index=False
    )


    # --------------------------------------------------------
    # Sheet 6
    # --------------------------------------------------------

    training_stats = (
        df
        .groupby(
            "optimizer",
            as_index=False
        )
        .agg(
            Mean_Training_Time_s=(
                "training_time",
                "mean"
            ),
            SD_Training_Time_s=(
                "training_time",
                "std"
            ),
            Min_Training_Time_s=(
                "training_time",
                "min"
            ),
            Max_Training_Time_s=(
                "training_time",
                "max"
            ),
            Mean_HPO_Time_s=(
                "hpo_time",
                "mean"
            )
        )
    )

    training_stats.to_excel(
        writer,
        sheet_name="06_Computational_Time",
        index=False
    )


    # --------------------------------------------------------
    # Sheet 7
    # --------------------------------------------------------

    if not stats_df.empty:

        stats_df.to_excel(
            writer,
            sheet_name="07_Wilcoxon_Holm",
            index=False
        )


    # --------------------------------------------------------
    # Sheet 8
    # --------------------------------------------------------

    green_base.to_excel(
        writer,
        sheet_name="08_Green_AI",
        index=False
    )


    # --------------------------------------------------------
    # Sheet 9
    # --------------------------------------------------------

    pareto_df.to_excel(
        writer,
        sheet_name="09_Pareto_Analysis",
        index=False
    )


    # --------------------------------------------------------
    # Sheet 10
    # --------------------------------------------------------

    convergence_df.to_excel(
        writer,
        sheet_name="10_Convergence_Data",
        index=False
    )


    # --------------------------------------------------------
    # Sheet 11
    # --------------------------------------------------------

    summary.to_excel(
        writer,
        sheet_name="11_Final_Summary",
        index=False
    )


print(
    f"\n   ✅ Excel created:"
)

print(
    f"   {excel_path}"
)


# ============================================================
# 32. LATEX TABLE
# ============================================================

print("\n" + "-" * 80)
print("STEP 9 - CREATING LATEX TABLE")
print("-" * 80)

latex_path = os.path.join(
    OUTPUT_DIR,
    "optimizer_comparison_table.tex"
)


latex_df = (
    optimizer_stats
    .copy()
)


with open(
    latex_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\\begin{table}[htbp]\n"
    )

    f.write(
        "\\centering\n"
    )

    f.write(
        "\\caption{Comparison of optimization algorithms "
        "across 20 independent runs.}\n"
    )

    f.write(
        "\\label{tab:optimizer_comparison}\n"
    )

    f.write(
        "\\begin{tabular}{lrrrrrr}\n"
    )

    f.write(
        "\\hline\n"
    )

    f.write(
        "Optimizer & Mean RMSE & SD & "
        "Mean MAE & Mean $R^2$ & "
        "HPO Time (s) & Model Size (MB) \\\\\n"
    )

    f.write(
        "\\hline\n"
    )


    for _, row in latex_df.iterrows():

        optimizer = str(
            row["optimizer"]
        )

        mean_rmse = row[
            "Mean_RMSE"
        ]

        sd_rmse = row[
            "SD_RMSE"
        ]

        mean_mae = row[
            "Mean_MAE"
        ]

        mean_r2 = row[
            "Mean_R2"
        ]

        hpo_time = row[
            "Mean_HPO_Time_s"
        ]

        model_size = row[
            "Mean_Model_Size_MB"
        ]


        def fmt(value):

            if pd.isna(value):
                return "--"

            return f"{value:.5f}"


        f.write(
            f"{optimizer} & "
            f"{fmt(mean_rmse)} & "
            f"{fmt(sd_rmse)} & "
            f"{fmt(mean_mae)} & "
            f"{fmt(mean_r2)} & "
            f"{fmt(hpo_time)} & "
            f"{fmt(model_size)} "
            "\\\\\n"
        )


    f.write(
        "\\hline\n"
    )

    f.write(
        "\\end{tabular}\n"
    )

    f.write(
        "\\end{table}\n"
    )


print(
    f"   ✅ {latex_path}"
)


# ============================================================
# 33. TEXT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL ANALYSIS SUMMARY")
print("=" * 80)


print(
    f"\n📌 Existing runs analyzed: "
    f"{len(loaded_runs)}"
)

print(
    f"📌 Optimized records analyzed: "
    f"{len(df)}"
)

print(
    f"📌 Batteries: "
    f"{', '.join(batteries)}"
)

print(
    f"📌 Models: "
    f"{', '.join(models)}"
)

print(
    f"📌 Optimizers: "
    f"{', '.join(optimizers)}"
)


# Best optimizer by mean RMSE

overall_optimizer = (
    df
    .groupby(
        "optimizer"
    )["test_RMSE"]
    .mean()
    .sort_values()
)


print(
    "\n🏆 Mean RMSE ranking:"
)

for rank, (
    optimizer,
    value
) in enumerate(
    overall_optimizer.items(),
    start=1
):

    print(
        f"   {rank}. "
        f"{optimizer}: "
        f"{value:.6f}"
    )


# Best Green AI

if not green_optimizer.empty:

    best_green = (
        green_optimizer.iloc[0]
    )

    print(
        "\n🌱 Best Green AI optimizer:"
    )

    print(
        f"   {best_green['optimizer']} "
        f"→ "
        f"{best_green['GreenAI_Score']:.6f}"
    )


# Pareto solutions

pareto_solutions = pareto_df[
    pareto_df["Pareto_Optimal"]
].copy()


print(
    "\n🎯 Pareto-optimal solutions:"
)

if pareto_solutions.empty:

    print(
        "   None detected."
    )

else:

    for _, row in pareto_solutions.iterrows():

        print(
            f"   {row['model']} + "
            f"{row['optimizer']} | "
            f"RMSE={row['test_RMSE_mean']:.6f} | "
            f"HPO={row['hpo_time_mean']:.3f}s | "
            f"Size={row['model_size_mb_mean']:.4f} MB"
        )


# Best run

if len(best_per_run) > 0:

    best_run_id = (
        best_per_run.idxmin()
    )

    best_run_value = (
        best_per_run.min()
    )

    print(
        "\n🥇 Best individual run:"
    )

    print(
        f"   Run {best_run_id:02d}"
        f" → "
        f"RMSE={best_run_value:.6f}"
    )


# ============================================================
# 34. OUTPUT INVENTORY
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

output_files = sorted(
    os.listdir(
        OUTPUT_DIR
    )
)

for filename in output_files:

    full_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.isfile(full_path):

        size_kb = (
            os.path.getsize(
                full_path
            )
            /
            1024
        )

        print(
            f"   📄 {filename:<50} "
            f"{size_kb:>10.1f} KB"
        )


# ============================================================
# 35. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETED")
print("=" * 80)

print(
    "\n❗ IMPORTANT:"
)

print(
    "Bu script hiçbir model eğitmedi."
)

print(
    "Bu script hiçbir optimizer çalıştırmadı."
)

print(
    "Bu script yeni run oluşturmadı."
)

print(
    "Mevcut run_01 ... run_20/results.json "
    "dosyaları doğrudan analiz edildi."
)

print(
    "\n📁 Tüm yeni çıktıların bulunduğu klasör:"
)

print(
    OUTPUT_DIR
)

print(
    "\n🎨 Figures:"
)

for i in range(1, 13):

    print(
        f"   {i:02d}. Figure"
    )

print(
    "\n📊 Excel:"
)

print(
    "   SCI_Makale_Tablolari_FINAL.xlsx"
)

print(
    "\n📄 LaTeX:"
)

print(
    "   optimizer_comparison_table.tex"
)

print(
    "\n" + "=" * 80
)