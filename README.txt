# Green AI-Driven Pareto Optimal Framework

A reproducible computational framework for battery health estimation,
metaheuristic optimization, Green AI evaluation, computational-cost
analysis, stability analysis, and Pareto-optimal solution identification.

---

## 1. Overview

This repository contains the Python source codes and experimental
results associated with the research study:

**Green AI-Driven Pareto Optimal Framework for Battery Health
Estimation and Optimization**

The purpose of this repository is to provide computational
transparency and reproducibility for the optimization experiments,
Green AI evaluation, stability analysis, and Pareto-optimality
assessment reported in the associated research.

The repository contains:

- Python source codes used in the experiments
- Experimental results from 20 independent optimization runs
- JSON-based result files
- RAM usage measurements
- Stability analysis results
- Visualization outputs
- Python package requirements
- Reproducibility instructions

The source codes provided in this repository correspond to the
computational versions used during the study.

---

# 2. Research Framework

The framework evaluates different optimization strategies according
to both predictive performance and computational efficiency.

The analysis considers multiple dimensions of computational
performance, including:

- prediction error,
- model performance,
- computational time,
- memory consumption,
- optimization stability,
- and Pareto optimality.

The framework therefore aims to identify optimization strategies that
provide a suitable balance between predictive performance and
computational cost.

---

# 3. Optimization Algorithms

The framework includes the following optimization approaches:

- Genetic Algorithm (GA)
- Whale Optimization Algorithm (WOA)
- Grey Wolf Optimizer (GWO)
- Particle Swarm Optimization (PSO)
- Bayesian Optimization

These algorithms are evaluated using multiple independent runs to
assess both performance and stability.

---

# 4. Evaluation Metrics

The computational experiments consider the following metrics:

### Prediction Performance

- RMSE
- MAE
- R²

### Computational Efficiency

- Training time
- RAM / memory usage
- Computational cost

### Optimization Robustness

- Results across independent runs
- Stability analysis
- Distribution of optimization results

### Multi-objective Analysis

- Pareto dominance
- Pareto-optimal solutions
- Trade-offs between prediction performance and computational cost

---

# 5. Repository Structure

The repository is organized as follows:

```text
Green_AI_Driven_Pareto_Optimal_Framework/
│
├── Optimization.py
├── kaynak_optimizasyon.py
├── Pareto Optimal.py
├── JSON Grafik.py
├── Json Excel.py
│
├── green_ai_multiple_runs/
│   │
│   ├── all_runs_summary.json
│   ├── experiment_results.json
│   ├── RAM_Usage_Report.xlsx
│   ├── stability_analysis.json
│   ├── stability_analysis_boxplot.png
│   │
│   ├── run_01/
│   ├── run_02/
│   ├── run_03/
│   ├── ...
│   ├── run_20/
│   │
│   ├── Runs json/
│   └── stability_visualizations/
│
├── requirements.txt
├── README.md
└── LICENSE