# Green AI-Driven Pareto Optimal Framework

A reproducible computational framework for battery State of Health (SOH) estimation, metaheuristic optimization, Green AI evaluation, computational-cost analysis, stability assessment, and Pareto-optimal solution identification.

---

## 1. Overview

This repository contains the Python source codes, experimental results, and supporting computational data associated with the research study:

**Green AI-Driven Pareto Optimal Framework for Battery Health Estimation and Optimization**

The primary purpose of this repository is to provide computational transparency and reproducibility for the optimization experiments, computational-efficiency assessment, Green AI evaluation, stability analysis, and Pareto-optimality analysis presented in the associated research.

The repository provides:

- Python source codes used in the computational workflow
- Experimental results from 20 independent optimization runs
- Aggregated JSON result files
- Individual optimization-run results
- RAM/memory usage measurements
- Stability-analysis results
- Visualization outputs
- Python package requirements
- Detailed instructions for reproducing the analyses

The source codes are provided in the form used during the research workflow in order to maintain consistency with the reported computational results.

---

# 2. Research Framework

The framework evaluates optimization strategies according to both predictive performance and computational efficiency.

Instead of considering prediction accuracy alone, the framework evaluates multiple dimensions of computational performance, including:

- prediction error,
- model performance,
- computational time,
- memory consumption,
- optimization stability,
- computational efficiency,
- and Pareto optimality.

The overall objective is to identify optimization strategies that provide an appropriate balance between predictive performance and computational cost.

---

# 3. Optimization Algorithms

The framework includes the following optimization approaches:

- Genetic Algorithm (GA)
- Whale Optimization Algorithm (WOA)
- Grey Wolf Optimizer (GWO)
- Particle Swarm Optimization (PSO)
- Bayesian Optimization

The optimization algorithms are evaluated using multiple independent runs in order to assess both their performance and stability.

---

# 4. Evaluation Metrics

The computational experiments consider multiple performance and computational-efficiency indicators.

## 4.1 Prediction Performance

The predictive performance assessment includes:

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

## 4.2 Computational Efficiency

The computational assessment includes:

- Training/optimization time
- RAM/memory usage
- Storage requirements
- Computational cost

## 4.3 Optimization Robustness

The robustness of the optimization algorithms is evaluated using:

- Independent optimization runs
- Run-to-run variability
- Statistical distributions
- Stability analysis

## 4.4 Multi-Objective Analysis

The framework uses Pareto analysis to identify nondominated solutions considering multiple objectives simultaneously.

The Pareto analysis can consider objectives such as:

- RMSE
- Computational time
- Memory usage

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