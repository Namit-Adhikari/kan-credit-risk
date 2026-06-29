# Project Context: KAN for Interpretable Credit Risk Modeling

**Document Purpose:** This file provides full technical context for AI coding agents (Claude Code, Cursor, Copilot, etc.) assisting on this project. Read in full before generating code, suggesting architecture, or modifying the pipeline.

---

## 1. Project Identity

| Field                   | Value                                                                                                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project Title**       | KAN for Interpretable Credit Risk Modeling                                                                                                                                                                          |
| **Researcher**          | B.Tech AI student, 4th Semester, Kathmandu University, Nepal                                                                                                                                                        |
| **Academic Context**    | Numerical Methods course (AIMA 203) — research presentation + practical evaluation (20/50 marks if published)                                                                                                       |
| **Research Domain**     | Financial ML — Credit Risk Scoring                                                                                                                                                                                  |
| **Core Novelty**        | Kolmogorov-Arnold Networks (KAN) applied to credit default prediction, benchmarked against standard ML/DL baselines, with comparative interpretability analysis (KAN native splines vs. SHAP/LIME post-hoc methods) |
| **Target Outcome**      | Publishable paper (arXiv minimum, journal stretch goal) + portfolio-grade GitHub repository                                                                                                                         |
| **Compute Environment** | Kaggle Notebooks — **Free Tier** (critical constraint, see §4)                                                                                                                                                      |

---

## 2. Research Motivation & Positioning

- **NM Syllabus alignment:** B-Splines (Unit 5.3), Interpolation (Unit 4), Least Squares (Unit 5.1), Error Analysis (Unit 1) — KAN's edge-wise learnable spline activations are the direct numerical-methods tie-in.
- **Research gap:** KANs (Liu et al., 2024) are new (post-2024); limited application papers exist in tabular financial risk modeling. Most KAN papers focus on physics/function-fitting, not finance.
- **Why interpretability matters financially:** Credit scoring is regulated (Basel III, SR 11-7 style requirements) — models must explain _why_ a decision was made. KAN's spline functions are **inherently visualizable per feature**, unlike black-box MLPs which need bolted-on tools (SHAP/LIME).
- **Core thesis to prove/disprove:**
  > "KANs achieve comparable or better predictive performance than gradient-boosted trees and MLPs on credit default prediction, while providing more direct, mathematically grounded interpretability than post-hoc explainability methods."

---

## 3. Dataset: Home Credit Default Risk (Kaggle)

**Source:** <https://www.kaggle.com/c/home-credit-default-risk>
**Size:** ~2.68 GB total across all files
**Decision:** Using **ALL files** (upgraded from initial application-only scope) for richer feature engineering and a stronger publication-grade result.

### 3.1 File Inventory & Roles

| File                                 | Granularity                                         | Role in Pipeline                                                                                                         | Join Key                      |
| ------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| `application_train.csv`              | 1 row per loan applicant                            | **Core table.** Contains `TARGET` (binary: 1=default, 0=repaid) + applicant demographic/financial features               | `SK_ID_CURR`                  |
| `application_test.csv`               | 1 row per loan applicant                            | Same schema as train, no `TARGET` — used for held-out evaluation / Kaggle leaderboard submission (optional)              | `SK_ID_CURR`                  |
| `bureau.csv`                         | 1 row per client's prior loan at OTHER institutions | External credit bureau history — aggregate into per-client features (count, mean, max of credit amounts, overdue status) | `SK_ID_CURR` → `SK_ID_BUREAU` |
| `bureau_balance.csv`                 | 1 row per month per bureau loan                     | Monthly repayment status of bureau loans — aggregate before joining to `bureau.csv`                                      | `SK_ID_BUREAU`                |
| `previous_application.csv`           | 1 row per client's prior Home Credit application    | Internal loan history with Home Credit itself — aggregate per client (approval rate, average amount, etc.)               | `SK_ID_CURR` → `SK_ID_PREV`   |
| `POS_CASH_balance.csv`               | 1 row per month per previous POS/cash loan          | Monthly balance/status of point-of-sale and cash loans — aggregate before joining                                        | `SK_ID_PREV`                  |
| `installments_payments.csv`          | 1 row per installment payment                       | Payment punctuality/amount history — aggregate (late payment count, avg delay days)                                      | `SK_ID_PREV`                  |
| `credit_card_balance.csv`            | 1 row per month per credit card                     | Monthly credit card utilization/balance — aggregate (avg utilization, max balance)                                       | `SK_ID_PREV`                  |
| `HomeCredit_columns_description.csv` | N/A                                                 | **Reference only** — data dictionary, not used in pipeline                                                               | —                             |
| `sample_submission.csv`              | N/A                                                 | **Reference only** — Kaggle submission format template                                                                   | —                             |

### 3.2 Relational Schema (Join Order)

```text
application_{train|test}.csv  (SK_ID_CURR = primary entity)
│
├── bureau.csv  (via SK_ID_CURR)
│   └── bureau_balance.csv  (via SK_ID_BUREAU)
│
└── previous_application.csv  (via SK_ID_CURR)
    ├── POS_CASH_balance.csv  (via SK_ID_PREV)
    ├── installments_payments.csv  (via SK_ID_PREV)
    └── credit_card_balance.csv  (via SK_ID_PREV)
```

**Aggregation strategy:** Bottom-up. Aggregate `bureau_balance` into `bureau`-level features first, then aggregate `bureau` into `SK_ID_CURR`-level features. Same pattern for the `previous_application` branch (aggregate `POS_CASH_balance`, `installments_payments`, `credit_card_balance` to `SK_ID_PREV`-level stats first if needed, then to `SK_ID_CURR`-level).

### 3.3 Class Imbalance Warning

`TARGET` distribution is approximately **92% (no default) / 8% (default)**. This is critical:

- Use **AUC-ROC** and **Precision-Recall AUC** as primary metrics — NOT accuracy
- Apply class weighting or resampling (SMOTE / class_weight='balanced') for baseline and KAN training
- Stratify all train/validation splits

---

## 4. Compute Environment Constraints (Kaggle Free Tier)

| Constraint                                                                   | Limit                                      | Implication                                                                                                                                                                     |
| ---------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPU                                                                          | Free tier T4 x2 / P100, ~30 hrs/week quota | Budget GPU time — don't waste on hyperparameter grid search; use smaller search spaces                                                                                          |
| RAM                                                                          | ~13 GB (CPU), ~16 GB (GPU sessions)        | Full merged dataset (all 9 files) may not fit in memory raw — **must aggregate before merging**, use `dtype` downcasting (`float32` instead of `float64`, categorical encoding) |
| Disk                                                                         | ~20 GB session disk                        | Use `del` + `gc.collect()` after each aggregation step; avoid holding multiple full-size DataFrames simultaneously                                                              |
| Session length                                                               | 9-12 hrs max per session                   | Checkpoint intermediate processed data to `/kaggle/working/` as `.parquet` (not `.csv` — faster I/O, smaller size)                                                              |
| No persistent storage between sessions (unless using Kaggle Datasets output) | —                                          | Save processed features as a Kaggle Dataset output for reuse across notebook runs                                                                                               |

**Mandatory engineering practice:** Process data in chunks where possible; use `reduce_mem_usage()` utility function (downcast int64→int32, float64→float32) immediately after each CSV load.

---

## 5. Modeling Pipeline

### 5.1 Phase Breakdown

```text
Phase 1: Data Engineering
  └── Load all 9 CSVs → reduce memory → aggregate child tables → merge to application-level

Phase 2: Baseline Models
  └── Logistic Regression | Random Forest | XGBoost | LightGBM | Standard MLP

Phase 3: KAN Implementation
  └── PyKAN-based architecture, matched input dimensionality to baselines

Phase 4: Interpretability Layer
  └── SHAP + LIME on baselines (XGBoost/LightGBM/MLP)
  └── Native spline visualization on KAN
  └── Comparative interpretability analysis (agreement/divergence study)

Phase 5: Extension Work (Publication-Grade)
  └── See §6 below

Phase 6: Writing & Packaging
  └── Paper draft + GitHub repo polish
```

### 5.2 Baseline Models (in order of implementation)

| Model               | Library    | Purpose                                                                     |
| ------------------- | ---------- | --------------------------------------------------------------------------- |
| Logistic Regression | `sklearn`  | Naive linear baseline                                                       |
| Random Forest       | `sklearn`  | Tree ensemble baseline                                                      |
| XGBoost             | `xgboost`  | Gradient boosting SOTA baseline (Kaggle-competition standard)               |
| LightGBM            | `lightgbm` | Faster gradient boosting, handles categorical natively                      |
| Standard MLP        | `torch`    | Direct architecture-level comparison point for KAN (same input/output dims) |

**Hyperparameter tuning:** Use `Optuna` (lightweight, GPU-budget-friendly) — NOT exhaustive GridSearchCV — limit to 30-50 trials per model given compute constraints.

### 5.3 KAN Implementation

- **Library:** `pykan` (official MIT implementation, `pip install pykan`)
- **Architecture matching:** Use same input feature count as MLP baseline; match KAN width/depth roughly to MLP's parameter budget for fair comparison
- **Key hyperparameters to tune:**
  - Grid size (number of spline intervals per edge)
  - Spline order (typically cubic, k=3 — ties back to NM coursework)
  - Network width/depth
- **Training:** KAN training is slower than MLP per-epoch — budget GPU hours accordingly; consider training on a feature-reduced subset first (top-K features via baseline feature importance) if full-dimensionality KAN training is too slow on free-tier compute

### 5.4 Evaluation Metrics (apply uniformly across ALL models)

- AUC-ROC (primary)
- Precision-Recall AUC (critical given class imbalance)
- F1-Score (at optimal threshold via PR curve)
- Brier Score (calibration quality — important for credit risk specifically)
- Training time / inference time (for a practical deployment discussion section)

### 5.5 Interpretability Layer

| Model               | Interpretability Method                                                      |
| ------------------- | ---------------------------------------------------------------------------- |
| Logistic Regression | Coefficients (baseline reference)                                            |
| XGBoost / LightGBM  | SHAP TreeExplainer                                                           |
| MLP                 | SHAP DeepExplainer or KernelExplainer + LIME                                 |
| **KAN**             | **Native spline function visualization** (per-edge learned activation plots) |

**Key analysis to perform:** Overlay/compare KAN's learned spline shape for a given feature (e.g., `EXT_SOURCE_1`, `AMT_INCOME_TOTAL`, `DAYS_EMPLOYED`) against that same feature's SHAP dependence plot from XGBoost. Do they agree on direction/shape of risk relationship? This comparison IS the paper's core contribution — frame results around this.

---

## 6. Extensions for Publication-Grade / Portfolio-Grade Quality

Implement as many of these as time permits, in priority order:

1. **Symbolic regression on KAN splines** — `pykan` supports extracting symbolic formulas from learned splines (`model.symbolic_formula()`). Extract closed-form risk equations for top features — this is a strong, novel publication angle (interpretability as an _equation_, not just a plot).
2. **Calibration analysis** — reliability diagrams comparing KAN vs. baselines; credit risk models must be well-calibrated, not just high-AUC.
3. **Ablation study** — KAN performance vs. grid size / spline order (ties directly to NM coursework, strengthens the numerical-methods angle of the paper).
4. **Robustness/concept drift test** — split data temporally if any time signal exists; test if KAN degrades differently than baselines under distribution shift.
5. **Feature importance agreement metric** — quantify (e.g., rank correlation) between KAN-derived importance and SHAP-derived importance across all models, not just visual comparison.
6. **Fairness audit** — check default predictions across `CODE_GENDER`, age bands for disparate impact — adds a responsible-AI angle, increases publication appeal in applied venues.
7. **Deployment-readiness discussion** — inference latency comparison (KAN vs. XGBoost) — relevant for a "financially lucrative" framing (production cost implications).

---

## 7. Deliverables

### 7.1 GitHub Repository Structure

```text
kan-credit-risk/
├── README.md                     # Project overview, results summary, how to run
├── notebooks/
│   ├── 01_data_engineering.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_kan_model.ipynb
│   ├── 04_interpretability_analysis.ipynb
│   └── 05_extensions.ipynb
├── src/
│   ├── data_utils.py              # reduce_mem_usage, aggregation functions
│   ├── feature_engineering.py
│   ├── models.py                  # baseline + KAN wrapper classes
│   └── interpretability.py        # SHAP/LIME/spline visualization helpers
├── results/
│   ├── figures/                   # spline plots, SHAP plots, ROC curves
│   └── metrics_summary.csv
├── paper/
│   └── draft.md or .tex
└── LICENSE
```

### 7.2 Paper Structure (target: arXiv preprint minimum)

1. Abstract
2. Introduction (credit risk + regulatory interpretability need)
3. Related Work (KAN architecture, credit risk XAI literature, SHAP/LIME)
4. Methodology (data pipeline, model architectures, training setup)
5. Results (performance table, calibration, ablations)
6. Interpretability Analysis (spline visualization vs. SHAP comparison — core contribution)
7. Discussion (limitations, deployment implications, future work — mention Nepal-specific financial inclusion application as future work)
8. Conclusion

---

## 8. Constraints & Conventions for AI Agents Assisting on This Project

- **Always account for Kaggle free-tier compute limits** (§4) when suggesting code — no unbounded grid searches, no loading all 9 raw CSVs into memory simultaneously without downcasting.
- **Prioritize reproducibility** — set random seeds everywhere (`numpy`, `torch`, `sklearn`, `lightgbm`).
- **Package Manager:** Exclusively use the `uv` Python package manager where possible, falling back to other tools like `pip`, `venv` etc. only when necessary.
- **Code style:** modular functions over notebook-only inline code where logic will be reused (data engineering, metrics) — mirror the `src/` structure in §7.1 even while prototyping in notebooks.
- **Every model comparison must use identical train/val/test splits** — use a fixed stratified split saved once, loaded everywhere.
- **Documentation discipline:** every notebook should have markdown cells explaining _why_, not just _what_ — this notebook commentary doubles as paper-writing material later.
- **Default to interpretability-first framing** in all outputs — this project's contribution is the comparison, not raw leaderboard performance.

---

## 9. Key References

1. Liu, Z. et al. (2024). _KAN: Kolmogorov-Arnold Networks._ arXiv:2404.19756
2. Lundberg, S. & Lee, S. (2017). _A Unified Approach to Interpreting Model Predictions (SHAP)._ NeurIPS
3. Ribeiro, M. et al. (2016). _"Why Should I Trust You?" Explaining the Predictions of Any Classifier (LIME)._ KDD
4. Home Credit Group. (2018). _Home Credit Default Risk._ Kaggle Competition. <https://www.kaggle.com/c/home-credit-default-risk>

---

**End of context file. Agents should treat §4 (compute constraints) and §5.5/§6 (interpretability + extensions) as the highest-priority sections when generating or reviewing code.**
