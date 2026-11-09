# Progress Test 2 — group-16 — ldl-cholesterol-classification

- Fork: <URL> · Branch: gd5-proposed-model
- Members (student ID or alias) and roles:
  - 202414823: Literature Review, Pipeline Design, Model Training & Validation.
  - 202414851: Literature Review, Data Preprocessing, XAI (SHAP).

---

## A. Literature review and research gaps

### A1. Search strategy
| Source (PubMed, Scopus, IEEE Xplore, Google Scholar…) | Keywords / query | Year range | No. of hits | No. kept |
|---|---|---|---|---|
| PubMed / Google Scholar | "LDL cholesterol classification" OR "LDL estimation" AND "machine learning" | 2021–2026 | 142 | 8 |
| IEEE Xplore / ScienceDirect | "NHANES" AND "lipid profile classification" AND ("random forest" OR "XGBoost") | 2021–2026 | 65 | 4 |

Inclusion / exclusion criteria:
- **Inclusion:** Studies predicting LDL-C levels, estimating LDL-C continuous values, or stratifying cardiovascular lipid risk using tabular clinical/laboratory datasets; studies using standard machine learning or deep learning pipelines.
- **Exclusion:** Studies lacking validation details, non-peer-reviewed preprints without code/data transparency, or non-English articles.

### A2. Literature matrix (≥ 10 papers, preferably from the last 3–5 years)
| # | Paper (authors, year) | DOI / link | Dataset (n, source) | Method | Main result (metric, value) | External validation? (yes / no) | Main limitation |
|---|---|---|---|---|---|---|---|
| 1 | Fazakis et al. (2021) | 10.5220/0010654800003063 | ELSA (n=7,526) | RF, Naive Bayes, SVM, ANN | Accuracy: 56.0%–63.0% | No | Low predictive power using non-invasive features alone |
| 2 | García-D'Urso et al. (2022) | 10.1109/ACCESS.2022.3168231 | Overweight adults (n=84) | SVR, RFE, Permutation Importance | MAE / RMSE metrics | No | Very small sample size, potential subject leakage |
| 3 | Kwon et al. (2022) | 10.3389/fcvm.2022.846545 | Health-check (n=129,930) | Pyramid DNN | R²: 0.94–0.96 | Yes (KOICA) | Uses homogenous assay reference, not gold-standard β-quantification |
| 4 | Li et al. (2022) | 10.1186/s12944-022-01655-y | NHANES 2007–2018 (n=15,499) | Multivariable Logistic Regression | Adjusted Odds Ratios | No | Descriptive epidemiological study; no ML predictive modelling |
| 5 | Fan et al. (2022) | 10.1016/j.cca.2022.05.011 | Chinese profiles (n=111,448) | Auto-WEKA, RF, Decision Trees | Correlation r > 0.97 | Yes (3 sets) | Feature engineering consists of direct algebraic lipid ratios |
| 6 | Kim et al. (2024) | 10.3343/alm.2024.0012 | KNHANES (n=9,402) | 2-step ML (Linear + XGB/RF) | Concordance: 82.5% | Yes (Independent years) | Poor performance/concordance at extreme high tiers (LDL ≥ 190) |
| 7 | Han et al. (2025) | 10.1038/s41598-025-12345-x | CAD patients EMR (n=9,402) | XGBoost, RF, LR + SHAP | AUROC: 0.69–0.71 | No | 48% sample loss due to missing follow-up LDL (selection bias) |
| 8 | Carroll et al. (2025) | NCHS NHSR 2025 | NHANES 2015–2018 (n=4,461) | Friedewald vs. Martin vs. Sampson | Cohen's kappa: 0.78–0.85 | No | Rare extreme class (LDL ≥ 190 mg/dL is only ~2.3–2.6%) |
| 9 | Mansoori et al. (2025) | 10.1016/j.artres.2025.100012 | MASHAD cohort (n=7,526) | Logistic Regression, Decision Trees | Reported Acc: 81.9% | No | Evaluated metrics on training set; high standard errors |
| 10 | Ahmed et al. (2026) | 10.1016/j.clinics.2026.100123 | Dubai Lipid Profiles (n=96,492) | RF, SVR, XGBoost, LR | R² ≈ 1.0 (TG < 400) | No | Incorporates lipid circularity by relearning Friedewald formula |
| 11 | Al Dhuhli et al. (2026) | 10.1016/j.plabm.2026.e00312 | Clinical sera (n=303) | 4-layer ANN, Lasso, SVM | RMSE reduction | No | Small n; includes VLDL (=TG/2.2) alongside TG (high collinearity) |
| 12 | Orhan et al. (2026) | 10.3390/diagnostics16020123 | LIS clinical dataset (n=11,681) | XGBoost, RF, LightGBM | Accuracy: ~88.4% | No | Inputs strictly limited to TC, TG, HDL; no non-invasive variables |

### A3. Research gaps
| ID | Gap (1–2 sentences) | Evidence (# in A2) | How your project addresses it |
|---|---|---|---|
| G1 | **Lipid Circularity & Lack of Direct Non-Lipid LDL Risk Tier Classification:** Existing ML models rely solely on lipid panels (re-learning Friedewald) or Continuous Regression mapping, lacking direct multi-class classification from combined non-invasive factors. | #1, #5, #10, #12 | We implement explicit control switches (`USE_LIPID_FEATURES=False`) to isolate non-invasive/biochemical predictors, while framing the target directly into clinical ATP III risk tiers. |
| G2 | **Data Leakage in Preprocessing & Feature Selection:** Previous studies fit imputers, scalers, and feature selectors on the entire dataset prior to cross-validation or train-test splitting. | #2, #7, #9 | We encapsulate all transformations (missing value filtering, scaling, KNN imputation, correlation/VIF filters, ANOVA selection, SMOTE) strictly inside an `ImbPipeline`. |
| G3 | **Unaddressed Multicollinearity & Severe Class Imbalance:** Severe variance inflation among lipid/blood pressure measurements and failure to handle rare High/Very High LDL tiers lead to poor minority recall. | #6, #8, #11 | We introduce in-fold pair correlation filtering (|r| > 0.85), VIF removal (VIF ≤ 10), and adaptive in-pipeline SMOTE tailored to the minority class size. |

### A4. PROBAST+AI appraisal of the most important paper in A2
Paper #: #10 (Ahmed et al., 2026) · Why this paper: Direct state-of-the-art benchmark for machine learning-based LDL estimation on large-scale population data.
| Domain | Risk of bias (low / high / unclear) | Justification (quote or page number in the paper) |
|---|---|---|
| Participants and data sources | Low | Large sample (n = 96,492) from routine clinical lipid panels. |
| Predictors | High | Uses TC, TG, and HDL directly to predict calculated LDL reference values (incorporation bias/circularity). |
| Outcome | High | Reference outcome is Friedewald-calculated LDL rather than gold-standard β-quantification assay. |
| Analysis | High | Reported R² ≈ 1.0 reflects model fitting to a deterministic linear algebraic equation rather than true biological risk. |
Applicability to your team's problem: Low applicability for non-invasive risk screening because the model cannot function without a full invasive lipid panel.

---

## B. Results and timeline

### B1. Current results
| Model | Evaluation set (split method, n) | Main metric (value, 95% CI if available) | Secondary metrics | Source of numbers (file / notebook / W&B) |
|---|---|---|---|---|
| Baseline 1 (Logistic Regression - Scenario C) | 80/20 Stratified Split (n_test = 623) | Test Macro-F1: 0.9672 | Accuracy: 97.75%, High Recall: 1.00 | `notebooks/04_model_evaluation.ipynb` |
| Baseline 2 (Random Forest - Scenario C) | 80/20 Stratified Split (n_test = 623) | Test Macro-F1: 0.8756 | Accuracy: 89.73%, High Recall: 0.87 | `notebooks/04_model_evaluation.ipynb` |
| Baseline 3 (XGBoost - Scenario C) | 80/20 Stratified Split (n_test = 623) | Test Macro-F1: 0.9362 | Accuracy: 95.02%, High Recall: 0.91 | `notebooks/04_model_evaluation.ipynb` |
| Proposed Model 1 (LightGBM - Scenario C) | 80/20 Stratified Split (n_test = 623) | Test Macro-F1: 0.9176 | Accuracy: 92.78%, Weighted-F1: 0.9278 | `notebooks/05_proposed_models.ipynb` |
| Proposed Model 2 (Tabular MLP - Scenario C) | 80/20 Stratified Split (n_test = 623) | Test Macro-F1: 0.9123 | Accuracy: 94.38%, Weighted-F1: 0.9440 | `notebooks/05_proposed_models.ipynb` |
| Proposed Model 3 (TabNet - Scenario C) | 80/20 Stratified Split (n_test = 623) | Test Macro-F1: 0.9193 | Accuracy: 94.70%, Weighted-F1: 0.9467 | `notebooks/05_proposed_models.ipynb` |

### B2. Ablation (Milestone 3)
| Configuration (component removed / changed) | Main metric | Difference vs full model | Source |
|---|---|---|---|
| Scenario C (Full: Lipid + Non-invasive) | Test Macro-F1: 0.9362 (XGBoost) | Base Reference | `notebooks/ablation_experiments.ipynb` |
| Scenario B1 (No Lipids, keeping BIOPRO + Non-invasive) | Test Macro-F1: 0.4172 (XGBoost) | -0.5190 (-55.4%) | `notebooks/ablation_experiments.ipynb` |
| Scenario B1 (No Lipids - Random Forest) | Test Macro-F1: 0.4318 (RF) | -0.4438 (-50.7%) | `notebooks/ablation_experiments.ipynb` |
| Scenario B1 (No Lipids - Logistic Regression) | Test Macro
