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
| PubMed / Google Scholar | "LDL cholesterol level classification" OR "LDL estimation" AND "machine learning " | 2021–2026 | 142 | 8 |
| IEEE Xplore / ScienceDirect | "NHANES" AND "lipid profile classification" AND ("random forest" OR "XGBoost") | 2021–2026 | 65 | 4 |

Inclusion / exclusion criteria:
- **Inclusion:** Studies predicting LDL-C levels, estimating LDL-C continuous values, or stratifying cardiovascular lipid risk using tabular clinical/laboratory datasets; studies using standard machine learning or deep learning pipelines.
- **Exclusion:** Studies lacking validation details, non-peer-reviewed preprints without code/data transparency, or non-English articles.

### A2. Literature matrix (≥ 10 papers, preferably from the last 3–5 years)
| # | Paper (authors, year) | DOI / link | Dataset (n, source) | Method | Main result (metric, value) | External validation? (yes / no) | Main limitation |
|---|---|---|---|---|---|---|---|
| 1 | Fazakis et al. (2021) | DOI:10.5220/0010727200003063 | ELSA (n=7,526) | 8 ML models, 10-fold CV, undersampling, 22 features (Gini & Stepwise) | Accuracy: 56.0%–63.0% | No | Low predictive power using non-invasive features alone |
| 2 | García-D'Urso et al. (2022) | DOI:10.1109/ACCESS.2022.3168231 | Overweight adults (n=84) | ETR, RFE, Hyperparameter Optimization, RFR | MAE / RMSE metrics | No | Very small sample size, potential subject leakage |
| 3 | Kwon et al. (2022) | DOI:10.3389/fcvm.2022.846545 | Health-check (n=129,930 GSHC + n=46,470 KOICA) | Pyramid DNN with 6 hidden layers (30 nodes/layer), 10-fold CV | Superior accuracy compared to Friedewald and NIH equations | Yes (KOICA) | Uses homogenous assay reference, not gold-standard β-quantification |
| 4 | Li et al. (2022) | DOI:10.1186/s12944-022-01721-y | NHANES 2007–2018 (n=15,499) | Multivariable Logistic Regression & hypertriglyceridemia across demographic subgroups | Adjusted Odds Ratios | No | Descriptive epidemiological study; no ML predictive modelling |
| 5 | Fan et al. (2022) | DOI:10.1016/j.cca.2022.08.007 | Chinese profiles (n=111,448) | RF, Decision Trees; trained on 5-fold CV, external validation on 3 separate datasets; compared with Friedewald, Martin/Hopkins, and Sampson equations | Lower error rates in Chinese population; less affected by age; highly accurate LDL-C estimation | Yes (3 external sets) | Lacks gold-standard β-quantification reference; direct algebraic lipid ratios used |
| 6 | Kim et al. (2024) | DOI:10.1093/labmed/lmad114 | KNHANES (n=9,402) | Regression, penalized regression, Random Forest, XGBoost, and a proposed two-step ML model (Linear + XGB/RF), compared with Friedewald, Martin, and Sampson equations | Concordance: 85.1%, lowest RMSE | Yes (Independent years) | Still relies on invasive laboratory measurements; cannot fully capture non-linear lipid relationships |
| 7 | Han et al. (2025) | DOI:10.1038/s41598-025-88693-y | Asan Medical Center EMR, Seoul | XGBoost, RF, Logistic Regression; evaluating performance metrics | average AUROC of ~0.695 (even after reducing feature count by over 43%) | No | 48% sample loss due to missing follow-up LDL (selection bias) |
| 8 | Mansoori et al. (2025) | DOI:10.1007/s44200-025-00089-5 | MASHAD cohort (n=7,526) | Logistic Regression, Decision Tree, and Bootstrap Forest | Identified key predictive factors: PAL, sex, hip circumference, arm circumference, and short arm span | No | Lacks longitudinal HDL-C trend data for the Mashad region; findings cannot be generalized to regions with different lifestyle habits |
| 9 | Meng et al. (2025) | DOI:10.7717/peerj.19248 | Clinical records from Yanbian University Hospital (n=120,174) | Linear regression, KNN, Decision Tree, Random Forest, XGBoost, MLP regressor, compared with Friedewald, Sampson, and other formulas| ML models outperformed traditional equations; Random Forest and XGBoost achieved highest accuracy (R^2 = 0.94, MSE = 89.25) | No | Decreased predictive accuracy in patients with high triglycerides (TG > 300 mg/dL) |
| 10 | Orhan et al. (2026) | DOI:10.3390/diagnostics16132031 | LIS clinical dataset (n=11,681) | Linear regression, Random Forest, SVR, XGBoost (10-fold CV) | XGBoost achieved lowest error and highest stability (MAE: 14.7 mg/dL, RMSE: 20.22 mg/dL, r = 0.780); higher clinical classification accuracy (up to 66%) | No | Moderate clinical classification accuracy (approx. one-third misclassified); limited accuracy in patients with severe hypertriglyceridemia |
| 11 | Ahmed et al. (2026) | DOI: 10.1016/j.clinsp.2026.101068 | Dubai Health lipid profiles (n=96,492) | RF, XGBoost, Neural Network, k-NN, and Bayesian Ridge; using total cholesterol, HDL-C, and triglycerides as predictors | All models achieved R^2 from 0.996 to 1.000 in females and 0.996 to 0.999 in males; minimal prediction error (MAE≤0.96 mg/dL) | No | Evaluated on large laboratory cohort; incorporates SHAP explanations but relies directly on standard lipid panel features |
| 12 | Al Dhuhli et al. (2026) | DOI: 10.1016/j.plabm.2026.e00524 | Clinical sera (n=303) | ANN, Linear Regression, Ridge Regression, compared with 10 conventional formulas (60:40 train-test split) | ANN, Linear, and Ridge regression achieved the highest correlation (R = 0.97); ANN showed the lowest mean bias (0.04 mM); Puavilai and DeLong formulas also showed strong correlation (r≈0.97) | No | Relies on laboratory datasets without broader external validation cohorts |
| 13 | 

### A3. Research gaps
| ID | Gap (1–2 sentences) | Evidence (# in A2) | How your project addresses it |
|---|---|---|---|
| G1 | **Existing lipid blood test-only models omit vital clinical and demographic variables, creating a gap in model personalization, subgroup stability, and cross-population generalizability​** |  #9, #10, | Adding non-invasive determinants and examines its role in LDL-C levels estimation. |
| G2 | **Boundary Error when predicting continuous LDL-C values via regression and manually map them into ATP III tiers ** | #13| Direct LDL-C levels classification​ |
| G3 | **Imputation, outlier handling and importance ranking are done before data splitting, no VIF or correlation filtering is reported, Undersampling and importance-based reduction are applied before cross-validation (Leakage Risk)** | #6, #8, #11 |  |
| G4 | **Only evaluate on basic machine learning models​, Deep learning is not tested when contains non-linear features (non- invasive determinants)** | #1 -> #12| We train and test on LightBGM for machine learning model and TabNet, MLP for deep learning |

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
| Baseline 1 (Milestone 2) - Logistic Regression | Stratified 5-fold CV, Test n=623 | Test Macro-F1: 0.9360 (CV: 0.9505 ± 0.0192) | Test Accuracy: 0.9615, Weighted-F1: 0.96 | `submissions/<LDL Cholesterol Risk Classification from Invasive And Non-Invasive Determinants>_<N16>/src/run_analysis.py` ([Colab Link](https://colab.research.google.com/drive/1NNghBDIziMYZxsnOCX0zA7aLdUUAun4F?usp=sharing)) |
| Baseline 2 - Random Forest | Stratified 5-fold CV, Test n=623 | Test Macro-F1: 0.8389 (CV: 0.7940 ± 0.0284) | Test Accuracy: 0.8764, Weighted-F1: 0.88 | `submissions/<LDL Cholesterol Risk Classification from Invasive And Non-Invasive Determinants>_<N16>/src/run_analysis.py` ([Colab Link](https://colab.research.google.com/drive/1NNghBDIziMYZxsnOCX0zA7aLdUUAun4F?usp=sharing)) |
| Proposed model (Milestone 3) - XGBoost | Stratified 5-fold CV, Test n=623 | Test Macro-F1: 0.9262 (CV: 0.8557 ± 0.0163) | Test Accuracy: 0.9390, Weighted-F1: 0.94 | `submissions/<LDL Cholesterol Risk Classification from Invasive And Non-Invasive Determinants>_<N16>/src/run_analysis.py` ([Colab Link](https://colab.research.google.com/drive/1NNghBDIziMYZxsnOCX0zA7aLdUUAun4F?usp=sharing)) |
| Proposed model (Milestone 3) - LightGBM | Stratified 5-fold CV, Test n=623 | Test Macro-F1: 0.8963 (CV: 0.8594 ± 0.0184) | Test Accuracy: 0.9294, Weighted-F1: 0.93 | `submissions/<LDL Cholesterol Risk Classification from Invasive And Non-Invasive Determinants>_<N16>/src/run_analysis.py` ([Colab Link](https://colab.research.google.com/drive/1NNghBDIziMYZxsnOCX0zA7aLdUUAun4F?usp=sharing)) |
| Proposed model (Milestone 3) - MLP | Stratified 5-fold CV, Test n=623 | Test Macro-F1: 0.9533 (CV: 0.9012 ± 0.0242) | Test Accuracy: 0.9551, Weighted-F1: 0.96 | `submissions/<LDL Cholesterol Risk Classification from Invasive And Non-Invasive Determinants>_<N16>/src/run_analysis.py` ([Colab Link](https://colab.research.google.com/drive/1NNghBDIziMYZxsnOCX0zA7aLdUUAun4F?usp=sharing)) |

### B2. Ablation (Milestone 3)
| Configuration (component removed / changed) | Main metric | Difference vs full model | Source |
|---|---|---|---|
| Scenario A (Lipid-only) | Test Macro-F1: 0.9089 | -0.0014 (-0.2%) |`submissions/<LDL Cholesterol Risk Classification from Invasive And Non-Invasive Determinants​>_<N16>/src/run_ablation.py`|
| Scenario B (Non-lipid) | Test Macro-F1:0.2343 | -0.6760 (-74.3%) | `submissions/<LDL Cholesterol Risk Classification from Invasive And Non-Invasive Determinants​>_<N16>/src/run_ablation.py` |
| Scenario C (All features) | Test Macro-F1:0.9Z246 | Base Reference | `submissions/<LDL Cholesterol Risk Classification from Invasive And Non-Invasive Determinants​>_<N16>/src/run_ablation.py` |

### B3. Comparison with the literature
| Paper | Their setting | Their result | Our result (B1, Scenario C) | What makes the comparison unfair |
|---|---|---|---|---|
| Orhan et al. (2026), #12 | LIS clinical data, n = 11,681, direct LDL, inputs TC/TG/HDL only, XGBoost/RF/LightGBM, 10-fold CV | Accuracy ≈ 88.4% | LightGBM 92.78%, TabNet 94.70% (5-class) | Different population and assay; their label is a measured LDL while ours is calculated from the same inputs; class definition and split differ (10-fold vs one 80/20 split). Our higher accuracy must **not** be presented as better performance. |
| Kim et al. (2024), #6 | KNHANES, direct LDL, 2-step model (linear + XGB/RF on residuals), other years as independent validation | Concordance 82.5%; poor at LDL ≥ 190 | Very High recall 0.82–0.91 (n = 11) | Different metric (concordance vs recall/macro-F1), different country and measured label; our Very High class has only 11 test cases (wide interval). Their independent-year validation is stronger than our single split. |
| Ahmed et al. (2026), #10 | Dubai lipid profiles, n = 96,492, TC/TG/HDL only; reference Friedewald for TG < 400 | R² ≈ 1.0 | Regression: R² 0.9999 (formula alone: 0.9999) | Same circularity as ours: the model relearns a deterministic formula. This is the fairest comparison and shows that high scores with lipid inputs carry no extra information. |
| Fazakis et al. (2021), #1 (non-invasive comparator) | ELSA, non-invasive features, binary incident high cholesterol at 2 years | Accuracy 56–63% | No-lipid scenarios (3-class): macro-F1 0.40–0.43, accuracy 0.43–0.57; regression R² 0.05–0.12 | Different task (binary, longitudinal) vs ours (cross-sectional, multi-class); only the order of magnitude is comparable. |

---

### B4. Timeline to completion
Planned dates are given by week; add calendar dates.

| Task | Owner (student ID / role) | Milestone (M3 week 10 · M4 week 12 · Final week 15) | Planned date | Status (Not started / In progress / Done + evidence) |
|---|---|---|---|---|
| Re-run 5-class baselines (add Logistic Regression) and proposed models on the same split | 202414823 / Model Training & Validation | M3 | Week 10 | In progress: RF, XGBoost, LightGBM, MLP, TabNet done; LR missing |
| Add QWK, ordinal MAE, per-class recall with bootstrap 95% CI, AUC (one-vs-rest), calibration and Brier | 202414823 / Model Training & Validation | M3 | Week 10 | Not started |
| Leakage checks: label permutation on the full pipeline, deliberately leaky pipeline, formula baseline | 202414823 / Pipeline Design | M4 | Week 12 | In progress: formula baseline done (MAE 0.246); other checks not started |
| SMOTE inside the pipeline (in-fold, adaptive k_neighbors) | 202414851 / Data Preprocessing | M3 | Week 10 | Done: `build_pipeline` in the code (SMOTE after feature selection, `SMOTE_K` capped by the smallest class) |
| Imbalance ablation: no balancing vs SMOTE vs class weights, and preprocessing ablation (add one step at a time). `class_weight` is not yet set in any model | 202414851 / Data Preprocessing | M4 | Week 12 | Not started: no run without SMOTE or with class weights yet |
| Feature-selection report (stage log, reasons, 5-fold stability) | 202414851 / Data Preprocessing | M3 | Week 10 | Done; outputs `feature_selection_report.csv`, `feature_selection_log.csv`, `feature_selection_stability.csv` |
| SHAP of the final model for classes | 202414851 / XAI (SHAP) | M4 | Week 12 | In progress: beeswarm for 5 models: Optimal, Near Optimal, Borderline High, High, Very high |
| Verify literature matrix (DOIs, sample sizes, numbers) and finish B3 | 202414823 and 202414851 / Literature Review | M3 | Week 10 | In progress |
| Datasheet, file checksums, FAIR and TRIPOD+AI checklist | 202414823 and 202414851 | M4 | Week 12 | Not started |
| Final report, slides, repository clean-up, AI-use declaration | 202414823 and 202414851 | Final | Week 15 | Not started |

---

### B5. Risks / blockers / questions for the instructor
- **Label circularity.** `LBDLDL` in NHANES is calculated from TC, HDL, and TG (our empirical verification on $n = 3,115$ participants shows that the Friedewald formula reproduces the label with an $R^2$ of 0.9999 and a median absolute difference of 0.20 mg/dL). With lipid inputs, high scores come from relearning the formula, and a non-invasive model scores macro-F1 $\approx 0.4$ ($R^2 \approx 0.05$–0.12). Question: is it acceptable to keep a calculated label, or should we look for a directly measured LDL source?
- **Very small rare classes.** The "Very High" class contains only 55 samples overall (11 in the test set, representing 1.8%); its metrics have very wide confidence intervals. Question: should we merge High and Very High, or report intervals only?
- **Single split, no external validation.** Results come from one 80/20 split of one NHANES cycle (2021–2023). We plan the 2017–March 2020 cycle (`P_` files, triglyceride variable `LBXTR`) as temporal validation if time allows.
- **Survey design ignored.** Weights and strata are not used, so results do not generalise to the US population. Question: is weighting required for this assignment?
- **Metrics.** Only F1 and accuracy are reported so far. Question: for a 5-class ordinal target, which of AUC, calibration, QWK is expected in the TRIPOD+AI table?

---

## C. Self-assessment against the FAIR, TRIPOD+AI, PROBAST+AI appendix

### C1. FAIR — data
| Item | Answer | Evidence |
|---|---|---|
| Source, version, download date; checksum of the training file | CDC NHANES 2021–2023 cycle, public-use `.xpt` files with suffix `_L` (DEMO, BMX, TRIGLY, TCHOL, HDL, SMQ, PAQ, SLQ, BIOPRO, BPXO). Files downloaded between 29/09/2026 and 03/10/2026. SHA-256 checksum | Documented in data collection logs and preprocessing execution scripts. |
| Licence / DUA; basis for being allowed to use the data | NHANES public-use files released by NCHS/CDC; CDC data-use terms apply (no attempt to identify participants). | Security & Compliance protocol in the repository documentation. |
| Datasheet: population, inclusion/exclusion criteria, variable definitions and units, limitations | **Population:** civilian non-institutionalised US residents, NHANES 2021–2023. **Unit:** participant (`SEQN`). **Size:** 3,996 participants in `TRIGLY_L` (fasting morning subsample); 3,115 adults ($\ge 18$ years) with valid LDL and BMI 10–90 after cleaning. **Inclusion/exclusion:** drop missing LDL, age < 18, implausible BMI; "refused/don't know" codes recoded to NaN. **Key variables:** TC, HDL, TG, LDL (mg/dL; LDL is calculated), BMI (kg/m²), diastolic BP and pulse (mmHg, bpm), age (years), income-to-poverty ratio, sleep hours, sedentary time. **Limitations:** calculated label; fasting subsample; survey weights not used; single cycle; no Vietnamese population. | Data cleaning scripts, exploratory data analysis summaries, and target distribution logs ($n = 3,115$; 61 features including 3 lipid, 58 non-lipid features). |
| No patient identifiers | Only the random public ID `SEQN`; no names or contact data | System security logs |

### C2. Reporting guideline
- **My research question:** Can the five ATP III LDL-C risk levels be classified from invasive (lipid profile) and non-invasive (demographic, anthropometric, blood-pressure, lifestyle) variables, and how much do non-invasive variables add once lipids are known?
- **Checklist items met / total; items still missing:** **[TO FILL after going through the official TRIPOD+AI checklist item by item]**. Likely missing: calibration and discrimination with confidence intervals, sample-size justification, handling of the survey design, fairness across subgroups, a model-availability statement, and an explicit statement on the intended use and the time of prediction.
- **Outcome, label, prediction time T0, variables available at T0:** Outcome: LDL-C risk level (0 Optimal < 100; 1 Near optimal 100–129; 2 Borderline high 130–159; 3 High 160–189; 4 Very high $\ge 190$ mg/dL). Label: ATP III cut points applied to `LBDLDL`, calculated from TC, HDL, and TG. T0: the single NHANES examination visit; all predictors are measured at the same visit. In lipid scenarios, TC/HDL/TG come from the same blood draw as the label. The non-invasive scenario uses only variables obtainable without a blood test.

### C3. PROBAST+AI — self-appraisal of your project
| Domain | Risk of bias (low / high / unclear) | Evidence in the repo (file, line) | What you will do to reduce the risk |
|---|---|---|---|
| Participants and data sources | Unclear | Cleaning and filtering steps in the pipeline script | Document the fasting-subsample selection ($n = 3,115$); state that weights are not used; check the effect of the inner join with `TRIGLY_L` |
| Predictors | High with lipid variables, low without | `USE_LIPID_FEATURES` switch in the pipeline code | Report the no-lipid scenario as the main non-invasive result; state clearly that TC/HDL/TG are used to compute the label |
| Outcome (watch for incorporation bias) | High | Label creation step (`np.digitize` on `LBDLDL`); formula baseline MAE 0.25 mg/dL, $R^2 \approx 0.9999$ | State the incorporation bias; compare with the formula baseline; look for a measured-LDL data source |
| Analysis (patient-level split, fit on train only, sample size, missing data) | Unclear | Stratified split and `ImbPipeline`: all fitted steps are in-fold; one record per `SEQN`, so a patient-level split is automatic | Add bootstrap confidence intervals, calibration and Brier score, nested tuning, a label-permutation check, and a sample-size justification (Very High $n = 55$) |
Applicability (population, setting where the model would be used): US adults in the fasting NHANES subsample; the lipid models only reproduce a formula and cannot work without a blood test; the non-invasive model has weak performance (macro-F1 $\approx 0.4$) and is not validated for Vietnamese or non-fasting populations.

### C4. Quick checklist (Appendix §7)
| Item | Met / Not yet | Evidence |
|---|---|---|
| Datasheet | Not yet (draft in C1) | C1 |
| Data version + checksum | Not yet | Command given in C1 |
| No identifiers; basis for data use | Met (confirm CDC wording) | Security documentation |
| Correct reporting guideline + completed checklist | Not yet (TRIPOD+AI chosen, checklist not completed) | C2 |
| Outcome, label, T0 defined | Met, with the caveat that the label is calculated | C2 |
| Patient-level split; preprocessing fit on train only | Met | Pipeline implementation scripts (`ImbPipeline`, stratified split first) |
| Logistic regression baseline (or a suitable baseline, with reason) | Partly: formula baseline done ($R^2 \approx 0.9999$); 5-class logistic regression evaluated with and without balancing | Model evaluation logs and comparative tables |
| AUC + calibration (curve, in-the-large, slope) + Brier, with 95% CI | Not yet | Timeline task list |
| Is the sample size sufficient; applicability limits | Not yet: $n = 3,115$ overall but Very High = 55 | B5, C3 |

---

## D. AI use declaration

## D. AI use declaration

| Tool | Used for which part | How the team verified the output |
|---|---|---|
| Claude (Anthropic) | Reviewing and revising the preprocessing and feature-selection code; building the regression and feature-selection report scripts; explaining metrics (macro-F1, QWK); drafting the Word documentation and this progress report | Ran the code on the real NHANES files and compared the printed outputs; recomputed accuracy and macro-F1 from the confusion matrices; checked paper claims and DOIs by hand against Crossref for the 5 selected literature sources (Orhan et al. 2026, Kim et al. 2024, Ahmed et al. 2026, Fan et al. 2022, Meng et al. 2025) and manually verified sample sizes and performance metrics. |
| Gemini (Google) | Critique of the research gaps and suggestions on methodology | Accepted the suggestion to add strict multicollinearity filtering (correlation threshold at 0.85 and VIF threshold at 10) inside an ImbPipeline to avoid data leakage; rejected the suggestion of using unverified feature interactions without checking literature, opting instead for univariate ANOVA F-test and Random Forest embedded importance as supported by Fan et al. (2022) and Orhan et al. (2026). |
