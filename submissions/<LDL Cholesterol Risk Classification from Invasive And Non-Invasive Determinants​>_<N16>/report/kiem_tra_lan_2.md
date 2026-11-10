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

### B3. Comparison with the literature
| Paper | Their setting | Their result | Our result (B1, Scenario C) | What makes the comparison unfair |
|---|---|---|---|---|
| Orhan et al. (2026), #12 | LIS clinical data, n = 11,681, direct LDL, inputs TC/TG/HDL only, XGBoost/RF/LightGBM, 10-fold CV | Accuracy ≈ 88.4% | LightGBM 92.78%, TabNet 94.70% (5-class) | Different population and assay; their label is a measured LDL while ours is calculated from the same inputs; class definition and split differ (10-fold vs one 80/20 split). Our higher accuracy must **not** be presented as better performance. |
| Kim et al. (2024), #6 | KNHANES, direct LDL, 2-step model (linear + XGB/RF on residuals), other years as independent validation | Concordance 82.5%; poor at LDL ≥ 190 | Very High recall 0.82–0.91 (n = 11) | Different metric (concordance vs recall/macro-F1), different country and measured label; our Very High class has only 11 test cases (wide interval). Their independent-year validation is stronger than our single split. |
| Ahmed et al. (2026), #10 | Dubai lipid profiles, n = 96,492, TC/TG/HDL only; reference Friedewald for TG < 400 | R² ≈ 1.0 | Regression: R² 0.9999 (formula alone: 0.9999) | Same circularity as ours: the model relearns a deterministic formula. This is the fairest comparison and shows that high scores with lipid inputs carry no extra information. |
| Fazakis et al. (2021), #1 (non-invasive comparator) | ELSA, non-invasive features, binary incident high cholesterol at 2 years | Accuracy 56–63% | No-lipid scenarios (3-class): macro-F1 0.40–0.43, accuracy 0.43–0.57; regression R² 0.05–0.12 | Different task (binary, longitudinal) vs ours (cross-sectional, multi-class); only the order of magnitude is comparable. |

### B4. Timeline to completion
Planned dates are given by week; add calendar dates.

| Task | Owner (student ID / role) | Milestone (M3 week 10 · M4 week 12 · Final week 15) | Planned date | Status (Not started / In progress / Done + evidence) |
|---|---|---|---|---|
| Re-run 5-class baselines (add Logistic Regression) and proposed models on the same split | 202414823 / Model Training & Validation | M3 | Week 10 | In progress: RF, XGBoost, LightGBM, MLP, TabNet done (slide 26); LR missing |
| Add QWK, ordinal MAE, per-class recall with bootstrap 95% CI, AUC (one-vs-rest), calibration and Brier | 202414823 / Model Training & Validation | M3 | Week 10 | Not started (QWK and recall derived from slide 27 only) |
| 5-class ablation of scenarios A, B2, C (and TG-free E, F) | 202414823 / Model Training & Validation | M3 | Week 10 | In progress: 3-class B1 done; regression A–F done; 5-class not run |
| Leakage checks: label permutation on the full pipeline, deliberately leaky pipeline, formula baseline | 202414823 / Pipeline Design | M4 | Week 12 | In progress: formula baseline done (MAE 0.246); other checks not started |
| Preprocessing ablation; SMOTE vs SMOTENC vs class weights | 202414851 / Data Preprocessing | M4 | Week 12 | Not started |
| Feature-selection report (stage log, reasons, 5-fold stability) | 202414851 / Data Preprocessing | M3 | Week 10 | Done: slides 20–24; outputs `feature_selection_report.csv`, `feature_selection_log.csv`, `feature_selection_stability.csv` |
| SHAP of the final model for Scenario C and B2, class-specific (High, Very High) | 202414851 / XAI (SHAP) | M4 | Week 12 | In progress: beeswarm for 5 models, class High (slide 28); B2 and Very High pending |
| Verify literature matrix (DOIs, sample sizes, numbers) and finish B3 | 202414823 and 202414851 / Literature Review | M3 | Week 10 | In progress |
| Datasheet, file checksums, FAIR and TRIPOD+AI checklist | 202414823 and 202414851 | M4 | Week 12 | Not started |
| Final report, slides, repository clean-up, AI-use declaration | 202414823 and 202414851 | Final | Week 15 | Not started |

### B5. Risks / blockers / questions for the instructor
- **Label circularity.** `LBDLDL` in NHANES is calculated from TC, HDL and TG (our check: the Friedewald formula reproduces it with MAE 0.25 mg/dL). With lipid inputs, high scores come from relearning the formula, and a "non-invasive" model scores macro-F1 ≈ 0.4 (R² ≈ 0.05–0.12). Question: is it acceptable to keep a calculated label, or should we look for a directly measured LDL source?
- **Very small rare classes.** Very High has 55 samples (11 in the test set, 1.8%); its metrics have very wide intervals. Question: should we merge High and Very High, or report intervals only?
- **Single split, no external validation.** Results come from one 80/20 split of one NHANES cycle (2021–2023). We plan the 2017–March 2020 cycle (`P_` files, triglyceride variable `LBXTR`) as temporal validation if time allows.
- **Survey design ignored.** Weights and strata are not used, so results do not generalise to the US population. Question: is weighting required for this assignment?
- **Metrics.** Only F1 and accuracy are reported so far. Question: for a 5-class ordinal target, which of AUC, calibration, QWK is expected in the TRIPOD+AI table?
- **Inconsistencies in the current slides to fix:** slide 21 (VIF "Partly", embedded "No") contradicts slide 24 (VIF and embedded in the pipeline); slide 10 writes "Very high: LDL-C <= 190" (should be ≥ 190); slide 9 says 141 features while slide 13 says 136 columns; slide 3 gives "roughly 85% of all cardiovascular deaths" for LDL-C and needs a citation; the "New" claims on slide 25 should be worded "among the papers we reviewed".
- **Literature matrix to verify.** Several DOIs in A2 look like placeholders (for example `10.1038/s41598-025-12345-x`, `10.1016/j.clinics.2026.100123`, `10.3390/diagnostics16020123`, `10.1016/j.artres.2025.100012`) and Kim 2024 (n = 9,402) has the same n as Han 2025. Each DOI must be opened and checked before submission.

## C. Self-assessment against the FAIR, TRIPOD+AI, PROBAST+AI appendix

### C1. FAIR — data
| Item | Answer | Evidence |
|---|---|---|
| Source, version, download date; checksum of the training file | CDC NHANES 2021–2023 cycle, public-use `.xpt` files with suffix `_L` (DEMO, BMX, TRIGLY, TCHOL, HDL, SMQ, PAQ, SLQ, BIOPRO, BPXO). Files dated 29/09/2026–03/10/2026 (Windows timestamps in the slide-12 screenshot). SHA-256 checksum: **[TO RUN]** | Slides 9, 12; run `import hashlib, glob; [print(f, hashlib.sha256(open(f,'rb').read()).hexdigest()) for f in sorted(glob.glob('*.xpt'))]`. `HDL_L.xpt` (29/09) and `HDL_L (1).xpt` (03/10) both exist: confirm the hashes match and that the code reads the intended file. The Kaggle link on slide 9 is a derived copy; state which source was actually used. |
| Licence / DUA; basis for being allowed to use the data | NHANES public-use files released by NCHS/CDC; CDC data-use terms apply (no attempt to identify participants). **[verify the exact wording on the CDC NHANES data-use page and cite it]** | Slide 12 (Security & Compliance) |
| Datasheet: population, inclusion/exclusion criteria, variable definitions and units, limitations | **Population:** civilian non-institutionalised US residents, NHANES 2021–2023. **Unit:** participant (`SEQN`). **Size:** 3,996 participants in `TRIGLY_L` (fasting morning subsample); 3,115 adults (≥ 18 years) with LDL and BMI 10–90 after cleaning. **Inclusion/exclusion:** drop missing LDL, age < 18, implausible BMI; "refused/don't know" codes recoded to NaN. **Key variables:** TC, HDL, TG, LDL (mg/dL; LDL is calculated), BMI (kg/m²), diastolic BP and pulse (mmHg, bpm), age (years), income-to-poverty ratio, sleep (hours), sedentary time (minutes). **Limitations:** calculated label; fasting subsample; survey weights not used; single cycle; no Vietnamese population. | Slides 13, 17, 18, 20; target definition slide 10 |
| No patient identifiers | Only the random public ID `SEQN`; no names or contact data | Slide 12 |

### C2. Reporting guideline
- **My research question:** Can the five ATP III LDL-C risk levels be classified from invasive (lipid profile) and non-invasive (demographic, anthropometric, blood-pressure, lifestyle) variables, and how much do non-invasive variables add once lipids are known?
- **Matching reporting guideline and why:** TRIPOD+AI, because the study develops and internally validates a multivariable machine-learning classification model. STARD-AI (accuracy of an existing AI test against a reference) and DECIDE-AI (early live clinical evaluation) do not fit. The task is cross-sectional (estimation at the time of the NHANES examination), which TRIPOD+AI also covers as a diagnostic prediction model.
- **Checklist items met / total; items still missing:** **[TO FILL after going through the official TRIPOD+AI checklist item by item]**. Likely missing: calibration and discrimination with confidence intervals, sample-size justification, handling of the survey design, fairness across subgroups, a model-availability statement, and an explicit statement on the intended use and the time of prediction.
- **Outcome, label, prediction time T0, variables available at T0:** Outcome: LDL-C risk level (0 Optimal < 100; 1 Near optimal 100–129; 2 Borderline high 130–159; 3 High 160–189; 4 Very high ≥ 190 mg/dL). Label: ATP III cut points applied to `LBDLDL`, which is calculated from TC, HDL and TG. T0: the single NHANES examination visit; all predictors are measured at the same visit, and in the lipid scenarios TC/HDL/TG come from the same blood draw as the label, so the lipid scenarios are not prospective prediction. The non-invasive scenario uses only variables obtainable without a blood test.

### C3. PROBAST+AI — self-appraisal of your project
| Domain | Risk of bias (low / high / unclear) | Evidence in the repo (file, line) | What you will do to reduce the risk |
|---|---|---|---|
| Participants and data sources | Unclear | Cleaning and filtering step of the pipeline notebook (add line numbers); slides 13, 18 | Document the fasting-subsample selection; state that weights are not used; check the effect of the inner join with `TRIGLY_L` |
| Predictors | High with lipid variables, low without | `USE_LIPID_FEATURES` switch in the pipeline code | Report the no-lipid scenario as the main non-invasive result; state clearly that TC/HDL/TG are used to compute the label |
| Outcome (watch for incorporation bias) | High | Label creation step (`np.digitize` on `LBDLDL`); formula baseline MAE 0.246 | State the incorporation bias; compare with the formula baseline; look for a measured-LDL data source |
| Analysis (patient-level split, fit on train only, sample size, missing data) | Unclear | Stratified split and `ImbPipeline` (slides 22–24): all fitted steps are in-fold; one record per `SEQN`, so a patient-level split is automatic | Add bootstrap confidence intervals, calibration and Brier score, nested tuning, a label-permutation check, and a sample-size justification (Very High n = 55) |
Applicability (population, setting where the model would be used): US adults in the fasting NHANES subsample; the lipid models only reproduce a formula and cannot work without a blood test; the non-invasive model has weak performance (macro-F1 ≈ 0.4) and is not validated for Vietnamese or non-fasting populations.

### C4. Quick checklist (Appendix §7)
| Item | Met / Not yet | Evidence |
|---|---|---|
| Datasheet | Not yet (draft in C1) | C1 |
| Data version + checksum | Not yet | Command given in C1 |
| No identifiers; basis for data use | Met (confirm CDC wording) | Slide 12 |
| Correct reporting guideline + completed checklist | Not yet (TRIPOD+AI chosen, checklist not completed) | C2 |
| Outcome, label, T0 defined | Met, with the caveat that the label is calculated | Slide 10, C2 |
| Patient-level split; preprocessing fit on train only | Met | Slides 22–24 (`ImbPipeline`, stratified split first) |
| Logistic regression baseline (or a suitable baseline, with reason) | Partly: formula baseline done; 5-class LR missing | B1, B2 supplementary table |
| AUC + calibration (curve, in-the-large, slope) + Brier, with 95% CI | Not yet | B4 |
| Is the sample size sufficient; applicability limits | Not yet: n = 3,115 overall but Very High = 55 | B5, C3 |

## D. AI use declaration
Write only what the team actually did; the entries marked **[fill]** must be completed by the team.

| Tool | Used for which part | How the team verified the output |
|---|---|---|
| Claude (Anthropic) | Reviewing and revising the preprocessing and feature-selection code; building the regression and feature-selection report scripts; explaining metrics (macro-F1, QWK); drafting the Word documentation and this progress report | Ran the code on the real NHANES files and compared the printed outputs; recomputed accuracy and macro-F1 from the confusion matrices (they matched slide 26); **[fill: literature claims and DOIs checked by hand]** |
| Gemini (Google) | Critique of the research gaps and suggestions on methodology | **[fill: which suggestions were accepted or rejected and why; claims such as "mutual information captures feature interactions" must be checked against the literature]** |
