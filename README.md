# Mapping Sustainable Rice Water-Management Practices in Punjab using Machine Learning

A CMPE 257 (Fall 2026) machine learning project that reproduces and extends the classification experiments of
Shah et al. (AAAI-26): identifying how rice fields in Punjab, India were sown and irrigated from Sentinel-1 radar features.

**Team:** Nhi Nguyen, Jack Sangiamputtakoon, Jordan Phillips

## Repository contents

| File | Description |
| --- | --- |
| `CMPE_257_Project.ipynb` | Full project notebook (Google Colab): data exploration, model training, evaluation and error analysis |
| `CMPE257_Rice_Presentation.pdf` | Final presentation slides |

## Prediction tasks

- **Sowing (DSR):** direct-seeded rice vs. not DSR (PTR and AWD fields grouped together).
- **Irrigation (AWD):** alternate wetting and drying vs. not AWD (PTR and DSR fields grouped together).
- **Three-class (ALL):** AWD vs. PTR vs. DSR in one model.

The dataset has 1,283 labeled fields: 452 AWD, 420 DSR and 411 PTR (puddled transplanted rice).

## Input features

The notebook uses the feature tables released with the reference paper:

| Set | Columns | Description |
| --- | --- | --- |
| HC | 61 | Handcrafted Sentinel-1 radar features |
| SE | 64 | Google AlphaEarth satellite embeddings |
| Presto (P) | 5,760 or 6,272 | Pretrained Presto embeddings (depends on the date window) |

Two feature combinations are compared: `HC_SE` (the paper's 125-feature set) and `HC_PCA64_SE`, which adds Presto
compressed to 64 PCA components.

## Method

- **Models:** Random Forest and LightGBM (the paper's models); Extra Trees, RBF-SVM, XGBoost and CatBoost (proposed);
  majority class, logistic regression and a decision tree (reference checks).
- **Leakage-safe split:** fields with identical handcrafted features are grouped and never split across partitions.
  About 10% of fields (123) are held out as the test set, the same for every task and feature set.
- **Model selection:** each model tries four settings with `GridSearchCV` on 5-fold grouped cross-validation repeated
  twice. The best mean weighted F1 picks the model before the test set is used.
- **Preprocessing inside each training fold:** median imputation; for Presto, scaling and PCA(64).
- **Evaluation:** weighted F1 (the paper's metric), macro F1, accuracy, positive-class recall, and a 95% bootstrap
  confidence interval on the test set.
- **Error analysis (DSR):** Random Forest vs. CatBoost confusion matrices, feature importance, and feature patterns
  of false negatives vs. false positives.

## Results summary

| Task | Features | Selected model | Validation F1 | Test F1 (95% CI) | Paper F1 |
| --- | --- | --- | --- | --- | --- |
| DSR | HC + SE | CatBoost | 0.736 ± 0.016 | 0.721 (0.640–0.797) | 0.796 |
| DSR | HC + Presto PCA + SE | CatBoost | 0.741 ± 0.022 | 0.670 (0.582–0.744) | – |
| AWD | HC + SE | CatBoost | 0.685 ± 0.024 | 0.674 (0.585–0.762) | 0.697 |
| AWD | HC + Presto PCA + SE | RBF-SVM | 0.729 ± 0.032 | 0.695 (0.605–0.777) | – |
| 3-class | HC + SE | RBF-SVM | 0.540 ± 0.039 | 0.535 (0.438–0.620) | 0.616 |
| 3-class | HC + Presto PCA + SE | RBF-SVM | 0.570 ± 0.037 | 0.611 (0.519–0.694) | – |

On the DSR test set, Random Forest reached the highest weighted F1 (0.750), although CatBoost was selected by
cross-validation. The paper used a different (random, ungrouped) split and tuning, so its scores are a reference
point rather than a direct comparison. See the slides for details.

## How to run

1. Open `CMPE_257_Project.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run section 1. If the data are not found, it clones the
   [reference repository](https://github.com/microsoft/rice-irrigation-mapping-s1s2) and loads the feature files from it.
3. Section 7 installs the training packages: `scikit-learn>=1.5,<2`, `lightgbm>=4.5,<5`, `xgboost>=2.1,<4`,
   `catboost>=1.2.7,<2`, `pyarrow>=16`.
4. In section 8, choose `TASK` (`"DSR"`, `"AWD"` or `"ALL"`) and `FEATURE_SET` (`"HC_SE"` or `"HC_PCA64_SE"`), then run
   sections 8–15. The saved notebook shows the `DSR` / `HC_PCA64_SE` run.
5. Outputs (split, settings, fitted models, validation and test tables, predictions) are saved to
   `simple_results/<RUN_NAME>/<TASK>/<FEATURE_SET>/`. Set `SAVE_TO_DRIVE = True` to save them to Google Drive instead.

Random seed: 42. The feature-error analysis in section 14.3 runs only for the DSR task.

### Notebook sections

| Sections | Content |
| --- | --- |
| 1–6 | Data loading, cleaning, correlation, PCA and Presto exploration |
| 7–12 | Task and feature setup, grouped split, model training, model comparison, test evaluation |
| 13–15 | Random Forest vs. CatBoost evaluation, feature importance, error analysis, inference |

## Reference

**Machine Learning for Sustainable Rice Production: Region-Scale Monitoring of Water-Saving Practices in Punjab, India**, Shah et al., AAAI-26.

- [Paper](https://ojs.aaai.org/index.php/AAAI/article/download/41270/45231)
- [Reference code and dataset](https://github.com/microsoft/rice-irrigation-mapping-s1s2)

The paper and its released resources are credited to their respective authors. This repository contains a separate
course project based on those resources.
