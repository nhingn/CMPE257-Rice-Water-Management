# CMPE257-Rice-Water-Management

A CMPE 257 machine learning project exploring how satellite-derived features can identify rice sowing and irrigation practices in Punjab, India.

## Project status

The team is developing the notebook and completing model experiments. This repository currently contains the project overview; code, notebooks, and results will be added later.

## Project goal

Evaluate several machine learning models and compare them with Random Forest and LightGBM, the model families used in the reference paper. Our goal is to investigate whether alternative models improve performance under a shared evaluation setup; improvements have not yet been established.

## Prediction tasks

- **Sowing (DSR):** Distinguish direct-seeded rice from transplanted rice, grouping Control and AWD fields as transplanted rice.
- **Irrigation (AWD):** Distinguish alternate wetting and drying from continuous flooding, grouping Control and DSR fields as continuously flooded for this task.
- **Three-class classification (ALL):** Predict AWD, Control/PTR, or DSR directly.

These labels follow the reference dataset's task definitions.

## Input features

The project uses feature tables released with the reference paper:

- **HC:** Handcrafted satellite features.
- **AlphaEarth (AE):** Learned satellite embeddings, called `SE` in the released filenames.
- **Presto (P):** Learned representations of satellite observations over time.

The planned comparisons include `HC_SE` and `HC_PCA64_SE`. The second combination adds Presto features reduced to 64 components using PCA.

## Models

| Role | Models |
| --- | --- |
| Reference model families | Random Forest, LightGBM |
| Alternative models | Extra Trees, RBF-SVM, XGBoost, CatBoost |
| Simple comparison models | Logistic Regression, Decision Tree |

## Evaluation approach

The notebook is being prepared for Google Colab. Models use a shared development/test split and the same validation folds. Records with identical handcrafted features are grouped together; these groups are proxies rather than verified field identities. Missing-value handling, scaling, and optional PCA are fitted within the training folds.

Hyperparameters are selected with `GridSearchCV` using weighted F1. Accuracy and macro F1 provide additional comparisons. The held-out test set is reserved for evaluation after development choices are finalized.

Published paper scores serve as contextual references. Differences in features, data splits, and tuning mean that a higher score alone does not establish an improvement over the paper. Retrained Random Forest and LightGBM models provide comparisons under our own evaluation setup.

## Reference

**Machine Learning for Sustainable Rice Production: Region-Scale Monitoring of Water-Saving Practices in Punjab, India** — Shah et al.

- [Paper](https://ojs.aaai.org/index.php/AAAI/article/download/41270/45231)
- [Reference code and dataset](https://github.com/microsoft/rice-irrigation-mapping-s1s2)

The paper and its released resources are credited to their respective authors. This repository contains a separate course project based on those resources.
