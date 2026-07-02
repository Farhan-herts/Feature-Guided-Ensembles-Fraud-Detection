# Feature-Guided Bagging Ensembles for Robust Credit Card Fraud Detection

## MSc Final Project

**Author:** Farhan Ali  
**University:** University of Hertfordshire  
**Programme:** MSc Data Science and Analytics  
**Year:** 2026

---

# Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a very small fraction of all transactions. Traditional machine learning models often struggle to detect minority-class fraud while maintaining a low false positive rate.

This project proposes a **Feature-Guided Bagging Ensemble Framework** that organizes features into meaningful domain-specific groups and evaluates both individual feature-view models and multiview soft-voting ensembles.

The framework investigates whether combining specialized feature groups produces better fraud detection performance than conventional single-view bagging models.

---

# Research Objectives

The project aims to:

- Develop robust Decision Tree Bagging models for fraud detection.
- Organize features into domain-guided feature views.
- Compare individual feature-view models with multiview ensembles.
- Investigate the impact of different class imbalance handling techniques.
- Select the optimal balancing strategy using validation data.
- Compare balanced and unbalanced training pipelines.
- Identify the strongest individual feature view.
- Identify the strongest multiview ensemble.

---

# Dataset

This project uses the **IEEE-CIS Fraud Detection Dataset** released on Kaggle.

Dataset:

https://www.kaggle.com/competitions/ieee-fraud-detection

Due to GitHub file size limitations and Kaggle licensing, the dataset is **NOT included** in this repository.

---

# Dataset Placement

After downloading the dataset from Kaggle, create the following directory:

```
project_root/
│
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   ├── test_identity.csv
│   └── sample_submission.csv
```

The scripts assume the dataset is stored inside the **data/** directory.

---

# Repository Structure

```
Feature-Guided-Ensembles-Fraud-Detection/

│
├── scripts/
│     01_load_merge.py
│     02_split.py
│     03_preprocess.py
│     04_baselines.py
│     05_feature_views.py
│     06_bagging_standard.py
│     07_multiview_bagging.py
│     08_prepare_feature_views_v2.py
│
│     09a_individual_A_tuned.py
│     09b_individual_B_tuned.py
│     09c_individual_T_tuned.py
│     09d_individual_I_tuned.py
│
│     10_ab_multiview.py
│     11_it_multiview.py
│     12_tbi_multiview.py
│     13_abi_multiview.py
│     14_atb_multiview.py
│     15_atbi_multiview.py
│
│     16_compare_all_experiments.py
│
│     17a_A_balance_method_comparison.py
│     17b_B_balance_method_comparison.py
│     17c_AB_balance_method_comparison.py
│
│     18_select_best_balance_method.py
│
│     19a_A_best_balanced.py
│     19b_B_best_balanced.py
│     19c_T_best_balanced.py
│     19d_I_best_balanced.py
│
│     20_ab_best_balanced.py
│     21_it_best_balanced.py
│     22_tbi_best_balanced.py
│     23_abi_best_balanced.py
│     24_atb_best_balanced.py
│     25_atbi_best_balanced.py
│
│     26_compare_balanced_vs_unbalanced.py
│
│     27_compare_standard_bagging_vs_multiview_ensemble.py
│
│     28_exploratory_analysis.py
│
├── results/
│
├── report/
│     MSc_Report.pdf
│
├── figures/
│
├── requirements.txt
│
└── README.md
```

---

# Feature Views

The project groups features into four meaningful feature views.

| View | Description |
|------|-------------|
| A | Aggregation Features |
| B | Behaviour Features |
| T | Temporal Features |
| I | Identity Features |

These feature groups are used to construct both individual bagging models and multiview ensembles.

---

# Machine Learning Pipeline

The complete workflow consists of:

1. Dataset loading and merging
2. Stratified train/validation/test split
3. Missing value imputation
4. Frequency encoding
5. Feature view construction
6. Standard Bagging models
7. Multiview Soft Voting Ensembles
8. Class imbalance experiments
9. Selection of best balancing strategy
10. Balanced vs Unbalanced comparison
11. Final Standard vs Multiview comparison

---

# Class Imbalance Handling Methods

The project compares four balancing approaches:

- Baseline (No Resampling)
- Random Under Sampling
- SMOTE
- Random Under Sampling + SMOTE

The optimal strategy is selected using validation performance before evaluating on the test set.

---

# Evaluation Metrics

The models are evaluated using:

- PR-AUC
- ROC-AUC
- Accuracy
- Precision
- Recall
- F1 Score
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- True Positive Rate (TPR)
- True Negative Rate (TNR)

For model selection, **PR-AUC** is treated as the primary evaluation metric because of the severe class imbalance.

---

# Experimental Workflow

```
01 Load Dataset
        ↓
02 Train / Validation / Test Split
        ↓
03 Data Preprocessing
        ↓
04 Baseline Models
        ↓
05 Feature View Construction
        ↓
06 Standard Bagging
        ↓
07 Multiview Soft Voting
        ↓
08 Feature Preparation
        ↓
09 Individual Feature View Models
        ↓
10–15 Multiview Models
        ↓
16 Compare All Experiments
        ↓
17 Compare Balancing Methods
        ↓
18 Select Best Balancing Method
        ↓
19–25 Re-run Using Selected Method
        ↓
26 Balanced vs Unbalanced Comparison
        ↓
27 Standard vs Multiview Comparison
        ↓
28 Exploratory Analysis
```

---

# Main Findings

The experiments demonstrate that:

- Behaviour (B) is the strongest individual feature view.
- The ATB (Aggregation + Temporal + Behaviour) ensemble achieves the strongest multiview performance.
- Baseline (no resampling) performs better overall than random under-sampling and SMOTE for this dataset.
- Multiview soft-voting ensembles generally provide more robust fraud ranking than individual feature-view models.

---

# Results

The repository includes:

- Performance metrics (CSV)
- Summary reports (JSON)
- Confusion matrices
- Performance visualizations
- Comparison plots
- Final experiment rankings

All generated outputs are available in the **results/** directory.

---

# Software Requirements

- Python 3.11+
- pandas
- numpy
- matplotlib
- scikit-learn
- imbalanced-learn
- scipy

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the scripts sequentially from **01** to **28**.

Each script generates intermediate outputs that are used by later stages.

---

# Citation

If you use this repository for academic purposes, please cite the associated MSc dissertation.

---

# License

This repository is intended for academic and research purposes only.

The IEEE-CIS Fraud Detection dataset is subject to Kaggle's licensing terms and is **not redistributed** in this repository.

---

# Contact

**Farhan Ali**

MSc Data Science and Analytics

University of Hertfordshire

GitHub:
https://github.com/farhanaleherts
