# Breast Tumor Classification

**Author:** Arezoo Jafari  
**Tools:** Python · scikit-learn · pandas · matplotlib · seaborn · Jupyter Notebook

---

## Overview

This project addresses a critical healthcare challenge:  
**Can we predict whether a breast tumor is benign or malignant based on cell characteristics — and do it in a way that is interpretable and clinically useful?**

Using the UCI Breast Cancer Wisconsin dataset, this notebook walks through a complete, end-to-end machine learning pipeline, focusing on both performance and **interpretability** — a key requirement in medical applications.

---

## 🔍 Problem Statement

Early and accurate detection of breast cancer is essential. While machine learning models can offer strong predictive performance, clinical decision-making requires more than just accuracy — **doctors need to understand the *why*** behind a prediction.

The main goal of this project was to:
- Build a reliable model that can distinguish between benign and malignant tumors
- Identify which features contribute most to the predictions
- Prioritize **interpretability and robustness** over black-box complexity

---

##  Dataset

The project uses the [UCI Breast Cancer Wisconsin (Original) Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)), consisting of ~700 clinical samples. Each instance describes 10 features extracted from cell images (e.g., clump thickness, cell shape, bare nuclei), plus a target class (benign or malignant).

---

## Methodology

1. **Data Preprocessing**
   - Handled missing values in the `Bare Nuclei` feature using **class-wise median imputation**
   - Checked class balance (approx. 65% benign, 35% malignant)

2. **Exploratory Data Analysis**
   - Visualized distributions and class separation via boxplots
   - Used **Mutual Information** to quantify individual feature relevance
   - Identified high correlation between features (e.g., Cell Size and Cell Shape)

3. **Modeling**
   - Trained a **Logistic Regression** model with **L1 regularization** for built-in feature selection and interpretability
   - Performed **hyperparameter tuning** using grid search
   - Validated performance with **5-fold cross-validation**

4. **Evaluation Metrics**
   - **AUC:** 0.995  
   - **Precision:** 95%  
   - **Recall:** 98%  
   - **Accuracy:** ~96.8% (cross-validated)

5. **Feature Importance & Interpretation**
   - Identified top predictors: **Clump Thickness**, **Bare Nuclei**, **Mitoses**
   - Compared **odds ratios** (from logistic regression) with **permutation importance** (model reliance)
   - Discovered **Marginal Adhesion** had weak importance despite a positive odds ratio → removing it improved AUC from **0.9951 to 0.9970**

---

## Key Takeaways

- **Interpretability matters:** Logistic regression with L1 regularization offered both performance and transparency
- **Not all “important-looking” features help the model:** Odds ratios can be misleading if a feature’s signal is redundant
- **Robust model design:** Final model relies on a small set of strong, non-redundant features

---

## 📁 Repository Structure

```bash
📦 breast_tumor_classification/
├── final_breast_cancer_notebook.ipynb   # Full analysis + modeling
├── README.md                            # Project overview
├── requirements.txt                     # Required packages (optional)
└── figures/                             # Plots and visuals (optional)
