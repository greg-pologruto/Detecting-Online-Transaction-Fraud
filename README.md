# Digital Online Transaction Fraud Detection

A comparison of statistical learning models for flagging fraudulent online transactions, built in R. The project covers data cleaning, exploratory analysis, handling severe class imbalance, threshold tuning, and evaluation with metrics that matter for fraud (sensitivity over raw accuracy).

**Author:** Greg Pologruto · **Date:** May 2026

---

## Overview

Fraud detection is a highly imbalanced classification problem: only about 6.5% of transactions in this dataset are fraudulent. A model that labels everything as legitimate would be ~93% accurate and completely useless. This project therefore puts the most weight on **sensitivity** (the share of fraud cases caught) while also tracking precision, specificity, F1, and balanced accuracy.

Models compared:

| Model | Imbalance handling | Threshold selection |
|---|---|---|
| Logistic Regression (with natural cubic splines) | Class weights (fraud = 5) | ROC-optimal cutoff |
| K-Nearest Neighbors (`kknn`) | Downsampling inside 10-fold CV | Max F1 subject to precision ≥ 0.10 |
| Random Forest (1,000 trees, `mtry = 3`) | Stratified sampling (`sampsize`) | Max F1 |
| Boosted Trees (`gbm`, 5,000 trees, depth 3, λ = 0.001) | None | ROC-optimal cutoff |
| BART (`pbart`, 100 trees) | Upsampling | Fixed cutoff of 0.05 |

## Dataset

- **7,500** synthetic digital payment transactions, one row per transaction
- **Target:** `fraud_label` (0 = legitimate, 1 = fraudulent) — 93.5% / 6.5%
- **Predictors** (after dropping `transaction_id` and `user_id`):

| Category | Variables |
|---|---|
| Financial | `transaction_amount`, `avg_transaction_amount`, `transaction_type`, `payment_mode`, `is_international` |
| Device | `device_type`, `device_location` |
| Behavioral | `account_age_days`, `transaction_hour`, `login_attempts_last_24h`, `previous_failed_attempts` |
| Risk | `ip_risk_score` |

The code expects the file `Digital_Payment_Fraud_Detection_Dataset.csv` in the project root.

## Methodology

1. **Preprocessing** – cleaned column names (`janitor::clean_names`), converted categorical fields to factors, removed ID columns.
2. **Splitting** – stratified 80/20 train/test split, with a further 80/20 train/validation split of the training data for tuning and threshold selection.
3. **Exploratory analysis** – boxplots, violin plots, and stacked bar charts comparing each predictor across classes.
4. **Modeling** – five classifiers, each paired with an imbalance strategy (see table above).
5. **Threshold tuning** – cutoffs chosen on validation data via ROC analysis or F1 search instead of the default 0.5.
6. **Evaluation** – confusion matrices and summary metrics on the held-out test set (1,500 transactions, 98 fraudulent).

The full report (`Final_Project_Report.pdf`) includes the mathematical background for each method, along with cross-validation and natural cubic splines.

## Results

Test-set performance, with **fraud as the positive class**:

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | Balanced Acc. |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.223 | **0.888** | 0.176 | 0.070 | 0.130 | 0.532 |
| KNN | 0.427 | 0.704 | 0.407 | 0.077 | 0.138 | **0.556** |
| BART | 0.267 | 0.735 | 0.235 | 0.063 | 0.116 | 0.485 |
| Random Forest | 0.908 | 0.011 | 0.967 | 0.021 | 0.014 | 0.489 |
| Boosted Trees | 0.849 | 0.061 | 0.904 | 0.043 | 0.050 | 0.483 |

### Key takeaways

- **Accuracy is misleading here.** Random Forest and Boosted Trees have the highest accuracy only because they label nearly everything as legitimate, missing almost all fraud.
- **Logistic regression was the preferred model.** It caught 87 of 98 fraudulent transactions (89% sensitivity), at the cost of many false positives — an acceptable trade-off when missed fraud is more costly than a flagged legitimate payment.
- **Signal in the data is weak.** Balanced accuracy stays close to 0.5 for every model, and the class distributions overlap heavily for most predictors (e.g., IP risk score means of 0.50 vs. 0.51). Account age showed the clearest visual difference between classes.

## Future Work

- Try more advanced gradient boosting (XGBoost, LightGBM) and neural networks
- Feature engineering, e.g., the ratio of transaction amount to the user's average amount
- Dimensionality reduction (PCA) and formal feature selection
- Cost-sensitive evaluation that weighs false negatives vs. false positives in dollar terms
- Precision–recall AUC as an additional imbalance-aware metric

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. *An Introduction to Statistical Learning: with Applications in R* (2nd ed.). Springer, 2023.
- [Deciding Threshold for GLM Logistic Regression Model in R](https://www.geeksforgeeks.org/machine-learning/deciding-threshold-for-glm-logistic-regression-model-in-r/) – GeeksforGeeks
- [ns: Generate a Basis Matrix for Natural Cubic Splines](https://www.rdocumentation.org/packages/splines/versions/3.6.2/topics/ns) – R Documentation
- [How to Calculate the OOB of Random Forest in R?](https://www.geeksforgeeks.org/r-machine-learning/how-to-calculate-the-oob-of-random-forest-in-r/) – GeeksforGeeks

## AI Usage

ChatGPT was used to generate the LaTeX for the methodology equations and to help style plots and tables.
