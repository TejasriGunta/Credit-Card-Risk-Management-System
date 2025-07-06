# Credit Card Risk Management System

## Introduction
This project was developed as part of the **IDFC Bank x IIT Bombay Case Study Competition**. The objective is to build a predictive model called a **Behaviour Score**, which estimates the probability of default for active credit card customers. This score can be used for risk management tasks such as credit limit adjustments, rate modifications, and targeted customer engagement.

## Problem Statement
Given anonymized credit card portfolio data from IDFC Bank, the task is to predict the probability of default (`bad_flag = 1`) for each customer. A robust Behaviour Score will help the bank proactively manage risk across its portfolio.

## Dataset Overview
The features in the dataset fall into several categories:
- `onus_attributes_*`: Internal bank attributes (e.g., credit limit).
- `transaction_attribute_*`: Transaction summaries across merchants/categories.
- `bureau_*`: Credit history and delinquencies.
- `bureau_enquiry_*`: Recent credit inquiry information.

Each account is uniquely identified by `account_number`.

## Data Cleaning
- Dropped columns with only unique values.
- Dropped columns with high missingness and less variance.
- Remaining `NaN` values were imputed with mean for numerical features and mode for categorical feautures.
- Since all features were numeric, no encoding was needed.
- removed highly correlated features to help reduce multicollinearity

## Model Selection
tested multiple industry-grade models and selected XGBoost for its high F1 score and recall.
              ![model_comparision_optimized](https://github.com/user-attachments/assets/03bc4511-b39d-4ae1-8333-878f3d4d4a80)

## Feature Selection & Importance Analysis
- Used base XGBoost model to derive feature importances and retained the **top 100** features.
- To ensure interpretability and better understanding of the data, calculated **SHAP** values.
              ![pic2_optimized](https://github.com/user-attachments/assets/df964f60-3196-4295-953b-227303d587bf)


## Final Model Pipeline with Results

### Stage 1: XGBoost
- Tuned using **Optuna** to maximize **recall** for stage1.
- Threshold strategy:
  - `>= 0.8`: High confidence fraud — flagged immediately.
  - `0.325 <= score < 0.8`: Borderline cases — sent to Stage 2.
  - `< 0.325`: Considered non-fraud.
![image](https://github.com/user-attachments/assets/38e2ab2b-1663-4186-8664-44f95b85e07e)

### Stage 2: Calibrated Gradient Boosting (High Precision)
- Borderline cases passed through a **GradientBoostingClassifier** wrapped in `CalibratedClassifierCV` (isotonic).
- Calibrated scores used to **rank borderline customers**.
- Top 30% highest-risk accounts among borderlines were flagged as fraud thus improving the precision and recall considerably.


## 📊 Final Results
```text
Final Precision: 0.0781
Final Recall:    0.9197
F1 Score:        0.1440
```

```text
Classification Report:
              precision    recall  f1-score   support
           0       1.00      0.84      0.91     19088
           1       0.08      0.92      0.14       274
    accuracy                           0.85     19362
   macro avg       0.54      0.88      0.53     19362
weighted avg       0.99      0.85      0.90     19362
```

This shows the **two-stage pipeline successfully achieved very high recall (91.9%)**, while maintaining considerably decent precision, ensuring to catch maximum defaulting customers.

##  Conclusion
The two-stage fraud detection system provides a **robust trade-off between recall and precision**:
- Stage 1 ensures **no major frauds are missed**.
- Stage 2 **controls false positives**, improving overall reliability.

Combined with explainability using **SHAP**, careful Optuna tuning, and calibrated risk scoring, this solution is **production-ready and scalable** for real-world financial risk management.

---

> Built for the IDFC Bank x IIT Bombay -Credit Risk Modeling Challenge

