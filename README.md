# Bankruptcy Prediction

Early detection of corporate bankruptcy allows banks, investors, and credit agencies 
to take preventive action instead of reacting after a default.

This project builds an ML model that predicts whether a company will go bankrupt 
based on 75 financial ratios from balance sheet, income statement, and cash flow statement.

---

## Dataset

- **Source:** [Kaggle — Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)  
  Originally from the Taiwan Economic Journal (1999–2009)
- **Size:** 6,819 companies, 95 features, binary target `Bankrupt?`
- **Class imbalance:** 3.2% bankrupt (220 out of 6,819) — shapes both modeling and evaluation

---

## Project Structure

```
├── data.csv
├── 01_EDA.ipynb                    # Exploratory data analysis
├── 02_Preprocessing_and_Modeling.ipynb   # Preprocessing, model training, evaluation
├── 03_Interpretation.ipynb         # SHAP values, error analysis
└── xgb_bankruptcy_model.joblib     # Saved final model
```

---

## Pipeline

```
Raw data → EDA → Drop multicollinear features → Train/test split
→ Outlier clipping (1st–99th percentile) → Model training → Threshold tuning → SHAP interpretation
```

---

## Results

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (threshold 0.5) | 0.84 | 0.19 | 0.31 | 0.946 |
| Random Forest (threshold 0.5) | 0.41 | 0.49 | 0.44 | 0.950 |
| XGBoost (threshold 0.5) | 0.48 | 0.55 | 0.51 | 0.964 |
| **XGBoost (threshold 0.2)** | **0.68** | **0.54** | **0.60** | **0.964** |

**Best model:** XGBoost with tuned threshold 0.20  
Catches 68% of real bankruptcies (30 out of 44) with precision of 0.54.

Lowering the threshold from 0.5 → 0.20 increased recall by +0.20 
with almost no drop in precision (0.55 → 0.54).

---

## Key Bankruptcy Signals (SHAP)

1. **Retained Earnings to Total Assets** — companies without accumulated reserves are most vulnerable
2. **Persistent EPS** — sustained earnings deterioration is the strongest individual signal
3. **Quick Ratio** — near-zero liquidity consistently drives bankruptcy predictions
4. **Total debt/Total net worth** — overleveraged companies are at high risk

---

## Why Not Accuracy?

Always predicting "not bankrupt" gives 96.8% accuracy while catching zero bankruptcies.  
Missing a real bankruptcy (false negative) is far costlier than a false alarm.  
This project focuses on **recall** for the bankrupt class as the primary metric.

---

## How to Use the Saved Model

```python
from joblib import load
import pandas as pd

# Load model
model = load('xgb_bankruptcy_model.joblib')

# Prepare your data (same 75 features, clipped at 1st–99th percentile)
X_new = pd.DataFrame([your_company_data])

# Predict with tuned threshold
proba = model.predict_proba(X_new)[:, 1]
prediction = (proba >= 0.20).astype(int)

# 1 = Bankrupt, 0 = Not Bankrupt
print(f'Probability: {proba[0]:.3f}')
print(f'Prediction: {"Bankrupt" if prediction[0] == 1 else "Not Bankrupt"}')
```

---

## Limitations

The model works only with financial ratios. 14 out of 44 test bankruptcies were missed —
these companies were financially indistinguishable from healthy ones.  
Bankruptcies caused by external shocks, fraud, or operational failures 
that do not appear in financial statements cannot be detected by this model.