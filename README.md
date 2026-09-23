# Customer Churn Prediction & Retention Analytics System

Predicts which telecom customers are likely to churn and quantifies the revenue impact, with an interactive web app for live predictions and portfolio-level risk analytics.

**🔗 Live demo:** [your-streamlit-url-here]
**📊 Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features)

---

## Problem Statement

Customer churn costs telecom companies significant recurring revenue. This project builds a machine learning system to identify at-risk customers *before* they churn, explain *why* they're at risk, and estimate the revenue impact of targeted retention efforts.

## Key Findings

- **Contract type is the strongest churn driver** — month-to-month customers churn at 42.7% vs. 2.8% for two-year contracts
- **Payment method matters** — electronic check users churn at 45.3%, nearly 3x the rate of automatic payment methods
- **Fiber optic customers churn 6x more** than customers with no internet service, suggesting pricing or service quality issues
- **Churn risk is front-loaded** — the majority of churn happens within the first 5 months of tenure

![EDA Overview](eda_churn_overview.png)

## Approach

1. **Data cleaning** — handled a hidden missing-data pattern in `TotalCharges` (blank values were logically $0 for brand-new customers, not random gaps)
2. **EDA** — identified key churn drivers across contract, payment, tenure, and service usage
3. **Feature engineering** — tenure buckets, services-count aggregation, one-hot encoding
4. **Class imbalance handling** — applied SMOTE oversampling (26.5% churn rate in raw data)
5. **Modeling** — compared Logistic Regression, Random Forest, and XGBoost
6. **Explainability** — used SHAP to identify and visualize feature-level drivers of individual predictions
7. **Retention analytics** — segmented customers into risk tiers and estimated revenue at risk
8. **Deployment** — built and deployed an interactive Streamlit app for live predictions

## Model Performance

| Model | ROC-AUC |
|---|---|
| Logistic Regression | [0.822] |
| Random Forest | [0.834] |
| XGBoost | [0.820] |

## Business Impact

- **$[$992,644.20] in annual revenue** identified as at-risk from high-churn-probability customers
- A targeted retention campaign saving even 20% of high-risk customers translates to an estimated **$[$198,528.84]** in recovered annual revenue

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · matplotlib/seaborn

## Project Structure

```
├── app.py                          # Streamlit app
├── requirements.txt                # Dependencies
├── notebooks/
│   └── churn_analysis.ipynb        # Full analysis: EDA → modeling → SHAP
├── data/
│   └── telco_churn_risk_scored.csv # Risk-scored customer data
├── models/
│   ├── churn_model.pkl             # Trained XGBoost model
│   └── model_features.pkl          # Feature list for inference
└── images/
    ├── eda_churn_overview.png
    ├── shap_summary_detail.png
    └── risk_by_contract.png
```

## Run Locally

```bash
git clone https://github.com/[your-username]/customer-churn-app.git
cd customer-churn-app
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements

- Add a "what-if" simulator to test the effect of retention offers on individual customers
- Incorporate customer support ticket data for richer features
- A/B test retention interventions on the highest-risk segment
