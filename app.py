import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Load model and feature list
model = joblib.load('churn_model.pkl')
feature_names = joblib.load('model_features.pkl')

st.title("📊 Customer Churn Prediction & Retention Analytics")
st.markdown("Predict churn risk for a telecom customer and see what's driving it.")

tab1, tab2 = st.tabs(["🔮 Predict Churn", "📈 Portfolio Analytics"])

# ---------------- TAB 1: Single customer prediction ----------------
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    with col3:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    if st.button("Predict Churn Risk", type="primary"):
        # Build a single-row input matching training feature format
        input_dict = {col: 0 for col in feature_names}
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly_charges
        input_dict['TotalCharges'] = monthly_charges * max(tenure, 1)
        input_dict['SeniorCitizen'] = 1 if senior == "Yes" else 0

        # Set one-hot flags that exist in feature_names
        for flag_col, value in [
            (f'Contract_{contract}', 1),
            (f'InternetService_{internet}', 1),
            (f'PaymentMethod_{payment}', 1),
            (f'TechSupport_{tech_support}', 1),
            (f'Partner_{partner}', 1),
            (f'PaperlessBilling_{paperless}', 1),
        ]:
            if flag_col in input_dict:
                input_dict[flag_col] = value

        input_df = pd.DataFrame([input_dict])[feature_names]

        proba = model.predict_proba(input_df)[0, 1]

        st.divider()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Churn Probability", f"{proba:.1%}")
            if proba >= 0.7:
                st.error("🔴 High Risk")
            elif proba >= 0.4:
                st.warning("🟡 Medium Risk")
            else:
                st.success("🟢 Low Risk")

        with col_b:
            st.markdown("**What's driving this prediction:**")
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)
            fig, ax = plt.subplots(figsize=(8, 3))
            shap.summary_plot(shap_vals, input_df, plot_type='bar', show=False)
            st.pyplot(fig)

# ---------------- TAB 2: Portfolio-level analytics ----------------
with tab2:
    st.subheader("Customer Base Risk Overview")
    try:
        risk_df = pd.read_csv('telco_churn_risk_scored.csv')

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{len(risk_df):,}")
        high_risk_count = (risk_df['risk_tier'] == 'High Risk').sum()
        col2.metric("High Risk Customers", f"{high_risk_count:,}")
        revenue_at_risk = risk_df[risk_df['risk_tier'] == 'High Risk']['MonthlyCharges'].sum() * 12
        col3.metric("Annual Revenue at Risk", f"${revenue_at_risk:,.0f}")

        st.bar_chart(risk_df['risk_tier'].value_counts())

        st.markdown("**High Risk Customers**")
        st.dataframe(
            risk_df[risk_df['risk_tier'] == 'High Risk']
            [['customerID', 'churn_probability', 'MonthlyCharges', 'Contract']]
            .sort_values('churn_probability', ascending=False)
            .head(20)
        )
    except FileNotFoundError:
        st.warning("Upload telco_churn_risk_scored.csv to this folder to see portfolio analytics.")
