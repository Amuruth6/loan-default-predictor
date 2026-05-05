import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')
feature_names = joblib.load('features.pkl')

st.title("🏦 Loan Default Predictor")
st.write("Fill in applicant details to predict default risk")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female", "Sex Not Available"])
    loan_limit = st.selectbox("Loan Limit", ["cf", "ncf"])
    approv_in_adv = st.selectbox("Pre-Approved?", ["pre", "nopre"])
    loan_type = st.selectbox("Loan Type", ["type1", "type2", "type3"])
    loan_purpose = st.selectbox("Loan Purpose", ["p1", "p2", "p3", "p4"])
    Credit_Worthiness = st.selectbox("Credit Worthiness", ["l1", "l2"])
    open_credit = st.selectbox("Open Credit", ["nopc", "opc"])
    business_or_commercial = st.selectbox("Business/Commercial", ["nob/c", "b/c"])
    loan_amount = st.number_input("Loan Amount", min_value=0, value=200000)
    rate_of_interest = st.number_input("Rate of Interest (%)", min_value=0.0, value=5.0)
    Interest_rate_spread = st.number_input("Interest Rate Spread", value=0.5)
    Upfront_charges = st.number_input("Upfront Charges", min_value=0.0, value=1000.0)
    term = st.number_input("Loan Term (months)", min_value=0, value=360)
    Neg_ammortization = st.selectbox("Negative Amortization", ["not_neg", "neg_amm"])

with col2:
    interest_only = st.selectbox("Interest Only", ["not_int", "int_only"])
    lump_sum_payment = st.selectbox("Lump Sum Payment", ["not_lpsm", "lpsm"])
    property_value = st.number_input("Property Value", min_value=0, value=300000)
    construction_type = st.selectbox("Construction Type", ["sb", "mh"])
    occupancy_type = st.selectbox("Occupancy Type", ["pr", "sr", "ir"])
    Secured_by = st.selectbox("Secured By", ["home", "land"])
    total_units = st.selectbox("Total Units", ["1U", "2U", "3U", "4U"])
    income = st.number_input("Income", min_value=0, value=5000)
    credit_type = st.selectbox("Credit Type", ["EXP", "EQUI", "CRIF", "CIB"])
    Credit_Score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    co_applicant_credit_type = st.selectbox("Co-Applicant Credit Type", ["EXP", "EQUI", "CRIF", "CIB"])
    age = st.selectbox("Age Group", ["25-34", "35-44", "45-54", "55-64", "65-74", "<25", ">74"])
    submission_of_application = st.selectbox("Submission", ["to_inst", "not_inst"])
    LTV = st.number_input("LTV Ratio", min_value=0.0, value=80.0)
    Region = st.selectbox("Region", ["south", "North", "central", "North-East"])
    Security_Type = st.selectbox("Security Type", ["direct", "Indriect"])
    dtir1 = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=40.0)

if st.button("🔍 Predict Loan Default Risk"):
    from sklearn.preprocessing import LabelEncoder

    input_dict = {
        'loan_limit': loan_limit,
        'Gender': Gender,
        'approv_in_adv': approv_in_adv,
        'loan_type': loan_type,
        'loan_purpose': loan_purpose,
        'Credit_Worthiness': Credit_Worthiness,
        'open_credit': open_credit,
        'business_or_commercial': business_or_commercial,
        'loan_amount': loan_amount,
        'rate_of_interest': rate_of_interest,
        'Interest_rate_spread': Interest_rate_spread,
        'Upfront_charges': Upfront_charges,
        'term': term,
        'Neg_ammortization': Neg_ammortization,
        'interest_only': interest_only,
        'lump_sum_payment': lump_sum_payment,
        'property_value': property_value,
        'construction_type': construction_type,
        'occupancy_type': occupancy_type,
        'Secured_by': Secured_by,
        'total_units': total_units,
        'income': income,
        'credit_type': credit_type,
        'Credit_Score': Credit_Score,
        'co-applicant_credit_type': co_applicant_credit_type,
        'age': age,
        'submission_of_application': submission_of_application,
        'LTV': LTV,
        'Region': Region,
        'Security_Type': Security_Type,
        'dtir1': dtir1
    }

    input_df = pd.DataFrame([input_dict])

    # Encode text columns
    for col in input_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col])

    # Ensure column order matches training
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.divider()
    if prediction == 1:
        st.error("❌ High Risk — Loan Likely to DEFAULT")
        st.write(f"Default Probability: **{probability[1]*100:.1f}%**")
    else:
        st.success("✅ Low Risk — Loan Likely to be REPAID")
        st.write(f"Safe Probability: **{probability[0]*100:.1f}%**")