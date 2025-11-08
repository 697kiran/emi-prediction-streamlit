# streamlit_app.py (resolved & updated)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

# optional plotting libs
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# mlflow is optional at runtime; guard import to avoid build failures if not present
try:
    import mlflow
    import mlflow.sklearn
except Exception:
    mlflow = None

st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Assessment",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# Custom CSS
# ------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .eligible {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .high-risk {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .not-eligible {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------
# Helpers: build feature vector + safe divisions
# ------------------------
def safe_div(a, b, default=0.0):
    try:
        if pd.isna(a) or pd.isna(b):
            return default
        return a / b if b != 0 else default
    except Exception:
        return default


def build_feature_vector(input_df: pd.DataFrame, feature_columns: list):
    """
    Ensure the input_df (1-row) matches the exact columns used at training time.
    Missing columns are created and set to 0, extra columns are ignored.
    Returns a single-row DataFrame with columns in exactly feature_columns order.
    """
    if feature_columns is None:
        # If feature_columns not supplied, fall back to input_df itself
        return input_df.copy().reset_index(drop=True)

    fv = pd.DataFrame(columns=feature_columns)
    fv.loc[0] = 0  # initialize to zeros

    for c in input_df.columns:
        if c in fv.columns:
            # maintain numeric dtype where possible
            fv.at[0, c] = input_df.at[0, c]

    # Ensure numeric types for scaler
    fv = fv.apply(pd.to_numeric, errors="coerce").fillna(0)
    return fv


# ------------------------
# Robust model loading
# ------------------------
@st.cache_resource
def load_models(model_dir: str = "models"):
    """
    Loads classifier, regressor, scaler, label_encoders, and feature_columns.
    Expects files:
      - models/best_classifier.pkl
      - models/best_regressor.pkl
      - models/scaler.pkl
      - models/label_encoders.pkl
      - models/feature_columns.json
    Returns tuple (classifier, regressor, scaler, label_encoders, feature_columns)
    Missing artifacts are returned as None and a friendly error message is displayed.
    """
    cls_path = os.path.join(model_dir, "best_classifier.pkl")
    reg_path = os.path.join(model_dir, "best_regressor.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    label_enc_path = os.path.join(model_dir, "label_encoders.pkl")
    feature_cols_path = os.path.join(model_dir, "feature_columns.json")

    missing = []
    for p in [cls_path, reg_path, scaler_path, label_enc_path, feature_cols_path]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        # Do not raise — return Nones so app can show a readable message
        return None, None, None, None, None

    try:
        classifier = joblib.load(cls_path)
        regressor = joblib.load(reg_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(label_enc_path)
        with open(feature_cols_path, "r") as f:
            feature_columns = json.load(f)
        return classifier, regressor, scaler, label_encoders, feature_columns
    except Exception as e:
        # If any loading step fails, return Nones (app will present an error)
        st.error(f"Model loading exception: {e}")
        return None, None, None, None, None


classifier, regressor, scaler, label_encoders, feature_columns = load_models()

# ------------------------
# Sidebar and pages (fixed names without leading spaces)
# ------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    [
        "Home",
        "EMI Prediction",
        "Data Analytics",
        "Model Performance",
        "Data Management",
    ],
)

# ------------------------
# Header
# ------------------------
st.markdown('<h1 class="main-header">EMIPredict AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligent Financial Risk Assessment Platform</p>',
    unsafe_allow_html=True,
)

# ------------------------
# Home page
# ------------------------
if page == "Home":
    st.markdown("## Welcome to EMIPredict AI")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h3>EMI Eligibility</h3>
            <p>Get instant eligibility assessment for your EMI application with AI-powered risk analysis.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Maximum EMI Amount</h3>
            <p>Calculate the maximum EMI amount you can afford based on your financial profile.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Risk Assessment</h3>
            <p>Comprehensive financial risk analysis using ML models.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## Business Use Cases")
    tab1, tab2, tab3 = st.tabs(["Financial Institutions", "FinTech Companies", "Banks & Credit Agencies"])
    with tab1:
        st.markdown(
            "- Automate loan approval processes\n- Reduce manual underwriting time\n- Real-time eligibility assessment for walk-in customers"
        )
    with tab2:
        st.markdown("- Instant pre-qualification checks for digital platforms\n- Automated risk scoring for loan applications")
    with tab3:
        st.markdown("- Data-driven loan amount recommendations\n- Portfolio risk management and default prediction")

# ------------------------
# EMI Prediction page
# ------------------------
elif page == "EMI Prediction":
    st.markdown("## EMI Eligibility & Amount Prediction")

    if classifier is None or regressor is None:
        st.error(
            "Models not loaded. Please ensure the 'models/' folder contains the artifacts: "
            "best_classifier.pkl, best_regressor.pkl, scaler.pkl, label_encoders.pkl, feature_columns.json"
        )
        st.stop()

    # Input form
    with st.form("prediction_form"):
        st.markdown("### Personal Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        with c2:
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=40, value=5)
        with c3:
            company_type = st.selectbox("Company Type", ["MNC", "Local", "Government", "Startup"])
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
            family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)

        st.markdown("### Financial Information")
        c4, c5, c6 = st.columns(3)
        with c4:
            monthly_salary = st.number_input("Monthly Salary (INR)", min_value=0, max_value=5_000_000, value=50_000)
            monthly_rent = st.number_input("Monthly Rent (INR)", min_value=0, max_value=100_000, value=10_000)
            bank_balance = st.number_input("Bank Balance (INR)", min_value=0, max_value=10_000_000, value=100_000)
        with c5:
            current_emi_amount = st.number_input("Current EMI Amount (INR)", min_value=0, max_value=200_000, value=5_000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
            emergency_fund = st.number_input("Emergency Fund (INR)", min_value=0, max_value=5_000_000, value=50_000)
        with c6:
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=8, value=1)
            existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])

        st.markdown("### EMI Details")
        c7, c8, c9 = st.columns(3)
        with c7:
            emi_scenario = st.selectbox(
                "EMI Scenario",
                ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"],
            )
            requested_amount = st.number_input("Requested Amount (INR)", min_value=1000, max_value=2_000_000, value=100_000)
        with c8:
            requested_tenure = st.number_input("Requested Tenure (months)", min_value=3, max_value=84, value=24)

        st.markdown("### Monthly Expenses")
        c10, c11, c12 = st.columns(3)
        with c10:
            school_fees = st.number_input("School Fees (INR)", min_value=0, max_value=50_000, value=2_000)
            college_fees = st.number_input("College Fees (INR)", min_value=0, max_value=100_000, value=0)
        with c11:
            travel_expenses = st.number_input("Travel Expenses (INR)", min_value=0, max_value=20_000, value=3_000)
            groceries_utilities = st.number_input("Groceries & Utilities (INR)", min_value=0, max_value=50_000, value=8_000)
        with c12:
            other_monthly_expenses = st.number_input("Other Monthly Expenses (INR)", min_value=0, max_value=30_000, value=5_000)

        submitted = st.form_submit_button("Predict EMI Eligibility & Amount", use_container_width=True)

        if submitted:
            # ---------- compute derived numeric features ----------
            debt_to_income_ratio = safe_div(current_emi_amount, monthly_salary)
            total_expenses = monthly_rent + school_fees + college_fees + travel_expenses + groceries_utilities + other_monthly_expenses
            expense_to_income_ratio = safe_div(total_expenses, monthly_salary)
            affordability_ratio = safe_div((monthly_salary - current_emi_amount - total_expenses), (requested_amount * requested_tenure)) * 100
            employment_stability_score = years_of_employment * 0.3 + (0.7 if employment_type == "Government" else 0.0)
            financial_stability_score = safe_div(bank_balance, monthly_salary) * 0.4 + safe_div(emergency_fund, monthly_salary) * 0.6
            credit_utilization = safe_div(current_emi_amount, max(1.0, credit_score / 100.0))
            dependency_ratio = safe_div(dependents, family_size)

            # ---------- compute categorical buckets used in training ----------
            if age <= 30:
                age_group_val = "Young"
            elif age <= 40:
                age_group_val = "Adult"
            elif age <= 50:
                age_group_val = "Middle_Age"
            else:
                age_group_val = "Senior"

            if monthly_salary <= 30000:
                income_cat_val = "Low"
            elif monthly_salary <= 60000:
                income_cat_val = "Medium"
            elif monthly_salary <= 100000:
                income_cat_val = "High"
            else:
                income_cat_val = "Premium"

            # ---------- base numeric + binary features ----------
            base_dict = {
                "age": age,
                "monthly_salary": monthly_salary,
                "years_of_employment": years_of_employment,
                "monthly_rent": monthly_rent,
                "family_size": family_size,
                "dependents": dependents,
                "school_fees": school_fees,
                "college_fees": college_fees,
                "travel_expenses": travel_expenses,
                "groceries_utilities": groceries_utilities,
                "other_monthly_expenses": other_monthly_expenses,
                "current_emi_amount": current_emi_amount,
                "credit_score": credit_score,
                "bank_balance": bank_balance,
                "emergency_fund": emergency_fund,
                "requested_amount": requested_amount,
                "requested_tenure": requested_tenure,
                "debt_to_income_ratio": debt_to_income_ratio,
                "expense_to_income_ratio": expense_to_income_ratio,
                "affordability_ratio": affordability_ratio,
                "employment_stability_score": employment_stability_score,
                "financial_stability_score": financial_stability_score,
                "credit_utilization": credit_utilization,
                "dependency_ratio": dependency_ratio,
                "gender_encoded": 1 if gender == "Male" else 0,
                "marital_status_encoded": 1 if marital_status == "Married" else 0,
                "existing_loans_encoded": 1 if existing_loans == "Yes" else 0,
            }

            # ---------- helper: set one-hot columns using feature_columns ----------
            def apply_one_hot(base_row_dict, feature_columns, mappings):
                """
                base_row_dict: numeric + binary values (dict)
                feature_columns: list of training feature names (ordered)
                mappings: dict of {prefix: value} e.g. {"education": "Graduate", "employment_type": "Private"}
                Returns a 1-row DataFrame with exactly feature_columns columns.
                """
                fv = pd.DataFrame(columns=feature_columns)
                fv.loc[0] = 0
                # fill numeric/binary values that are present in feature_columns
                for k, v in base_row_dict.items():
                    if k in fv.columns:
                        try:
                            fv.at[0, k] = float(v)
                        except Exception:
                            fv.at[0, k] = v
                # For each mapping, set the matching one-hot column to 1 if it exists
                for prefix, val in mappings.items():
                    candidates = [
                        f"{prefix}_{val}",
                        f"{prefix}_{val.replace(' ', '_')}",
                        f"{prefix}_{val.replace('-', '_')}",
                        f"{prefix}_{val.replace(' ', '')}",
                    ]
                    candidates += [f"{prefix}_Unknown", f"{prefix}_nan", f"{prefix}_NA"]
                    found = False
                    for c in candidates:
                        if c in fv.columns:
                            fv.at[0, c] = 1
                            found = True
                            break
                    if not found:
                        for col in feature_columns:
                            if col.startswith(prefix + "_") and val.lower() in col.lower():
                                fv.at[0, col] = 1
                                found = True
                                break
                fv = fv.apply(pd.to_numeric, errors="coerce").fillna(0)
                return fv

            cat_mappings = {
                "education": education,
                "employment_type": employment_type,
                "company_type": company_type,
                "house_type": house_type,
                "emi_scenario": emi_scenario,
                "age_group": age_group_val,
                "income_category": income_cat_val,
            }

            # build the final input vector DataFrame (exactly match feature_columns)
            try:
                input_vector = apply_one_hot(base_dict, feature_columns, cat_mappings)
            except Exception as e:
                st.error(f"Error building feature vector: {e}")
                st.stop()

            # ---------- debug: check mismatch between input and model expectation (non-fatal) ----------
            model_expected = None
            if hasattr(classifier, "feature_names_in_"):
                model_expected = list(classifier.feature_names_in_)
            else:
                try:
                    booster = classifier.get_booster()
                    model_expected = booster.feature_names
                except Exception:
                    model_expected = None

            if model_expected is not None:
                set_expected = set(model_expected)
                set_input = set(input_vector.columns)
                extra = set_input - set_expected
                missing = set_expected - set_input
                if extra or missing:
                    st.warning(
                        "Feature name mismatch detected. Extra cols (not in model): "
                        f"{list(sorted(extra))[:8]} ...; Missing cols (expected by model): {list(sorted(missing))[:8]} ..."
                    )

            # ---------- scale, predict and display ----------
            try:
                if scaler is not None:
                    input_scaled = scaler.transform(input_vector)
                else:
                    input_scaled = input_vector.values

                eligibility_pred_raw = classifier.predict(input_scaled)[0]
                try:
                    eligibility_proba = classifier.predict_proba(input_scaled)[0]
                except Exception:
                    eligibility_proba = None
                max_emi_pred = regressor.predict(input_scaled)[0]

                # map numeric labels via label encoder if present
                pred_label = None
                try:
                    if isinstance(eligibility_pred_raw, (int, np.integer)) and isinstance(label_encoders, dict) and label_encoders.get("emi_eligibility") is not None:
                        pred_label = label_encoders["emi_eligibility"].inverse_transform([int(eligibility_pred_raw)])[0]
                    else:
                        pred_label = str(eligibility_pred_raw)
                except Exception:
                    pred_label = str(eligibility_pred_raw)

                # Display results
                st.markdown("---")
                st.markdown("## Prediction Results")

                col_r1, col_r2 = st.columns(2)

                with col_r1:
                    if pred_label.lower() == "eligible":
                        st.markdown('<div class="prediction-result eligible">✅ EMI ELIGIBLE</div>', unsafe_allow_html=True)
                    elif "high" in pred_label.lower() or "risk" in pred_label.lower():
                        st.markdown('<div class="prediction-result high-risk">⚠️ HIGH RISK</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="prediction-result not-eligible">❌ NOT ELIGIBLE</div>', unsafe_allow_html=True)

                    # Probability distribution (if available)
                    if eligibility_proba is not None:
                        try:
                            prob_df = pd.DataFrame({"Category": classifier.classes_, "Probability": eligibility_proba})
                            fig_prob = px.bar(prob_df, x="Category", y="Probability", title="Eligibility Probability Distribution")
                            st.plotly_chart(fig_prob, use_container_width=True)
                        except Exception:
                            st.write("Probability plot unavailable.")
                    else:
                        st.write("Probability scores not available for this classifier.")

                with col_r2:
                    st.markdown(f'<div class="prediction-result eligible">💰 Maximum EMI: ₹{max_emi_pred:,.0f}</div>', unsafe_allow_html=True)

                    financial_summary = {
                        "Monthly Income": f"₹{int(monthly_salary):,}",
                        "Current EMI": f"₹{int(current_emi_amount):,}",
                        "Predicted Max EMI": f"₹{int(round(max_emi_pred)):,.0f}",
                        "Available EMI Capacity": f"₹{int(round(max_emi_pred - current_emi_amount)):,.0f}",
                        "Debt-to-Income Ratio": f"{debt_to_income_ratio:.2%}",
                        "Credit Score": credit_score,
                    }

                    for key, value in financial_summary.items():
                        st.metric(key, value)

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ------------------------
# Data Analytics (static demo)
# ------------------------
elif page == "Data Analytics":
    st.markdown("## Financial Data Analytics Dashboard")
    st.markdown("### EMI Scenario Distribution")
    sample_data = {
        "EMI Scenario": ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"],
        "Count": [80000, 80000, 80000, 80000, 80000],
        "Average Amount": [50000, 100000, 500000, 300000, 200000],
    }
    sample_df = pd.DataFrame(sample_data)
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(sample_df, values="Count", names="EMI Scenario", title="EMI Applications by Scenario")
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_bar = px.bar(sample_df, x="EMI Scenario", y="Average Amount", title="Average EMI Amount by Scenario")
        st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------
# Model Performance (static demo)
# ------------------------
elif page == "Model Performance":
    st.markdown("## Model Performance Dashboard")
    st.markdown("### Classification Model Performance")
    classification_metrics = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "SVC", "Decision Tree", "Gradient Boosting"],
        "Accuracy": [0.85, 0.92, 0.94, 0.87, 0.81, 0.90],
        "F1-Score": [0.84, 0.91, 0.93, 0.86, 0.80, 0.89],
        "Precision": [0.85, 0.92, 0.94, 0.88, 0.82, 0.90],
        "Recall": [0.84, 0.91, 0.93, 0.86, 0.80, 0.89],
    }
    class_df = pd.DataFrame(classification_metrics)
    st.dataframe(class_df, use_container_width=True)
    fig_class = px.bar(class_df, x="Model", y=["Accuracy", "F1-Score", "Precision", "Recall"], title="Classification Model Comparison", barmode="group")
    st.plotly_chart(fig_class, use_container_width=True)

    st.markdown("### Regression Model Performance")
    regression_metrics = {
        "Model": ["Linear Regression", "Random Forest", "XGBoost", "SVR", "Decision Tree", "Gradient Boosting"],
        "RMSE": [2500, 1800, 1600, 2200, 2800, 1900],
        "MAE": [1800, 1200, 1100, 1600, 2000, 1300],
        "R² Score": [0.75, 0.85, 0.88, 0.78, 0.70, 0.83],
    }
    reg_df = pd.DataFrame(regression_metrics)
    st.dataframe(reg_df, use_container_width=True)
    fig_reg = px.bar(reg_df, x="Model", y=["RMSE", "MAE"], title="Regression Model Error Comparison", barmode="group")
    st.plotly_chart(fig_reg, use_container_width=True)

# ------------------------
# Data management
# ------------------------
elif page == "Data Management":
    st.markdown("## Data Management Interface")
    st.markdown("### Upload New Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        st.success(f"Uploaded file with {df_new.shape[0]} records and {df_new.shape[1]} columns")
        st.dataframe(df_new.head(), use_container_width=True)

    st.markdown("### Data Processing Options")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Clean Data", use_container_width=True):
            st.success("Data cleaning initiated!")
    with c2:
        if st.button("Feature Engineering", use_container_width=True):
            st.success("Feature engineering completed!")
    with c3:
        if st.button("Retrain Models", use_container_width=True):
            st.success("Model retraining started!")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 2rem;">
    <p> EMIPredict AI - Powered by Advanced Machine Learning</p>
    <p>Built with Streamlit • MLflow (optional) • XGBoost • Random Forest</p>
</div>
""",
    unsafe_allow_html=True,
)
