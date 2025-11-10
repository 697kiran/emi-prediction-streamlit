# streamlit_app.py (final integrated)
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
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 0.25rem solid #1f77b4;}
    .prediction-result { font-size: 1.5rem; font-weight: bold; padding: 1rem; border-radius: 0.5rem; text-align: center; margin: 1rem 0; }
    .eligible { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .high-risk { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
    .not-eligible { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------
# Helpers
# ------------------------
def safe_div(a, b, default=0.0):
    try:
        if pd.isna(a) or pd.isna(b):
            return default
        return a / b if b != 0 else default
    except Exception:
        return default


def build_input_vector(base_dict: dict, cat_mappings: dict, model_feature_names: list):
    """
    Build a 1-row DataFrame with exactly model_feature_names columns.
    - base_dict: numeric and pre-encoded binary columns {col: value}
    - cat_mappings: {prefix: value} for categorical features (e.g., {"age_group": "Young"})
    - model_feature_names: canonical list (from classifier.feature_names_in_ or booster.feature_names)
    """
    fv = pd.DataFrame(columns=model_feature_names)
    fv.loc[0] = 0

    # Fill numeric/binary base columns that match
    for k, v in base_dict.items():
        if k in fv.columns:
            fv.at[0, k] = v

    # Detect expected dummy prefixes from model_feature_names
    prefixes = {}
    for col in model_feature_names:
        if "_" in col:
            prefix = col.split("_", 1)[0]
            prefixes.setdefault(prefix, []).append(col)

    # For each categorical mapping, set the exact dummy expected by the model if model uses dummies
    for prefix, val in cat_mappings.items():
        val_str = "Unknown" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)
        if prefix in prefixes:
            # Model expects one-hot dummies for this prefix
            for expected_col in prefixes[prefix]:
                # compare expected suffix to val (try multiple transforms)
                suffix = expected_col.split(prefix + "_", 1)[1]
                matches = (
                        suffix.lower() == val_str.lower()
                        or suffix.lower() == val_str.replace(" ", "_").lower()
                        or suffix.lower() == val_str.replace("-", "_").lower()
                )
                fv.at[0, expected_col] = 1 if matches else 0
        else:
            # Model expects a plain column like 'age_group'
            if prefix in fv.columns:
                fv.at[0, prefix] = val_str

    # Ensure numeric types, fill NaNs with 0 where appropriate
    for c in fv.columns:
        try:
            fv[c] = pd.to_numeric(fv[c], errors="ignore")
        except Exception:
            pass

    fv = fv.where(pd.notnull(fv), 0)
    return fv


# ------------------------
# Robust load_models (permanent: model-driven features + XGBoost compatibility adjustments)
# ------------------------
# Replace your existing load_models with this function (drop-in)
@st.cache_resource
def load_models(model_dir: str = "models"):
    """
    Robust loader that wraps XGBoost models to avoid cross-version pickle errors.
    Returns: classifier, regressor, scaler, label_encoders, feature_columns, message
    """
    cls_path = os.path.join(model_dir, "best_classifier.pkl")
    reg_path = os.path.join(model_dir, "best_regressor.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    label_enc_path = os.path.join(model_dir, "label_encoders.pkl")
    feature_cols_path = os.path.join(model_dir, "feature_columns.json")

    # required checks
    for p in [cls_path, reg_path]:
        if not os.path.exists(p):
            return None, None, None, None, None, f"Missing required artifact: {p}"

    # load artifacts
    try:
        classifier_raw = joblib.load(cls_path)
    except Exception as e:
        return None, None, None, None, None, f"Failed loading classifier: {e}"

    try:
        regressor = joblib.load(reg_path)
    except Exception as e:
        return None, None, None, None, None, f"Failed loading regressor: {e}"

    scaler = None
    scaler_msg = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            scaler = None
            scaler_msg = f"Failed loading scaler.pkl: {e}"
    else:
        scaler_msg = "scaler.pkl not found"

    label_encoders = None
    if os.path.exists(label_enc_path):
        try:
            label_encoders = joblib.load(label_enc_path)
        except Exception:
            label_encoders = None

    feature_columns = None
    if os.path.exists(feature_cols_path):
        try:
            with open(feature_cols_path, "r") as f:
                feature_columns = json.load(f)
        except Exception:
            feature_columns = None

    # Helper wrapper around XGBoost models to avoid attribute errors across versions
    class SafeXGBWrapper:
        def __init__(self, model):
            self.model = model
            # Attempt to set safe defaults for common missing attrs
            for attr, default in [
                ("use_label_encoder", False),
                ("gpu_id", None),
                ("tree_method", None),
                ("n_gpus", None),
                ("eval_metric", "logloss"),
            ]:
                try:
                    if not hasattr(self.model, attr):
                        setattr(self.model, attr, default)
                except Exception:
                    # ignore if cannot set
                    pass

        def predict(self, X, **kwargs):
            # Primary attempt: let sklearn wrapper handle it
            try:
                return self.model.predict(X, **kwargs)
            except AttributeError as ae:
                # if missing attribute raised, try to set common attrs then retry
                missing = str(ae)
                for attr in ["gpu_id", "use_label_encoder", "tree_method", "n_gpus"]:
                    if not hasattr(self.model, attr):
                        try:
                            setattr(self.model, attr, None)
                        except Exception:
                            pass
                # second attempt
                try:
                    return self.model.predict(X, **kwargs)
                except Exception:
                    # fallback: if underlying booster available, run booster.predict on DMatrix
                    try:
                        import xgboost as xgb
                        booster = None
                        try:
                            booster = self.model.get_booster()
                        except Exception:
                            # if wrapped sklearn estimator stores .get_booster differently, try attribute
                            booster = getattr(self.model, "booster_", None)
                        if booster is not None:
                            dmat = xgb.DMatrix(X)
                            # For classifiers, booster.predict returns raw scores; this is a fallback
                            return booster.predict(dmat)
                    except Exception:
                        pass
                    # re-raise the original if nothing works
                    raise

        def predict_proba(self, X, **kwargs):
            # Not all XGBoost wrappers implement predict_proba; try predict then transform if needed
            try:
                if hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(X, **kwargs)
                # else fallback to predict and try to shape into probabilities if multiclass
                raw = self.predict(X, **kwargs)
                # if raw already 2D (proba-like), return
                arr = np.array(raw)
                if arr.ndim == 2:
                    return arr
                # if 1D, try to convert via softmax for multiclass (unknown classes) or two-class mapping
                # attempt safe softmax
                try:
                    if arr.ndim == 1:
                        # if classifier was trained as binary and returns single score, map to two columns
                        from scipy.special import softmax
                        # attempt softmax over [ -arr, arr ] to produce 2-col probabilities
                        two_col = np.vstack([ -arr, arr ]).T
                        probs = softmax(two_col, axis=1)
                        return probs
                except Exception:
                    pass
                # otherwise return raw wrapped as 2D
                return arr.reshape((arr.shape[0], -1))
            except Exception:
                # bubble up useful message
                raise

        @property
        def classes_(self):
            # expose classes_ if underlying model has them
            return getattr(self.model, "classes_", None)

    # If classifier looks like an XGBoost object (sklearn wrapper), wrap it
    safe_classifier = classifier_raw
    try:
        tname = type(classifier_raw).__name__.lower()
        if "xgb" in tname or "xgboost" in tname:
            # Best-effort: set some defaults and wrap
            try:
                if not hasattr(classifier_raw, "use_label_encoder"):
                    try:
                        classifier_raw.use_label_encoder = False
                    except Exception:
                        pass
                if not hasattr(classifier_raw, "eval_metric"):
                    try:
                        classifier_raw.eval_metric = "logloss"
                    except Exception:
                        pass
            except Exception:
                pass
            safe_classifier = SafeXGBWrapper(classifier_raw)
    except Exception:
        safe_classifier = classifier_raw

    # model_expected features derivation (if needed)
    model_expected = None
    try:
        # if wrapped we still attempt to fetch from the original object
        orig = getattr(classifier_raw, "model", classifier_raw)
        if hasattr(orig, "feature_names_in_"):
            model_expected = list(orig.feature_names_in_)
        else:
            try:
                booster = orig.get_booster()
                model_expected = booster.feature_names
            except Exception:
                model_expected = None
    except Exception:
        model_expected = None

    # choose canonical feature_columns
    if model_expected is not None:
        if feature_columns is None:
            feature_columns = model_expected
        else:
            set_feat = set(feature_columns)
            set_model = set(model_expected)
            if len(set_feat & set_model) < max(1, len(set_model) // 10):
                feature_columns = model_expected

    # compose return message (scaler load message if any)
    msg = scaler_msg

    # Return the safe wrapper (so app uses safe_classifier), regressor, scaler, label_encoders, feature_columns, msg
    return safe_classifier, regressor, scaler, label_encoders, feature_columns, msg



classifier, regressor, scaler, label_encoders, feature_columns, load_msg = load_models()

# ------------------------
# Sidebar + Header
# ------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "EMI Prediction", "Data Analytics", "Model Performance", "Data Management"],
)

st.markdown('<h1 class="main-header">EMIPredict AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligent Financial Risk Assessment Platform</p>', unsafe_allow_html=True)

# ------------------------
# Home page
# ------------------------
if page == "Home":
    st.markdown("## Welcome to EMIPredict AI")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-card'><h3>EMI Eligibility</h3><p>Get instant eligibility assessment for your EMI application with AI-powered risk analysis.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><h3>Maximum EMI Amount</h3><p>Calculate the maximum EMI amount you can afford based on your financial profile.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><h3>Risk Assessment</h3><p>Comprehensive financial risk analysis using ML models.</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## Business Use Cases")
    tab1, tab2, tab3 = st.tabs(["Financial Institutions", "FinTech Companies", "Banks & Credit Agencies"])
    with tab1:
        st.markdown("- Automate loan approval processes\n- Reduce manual underwriting time\n- Real-time eligibility assessment for walk-in customers")
    with tab2:
        st.markdown("- Instant pre-qualification checks for digital platforms\n- Automated risk scoring for loan applications")
    with tab3:
        st.markdown("- Data-driven loan amount recommendations\n- Portfolio risk management and default prediction")

# ------------------------
# EMI Prediction page
# ------------------------
elif page == "EMI Prediction":
    st.markdown("## EMI Eligibility & Amount Prediction")

    if load_msg:
        st.info(f"Model load message: {load_msg}")

    if classifier is None or regressor is None or feature_columns is None or scaler is None:
        st.error("Models or canonical feature list not loaded. Ensure 'models/' contains: best_classifier.pkl, best_regressor.pkl, scaler.pkl, label_encoders.pkl and feature_columns.json.")
        st.stop()

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
            emi_scenario = st.selectbox("EMI Scenario", ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"])
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
            # Derived numeric features
            debt_to_income_ratio = safe_div(current_emi_amount, monthly_salary)
            total_expenses = monthly_rent + school_fees + college_fees + travel_expenses + groceries_utilities + other_monthly_expenses
            expense_to_income_ratio = safe_div(total_expenses, monthly_salary)
            affordability_ratio = safe_div((monthly_salary - current_emi_amount - total_expenses), (requested_amount * requested_tenure)) * 100
            employment_stability_score = years_of_employment * 0.3 + (0.7 if employment_type == "Government" else 0.0)
            financial_stability_score = safe_div(bank_balance, monthly_salary) * 0.4 + safe_div(emergency_fund, monthly_salary) * 0.6
            credit_utilization = safe_div(current_emi_amount, max(1.0, credit_score / 100.0))
            dependency_ratio = safe_div(dependents, family_size)

            # Buckets
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

            # Base numeric + binaries
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

            cat_mappings = {
                "education": education,
                "employment_type": employment_type,
                "company_type": company_type,
                "house_type": house_type,
                "emi_scenario": emi_scenario,
                "age_group": age_group_val,
                "income_category": income_cat_val,
            }

            # Build exact input_vector using model-driven canonical feature list
            try:
                input_vector = build_input_vector(base_dict, cat_mappings, feature_columns)
            except Exception as e:
                st.error(f"Error building input vector: {e}")
                st.stop()

            # Final check: input_vector columns should equal feature_columns set
            set_input = set(input_vector.columns)
            set_model = set(feature_columns)
            missing_for_model = sorted(list(set_model - set_input))[:10]
            extra_for_model = sorted(list(set_input - set_model))[:10]
            if missing_for_model or extra_for_model:
                st.error(f"Feature mismatch vs canonical model features. Missing (sample): {missing_for_model}; Extra (sample): {extra_for_model}")
                st.stop()
            else:
                st.success("Input columns now match the model expected features (set equality).")

            # Scale and predict
            try:
                input_scaled = scaler.transform(input_vector) if scaler is not None else input_vector.values

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

                # Display
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
# other pages (static demos)
# ------------------------
elif page == "Data Analytics":
    st.markdown("## Financial Data Analytics Dashboard")
    st.markdown("### EMI Scenario Distribution")
    sample_data = {"EMI Scenario": ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"], "Count": [80000]*5, "Average Amount": [50000,100000,500000,300000,200000]}
    sample_df = pd.DataFrame(sample_data)
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(sample_df, values="Count", names="EMI Scenario", title="EMI Applications by Scenario")
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_bar = px.bar(sample_df, x="EMI Scenario", y="Average Amount", title="Average EMI Amount by Scenario")
        st.plotly_chart(fig_bar, use_container_width=True)

elif page == "Model Performance":
    st.markdown("## Model Performance Dashboard")
    classification_metrics = {"Model": ["Logistic Regression", "Random Forest", "XGBoost"], "Accuracy":[0.85,0.92,0.94], "F1-Score":[0.84,0.91,0.93]}
    class_df = pd.DataFrame(classification_metrics)
    st.dataframe(class_df, use_container_width=True)
    fig_class = px.bar(class_df, x="Model", y=["Accuracy","F1-Score"], title="Classification Model Comparison", barmode="group")
    st.plotly_chart(fig_class, use_container_width=True)

elif page == "Data Management":
    st.markdown("## Data Management Interface")
    st.markdown("### Upload New Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        st.success(f"Uploaded file with {df_new.shape[0]} records and {df_new.shape[1]} columns")
        st.dataframe(df_new.head(), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; padding: 2rem;'><p> EMIPredict AI - Powered by Advanced Machine Learning</p><p>Built with Streamlit • MLflow (optional) • XGBoost • Random Forest</p></div>", unsafe_allow_html=True)
