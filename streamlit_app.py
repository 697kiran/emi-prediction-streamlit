# streamlit_app.py
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

st.set_page_config(page_title="EMIPredict AI - Financial Risk Assessment",
                   page_icon="💰", layout="wide")

# -------------------------
# Small helpers
# -------------------------
def safe_div(a, b, default=0.0):
    try:
        if pd.isna(a) or pd.isna(b):
            return default
        return a / b if b != 0 else default
    except Exception:
        return default

def build_input_vector(base_dict: dict, cat_map: dict, model_features: list):
    """
    Build single-row DataFrame with columns ordered exactly like model_features.
    It fills numeric/binary keys from base_dict and creates one-hot/explicit
    columns expected by model_features from cat_map.
    """
    fv = pd.DataFrame(columns=model_features)
    fv.loc[0] = 0
    # fill numeric/binary present in feature names
    for k, v in base_dict.items():
        if k in fv.columns:
            fv.at[0, k] = v

    # detect prefixes for one-hot columns (prefix_suffix)
    prefixes = {}
    for c in model_features:
        if "_" in c:
            p = c.split("_", 1)[0]
            prefixes.setdefault(p, []).append(c)

    # fill categorical mappings
    for prefix, val in cat_map.items():
        val_str = "Unknown" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)
        if prefix in prefixes:
            # model expects dummies: set the matching dummy to 1 if exists
            for expected in prefixes[prefix]:
                suffix = expected[len(prefix) + 1 :]
                if suffix.lower() == val_str.lower() or suffix.lower() == val_str.replace(" ", "_").lower():
                    fv.at[0, expected] = 1
                else:
                    # remain zero
                    pass
        else:
            # model expects plain column (like 'age_group')
            if prefix in fv.columns:
                fv.at[0, prefix] = val_str

    # convert numeric-like columns to numeric (where possible)
    for c in fv.columns:
        try:
            fv[c] = pd.to_numeric(fv[c], errors="ignore")
        except Exception:
            pass

    fv = fv.where(pd.notnull(fv), 0)
    return fv

# -------------------------
# Robust model loading
# -------------------------
@st.cache_resource
def load_models(model_dir: str = "models"):
    """
    Loads classifier, regressor, scaler, label encoders and canonical feature list.
    Heuristics:
     - If feature_columns.json present use it (but override if model has feature_names_in_)
     - If scaler.n_features_in_ doesn't match canonical list, report it (don't silently change scaler)
     - Apply small XGBoost attribute compatibility fixes
    """
    cls_p = os.path.join(model_dir, "best_classifier.pkl")
    reg_p = os.path.join(model_dir, "best_regressor.pkl")
    scl_p = os.path.join(model_dir, "scaler.pkl")
    le_p = os.path.join(model_dir, "label_encoders.pkl")
    feat_p = os.path.join(model_dir, "feature_columns.json")

    # must have classifier & regressor (scaler & feature json ideally present)
    for req in [cls_p, reg_p]:
        if not os.path.exists(req):
            return None, None, None, None, None, f"Missing required artifact: {req}"

    try:
        classifier = joblib.load(cls_p)
    except Exception as e:
        return None, None, None, None, None, f"Failed loading classifier: {e}"

    try:
        regressor = joblib.load(reg_p)
    except Exception as e:
        return classifier, None, None, None, None, f"Failed loading regressor: {e}"

    scaler = None
    if os.path.exists(scl_p):
        try:
            scaler = joblib.load(scl_p)
        except Exception as e:
            # keep going, but report
            scaler = None
            scaler_err = f"Failed loading scaler.pkl: {e}"
    else:
        scaler_err = "scaler.pkl not found"

    label_encoders = None
    if os.path.exists(le_p):
        try:
            label_encoders = joblib.load(le_p)
        except Exception:
            label_encoders = None

    feature_columns = None
    if os.path.exists(feat_p):
        try:
            with open(feat_p, "r") as f:
                feature_columns = json.load(f)
        except Exception:
            feature_columns = None

    # derive canonical feature names from classifier if available
    model_expected = None
    try:
        if hasattr(classifier, "feature_names_in_"):
            model_expected = list(classifier.feature_names_in_)
        else:
            try:
                booster = classifier.get_booster()
                model_expected = booster.feature_names
            except Exception:
                model_expected = None
    except Exception:
        model_expected = None

    # If model_expected exists, prefer that if JSON is missing or highly mismatched
    if model_expected is not None:
        if feature_columns is None:
            feature_columns = model_expected
        else:
            set_feat = set(feature_columns)
            set_model = set(model_expected)
            # If overlap tiny, override
            if len(set_feat & set_model) < max(1, len(set_model) // 10):
                feature_columns = model_expected

    # XGBoost compatibility small patch (avoid attribute lookups that break across versions)
    try:
        tname = type(classifier).__name__.lower()
        if "xgb" in tname or "xgboost" in tname:
            for bad in ["use_label_encoder", "gpu_id", "n_gpus"]:
                if hasattr(classifier, bad):
                    try:
                        delattr(classifier, bad)
                    except Exception:
                        pass
            # ensure eval_metric exists
            if not hasattr(classifier, "eval_metric"):
                try:
                    classifier.eval_metric = "logloss"
                except Exception:
                    pass
    except Exception:
        pass

    # final return with optional scaler_err message
    msg = None
    if scaler is None:
        msg = scaler_err if "scaler_err" in locals() else None

    return classifier, regressor, scaler, label_encoders, feature_columns, msg

# load
classifier, regressor, scaler, label_encoders, feature_columns, load_msg = load_models()

# -------------------------
# UI + pages
# -------------------------
st.title("EMIPredict AI — Financial Risk Assessment")

if load_msg:
    st.warning(f"Model load message: {load_msg}")

if classifier is None or regressor is None:
    st.error("Models not loaded. Ensure models/best_classifier.pkl and models/best_regressor.pkl exist.")
    st.stop()

# minimal form (same input fields as you had)
with st.form("pred_form"):
    st.header("Applicant details")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 80, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    with col2:
        monthly_salary = st.number_input("Monthly Salary (INR)", 0, 5_000_000, 50_000)
        monthly_rent = st.number_input("Monthly Rent (INR)", 0, 100_000, 10_000)
        bank_balance = st.number_input("Bank Balance (INR)", 0, 10_000_000, 100_000)
    with col3:
        current_emi_amount = st.number_input("Current EMI Amount (INR)", 0, 200_000, 5_000)
        credit_score = st.number_input("Credit Score", 300, 850, 700)
        emergency_fund = st.number_input("Emergency Fund (INR)", 0, 5_000_000, 50_000)

    requested_amount = st.number_input("Requested Amount (INR)", 1000, 2_000_000, 100_000)
    requested_tenure = st.number_input("Requested Tenure (months)", 3, 84, 24)
    submitted = st.form_submit_button("Predict")

if submitted:
    # derived features
    debt_to_income_ratio = safe_div(current_emi_amount, monthly_salary)
    total_expenses = monthly_rent
    expense_to_income_ratio = safe_div(total_expenses, monthly_salary)
    affordability_ratio = safe_div((monthly_salary - current_emi_amount - total_expenses), (requested_amount * requested_tenure)) * 100

    base = {
        "age": age,
        "monthly_salary": monthly_salary,
        "monthly_rent": monthly_rent,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "debt_to_income_ratio": debt_to_income_ratio,
        "expense_to_income_ratio": expense_to_income_ratio,
        "affordability_ratio": affordability_ratio,
    }

    cat_map = {
        "gender": gender,
        "marital_status": marital_status
    }

    # If canonical feature list not available, try to use model's feature_names_in_
    if feature_columns is None:
        st.error("feature_columns.json not found and model did not expose feature names. Add feature_columns.json to models/ or retrain to include feature_names_in_.")
        st.stop()

    # Build input vector exactly matching canonical order
    input_vector = build_input_vector(base, cat_map, feature_columns)

    # check scaler shape alignment
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        if scaler.n_features_in_ != input_vector.shape[1]:
            st.error(f"Prediction error: X has {input_vector.shape[1]} features, but StandardScaler expects {scaler.n_features_in_} features as input.")
            st.write("Please ensure you uploaded the scaler.pkl that was fitted on the same columns as feature_columns.json.")
            st.stop()

    # final predict
    try:
        X_for_pred = input_vector.values if scaler is None else scaler.transform(input_vector)
        elig_raw = classifier.predict(X_for_pred)
        try:
            proba = classifier.predict_proba(X_for_pred)[0]
        except Exception:
            proba = None
        max_emi = regressor.predict(X_for_pred)[0]

        label = str(elig_raw[0])
        st.success(f"Predicted eligibility: {label}")
        st.metric("Predicted max EMI", f"₹{int(round(max_emi)):,}")

        if proba is not None:
            dfp = pd.DataFrame({"class": list(classifier.classes_), "prob": proba})
            st.bar_chart(dfp.set_index("class"))

    except Exception as e:
        st.error(f"Prediction error: {e}")
