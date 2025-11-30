# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from src.data import load_data
from src.preprocess import basic_preprocess
from src.model import build_model  # used if we need to train fallback

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write("Simple demo: enter passenger info and the ML model will predict whether they survived.")
st.markdown("---")


@st.cache_data(show_spinner=False)
def load_model_and_features():
    """
    Load saved model and feature column list.
    If model not present, train quickly (fallback) and save both model and feature list.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    model = None
    features = None

    # Try to load model
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None

    # Try to load feature columns
    if FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, "r") as f:
                features = json.load(f)
        except Exception:
            features = None

    # If either is missing, train quickly and save
    if model is None or features is None:
        st.info("Training model on server (this runs once). Please wait a bit...")
        df = load_data()
        X_train, X_test, y_train, y_test = basic_preprocess(df)
        model = build_model()
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

        features = list(X_train.columns)
        with open(FEATURES_PATH, "w") as f:
            json.dump(features, f)
        st.success("Model trained and saved.")

    return model, features


model, FEATURE_NAMES = load_model_and_features()

st.sidebar.header("Input passenger details")

# Input fields
pclass = st.sidebar.selectbox("Ticket class (Pclass)", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.number_input("Age", min_value=0.0, max_value=120.0, value=25.0, step=0.5)
sibsp = st.sidebar.number_input("SibSp (siblings/spouse)", min_value=0, max_value=10, value=0, step=1)
parch = st.sidebar.number_input("Parch (parents/children)", min_value=0, max_value=10, value=0, step=1)
fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=1000.0, value=7.25, step=0.1)
embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])
name = st.sidebar.text_input("Name (for Title extraction)", value="Mr. John Doe")

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: click **Predict** after filling details.")

# Build a single-row input DataFrame (we keep Survived=0 dummy because preprocess expects it)
input_df = pd.DataFrame([{
    "PassengerId": 0,
    "Survived": 0,         # dummy, preprocess expects Survived column
    "Pclass": pclass,
    "Name": name,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Ticket": "NONE",
    "Fare": fare,
    "Cabin": np.nan,
    "Embarked": embarked
}])


def preprocess_single_row_by_appending(sample_df: pd.DataFrame, df_reference: pd.DataFrame):
    combined = pd.concat([df_reference, sample_df], ignore_index=True)
    X_train, X_test, y_train, y_test = basic_preprocess(combined)
    if len(X_test) > 0:
        return X_test.tail(1)
    else:
        return X_train.tail(1)


if st.button("Predict"):
    try:
        # load original dataset as reference (ensures same encoding categories)
        df_full = load_data()
        X_input = preprocess_single_row_by_appending(input_df, df_full)

        # Now enforce the saved feature order to avoid unseen/missing columns
        # Create dataframe reindexed to FEATURE_NAMES (missing -> 0, extra -> dropped)
        X_input = X_input.copy()
        # Add any missing expected columns with 0
        for col in FEATURE_NAMES:
            if col not in X_input.columns:
                X_input[col] = 0
        # Reindex to the exact feature order
        X_input = X_input.reindex(columns=FEATURE_NAMES, fill_value=0)

        # Predict
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[:, 1][0]
        else:
            proba = None
        pred = model.predict(X_input)[0]

    except Exception as e:
        st.error("Model prediction failed: " + str(e))
        st.stop()

    st.write("### Prediction result")
    st.write("**Predicted label:**", "Survived" if pred == 1 else "Not survived")
    if proba is not None:
        st.write(f"**Survival probability:** {proba:.2f}")

    st.markdown("---")
    st.write("### Model info & features used")
    st.write(f"Model type: `{type(model).__name__}`")
    st.write("Features the model expects:")
    st.write(", ".join(FEATURE_NAMES))
    st.info("Note: For consistent results, try realistic values similar to dataset distribution.")
