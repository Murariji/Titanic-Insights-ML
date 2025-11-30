# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path

# Import project code 
from src.data import load_data
from src.preprocess import basic_preprocess
from src.model import build_model  
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.pkl"

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Simple demo: enter passenger info and the ML model will predict whether they survived."
)
st.markdown("---")

@st.cache_data
def load_or_train_model():
    MODELS_DIR.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
           
        except Exception:
            model = None
    else:
        model = None

    if model is None:
        st.info("Training model on server (this runs once). Please wait a bit...")
        # Train quickly using existing pipeline
        df = load_data()
        X_train, X_test, y_train, y_test = basic_preprocess(df)
        model = build_model()
        model.fit(X_train, y_train)
        # persist
        joblib.dump(model, MODEL_PATH)
        st.success("Model trained and saved.")
        return model, X_train.columns.to_list()
    else:
        # Need feature names — derive from preprocessing on full dataset
        df = load_data()
        X_train, X_test, y_train, y_test = basic_preprocess(df)
        return model, X_train.columns.to_list()

model, FEATURE_NAMES = load_or_train_model()

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

if st.button("Predict"):
    # Build a 1-row dataframe consistent with training preprocess expectations
    input_df = pd.DataFrame([{
        "PassengerId": 0,
        "Survived": 0,         # dummy, preprocess expects Survived
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

    try:
        X_all = basic_preprocess(input_df)  # returns train_test_split tuple
        # basic_preprocess returns train_test_split -> (X_train, X_test, y_train, y_test)
        # but for a single-row input we want processed features; safer to call internals:
    except Exception:
        # fallback: run full pipeline on the original dataset + merge single sample
        df_full = load_data()
        # append the new row to full df to ensure same encoding/columns
        df_full = pd.concat([df_full, input_df], ignore_index=True)
        X_train, X_test, y_train, y_test = basic_preprocess(df_full)
        # last row will correspond to our input (since we appended), locate by index -1
        X_input = X_test.tail(1) if len(X_test) > 0 else X_train.tail(1)

    else:
        # If preprocess didn't throw, it returned tuples; extract the processed features
        # We'll put the single row through the same preprocessing by appending to train set:
        df_full = load_data()
        df_full = pd.concat([df_full, input_df], ignore_index=True)
        X_train, X_test, y_train, y_test = basic_preprocess(df_full)
        X_input = X_test.tail(1) if len(X_test) > 0 else X_train.tail(1)

    # Ensure model and columns match
    try:
        proba = model.predict_proba(X_input)[:, 1][0]
        pred = model.predict(X_input)[0]
    except Exception as e:
        st.error("Model prediction failed: " + str(e))
        st.stop()

    st.write("### Prediction result")
    st.write("**Predicted label:**", "Survived" if pred == 1 else "Not survived")
    st.write(f"**Survival probability:** {proba:.2f}")

    st.markdown("---")
    st.write("### Model info & features used")
    st.write(f"Model type: `{type(model).__name__}`")
    st.write("Features the model expects:")
    st.write(", ".join(FEATURE_NAMES))
    st.write("")
    st.info("Note: For consistent results, try realistic values similar to dataset distribution.")
