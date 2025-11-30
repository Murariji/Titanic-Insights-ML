from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

# model will be saved in /models/model.pkl
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
def build_model():
    return RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)
