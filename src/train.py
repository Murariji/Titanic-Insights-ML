from pathlib import Path
import json

from src.data import load_data
from src.preprocess import basic_preprocess
from src.model import build_model, save_model
from src.evaluate import evaluate_model

def main():
    df = load_data()
    split = basic_preprocess(df)

    # robust check: must be a tuple of length 4 (X_train, X_test, y_train, y_test)
    if isinstance(split, tuple) and len(split) == 4:
        X_train, X_test, y_train, y_test = split
    else:
        raise RuntimeError("Dataset missing 'Survived' column or preprocess returned X only.")

    model = build_model()
    model.fit(X_train, y_train)

    # ensure models directory exists if save_model expects it
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    save_model(model)

    metrics = evaluate_model(model, X_test, y_test)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training finished. Metrics:", metrics)

if __name__ == "__main__":
    main()
