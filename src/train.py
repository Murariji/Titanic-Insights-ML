from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data
from src.preprocess import basic_preprocess
from src.model import build_model, save_model
from src.evaluate import evaluate_model


def main():
    df = load_data()

    split = basic_preprocess(df)
    X_train, X_test, y_train, y_test = split
    
    model = build_model()
    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    save_model(model)

    metrics = evaluate_model(model, X_test, y_test)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    cm = np.array(metrics["confusion_matrix"])

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Add cell numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix.png")
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        labels = X_train.columns.to_numpy()

        # sort by importance
        idx = np.argsort(importances)[::-1]
        importances = importances[idx]
        labels = labels[idx]

        plt.figure(figsize=(6, 4))
        plt.bar(range(len(importances)), importances)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("results/figures/feature_importance.png")
        plt.close()

    print("Training finished successfully.")
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
