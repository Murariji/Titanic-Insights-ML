from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_data(path: str = None):
    if path is None:
        path = ROOT / "titanic.csv"
    return pd.read_csv(path)
