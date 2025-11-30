import pandas as pd
from sklearn.model_selection import train_test_split
import re

def basic_preprocess(df: pd.DataFrame):
    df = df.copy()

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"] + df["Parch"]
        df["IsAlone"] = (df["FamilySize"] == 0).astype(int)
    else:
        df["FamilySize"] = 0
        df["IsAlone"] = 1

    def extract_title(name):
        if not isinstance(name, str):
            return "None"
        match = re.search(r" ([A-Za-z]+)\.", name)
        if match:
            return match.group(1)
        return "None"

    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)
        df["Title"] = df["Title"].replace(
            ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
             "Major", "Rev", "Sir", "Jonkheer", "Dona"],
            "Rare"
        )
        df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    else:
        df["Title"] = "None"

    survival_cols = [c for c in df.columns if c.strip().lower() == "survived"]
    if len(survival_cols) == 0:
        raise ValueError("Survived column not found. Columns: " + str(df.columns.tolist()))

    survived_col = survival_cols[0]
    y = df[survived_col]

    cat_cols = []
    if "Sex" in df.columns:
        cat_cols.append("Sex")
    if "Embarked" in df.columns:
        cat_cols.append("Embarked")
    if "Title" in df.columns:
        cat_cols.append("Title")

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    features = []

    for col in ["Pclass", "Age", "Fare", "FamilySize", "IsAlone"]:
        if col in df.columns:
            features.append(col)

    features += [c for c in df.columns if c.startswith("Sex_")]
    features += [c for c in df.columns if c.startswith("Embarked_")]
    features += [c for c in df.columns if c.startswith("Title_")]

    if len(features) == 0:
        raise ValueError("No valid features created. Columns: " + str(df.columns.tolist()))
