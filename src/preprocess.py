import pandas as pd
from sklearn.model_selection import train_test_split
import re

def basic_preprocess(df: pd.DataFrame):
    df = df.copy()

    # Fill missing values
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Feature Engineering
    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"] + df["Parch"]
        df["IsAlone"] = (df["FamilySize"] == 0).astype(int)
    else:
        df["FamilySize"] = 0
        df["IsAlone"] = 1

    # Extract Title from Name
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

    # Ensure Survived exists
    if "Survived" not in df.columns:
        raise ValueError("'Survived' column missing. Available columns: " + str(df.columns.tolist()))

    y = df["Survived"]

    # One-hot encode
    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Title"], drop_first=True)

    # Feature list
    features = ["Pclass", "Age", "Fare", "FamilySize", "IsAlone"]
    features += [c for c in df.columns if c.startswith("Sex_")]
    features += [c for c in df.columns if c.startswith("Embarked_")]
    features += [c for c in df.columns if c.startswith("Title_")]

    X = df[features]

    return train_test_split(X, y, test_size=0.2, random_state=42)
