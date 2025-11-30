# Titanic Survival Prediction (Machine Learning Project)

This project builds a complete end-to-end Machine Learning pipeline on the popular Titanic dataset.  
It includes data preprocessing, feature engineering, model training, evaluation, and visualization of results.

---

##  Project Highlights

- Clean modular ML pipeline (scikit-learn)
- Custom preprocessing (missing values, feature engineering, categorical handling)
- Feature importance visualization
- Confusion matrix to evaluate model performance
- Easily reproducible training script
- Organized folder structure used in real-world ML projects

---

## 📂 Project Structure
Titanic-Insights-ML/
│
├── src/
│ ├── data.py → Loads titanic.csv
│ ├── preprocess.py → Cleans & transforms the data
│ ├── model.py → Builds the ML model (Random Forest)
│ ├── evaluate.py → Generates metrics
│ └── train.py → Main training script
│
├── results/
│ ├── metrics.json
│ └── figures/
│ ├── confusion_matrix.png
│ └── feature_importance.png
│
├── models/ → (ignored in Git) contains model.pkl after training
├── titanic.csv
├── requirements.txt
└── .gitignore


---

## 🧠 Model Performance

| Metric      | Score      |
|-------------|------------|
| Accuracy    | **0.8324** |
| Precision   | **0.7973** |
| Recall      | **0.7973** |
| F1 Score    | **0.7973** |

📌 Full metrics available in: `results/metrics.json`

---

## 📊 Confusion Matrix

![Confusion Matrix](results/figures/confusion_matrix.png)

---

## 📈 Feature Importance

![Feature Importance](results/figures/feature_importance.png)

---

## 🛠️ How to Run Locally (Windows PowerShell)
git clone -b tidy-titanic https://github.com/Murariji/Titanic-Insights-ML.git

cd Titanic-Insights-ML

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python -m src.train



This will:
- preprocess the dataset  
- train the model  
- save the model to `models/model.pkl`  
- generate evaluation results in `results/`  

---

## 🧩 Feature Engineering Used

- Title extraction from passenger names  
- Family size & IsAlone indicators  
- One-hot encoding for categorical features  
- Missing value imputation  
- Fare & Age normalization via median strategy  

---

## 📘 What This Project Demonstrates (Good for Interviews)

- Understanding of ML workflow  
- Ability to implement clean modular code (src/ structure)  
- Concept of evaluation metrics & trade-offs  
- Real-world preprocessing (title extraction, family size engineering)  
- Clear model explainability through feature importance  

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- SHAP-based explainability  
- Add cross-validation  
- Try alternative models (XGBoost, Logistic Regression baseline)  

---

## ✨ About This Repo
This branch (`tidy-titanic`) contains the clean, production-ready version of the Titanic ML pipeline with proper structure, results, and documentation.



