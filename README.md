# 💰 Adult Income Prediction (Census Data)

## 📌 Project Overview
This project is an end-to-end machine learning pipeline that predicts whether an individual's income exceeds $50K/year using census data.

The system is built using a clean Object-Oriented Programming (OOP) structure and covers the full ML workflow including preprocessing, training, evaluation, and prediction export.

---

## 🚀 Key Features

Smart Data Management: Automatically loads dataset (OpenML or local CSV cache) and avoids repeated downloads.

Automated Preprocessing: Handles missing values and encodes categorical features into numerical format.

Robust Modeling: Uses a Random Forest Classifier for stable and accurate predictions.

Balanced Data Split: Maintains class distribution using stratified train/test splitting.

Reusable Pipeline: Trained model can be saved and reused for future predictions.

SQLite Export: Saves predictions and dataset into a local database.

---

## 🧠 Machine Learning Pipeline

1. Load dataset (with automatic caching)
2. Handle missing values (median / most frequent)
3. Encode categorical variables
4. Split dataset into training and testing sets
5. Train Random Forest model
6. Evaluate model performance
7. Save trained model
8. Store results in SQLite database

---

## 📊 Model Performance

- Accuracy: 0.86
- Precision (>50K): 0.81
- Recall (>50K): 0.53

---

## 📁 Output Files

- adult_data_cleaned.csv
- income_model.pkl
- income_results.db

---

## 🚀 Quick Start

Install dependencies:

pip install pandas scikit-learn joblib

Run the project:

python Main.py

---

## 📌 Notes

- Dataset is automatically downloaded or loaded from local storage.
- No manual preprocessing is required.
- Fully portable across different systems.

---

## 👤 Author

Elyas Nematvand
