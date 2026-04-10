![Python](https://img.shields.io/badge/Python-3.9-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

# 💰 Adult Income Prediction (Census Data)

## 📌 Project Overview
This project builds an end-to-end machine learning pipeline for predicting whether an individual's income exceeds $50K/year using census data. The workflow includes data preprocessing, feature engineering, model training, evaluation, and reusable prediction functions.

A professional machine learning implementation using **Object-Oriented Programming (OOP)**.  
This project leverages **Pandas** for data engineering and **Scikit-learn** for predictive modeling.

---

## 🚀 Key Features
* **Smart Data Management:** Automatically fetches datasets from the UCI Machine Learning Repository and implements local caching for performance.
* **Automated Preprocessing:** Uses `LabelEncoder` to transform categorical text features into model-ready numerical formats.
* **Robust Modeling:** Implements a **Random Forest Classifier** with 100 estimators and `max_depth=10`.
* **Balanced Splitting:** Uses `stratify` to keep class distribution consistent.

## 🛠 Tech Stack
* Python 3.x
* Pandas
* Scikit-learn (RandomForest, Train_Test_Split, Preprocessing)

## 📊 Quick Start
```bash
pip install pandas scikit-learn
python Main.py
