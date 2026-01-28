Risk Level Prediction using Machine Learning
📌 Overview

This project presents a complete end-to-end machine learning pipeline for multi-class risk level prediction using structured (tabular) clinical data.
It demonstrates real-world ML practices including data cleaning, class imbalance handling, feature importance analysis, dimensionality reduction, and comparative evaluation of multiple machine learning models.

The project is designed for learning, experimentation, and portfolio demonstration.

🎯 Problem Statement

Given a dataset containing multiple clinical features, the objective is to predict the risk level of each instance into one of the following categories:

Low Risk (0)
Mid Risk (1)
High Risk (2)

🛠️ Technologies & Libraries

Python
Pandas
NumPy
Scikit-learn
Imbalanced-learn (SMOTE)
XGBoost
Matplotlib

Seaborn

📂 Project Structure
Risk-Prediction-with-PCA/
│
├─ data.csv                 # Tabular dataset
├─ Case1.py                 # Feature Importance: Random Forest, Prediction: Logistic Regression
├─ Case2.py                 # Feature Importance: XGBoost, Prediction: Random Forest
├─ Case3.py                 # PCA-based Feature Extraction; Models: RF, XGBoost, Logistic Regression; Visualization
└─ README.md                # Project documentation

🧩 Workflow

All three cases follow the same core pipeline:
Data Loading & Preprocessing
Load data.csv
Remove duplicates
Remove age group 10–18
Map risk levels to numeric (0 = low, 1 = mid, 2 = high)
Train/Test Split
80% training / 20% testing
Stratified sampling to maintain class distribution
Class Imbalance Handling
Apply SMOTE on training data

Feature Importance / Dimensionality Reduction
Case 1: Feature importance via Random Forest
Case 2: Feature importance via XGBoost
Case 3: PCA (3 components)

Model Training & Prediction
Case 1: Logistic Regression
Case 2: Random Forest
Case 3: Random Forest, XGBoost, Logistic Regression (on PCA-transformed features)

Evaluation Metrics
Accuracy
Confusion matrix
Classification report

Visualizations (Case 3)
Rows remaining after each preprocessing step
Class distribution before and after SMOTE
Confusion matrix heatmap
Feature importance of PCA components
Precision, Recall, F1-score per class

🚀 Getting Started
1️⃣ Install Dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

2️⃣ Run Scripts
Case 1: python Case1.py
Case 2: python Case2.py
Case 3: python Case3.py
Ensure data.csv is in the same folder as the scripts.

📊 Results Overview

Case 1: Feature importance using Random Forest; Logistic Regression predictions.
Case 2: Feature importance using XGBoost; Random Forest predictions.
Case 3: PCA-based feature extraction; multiple classifier training; detailed visualizations.

📝 Conclusion
This repository demonstrates a robust machine learning workflow for multi-class risk prediction:
Data preprocessing and cleaning
Handling class imbalance with SMOTE
Feature importance analysis and dimensionality reduction
Training and evaluating multiple classifiers
Clear visualizations for insights

It can be extended to other clinical datasets or multi-class classification problems
