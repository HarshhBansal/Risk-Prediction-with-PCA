# Case 2: Feature Importance XGbOost : Random Forest

import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Load and preprocess data 
df = pd.read_csv('data.csv')

# Remove duplicates
df = df.drop_duplicates()

# Remove age group 10-18
df = df[~((df['Age'] >= 10) & (df['Age'] <= 18))]

# Map risk levels to numeric
risk_map = {'high risk': 2, 'mid risk': 1, 'low risk': 0}
df['RiskLevel'] = df['RiskLevel'].replace(risk_map)

# Split features and target
X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

# ---- Split data into training and testing sets -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Original training set class distribution: {Counter(y_train)}")
print(f"Testing set class distribution: {Counter(y_test)}")

# ----Apply SMOTE on training set -----
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Resampled training set class distribution: {Counter(y_train_resampled)}")

# --- Feature Importance with XGBoost -----
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (XGBoost):")
print(feature_importance_df)

# ---- Train Random Forest for Prediction 
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Evaluate Random Forest on test set -----
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("\nAccuracy of Random Forest on test set:", accuracy_rf)

print("\nRandom Forest Evaluation (Prediction on Test Set):")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
