
# Case 3: Feature Extraction using PCA
# Models: Random Forest, XGBoost, Logistic Regression

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('data.csv')
print("Initial data shape:", df.shape)
print("Data sample:\n", df.head())

missing_values_count = df.isnull().sum()
print("\nMissing values per column:\n", missing_values_count)
print("Total missing values:", missing_values_count.sum())

df = df.drop_duplicates()
print("\nAfter removing duplicates, shape:", df.shape)

df = df[~((df['Age'] >= 10) & (df['Age'] <= 18))]
print("After removing age 10-18, shape:", df.shape)

risk_map = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
df['RiskLevel'] = df['RiskLevel'].map(risk_map).astype(int)
print("\nRisk level counts:\n", df['RiskLevel'].value_counts())

X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("\nTraining set class distribution:", Counter(y_train))
print("Testing set class distribution:", Counter(y_test))

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("\nResampled training set class distribution:", Counter(y_train_res))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

#  Apply PCA 
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"\nPCA applied. Number of components: {X_train_pca.shape[1]}")

#  Train Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_pca, y_train_res)
    y_pred = model.predict(X_test_pca)
    
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Low Risk','Mid Risk','High Risk']))

#  Visualizations 

# 1️ Bar chart: Rows at each preprocessing step
steps = ['Original', 'No Duplicates', 'Age Filter']
rows = [df.shape[0] + len(df[df.duplicated()]) + len(df[(df['Age']>=10)&(df['Age']<=18)]),
        df.shape[0] + len(df[(df['Age']>=10)&(df['Age']<=18)]),
        df.shape[0]]

plt.figure(figsize=(8,5))
sns.barplot(x=steps, y=rows, color='skyblue')
plt.ylabel("Number of Rows")
plt.title("Rows after each preprocessing step")
plt.show()

# 2️ Class distribution before and after SMOTE
plt.figure(figsize=(8,5))
before = Counter(y_train)
after = Counter(y_train_res)
plt.bar(before.keys(), before.values(), alpha=0.5, label='Before SMOTE')
plt.bar(after.keys(), after.values(), alpha=0.5, label='After SMOTE')
plt.xticks([0,1,2], ['Low Risk','Mid Risk','High Risk'])
plt.ylabel("Number of Samples")
plt.title("Class Distribution Before and After SMOTE")
plt.legend()
plt.show()

# 3️ Random Forest Confusion Matrix
rf = models['Random Forest']
y_pred_rf = rf.predict(X_test_pca)
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low','Mid','High'], yticklabels=['Low','Mid','High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

# 4️ Random Forest Feature Importance (PCA components)
importances = rf.feature_importances_
pc_labels = ['PC1', 'PC2', 'PC3']
plt.figure(figsize=(6,4))
sns.barplot(x=pc_labels, y=importances, palette='viridis', legend=False)
plt.ylabel("Importance")
plt.title("Random Forest PCA Feature Importance")
plt.show()

# 5️ Precision, Recall, F1 per class
report = classification_report(y_test, y_pred_rf, target_names=['Low','Mid','High'], output_dict=True)
metrics_df = pd.DataFrame(report).transpose().iloc[:3, :3]  # Only classes
metrics_df.plot(kind='bar', figsize=(8,5))
plt.title("Random Forest: Precision, Recall, F1-Score per Class")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.show()
