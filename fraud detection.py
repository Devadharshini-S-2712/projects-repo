
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
data = pd.read_csv("creditcard_2023.csv") #"C:\Users\dhars\Desktop\New folder\creditcard_2023.csv"
print("Dataset shape:", data.shape)
print("\nClass distribution:\n", data['Class'].value_counts())
print("\nColumns available:", data.columns.tolist())
if 'Amount' in data.columns:
    scaler = StandardScaler()
    data['NormalizedAmount'] = scaler.fit_transform(data[['Amount']])
    data.drop('Amount', axis=1, inplace=True)

if 'id' in data.columns:
    data.drop('id', axis=1, inplace=True)



sns.countplot(x='Class', data=data)
plt.title("Class Distribution: 0 = Normal, 1 = Fraud")
plt.show()

fraud = data[data['Class'] == 1]
non_fraud = data[data['Class'] == 0].sample(n=len(fraud), random_state=42)
balanced_data = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42)

X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print(classification_report(y_test, lr_preds))
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
fpr, tpr, _ = roc_curve(y_test, rf_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.show()
