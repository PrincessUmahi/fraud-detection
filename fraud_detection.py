# =============================================
# CREDIT CARD FRAUD DETECTION
# Author: Adaeze (Princess) Umahi
# Dataset: Kaggle Credit Card Fraud Detection
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    roc_auc_score, precision_score, recall_score, f1_score
)

# =============================================
# STEP 1: LOAD AND EXPLORE THE DATA
# =============================================
df = pd.read_csv('creditcard.csv')
print("Dataset shape:", df.shape)
print("\nClass distribution (0 = Legitimate, 1 = Fraud):")
print(df['Class'].value_counts())
fraud_rate = df['Class'].mean() * 100
print(f"\nFraud rate: {fraud_rate:.3f}% of all transactions")

# Why accuracy is misleading: a model that always predicts "legitimate"
# would score ~99.83% accuracy while catching ZERO fraud. We therefore
# focus on PRECISION and RECALL for the fraud class, not accuracy.

# =============================================
# STEP 2: VISUALISE CLASS DISTRIBUTION
# =============================================
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Legitimate Transactions')
plt.xlabel('Class (0 = Legitimate, 1 = Fraud)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('fraud_vs_legitimate.png', dpi=150)
print("\nChart saved as fraud_vs_legitimate.png")

# =============================================
# STEP 3: PREPARE THE DATA
# =============================================
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("Training fraud cases:", y_train.sum())
print("Testing fraud cases:", y_test.sum())

# =============================================
# STEP 4: TRAIN TWO MODELS — BASELINE vs BALANCED
# The imbalance is the core challenge. We train a standard model and a
# class-weighted model that penalises missing the rare fraud class more
# heavily, then compare them.
# =============================================
print("\nTraining baseline model...")
baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
baseline.fit(X_train, y_train)

print("Training class-weighted (balanced) model...")
model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1,
    class_weight='balanced'
)
model.fit(X_train, y_train)
print("Training complete.")

# =============================================
# STEP 5: EVALUATE AND COMPARE
# =============================================
def report(name, clf):
    pred = clf.predict(X_test)
    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print(f"Fraud precision: {precision_score(y_test, pred):.3f}")
    print(f"Fraud recall:    {recall_score(y_test, pred):.3f}")
    print(f"Fraud F1:        {f1_score(y_test, pred):.3f}")
    return pred

pred_baseline = report("BASELINE (no class weight)", baseline)
pred_balanced = report("BALANCED (class_weight='balanced')", model)

# ROC-AUC and PR-AUC (better than accuracy for imbalanced data)
proba = model.predict_proba(X_test)[:, 1]
print(f"\nROC-AUC (balanced model): {roc_auc_score(y_test, proba):.4f}")
print(f"PR-AUC  (balanced model): {average_precision_score(y_test, proba):.4f}")

# =============================================
# STEP 6: CONFUSION MATRIX (balanced model)
# =============================================
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, pred_balanced),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('Confusion Matrix — Fraud Detection Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("\nConfusion matrix chart saved!")

# =============================================
# STEP 7: FEATURE IMPORTANCE — which signals drive fraud detection
# =============================================
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)
plt.figure(figsize=(7, 4))
sns.barplot(x=top10.values, y=top10.index, color='#1f4e79')
plt.title('Top 10 Features Driving Fraud Detection')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("Feature importance chart saved!")
print("\nTop 10 features:")
print(top10)

# =============================================
# STEP 8: PRECISION-RECALL CURVE + THRESHOLD TUNING
# The default 0.5 cutoff is rarely optimal for fraud. This curve shows the
# trade-off: lowering the threshold catches more fraud (higher recall) at
# the cost of more false alarms (lower precision).
# =============================================
prec, rec, thresholds = precision_recall_curve(y_test, proba)
plt.figure(figsize=(7, 4))
plt.plot(rec, prec, color='#c0392b')
plt.title('Precision-Recall Curve — Fraud Class')
plt.xlabel('Recall (fraud caught)')
plt.ylabel('Precision (alarms that are real)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=150)
print("Precision-recall curve saved!")

print("\nProject complete!")