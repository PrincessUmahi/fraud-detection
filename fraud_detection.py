# =============================================
# CREDIT CARD FRAUD DETECTION
# Author: Adaeze (Princess) Umahi
# Dataset: Kaggle Credit Card Fraud Detection
# =============================================

# IMPORTS — all at the top (best practice)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =============================================
# STEP 1: LOAD AND EXPLORE THE DATA
# =============================================

df = pd.read_csv('creditcard.csv')

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution (0 = Legitimate, 1 = Fraud):")
print(df['Class'].value_counts())

# =============================================
# STEP 2: VISUALISE CLASS DISTRIBUTION
# =============================================

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Legitimate Transactions')
plt.xlabel('Class (0 = Legitimate, 1 = Fraud)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('fraud_vs_legitimate.png')
print("\nChart saved as fraud_vs_legitimate.png")

# =============================================
# STEP 3: PREPARE THE DATA
# =============================================

# Separate features (X) from the target label (y)
X = df.drop('Class', axis=1)  # Everything except the Class column
y = df['Class']               # Just the Class column (0 or 1)

# Split into training data (80%) and testing data (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("\nTraining fraud cases:", y_train.sum())
print("Testing fraud cases:", y_test.sum())

# =============================================
# STEP 4: TRAIN THE MODEL
# =============================================

print("\nTraining the fraud detection model... (this may take a minute)")

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Model training complete!")

# =============================================
# STEP 5: EVALUATE THE MODEL
# =============================================

y_pred = model.predict(X_test)

print("\nModel Results:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =============================================
# STEP 6: VISUALISE THE CONFUSION MATRIX
# =============================================

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('Confusion Matrix — Fraud Detection Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix chart saved!")
print("\nProject complete!")