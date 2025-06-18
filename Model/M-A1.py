import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("apachejit_total.csv")  # atau ghpr.csv

# Pilih fitur penting dan target
features = ['la', 'ld', 'nf']
target = 'buggy'

# Handle missing values
df = df.dropna(subset=features + [target])

# Split X dan y
X = df[features]
y = df[target]

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model 1: Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Model 2: Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Evaluasi
import os
from datetime import datetime
from io import StringIO
import sys

# Tangkap output evaluasi
buffer = StringIO()
original_stdout = sys.stdout

# Cetak evaluasi ke terminal
print("=== Random Forest ===")
rf_report = classification_report(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(rf_report)
print("ROC AUC:", rf_auc)

print("\n=== Logistic Regression ===")
lr_report = classification_report(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
print(lr_report)
print("ROC AUC:", lr_auc)

# Simpan output juga ke buffer
sys.stdout = buffer
print("=== Random Forest ===")
print(rf_report)
print("ROC AUC:", rf_auc)
print("\n=== Logistic Regression ===")
print(lr_report)
print("ROC AUC:", lr_auc)
sys.stdout = original_stdout

# Buat folder output jika belum ada
output_dir = "A1output"
os.makedirs(output_dir, exist_ok=True)

# Simpan ke file dengan tanggal
today = datetime.now().strftime("%Y-%m-%d")
output_path = os.path.join(output_dir, f"M-A1-Output-{today}.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(buffer.getvalue())

