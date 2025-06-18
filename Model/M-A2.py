import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from io import StringIO
import sys

# ====== Load Dataset ======
df = pd.read_csv("apachejit_total.csv")
features = ['la', 'ld', 'nf']
target = 'buggy'
df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

# ====== Scaling ======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====== Train-Test Split ======
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ====== Grid Search: Random Forest ======
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
}
grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# ====== Grid Search: Logistic Regression ======
lr = LogisticRegression(solver='liblinear')
param_grid_lr = {
    'C': [0.1, 1.0, 10.0]
}
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)

# ====== Evaluasi + Output ======
buffer = StringIO()
original_stdout = sys.stdout

print("=== Random Forest (Best Params) ===")
rf_report = classification_report(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1])
print(rf_report)
print("ROC AUC:", rf_auc)

print("\n=== Logistic Regression (Best Params) ===")
lr_report = classification_report(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, best_lr.predict_proba(X_test)[:,1])
print(lr_report)
print("ROC AUC:", lr_auc)

# Tampung ke buffer juga
sys.stdout = buffer
print("=== Random Forest (Best Params) ===")
print(rf_report)
print("ROC AUC:", rf_auc)
print("\n=== Logistic Regression (Best Params) ===")
print(lr_report)
print("ROC AUC:", lr_auc)
sys.stdout = original_stdout

# Buat folder output A2
output_dir = "A2output"
os.makedirs(output_dir, exist_ok=True)

# Simpan evaluasi ke file dalam folder A2
today = datetime.now().strftime("%Y-%m-%d")
output_filename = os.path.join(output_dir, f"M-A2-Output-{today}.txt")
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(buffer.getvalue())

# Simpan gambar-gambar evaluasi
def plot_confusion(title, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ConfusionMatrix-{title}-{today}.png"))
    plt.close()

def plot_roc_curve(title, model, X_test, y_test):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve - {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ROCcurve-{title}-{today}.png"))
    plt.close()

# Buat visualisasi
plot_confusion("RandomForest", y_test, y_pred_rf)
plot_roc_curve("RandomForest", best_rf, X_test, y_test)

plot_confusion("LogisticRegression", y_test, y_pred_lr)
plot_roc_curve("LogisticRegression", best_lr, X_test, y_test)
