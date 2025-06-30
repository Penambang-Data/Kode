import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

# ===================================================================
#                      PROSES PREPROCESSING
# ===================================================================

# ====== Load Dataset ======
df = pd.read_csv("apachejit_total.csv")
# ====== 1. Pembersihan Data ======
# Penjelasan: Menghapus baris dengan nilai kosong pada kolom penting.
# Kolom yang digunakan untuk model adalah la, ld, nf, dan target buggy.
initial_rows = len(df)
df = df.dropna(subset=['la', 'ld', 'nf', 'buggy'])
print(f"Data Cleaning: Menghapus {initial_rows - len(df)} baris dengan nilai NaN.")

# Penjelasan: Mengonversi author_date ke format datetime.
# Walaupun tidak digunakan sebagai fitur, ini adalah bagian dari pembersihan.
df['author_date'] = pd.to_datetime(df['author_date'], errors='coerce')
df = df.dropna(subset=['author_date']) # Hapus jika ada error konversi

# ====== 2. Pembuatan Fitur Baru ======
# Penjelasan: Menambahkan fitur code_churn = la + ld sebagai fitur tambahan.
df['code_churn'] = df['la'] + df['ld']
print("Feature Engineering: Menambahkan fitur 'code_churn'.")

# ====== 3. Reduksi & Seleksi Data ======
# Penjelasan: Memilih hanya fitur penting: la, ld, nf, dan fitur baru code_churn.
features = ['la', 'ld', 'nf', 'code_churn']
target = 'buggy'

X = df[features]
y = df[target]

# ====== 4. Transformasi Data (Normalisasi) ======
# Penjelasan: Fitur numerik dinormalisasi menggunakan StandardScaler.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data Transformation: Fitur numerik telah di-scale menggunakan StandardScaler.")


# ==============================================================================
#                      PROSES TRAINING & EVALUASI MODEL
# ==============================================================================

# ====== Train-Test Split ======
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ====== Grid Search: Random Forest ======
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# ====== Grid Search: Logistic Regression ======
lr = LogisticRegression(solver='liblinear', class_weight='balanced')
param_grid_lr = {
    'C': [0.1, 1.0, 10.0]
}
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='roc_auc', n_jobs=-1)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)

# ====== Evaluasi + Output ======
# Menyiapkan buffer untuk menyimpan output teks
buffer = StringIO()
original_stdout = sys.stdout
sys.stdout = buffer

print("=== Random Forest (Best Params) ===")
rf_report = classification_report(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1])
print(f"Best Parameters: {grid_rf.best_params_}")
print(rf_report)
print(f"ROC AUC: {rf_auc:.4f}")

print("\n=== Logistic Regression (Best Params) ===")
lr_report = classification_report(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, best_lr.predict_proba(X_test)[:,1])
print(f"Best Parameters: {grid_lr.best_params_}")
print(lr_report)
print(f"ROC AUC: {lr_auc:.4f}")

# Mengembalikan output ke konsol dan menyimpan hasil dari buffer
sys.stdout = original_stdout
output_text = buffer.getvalue()
print(output_text)

# Buat folder output A2 jika belum ada
output_dir = "A2output"
os.makedirs(output_dir, exist_ok=True)

# Simpan evaluasi ke file dalam folder A2
today = datetime.now().strftime("%Y-%m-%d")
output_filename = os.path.join(output_dir, f"M-A2-Output-{today}.txt")
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(output_text)
print(f"\nHasil evaluasi teks disimpan di: {output_filename}")

# Fungsi untuk menyimpan gambar-gambar evaluasi
def plot_confusion(title, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Buggy', 'Buggy'], yticklabels=['Not Buggy', 'Buggy'])
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"ConfusionMatrix-{title}-{today}.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_roc_curve(title, model, X_test, y_test):
    plt.figure()
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve - {title}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"ROCcurve-{title}-{today}.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

# Buat dan simpan visualisasi
cm_rf_path = plot_confusion("RandomForest", y_test, y_pred_rf)
roc_rf_path = plot_roc_curve("RandomForest", best_rf, X_test, y_test)

cm_lr_path = plot_confusion("LogisticRegression", y_test, y_pred_lr)
roc_lr_path = plot_roc_curve("LogisticRegression", best_lr, X_test, y_test)

print(f"Visualisasi disimpan di folder '{output_dir}':")
print(f"- {os.path.basename(cm_rf_path)}")
print(f"- {os.path.basename(roc_rf_path)}")
print(f"- {os.path.basename(cm_lr_path)}")
print(f"- {os.path.basename(roc_lr_path)}")