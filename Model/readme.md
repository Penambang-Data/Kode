Spesifikasi Model 

Komponen	        Implementasi
======================================================================================================
Input	         |   Dataset primer yang sudah dibersihkan (apachejit_total.csv, ghpr.csv)
Preprocessing	 |  Feature selection, normalisasi, label encoding
Algoritma	     |  Random Forest Classifier + Logistic Regression (dibandingkan)
Output	       |    Model prediksi defect: 0 = clean, 1 = buggy
Evaluasi	     |  Accuracy, Precision, Recall, F1-Score, ROC-AUC
Tools	         |  Python + Scikit-learn + Pandas + Matplotlib + Seaborn

persyaratan untuk Debuggin model dan instalasi library :
-"pip install scikit-learn"
-"python -m pip install scikit-learn"
-"pip install pandas matplotlib seaborn"
-pastikan kedua file dataset GHRP "ghprdata.csv" dan ApacheJIT "apachejit_total.csv" 
sudah berada di folder yang sama dengan model A1 atau A2

Model-Alpha 1
======================================================================================================
Model ini akan menghasilkan metrik performa seperti:

-Akurasi tinggi (>80% pada dataset seimbang)
-ROC-AUC sebagai metrik utama, karena dataset bisa saja imbalanced
-F1-Score yang baik untuk menangani distribusi kelas

Untuk output pada model ini, akan tercetak di folder A1Output

Model-Alpha 2
======================================================================================================
Model ini adalah model Alpha 1 yang sudah diimprovisasi dengan mengubah 
Model menjadi cross-validation + grid search juga ditambahkan visualisasi
confusion matrix dan ROC curve pada output yang dihasilkan.

berikut adalah output yang dihasilkan :

-M-A2-Output-YYYY-MM-DD.txt — hasil evaluasi
-ConfusionMatrix-RandomForest-YYYY-MM-DD.png
-ROCcurve-RandomForest-YYYY-MM-DD.png
-ConfusionMatrix-LogisticRegression-YYYY-MM-DD.png
-ROCcurve-LogisticRegression-YYYY-MM-DD.png

Untuk output pada model ini, akan tercetak di folder A2Output

