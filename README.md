# Klasifikasi Multiclass pada Dataset Mental Disorder

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gzaa19/Klasifikasi_Multiclass_pada_Dataset_Mental_Disorder/blob/main/Code_Klasifikasi_Multiclass_pada_Dataset_Mental_Disorder.ipynb)

## 📋 Deskripsi Proyek

Proyek ini bertujuan untuk melakukan **klasifikasi multiclass** pada dataset gangguan mental (Mental Disorder) menggunakan berbagai algoritma *machine learning*. Dataset berisi informasi gejala-gejala pasien yang digunakan untuk memprediksi diagnosis gangguan mental.

## 📊 Dataset

Dataset yang digunakan adalah **Mental Disorders Dataset** dari Kaggle:
- **Sumber**: [Mental Disorders Dataset - Kaggle](https://www.kaggle.com/datasets/mdsultanulislamovi/mental-disorders-dataset/data)
- **Ukuran**: 120 baris × 19 kolom
- **Target Variable**: `Expert Diagnose`

### Kelas Target (4 Kelas):
| Kelas | Jumlah Data |
|-------|-------------|
| Bipolar Type-2 | 31 |
| Depression | 31 |
| Normal | 30 |
| Bipolar Type-1 | 28 |

### Fitur Dataset:
- **Frekuensi Gejala** (Seldom, Sometimes, Usually, Most-Often):
  - Sadness, Euphoric, Exhausted, Sleep dissorder

- **Fitur Biner** (YES/NO):
  - Mood Swing, Suicidal thoughts, Anorxia, Authority Respect
  - Try-Explanation, Aggressive Response, Ignore & Move-On
  - Nervous Break-down, Admit Mistakes, Overthinking

- **Fitur Skala** (1-10):
  - Sexual Activity, Concentration, Optimisim

## 🛠️ Teknologi & Library

```python
# Manipulasi Data
import numpy as np
import pandas as pd

# Visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
```

## 🔄 Alur Kerja (Workflow)

### 1. **Import Library**
Mengimpor library yang diperlukan untuk manipulasi data, visualisasi, dan pemodelan.

### 2. **Load Dataset**
Mengunduh dataset dari Kaggle menggunakan `kagglehub`.

### 3. **Exploratory Data Analysis (EDA)**
- Melihat struktur data (`shape`, `head`, `tail`, `info`, `describe`)
- Menganalisis distribusi kelas target

### 4. **Preprocessing Data**
- **Penghapusan Fitur**: Menghapus kolom `Patient Number` yang tidak relevan
- **Penanganan Konsistensi**: Memperbaiki nilai inkonsisten pada kolom `Suicidal thoughts` ('YES ' → 'YES')
- **Penanganan Missing Value**: Memeriksa dan menangani nilai null

### 5. **Ekstraksi Fitur**
- **Frequency Mapping**: Mengubah nilai frekuensi menjadi numerik (Seldom=0, Sometimes=1, Usually=2, Most-Often=3)
- **Binary Mapping**: Mengubah YES/NO menjadi 1/0
- **Binning**: Mengubah nilai skala 1-10 menjadi kategori (Low, Mid, High)
- **Label Encoding**: Mengenkode fitur kategorikal

### 6. **Visualisasi Korelasi**
Membuat heatmap korelasi antar fitur.

### 7. **Pemodelan & Evaluasi**
Menggunakan **Stratified K-Fold Cross-Validation (K=4)** dengan 4 model:

| Model | Deskripsi |
|-------|-----------|
| **K-Nearest Neighbors (KNN)** | Algoritma berbasis jarak |
| **Decision Tree** | Algoritma berbasis aturan keputusan |
| **Support Vector Classifier (SVC)** | Algoritma berbasis hyperplane |
| **Gaussian Naive Bayes** | Algoritma probabilistik |

### 8. **Metrik Evaluasi**
- **Accuracy**: Akurasi keseluruhan
- **ROC-AUC**: Area Under the ROC Curve (One-vs-Rest)
- **Classification Report**: Precision, Recall, F1-Score
- **Confusion Matrix**: Matriks kebingungan

## 📈 Hasil Evaluasi

Berdasarkan evaluasi K-Fold Cross-Validation, model terbaik adalah:

### 🏆 **Support Vector Classifier (SVC)**

| Fold | Accuracy | ROC-AUC |
|------|----------|---------|
| Fold 1 | 0.7500 | 0.9856 |
| Fold 2 | 0.8750 | 0.9468 |
| Fold 3 | 0.8750 | 0.9630 |
| Fold 4 | 0.8333 | 0.9719 |
| **Rata-rata** | **~0.83** | **~0.97** |

## 🚀 Cara Menjalankan

### Opsi 1: Google Colab (Direkomendasikan)
Klik badge "Open In Colab" di atas untuk langsung menjalankan notebook.

### Opsi 2: Lokal
```bash
# Clone repository
git clone https://github.com/Gzaa19/Klasifikasi_Multiclass_pada_Dataset_Mental_Disorder.git
cd Klasifikasi_Multiclass_pada_Dataset_Mental_Disorder

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub

# Jalankan notebook
jupyter notebook Code_Klasifikasi_Multiclass_pada_Dataset_Mental_Disorder.ipynb
```

## 📁 Struktur Proyek

```
Klasifikasi_Multiclass_pada_Dataset_Mental_Disorder/
├── Code_Klasifikasi_Multiclass_pada_Dataset_Mental_Disorder.ipynb  # Notebook utama
└── README.md                                                        # Dokumentasi proyek
```

## 📝 Kesimpulan

1. **Support Vector Classifier (SVC)** menunjukkan performa terbaik dengan rata-rata akurasi ~83% dan ROC-AUC ~97%
2. Preprocessing yang tepat (encoding, binning) sangat penting untuk performa model
3. Dataset Mental Disorder dapat diklasifikasikan dengan tingkat akurasi yang baik menggunakan metode machine learning tradisional

## 📜 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan penelitian.

---

*Dibuat oleh [Gzaa19](https://github.com/Gzaa19)*
