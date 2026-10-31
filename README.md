# 🫀 Heart Disease Risk Prediction & Classification

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-00A389?style=for-the-badge)](https://shap.readthedocs.io)
[![Certification](https://img.shields.io/badge/Standard-BNSP%20Data%20Science-blue?style=for-the-badge)](https://bnsp.go.id)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> 🔬 **Fokus:** *Healthcare Analytics, Predictive Modeling, Explainable AI (XAI), Automated Machine Learning (AutoML)*

---

## 📋 Deskripsi Proyek

Penyakit kardiovaskular (Cardiovascular Diseases / CVDs) merupakan salah satu penyebab kematian tertinggi di dunia. Diagnosis dini yang akurat dan berbasis data klinis sangat krusial untuk mencegah komplikasi fatal.

Proyek ini membangun sistem klasifikasi risiko penyakit jantung berbasis *Machine Learning* menggunakan dataset klinis **Cleveland Heart Disease**. Sistem dikembangkan mengikuti seluruh unit kompetensi standar **SKKNI Data Science BNSP**, mulai dari penelaahan data, pembersihan data (*data cleaning*), rekayasa fitur (*feature engineering*), interpretasi fitur berbasis **Explainable AI (SHAP)**, perbandingan multi-algoritma (*Logistic Regression, Decision Tree, Random Forest, XGBoost*), hingga optimasi otomatis menggunakan **AutoML**.

---

## 📊 Dataset & Variabel Klinis

Dataset terdiri dari **1.025 baris data rekam medis pasien** dengan **14 variabel klinis**:

| Fitur | Deskripsi | Tipe Data |
| :--- | :--- | :--- |
| `age` | Usia pasien (tahun) | Numerik |
| `sex` | Jenis kelamin (1 = Pria, 0 = Wanita) | Kategorikal |
| `cp` | Jenis nyeri dada (Chest Pain Type: 0–3) | Kategorikal |
| `trestbps` | Tekanan darah istirahat saat masuk RS (mm Hg) | Numerik |
| `chol` | Kolesterol serum (mg/dl) | Numerik |
| `fbs` | Gula darah puasa > 120 mg/dl (1 = Benar, 0 = Salah) | Biner |
| `restecg` | Hasil elektrokardiografi istirahat (0, 1, 2) | Kategorikal |
| `thalach` | Detak jantung maksimum yang tercapai (*Max Heart Rate*) | Numerik |
| `exang` | Angina akibat olahraga / *exercise induced angina* (1 = Ya, 0 = Tidak) | Biner |
| `oldpeak` | Depresi ST yang diinduksi olahraga relatif terhadap istirahat | Numerik |
| `slope` | Kemiringan puncak segmen ST saat latihan (0, 1, 2) | Kategorikal |
| `ca` | Jumlah pembuluh darah utama (0–3) yang diwarnai fluoroskopi | Numerik |
| `thal` | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) | Kategorikal |
| **`target`** | **Diagnosis Penyakit Jantung (1 = Berisiko/Sakit, 0 = Normal)** | **Target (Biner)** |

---

## 🏗️ Metodologi & Pipeline Pengembangan (Standar SKKNI BNSP)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA SCIENCE PIPELINE (SKKNI)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [1] Penelaahan Data (Data Understanding)                                   │
│      ├── Audit tipe data, statistik deskriptif & sebaran label target       │
│                                                                             │
│  [2] Validasi & Pembersihan Data (Data Cleaning)                            │
│      ├── Deteksi & penanganan missing values & duplikasi                    │
│      └── Deteksi outlier menggunakan metode IQR & Z-Score                   │
│                                                                             │
│  [3] Explainable AI (XAI) & Seleksi Fitur via SHAP                         │
│      ├── Pelatihan baseline XGBoost Classifier                              │
│      └── Analisis kontribusi fitur menggunakan TreeSHAP Explainer           │
│                                                                             │
│  [4] Konstruksi & Transformasi Data                                         │
│      ├── Encoding data kategorikal & standarisasi fitur numerik             │
│      └── Data splitting: 80% Train Set (819) vs 20% Test Set (205)         │
│                                                                             │
│  [5] Pemodelan Komparatif & Hyperparameter Tuning                          │
│      ├── Pruned Decision Tree (Optimasi max_depth)                          │
│      ├── Logistic Regression (Regularized)                                  │
│      ├── Random Forest Classifier (Ensemble)                                │
│      └── Automated Machine Learning (AutoML MLJAR)                          │
│                                                                             │
│  [6] Evaluasi Komprehensif                                                  │
│      └── Accuracy, Precision, Recall, F1-Score, dan ROC-AUC Curve           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Hasil Evaluasi Performa Model

Evaluasi dilakukan pada **data pengujian independen (Test Set)**:

| Algoritma / Model | Accuracy | Precision | Recall (Sensitivitas) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **84.62%** | **85.44%** | **87.13%** | **86.27%** | **88.43%** |
| **Decision Tree (Pruned max_depth=4)** | 84.07% | 82.73% | 90.10% | 86.26% | 87.77% |
| **AutoML Best Model (Ensemble)** | 82.97% | 82.41% | 88.12% | 85.17% | **90.37%** |
| **Random Forest (n=20)** | 81.32% | 82.52% | 84.16% | 83.33% | 86.98% |
| **Decision Tree (max_depth=3)** | 82.42% | 82.24% | 87.13% | 84.62% | 86.90% |

> 💡 **Temuan Utama:**  
> - **Logistic Regression** menghasilkan akurasi dan F1-Score tertinggi secara keseluruhan (**84.62%** dan **86.27%**).
> - **AutoML Ensemble** memberikan kemampuan diskriminasi probabilitas terbaik dengan skor **ROC-AUC mencapai 90.37%**.
> - Pada konteks medis, metrik **Recall** sangat penting untuk meminimalkan *False Negative* (pasien sakit yang terlewat). Model **Decision Tree (depth=4)** memberikan recall tertinggi sebesar **90.10%**.

---

## 🔍 Wawasan Explainable AI (SHAP Insights)

Melalui analisis **SHAP (SHapley Additive exPlanations)**:
1. **`cp` (Chest Pain Type)**: Fitur paling berpengaruh; jenis nyeri dada tipe non-anginal / atypical secara konsisten meningkatkan probabilitas penyakit jantung.
2. **`thalach` (Max Heart Rate)**: Nilai detak jantung maksimum yang lebih tinggi berkorelasi positif dengan risiko jantung pada populasi pasien uji.
3. **`ca` (Major Vessels Colored)**: Semakin banyak pembuluh darah yang mengalami penyumbatan, probabilitas penyakit jantung meningkat secara signifikan.

---

## 🗂️ Struktur Direktori

```text
├── 📓 heart_disease_classification.ipynb  # Notebook analisis end-to-end lengkap
├── 📊 heart.csv                          # Dataset rekam medis pasien Cleveland
├── 📄 requirements.txt                   # Dependensi pustaka Python
├── 📄 .gitignore                         # File konfigurasi Git
└── 📄 README.md                          # Dokumentasi proyek
```
