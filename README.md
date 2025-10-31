# Perbandingan Model Klasifikasi Machine Learning

Repository ini berisi implementasi dan perbandingan beberapa algoritma klasifikasi machine learning umum. Proyek ini mencakup alur kerja *end-to-end* sederhana: mulai dari pemuatan data, preprocessing, pelatihan model, hingga evaluasi performa.

## 🎯 Tujuan

Proyek ini bertujuan untuk:
* Mendemonstrasikan implementasi pipeline klasifikasi machine learning.
* Menerapkan teknik preprocessing data standar.
* Membandingkan performa dari beberapa model klasifikasi populer pada dataset yang sama.
* Mengevaluasi model menggunakan metrik standar (Accuracy, Precision, Recall, F1-Score, dll.).

## 📊 Dataset

* **Nama Dataset:** [Nama Dataset Anda, misal: Iris Dataset]
* **Sumber:** [Sumber dataset, misal: `sklearn.datasets.load_iris()` atau link ke file CSV]
* **Deskripsi Singkat:** [Jelaskan dataset Anda. Misal: Dataset ini terdiri dari 150 sampel dari 3 spesies bunga Iris (Setosa, Versicolor, Virginica) yang diukur berdasarkan 4 fitur: panjang & lebar kelopak dan mahkota.]
* **Variabel Target:** [Sebutkan kolom targetnya, misal: `species` (3 kelas)]

## 🤖 Model yang Dibandingkan

Algoritma klasifikasi yang diimplementasikan dalam proyek ini:

1.  **Logistic Regression**
2.  **Decision Tree Classifier**
3.  **[Tambahkan model lain jika ada, misal: K-Nearest Neighbors]**
4.  **[Tambahkan model lain jika ada, misal: Support Vector Machine (SVM)]**

## 🛠️ Teknologi & Library

Proyek ini dibuat menggunakan **Python** dan library berikut:

* **uv**: Untuk manajemen *virtual environment* dan instalasi *package* yang cepat.
* **Scikit-learn**: Untuk implementasi model, preprocessing, dan metrik evaluasi.
* **Pandas**: Untuk manipulasi dan analisis data (DataFrame).
* **Matplotlib & Seaborn**: Untuk visualisasi data dan hasil (EDA, Confusion Matrix, ROC Curve).
* **Jupyter Notebook**: Sebagai environment untuk pengembangan interaktif.

## 📈 Hasil Evaluasi (Contoh)

Performa model dievaluasi pada data uji (test set). Metrik yang digunakan adalah *Macro Average* untuk F1-Score, Precision, dan Recall agar setiap kelas memiliki bobot yang sama.

| Metrik | Model 1: Logistic Regression | Model 2: Decision Tree |
| :--- | :---: | :---: |
| **Accuracy** | [Hasil Accuracy, misal: 0.97] | [Hasil Accuracy, misal: 1.00] |
| **Macro Avg F1-Score**| [Hasil F1, misal: 0.97] | [Hasil F1, misal: 1.00] |
| **Macro Avg Precision**| [Hasil Precision, misal: 0.97]| [Hasil Precision, misal: 1.00]|
| **Macro Avg Recall** | [Hasil Recall, misal: 0.97] | [Hasil Recall, misal: 1.00] |

### Kesimpulan
[Tulis kesimpulan general Anda di sini. Contoh: Pada dataset ini, model Decision Tree menunjukkan performa yang sempurna dan sedikit mengungguli Logistic Regression. Hal ini kemungkinan karena batas keputusan (decision boundaries) pada data ini lebih mudah ditangkap oleh aturan non-linear dari Decision Tree.]
