# Laporan Mini Project Deep Learning

- **NIM**    : 105841110523
- **Nama**   : Muhammad Zaky Farhan
- **Kelas**  : 6AI-B
- **Judul**  : Klasifikasi Fashion-MNIST Menggunakan Baseline Linear, MLP, dan CNN

## 1. Pendahuluan

(masalah, motivasi, tujuan)

Proyek ini bertujuan mengklasifikasikan citra pakaian Fashion-MNIST ke dalam 10 kelas. Tujuannya adalah membandingkan baseline linear, MLP, dan CNN pada data yang sama untuk melihat pengaruh struktur model terhadap performa klasifikasi citra.

## 2. Data & Praproses

(sumber, ukuran, pembagian latih/uji, normalisasi/praproses yang dilakukan)

Dataset yang digunakan adalah Fashion-MNIST dari `tensorflow.keras.datasets.fashion_mnist`. Dataset asli berisi 60.000 data latih dan 10.000 data uji, masing-masing berupa citra grayscale 28x28 piksel. Kode eksperimen memakai subset berimbang 12.000 data latih dan 3.000 data uji agar training tetap ringan di komputer lokal. Nilai piksel dinormalisasi dari rentang 0-255 menjadi 0-1, lalu citra dibentuk menjadi `(28, 28, 1)` untuk CNN.

## 3. Model

### 3.1 Baseline

(deskripsi + alasan pemilihan)

Baseline yang digunakan adalah SGD Logistic Classifier pada pixel yang diratakan menjadi 784 fitur. Model ini dipilih karena sederhana dan dapat menunjukkan performa dasar tanpa memanfaatkan struktur spasial citra.

### 3.2 Model Utama

(arsitektur, hyperparameter, alasan desain)

Model pembanding adalah MLP Keras dengan Dense 128, Dropout 0.2, Dense 64, dan output Softmax 10 kelas. Model utama adalah CNN Keras dengan Conv2D 32 filter, MaxPooling, Conv2D 64 filter, MaxPooling, Dense 128, Dropout 0.2, dan output Softmax. CNN dipilih sebagai model utama karena konvolusi dapat menangkap pola spasial pada citra pakaian.

## 4. Hasil

| Model | Metrik utama | Nilai |
|-------|--------------|-------|
| Baseline SGD Logistic | Akurasi | 82.33% |
| MLP | Akurasi | 83.80% |
| CNN | Akurasi | 87.37% |

| Model | F1-score macro |
|-------|----------------|
| Baseline SGD Logistic | 82.10% |
| MLP | 83.52% |
| CNN | 87.27% |

| Model | Precision macro | Recall macro | Waktu latih |
|-------|-----------------|--------------|-------------|
| Baseline SGD Logistic | 82.91% | 82.33% | sekitar 9 s |
| MLP | 84.19% | 83.80% | sekitar 2 s |
| CNN | 87.29% | 87.37% | sekitar 10 s |

(sisipkan plot kurva loss / confusion matrix / prediksi)

Confusion matrix model terbaik menunjukkan sebagian besar prediksi berada pada diagonal utama. Kesalahan terbesar CNN terjadi pada kelas pakaian yang bentuknya mirip, seperti Shirt diprediksi T-shirt/top, Shirt diprediksi Coat, Pullover diprediksi Coat, T-shirt/top diprediksi Shirt, dan Pullover diprediksi Shirt.

## 5. Analisis Kesalahan

(minimal 2 contoh prediksi salah + dugaan penyebab)

Contoh pertama, label asli Pullover diprediksi sebagai Coat. Dugaan penyebabnya adalah kedua kelas sama-sama memiliki bentuk lengan dan badan pakaian yang mirip pada citra grayscale kecil.

Contoh kedua, label asli T-shirt/top diprediksi sebagai Shirt. Dugaan penyebabnya adalah bentuk kerah dan lengan pada resolusi 28x28 tampak mirip, sehingga batas antar kelas menjadi sulit.

Contoh ketiga, label asli Dress diprediksi sebagai Shirt. Kesalahan ini dapat terjadi jika siluet bawah dress tidak terlihat jelas atau tampak pendek sehingga menyerupai pakaian atasan.

## 6. Kesimpulan & Saran

CNN menjadi model terbaik dengan akurasi 87.37% dan F1 macro 87.27%, lebih baik daripada MLP dan baseline linear. Hasil ini sesuai dengan karakter data citra karena CNN memanfaatkan pola lokal dan struktur spasial. Jika proyek dikembangkan lebih lanjut, eksperimen dapat memakai seluruh data Fashion-MNIST, tuning hyperparameter, augmentasi data, dan early stopping.

## 7. Referensi

- TensorFlow/Keras documentation: `fashion_mnist`, `Sequential`, `Conv2D`, `MaxPooling2D`, dan `Dense`.
- Scikit-learn documentation: `SGDClassifier` dan metrik evaluasi klasifikasi.
- Materi praktikum Deep Learning pertemuan 1 sampai 7.
