# Proposal Mini Project Deep Learning

- **NIM**    : 105841110523
- **Nama**   : Muhammad Zaky Farhan
- **Kelas**  : 6AI-B

## 1. Judul Proyek

Klasifikasi Fashion-MNIST Menggunakan Baseline Linear, MLP, dan CNN

## 2. Latar Belakang & Rumusan Masalah

(2-3 kalimat: masalah apa yang diselesaikan dan mengapa menarik)

Proyek ini menyelesaikan masalah klasifikasi citra pakaian Fashion-MNIST ke dalam 10 kelas, seperti T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, dan Ankle boot. Masalah ini menarik karena beberapa kelas memiliki bentuk visual yang mirip, sehingga cocok untuk membandingkan baseline linear, MLP, dan CNN.

## 3. Dataset

- Sumber: `tensorflow.keras.datasets.fashion_mnist`
- Jumlah sampel: 60.000 data latih dan 10.000 data uji pada dataset asli
- Fitur/format: citra grayscale 28x28 piksel dengan 10 kelas pakaian
- Catatan eksperimen: agar ringan dijalankan di komputer lokal, kode memakai subset berimbang 12.000 data latih dan 3.000 data uji.

## 4. Model yang Direncanakan

- Baseline: SGD Logistic Classifier pada pixel yang diratakan
- Model pembanding: MLP Keras dengan Dense 128, Dropout, Dense 64, dan Softmax
- Model utama: CNN Keras dengan Conv2D, MaxPooling, Dense, Dropout, dan Softmax

## 5. Metrik Evaluasi

Akurasi, precision macro, recall macro, dan F1-score macro pada data uji. Confusion matrix dan contoh prediksi salah digunakan untuk menganalisis kelas pakaian yang paling sering tertukar.

## 6. Rencana Kerja (di dalam sesi)

| Tahap | Estimasi waktu |
|-------|----------------|
| Persiapan data dan EDA | 20 menit |
| Training baseline dan MLP | 25 menit |
| Training CNN | 25 menit |
| Evaluasi, analisis error, dan laporan | 50 menit |

---
