# Praktikum Deep Learning

Repository ini berisi pengerjaan praktikum Deep Learning dari pertemuan 1 sampai pertemuan 8. Setiap pertemuan disusun sebagai latihan terpisah, sedangkan pertemuan 8 menjadi mini project akhir yang menggabungkan proses pemuatan data, preprocessing, pelatihan model, evaluasi, notebook eksplorasi, dan laporan.

## Ringkasan Materi

| Pertemuan | Fokus Pengerjaan | Berkas Utama |
|---|---|---|
| 01 | Dasar NumPy, operasi matriks, normalisasi, dan fungsi aktivasi | `pertemuan-01/src/main.py` |
| 02 | Regresi linear, MSE, gradient descent, dan training loop | `pertemuan-02/src/regresi.py` |
| 03 | Perceptron, sigmoid, dan forward pass MLP sederhana | `pertemuan-03/src/perceptron.py` |
| 04 | Backpropagation untuk kasus XOR | `pertemuan-04/src/backprop.py` |
| 05 | Klasifikasi digit dengan MLP dan evaluasi model | `pertemuan-05/src/digit.py` |
| 06 | Operasi dasar CNN, konvolusi, ReLU, dan max pooling | `pertemuan-06/src/cnn.py` |
| 07 | Dasar RNN, sequence window, forward pass, dan vanishing gradient | `pertemuan-07/src/rnn.py` |
| 08 | Mini project klasifikasi citra Fashion-MNIST | `pertemuan-08/src/proyek.py` |

## Final Project Pertemuan 8

Topik final project adalah klasifikasi citra Fashion-MNIST menggunakan tiga pendekatan:

1. Baseline SGD Logistic pada pixel yang diratakan.
2. Multi-Layer Perceptron (MLP) sebagai pembanding neural network.
3. Convolutional Neural Network (CNN) sebagai model utama untuk data citra.

Dataset Fashion-MNIST dimuat melalui `tensorflow.keras.datasets.fashion_mnist`. Dataset asli berisi 60.000 data latih dan 10.000 data uji. Untuk menjaga eksperimen tetap ringan, project menggunakan subset berimbang 12.000 data latih dan 3.000 data uji.

Hasil utama pada data uji:

| Model | Accuracy | F1 Macro |
|---|---:|---:|
| Baseline SGD Logistic | 0,8233 | 0,8210 |
| MLP | 0,8380 | 0,8352 |
| CNN | 0,8737 | 0,8727 |

CNN menjadi model terbaik pada konfigurasi ini. Kesalahan terbesar masih terjadi pada kelas pakaian bagian atas, terutama `Shirt`, `T-shirt/top`, `Pullover`, dan `Coat`, karena bentuk visualnya cukup mirip pada citra grayscale 28 x 28 pixel.

## Struktur Repository

Setiap folder pertemuan umumnya berisi:

```text
pertemuan-xx/
|-- README.md
|-- requirements.txt
|-- src/
|-- notebooks/
|-- tests/
`-- SUBMISSION.md
```

Khusus pertemuan 8, terdapat beberapa berkas tambahan:

```text
pertemuan-08/
|-- PROPOSAL.md
|-- LAPORAN.md
|-- LAPORAN-AKADEMIK-LENGKAP.docx
|-- SUBMISSION.md
|-- src/proyek.py
|-- notebooks/eksplorasi.ipynb
`-- assets/
```

## Cara Menjalankan

Gunakan Python 3.12 atau versi yang kompatibel dengan dependensi pada masing-masing pertemuan.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r pertemuan-08/requirements.txt
```

Menjalankan final project:

```powershell
cd pertemuan-08
python src/proyek.py
```

Menjalankan test struktur pertemuan 8:

```powershell
cd pertemuan-08
pytest -q
```

Untuk pertemuan lain, masuk ke folder pertemuan yang ingin dicek, install `requirements.txt` jika diperlukan, lalu jalankan `pytest -q`.

## Catatan

Notebook digunakan untuk eksplorasi dan visualisasi hasil, sedangkan kode utama disimpan di folder `src`. File `SUBMISSION.md` berisi jawaban dan ringkasan pengumpulan setiap pertemuan. Laporan lengkap final project tersedia dalam format Word pada `pertemuan-08/LAPORAN-AKADEMIK-LENGKAP.docx`.