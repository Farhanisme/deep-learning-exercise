# Pertemuan 5 — Klasifikasi Digit dengan MLP: Softmax & Cross-Entropy

> **Tujuan**: Mahasiswa mampu mengimplementasikan softmax dan cross-entropy
> loss untuk klasifikasi multikelas, serta melatih MLP pada dataset digit
> tulisan tangan (8×8) dari scikit-learn.

## Ringkasan

Dari klasifikasi biner (XOR) kita naik ke multikelas: mengenali digit 0–9
dari citra 8×8 piksel. Kamu mengimplementasikan softmax + cross-entropy dari
nol, lalu melatih MLP `MLPClassifier` scikit-learn pada dataset `load_digits`
— versi mini dari MNIST yang ringan dijalankan di komputer lab.

## Dasar Teori Singkat

- **Softmax** mengubah skor mentah (logits) menjadi distribusi probabilitas:
  $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$.
  Trik stabilitas numerik: kurangi $\max(z)$ sebelum eksponen.
- **Cross-entropy** mengukur jarak prediksi ke label sebenarnya:
  $L = -\log(p_{\text{kelas benar}})$. Prediksi yakin & benar → loss ≈ 0;
  yakin tapi salah → loss besar.
- **Flatten** — citra 8×8 diratakan jadi vektor 64 dimensi sebagai input MLP.

## Alat & Bahan

- Python ≥ 3.10, `numpy`, `scikit-learn`, `matplotlib`, `pytest`
- Dataset: `sklearn.datasets.load_digits` (1797 citra 8×8, built-in)

## Langkah-Langkah Praktikum

1. **Lengkapi `src/digit.py`:**

   | Fungsi | Deskripsi |
   |--------|-----------|
   | `softmax(z)` | stabil numerik (kurangi max) |
   | `cross_entropy(prob, label)` | $-\log(p_{label})$, clip prob min 1e-12 |
   | `siapkan_digits()` | load, normalisasi /16, train_test_split (25%, seed 42) |
   | `latih_mlp(X_train, y_train)` | `MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)` |
   | `evaluasi(model, X_test, y_test)` | akurasi |

2. Latih dan evaluasi (target akurasi > 90%):

   ```bash
   python src/digit.py
   pytest -q
   ```

3. **Eksplorasi** — di notebook, tampilkan 10 citra yang **salah**
   diklasifikasi. Apakah kesalahannya "masuk akal" bagi manusia?

## Lembar Kerja / Hasil Pengamatan

| Hidden layer | Akurasi test | Waktu latih (s) |
|--------------|--------------|------------------|
| (16,) | ... | ... |
| (32,) | ... | ... |
| (64, 32) | ... | ... |

## Tugas / Latihan Praktikum

1. Isi tabel di atas dengan 3 konfigurasi hidden layer — mana terbaik?
2. Berapa jumlah parameter (bobot+bias) MLP (64 → 32 → 10)? Tunjukkan
   perhitunganmu di `SUBMISSION.md`.

## Format Pelaporan

Isi `SUBMISSION.md` lalu push ke `main`.
