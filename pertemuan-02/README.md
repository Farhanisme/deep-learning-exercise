# Pertemuan 2 — Regresi Linear & Gradient Descent dari Nol

> **Tujuan**: Mahasiswa mampu mengimplementasikan regresi linear beserta
> algoritma gradient descent dari nol menggunakan NumPy, dan memahami peran
> learning rate terhadap konvergensi.

## Ringkasan

Gradient descent adalah mesin di balik semua pelatihan neural network.
Pertemuan ini mengimplementasikannya pada kasus paling sederhana — regresi
linear — sehingga kamu melihat persis bagaimana model "belajar" dari error.
Materi mengadaptasi BAB 3 (Implementasi Algoritma Regresi) dan BAB 11
(Algoritma Optimasi — Gradient Descent) Buku Ajar.

## Dasar Teori Singkat

Model: $\hat{y} = wx + b$. Ukuran kesalahan (loss) memakai **MSE**:

$$L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

**Gradient descent** memperbarui parameter berlawanan arah gradien:

$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}, \qquad
b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

dengan $\frac{\partial L}{\partial w} = \frac{2}{n}\sum(\hat{y}-y)x$ dan
$\frac{\partial L}{\partial b} = \frac{2}{n}\sum(\hat{y}-y)$.
$\alpha$ (learning rate) terlalu besar → divergen; terlalu kecil → lambat.

## Alat & Bahan

- Python ≥ 3.10, `numpy`, `matplotlib`, `pytest`
- Dataset sintetis dibuat di kode (`y = 3x + 2 + noise`)

## Langkah-Langkah Praktikum

1. **Lengkapi `src/regresi.py`:**

   | Fungsi | Deskripsi |
   |--------|-----------|
   | `prediksi(X, w, b)` | $\hat{y} = wX + b$ |
   | `mse(y_pred, y)` | mean squared error |
   | `hitung_gradien(X, y, w, b)` | `(dw, db)` sesuai rumus di atas |
   | `latih(X, y, lr, epochs)` | loop GD, return `(w, b, riwayat_loss)` |

2. Latih dan amati loss menurun:

   ```bash
   python src/regresi.py
   pytest -q
   ```

   Model harus menemukan $w \approx 3$, $b \approx 2$.

3. **Eksplorasi** — di notebook, plot kurva loss untuk
   `lr ∈ {0.001, 0.01, 0.1, 1.0}`. Mana yang divergen?

## Lembar Kerja / Hasil Pengamatan

| Learning rate | Loss akhir | Konvergen? | w, b akhir |
|---------------|-----------|------------|------------|
| 0.001 | ... | ... | ... |
| 0.01 | ... | ... | ... |
| 0.1 | ... | ... | ... |
| 1.0 | ... | ... | ... |

## Tugas / Latihan Praktikum

1. Modifikasi `latih` agar berhenti dini (early stopping) jika selisih loss
   antar-epoch < 1e-9. Berapa epoch yang dihemat?
2. Ganti data menjadi $y = -2x + 5$ dan buktikan model tetap menemukan
   parameter yang benar.

## Format Pelaporan

Isi `SUBMISSION.md` lalu push ke `main`.
