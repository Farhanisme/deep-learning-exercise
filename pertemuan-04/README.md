# Pertemuan 4 — Backpropagation dari Nol & Pengenalan Keras

> **Tujuan**: Mahasiswa mampu mengimplementasikan backpropagation untuk MLP
> satu hidden layer dengan NumPy hingga menyelesaikan XOR, lalu membandingkan
> dengan implementasi Keras.

## Ringkasan

Pertemuan 3 memakai bobot XOR "hadiah". Kali ini jaringan **belajar sendiri**
bobot tersebut via backpropagation — algoritma yang menghitung gradien loss
terhadap semua bobot dengan aturan rantai (chain rule). Setelah paham
mekanismenya, kamu akan melihat betapa singkatnya kode ekuivalen di Keras.

## Dasar Teori Singkat

MLP 2-2-1 dengan sigmoid dan loss MSE. Forward:
$h = \sigma(W_1 x + b_1)$, $\hat{y} = \sigma(W_2 h + b_2)$.

Backward (chain rule), dengan $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

$$\delta_2 = (\hat{y} - y)\,\hat{y}(1-\hat{y})$$
$$\delta_1 = (W_2^\top \delta_2) \odot h(1-h)$$

Gradien: $\nabla W_2 = \delta_2 h^\top$, $\nabla b_2 = \delta_2$,
$\nabla W_1 = \delta_1 x^\top$, $\nabla b_1 = \delta_1$. Semua bobot
di-update dengan gradient descent seperti pertemuan 2.

## Alat & Bahan

- Python ≥ 3.10, `numpy`, `pytest`
- Opsional: `tensorflow` (bagian intro Keras — jalankan di komputer sendiri
  atau Google Colab jika komputer lab tidak memadai)

## Langkah-Langkah Praktikum

1. **Lengkapi `src/backprop.py`:**

   | Fungsi | Deskripsi |
   |--------|-----------|
   | `sigmoid(z)`, `turunan_sigmoid(z)` | aktivasi + turunannya |
   | `forward(x, params)` | return `(h, y_hat)` |
   | `backward(x, y, params)` | return dict gradien `dW1, db1, dW2, db2` |
   | `latih_xor(lr, epochs, seed)` | training loop; return `(params, riwayat_loss)` |

2. Latih XOR sampai tuntas:

   ```bash
   python src/backprop.py     # loss harus turun, 4/4 prediksi benar
   pytest -q
   ```

3. **Intro Keras** — buka `notebooks/eksplorasi.ipynb`: model XOR yang sama
   ditulis dalam ±6 baris Keras. Bandingkan jumlah baris kodenya!

## Lembar Kerja / Hasil Pengamatan

| Implementasi | Epoch sampai 4/4 benar | Baris kode inti |
|--------------|------------------------|------------------|
| NumPy manual | ... | ... |
| Keras | ... | ... |

## Tugas / Latihan Praktikum

1. Ubah jumlah neuron hidden dari 2 → 4. Apakah konvergensi lebih cepat?
2. Ganti seed inisialisasi 3 kali — apakah selalu konvergen? Tulis analisismu
   tentang pengaruh inisialisasi bobot.

## Format Pelaporan

Isi `SUBMISSION.md` lalu push ke `main`.
