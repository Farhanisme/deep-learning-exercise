# Pertemuan 3 — Perceptron & Multi-Layer Perceptron dari Nol

> **Tujuan**: Mahasiswa mampu mengimplementasikan perceptron beserta aturan
> pembelajarannya dan forward pass MLP dengan NumPy, serta membuktikan mengapa
> perceptron tunggal gagal pada masalah XOR.

## Ringkasan

Perceptron (Rosenblatt, 1958) adalah neuron buatan pertama — dan batu penjuru
seluruh deep learning. Pertemuan ini melatih perceptron pada gerbang logika
AND/OR, menunjukkan kegagalannya pada XOR, lalu membangun forward pass MLP
yang menyelesaikan XOR — motivasi utama jaringan berlapis.

## Dasar Teori Singkat

**Perceptron**: $\hat{y} = \text{step}(w \cdot x + b)$, dengan step = 1 jika
input ≥ 0, selain itu 0. Aturan belajar untuk tiap sampel:

$$w \leftarrow w + \alpha\,(y - \hat{y})\,x, \qquad b \leftarrow b + \alpha\,(y - \hat{y})$$

Perceptron hanya bisa memisahkan data **linearly separable**. XOR tidak bisa
dipisahkan garis lurus → butuh **hidden layer**. MLP 2-2-1 dengan aktivasi
sigmoid $\sigma(z) = 1/(1+e^{-z})$ mampu menyelesaikannya.

## Alat & Bahan

- Python ≥ 3.10, `numpy`, `matplotlib`, `pytest`

## Langkah-Langkah Praktikum

1. **Lengkapi `src/perceptron.py`:**

   | Fungsi/Kelas | Deskripsi |
   |--------------|-----------|
   | `step(z)` | fungsi aktivasi step |
   | `Perceptron.prediksi(x)` | step(w·x + b) |
   | `Perceptron.latih(X, y, lr, epochs)` | aturan belajar perceptron |
   | `sigmoid(z)` | $1/(1+e^{-z})$ |
   | `forward_mlp(x, W1, b1, W2, b2)` | forward pass MLP 1 hidden layer |

2. Latih pada gerbang logika:

   ```bash
   python src/perceptron.py
   pytest -q
   ```

   Perceptron harus 100% benar di AND & OR, tetapi **gagal** di XOR.

3. **Eksplorasi** — notebook memuat bobot MLP siap-pakai untuk XOR;
   visualisasikan decision boundary perceptron vs MLP.

## Lembar Kerja / Hasil Pengamatan

| Gerbang | Akurasi perceptron | Epoch sampai konvergen |
|---------|--------------------|-----------------------|
| AND | ... | ... |
| OR | ... | ... |
| XOR | ... | (tidak konvergen) |

## Tugas / Latihan Praktikum

1. Latih perceptron untuk gerbang NAND, laporkan bobot akhirnya.
2. Di `SUBMISSION.md`, gambarkan (boleh sketsa/foto) mengapa XOR tak bisa
   dipisahkan satu garis lurus.

## Format Pelaporan

Isi `SUBMISSION.md` lalu push ke `main`.
