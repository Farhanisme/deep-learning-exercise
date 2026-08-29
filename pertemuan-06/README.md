# Pertemuan 6 — Convolutional Neural Network: Konvolusi dari Nol

> **Tujuan**: Mahasiswa mampu mengimplementasikan operasi konvolusi 2D dan
> max pooling dengan NumPy, memahami cara kernel mendeteksi fitur citra,
> serta menyusun arsitektur CNN sederhana dengan Keras.

## Ringkasan

MLP memperlakukan piksel sebagai fitur lepas — informasi spasial hilang.
CNN mempertahankan struktur 2D lewat **konvolusi**: kernel kecil yang
digeser ke seluruh citra untuk mendeteksi tepi, sudut, dan pola. Kamu
mengimplementasikan konvolusi & pooling dari nol, mengamati efek kernel
klasik (edge detection, blur, sharpen), lalu menyusun CNN di Keras.

## Dasar Teori Singkat

- **Konvolusi 2D** (valid): output$[i,j] = \sum_{u,v}$ citra$[i+u, j+v]\cdot$kernel$[u,v]$.
  Ukuran output: $(H-k+1) \times (W-k+1)$.
- **Max pooling 2×2**: ambil nilai maksimum tiap blok 2×2 → ukuran ½,
  tahan terhadap pergeseran kecil.
- **Arsitektur khas**: `Conv → ReLU → Pool` berulang, diakhiri `Flatten → Dense`.
- Kernel klasik: sobel (tepi), blur (rata-rata), sharpen.

## Alat & Bahan

- Python ≥ 3.10, `numpy`, `matplotlib`, `scikit-learn` (citra contoh), `pytest`
- Opsional: `tensorflow` (arsitektur CNN Keras — bisa di Google Colab)

## Langkah-Langkah Praktikum

1. **Lengkapi `src/cnn.py`:**

   | Fungsi | Deskripsi |
   |--------|-----------|
   | `konvolusi2d(citra, kernel)` | konvolusi valid, loop atau vektorisasi |
   | `relu(x)` | max(0, x) element-wise |
   | `max_pooling(citra, ukuran=2)` | non-overlapping max pool |
   | `hitung_ukuran_output(h, w, k, pool)` | ukuran fitur map akhir |

2. Uji dengan kernel deteksi tepi:

   ```bash
   python src/cnn.py
   pytest -q
   ```

3. **Eksplorasi** — notebook menerapkan kernel sobel/blur/sharpen pada citra
   digit dan (opsional) melatih CNN Keras kecil.

## Lembar Kerja / Hasil Pengamatan

| Kernel | Efek visual yang diamati |
|--------|--------------------------|
| Sobel X | ... |
| Sobel Y | ... |
| Blur 3×3 | ... |
| Sharpen | ... |

## Tugas / Latihan Praktikum

1. Rancang kernel 3×3 buatanmu sendiri dan jelaskan efeknya.
2. Citra 28×28 → Conv 3×3 (8 filter) → Pool 2×2 → Conv 3×3 (16 filter) →
   Pool 2×2. Hitung ukuran feature map di tiap tahap (tulis di `SUBMISSION.md`).

## Format Pelaporan

Isi `SUBMISSION.md` lalu push ke `main`.
