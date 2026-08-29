# Pertemuan 7 — Recurrent Neural Network & Data Sekuensial

> **Tujuan**: Mahasiswa mampu mengimplementasikan sel RNN sederhana dengan
> NumPy, menyiapkan data deret waktu dengan sliding window, dan memahami
> masalah vanishing gradient yang memotivasi LSTM.

## Ringkasan

CNN unggul di data spasial; RNN dirancang untuk data **sekuensial** — teks,
sinyal, deret waktu. RNN membaca input satu langkah demi satu langkah sambil
membawa "ingatan" (hidden state). Kamu mengimplementasikan satu sel RNN dari
nol, memakainya memprediksi deret sinus, dan mengamati mengapa RNN vanilla
sulit mengingat jangka panjang (vanishing gradient) — masalah yang dijawab
LSTM.

## Dasar Teori Singkat

- **Sel RNN**: $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$. Hidden state $h_t$
  adalah ringkasan seluruh input sampai waktu $t$.
- **Sliding window**: deret $[s_1..s_n]$ diubah jadi pasangan
  (window sepanjang $w$) → (nilai berikutnya) untuk supervised learning.
- **Vanishing gradient**: gradien melewati banyak perkalian $\tanh' \cdot W_h$;
  jika nilainya < 1, gradien menyusut eksponensial → ingatan jangka panjang
  hilang. **LSTM** menambahkan cell state + gerbang (forget/input/output)
  agar informasi bisa lewat tanpa menyusut.

## Alat & Bahan

- Python ≥ 3.10, `numpy`, `matplotlib`, `pytest`
- Opsional: `tensorflow` (LSTM Keras — bisa di Google Colab)

## Langkah-Langkah Praktikum

1. **Lengkapi `src/rnn.py`:**

   | Fungsi | Deskripsi |
   |--------|-----------|
   | `buat_dataset_window(deret, w)` | sliding window → (X, y) |
   | `rnn_cell(x_t, h_prev, Wx, Wh, b)` | satu langkah RNN |
   | `rnn_forward(deret, Wx, Wh, b)` | proses seluruh sekuens, return semua h |
   | `demo_vanishing(Wh, langkah)` | norma gradien ~ produk |Wh·tanh'| |

2. Jalankan dan amati:

   ```bash
   python src/rnn.py
   pytest -q
   ```

3. **Eksplorasi** — notebook memvisualisasikan hidden state pada deret sinus
   dan kurva vanishing gradient; bagian opsional melatih LSTM Keras.

## Lembar Kerja / Hasil Pengamatan

| |Wh| | Norma "gradien" setelah 50 langkah | Kesimpulan |
|-------|--------------------------------------|------------|
| 0.5 | ... | ... |
| 1.0 | ... | ... |
| 1.5 | ... | ... |

## Tugas / Latihan Praktikum

1. Isi tabel di atas dan jelaskan hubungannya dengan vanishing/exploding
   gradient.
2. Sebutkan 3 gerbang LSTM dan fungsi masing-masing (rujuk teori + notebook).

## Format Pelaporan

Isi `SUBMISSION.md` lalu push ke `main`.
