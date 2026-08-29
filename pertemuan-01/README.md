# Pertemuan 1 — Orientasi Lab & Fondasi NumPy untuk Deep Learning

> **Tujuan**: Mahasiswa mampu menyiapkan lingkungan kerja deep learning
> (numpy, matplotlib, TensorFlow/Keras) dan menguasai operasi tensor dasar
> yang menjadi fondasi seluruh jaringan saraf.

## Ringkasan

Deep learning pada intinya adalah rangkaian operasi matriks. Pertemuan pertama
menyiapkan lingkungan kerja dan melatih operasi vektor/matriks NumPy yang akan
dipakai di semua pertemuan berikutnya. Orientasi K3/SOP mengacu pada BAB 1–2
Buku Ajar Praktikum Kecerdasan Buatan.

## Dasar Teori Singkat

- **Tensor** = generalisasi skalar (0D), vektor (1D), matriks (2D), dst.
  Semua data di neural network (input, bobot, output) adalah tensor.
- **Perkalian matriks** $C = A \cdot B$: elemen $c_{ij}$ = dot product baris
  $i$ dari $A$ dengan kolom $j$ dari $B$. Inilah operasi inti layer neural
  network: $y = Wx + b$.
- **Vectorization** — operasi array NumPy jauh lebih cepat daripada loop
  Python karena dieksekusi di level C.

## Alat & Bahan

- Python ≥ 3.10, `numpy`, `matplotlib`, `pytest`
- Opsional (untuk pertemuan 4+): `tensorflow` — cek dengan `src/cek_env.py`

## Keselamatan & Etika Kerja (K3/SOP)

- Ikuti tata tertib laboratorium; laporkan kendala ke instruktur.
- Training model besar memakan resource — jangan jalankan job berat di
  komputer lab tanpa izin instruktur.

## Langkah-Langkah Praktikum

1. **Setup environment:**

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python src/cek_env.py
   ```

2. **Lengkapi `src/main.py`:**

   | Fungsi | Deskripsi |
   |--------|-----------|
   | `buat_matriks(baris, kolom, nilai)` | ndarray konstanta |
   | `kali_matriks_manual(A, B)` | perkalian matriks dengan loop (tanpa `@`/`np.dot`) |
   | `kali_matriks_numpy(A, B)` | perkalian dengan `np.dot` |
   | `normalisasi_minmax(x)` | skala array ke rentang [0, 1] |
   | `relu(x)` | fungsi aktivasi `max(0, x)` element-wise |

3. Verifikasi hasil manual vs NumPy **identik**, lalu ukur kecepatan:

   ```bash
   python src/main.py
   pytest -q
   ```

4. **Eksplorasi** — di notebook, bandingkan waktu `kali_matriks_manual` vs
   `kali_matriks_numpy` untuk ukuran matriks 10, 50, 100. Plot hasilnya.

## Lembar Kerja / Hasil Pengamatan

| Ukuran matriks | Waktu manual (ms) | Waktu NumPy (ms) | Speedup |
|----------------|-------------------|------------------|---------|
| 10×10 | ... | ... | ... |
| 50×50 | ... | ... | ... |
| 100×100 | ... | ... | ... |

## Tugas / Latihan Praktikum

1. Implementasikan `sigmoid(x)` = $1/(1+e^{-x})$ dan plot kurvanya untuk
   $x \in [-10, 10]$.
2. Jelaskan di `SUBMISSION.md` mengapa ReLU dan sigmoid disebut fungsi
   aktivasi dan apa perannya di neural network.

## Format Pelaporan

Isi `SUBMISSION.md` lalu push ke `main`. Autograding berjalan otomatis.
