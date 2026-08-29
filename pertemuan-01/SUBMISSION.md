# Submission Pertemuan 01

- **NIM**    : 105841110523
- **Nama**   : Muhammad Zaky Farhan
- **Kelas**  : 6AI-B

## Lembar Kerja

| Ukuran matriks | Waktu manual (ms) | Waktu NumPy (ms) | Speedup |
|----------------|-------------------|------------------|---------|
| 10×10 | 0.3 | 0.016 | 16x |
| 50×50 | 31.0 | 0.062 | 499x |
| 100×100 | 244.2 | 0.146 | 1672x |

## Bukti pengerjaan

(screenshot output + pytest + plot sigmoid)

## Jawaban refleksi

1. Mengapa NumPy jauh lebih cepat dari loop Python murni?
2. Apa peran fungsi aktivasi di neural network? Apa yang terjadi tanpa aktivasi?
3. Status TensorFlow di komputermu (terpasang/tidak): terpasang.

### Jawaban

1. NumPy lebih cepat karena operasi array dijalankan oleh kode low-level yang sudah dioptimasi, sedangkan loop Python menjalankan instruksi satu per satu dengan overhead lebih besar.
2. Fungsi aktivasi memberi unsur non-linear pada neural network. Tanpa aktivasi, banyak layer hanya menjadi gabungan operasi linear sehingga model sulit mempelajari pola kompleks.
3. Status TensorFlow di komputer: terpasang.
