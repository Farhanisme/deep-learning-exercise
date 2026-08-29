# Submission Pertemuan 05

- **NIM**    : 105841110523
- **Nama**   : Muhammad Zaky Farhan
- **Kelas**  : 6AI-B

## Lembar Kerja

| Hidden layer | Akurasi test | Waktu latih (s) |
|--------------|--------------|------------------|
| (16,) | 97.77% | 1.954 |
| (32,) | 97.55% | 1.735 |
| (64, 32) | 98.66% | 1.602 |

## Bukti pengerjaan

(screenshot output + pytest + citra yang salah klasifikasi)

## Jawaban refleksi

1. Perhitungan jumlah parameter MLP 64→32→10: 2410 parameter.
2. Dari citra yang salah diklasifikasi, apakah kesalahan model "masuk akal"?
   Beri 2 contoh.
3. Mengapa softmax perlu trik pengurangan max? Apa yang terjadi tanpa itu?

### Jawaban

1. Jumlah parameter MLP `64 -> 32 -> 10` adalah `(64 x 32 + 32) + (32 x 10 + 10) = 2080 + 330 = 2410` parameter.
2. Kesalahan model masih masuk akal karena citra digit 8 bisa terlihat mirip 1 jika bentuknya tipis, dan digit 5 bisa terlihat seperti 9 jika lengkungan atas/bawahnya kurang jelas. Contoh hasil salah: label asli 8 diprediksi 1, label asli 5 diprediksi 9.
3. Softmax perlu pengurangan nilai maksimum agar nilai eksponensial tidak terlalu besar. Tanpa trik ini, input besar seperti 1000 dapat menyebabkan overflow dan probabilitas menjadi tidak valid.
