# Submission Pertemuan 08

- **NIM**    : 105841110523
- **Nama**   : Muhammad Zaky Farhan
- **Kelas**  : 6AI-B

## Ringkasan Proyek

- Judul: Klasifikasi Fashion-MNIST Menggunakan Baseline Linear, MLP, dan CNN
- Topik yang dipilih: Klasifikasi citra pakaian Fashion-MNIST
- Hasil metrik utama: baseline = 82.33%, MLP = 83.80%, CNN = 87.37%
- Model terbaik: CNN dengan F1 macro = 87.27%

## Checklist

- [ ] PROPOSAL.md disetujui instruktur
- [x] src/proyek.py berjalan tanpa error
- [x] LAPORAN.md lengkap (termasuk analisis kesalahan)
- [x] Plot/bukti hasil tersedia di notebook eksplorasi

## Refleksi Singkat

1. Bagian tersulit dari proyek ini: menjaga perbandingan baseline, MLP, dan CNN tetap adil karena bentuk input model berbeda. Solusinya, preprocessing dibuat satu pintu di `src/proyek.py`, lalu baseline/MLP memakai versi flattened dan CNN memakai citra 2D.
2. Jika punya waktu 1 minggu lagi, apa yang akan kamu perbaiki? Saya akan memakai seluruh data Fashion-MNIST, menambah augmentasi data, mencoba early stopping, dan tuning arsitektur CNN agar performanya lebih stabil.
