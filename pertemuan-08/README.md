# Pertemuan 8 — Mini Project Deep Learning

> **Tujuan**: Mahasiswa mampu merancang, mengimplementasikan, dan melaporkan
> proyek deep learning ujung-ke-ujung — dari perumusan masalah, persiapan
> data, pelatihan model, hingga evaluasi dan analisis kesalahan.

## Ringkasan

Pertemuan terakhir menggabungkan seluruh materi (P1–P7) dalam satu proyek
mandiri. Kamu memilih satu topik, menulis proposal singkat, membangun model,
dan menulis laporan akhir. Proyek dinilai dari proses dan analisis — bukan
sekadar angka akurasi.

## Pilihan Topik (pilih satu, atau usulkan sendiri ke instruktur)

1. **Klasifikasi citra** — digit/fashion (load_digits, Fashion-MNIST via
   Keras/Colab); bandingkan MLP vs CNN.
2. **Prediksi deret waktu** — suhu/harga/sinyal sintetis; bandingkan regresi
   linear (P2) vs RNN/LSTM (P7).
3. **Klasifikasi tabular** — dataset UCI/Kaggle pilihanmu; MLP vs baseline
   sederhana (regresi logistik).
4. **Studi eksperimen** — pengaruh hyperparameter (learning rate, arsitektur,
   epoch) pada satu masalah; laporkan sebagai eksperimen sistematis.

## Ketentuan

- Model **baseline sederhana wajib ada** sebagai pembanding.
- Data dibagi latih/uji dengan benar; tidak ada kebocoran data.
- Evaluasi memakai metrik yang tepat (akurasi/F1/MSE sesuai masalah)
  + analisis kesalahan (contoh prediksi salah, dugaan penyebab).
- Boleh melatih di Google Colab; simpan hasil (plot, metrik) di repo.

## Langkah-Langkah Praktikum

1. Isi `PROPOSAL.md` — kumpulkan di pertemuan ini juga (paling lambat
   pertengahan sesi) untuk disetujui instruktur.
2. Implementasikan di `src/proyek.py` (fungsi terstruktur, bukan satu skrip
   panjang) + eksplorasi di `notebooks/eksplorasi.ipynb`.
3. Tulis `LAPORAN.md` mengikuti template yang tersedia.
4. Pastikan `pytest -q` lulus (tes struktural) dan push ke `main`.

## Format Pelaporan

- `PROPOSAL.md` terisi lengkap
- `LAPORAN.md` terisi lengkap
- `src/proyek.py` berjalan tanpa error
- `SUBMISSION.md` terisi
