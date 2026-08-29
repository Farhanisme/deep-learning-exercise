# Submission Pertemuan 04

- **NIM**    : 105841110523
- **Nama**   : Muhammad Zaky Farhan
- **Kelas**  : 6AI-B

## Lembar Kerja

| Implementasi | Epoch sampai 4/4 benar | Baris kode inti |
|--------------|------------------------|------------------|
| NumPy manual | 712 | sekitar 20 baris kode inti |
| Keras | dapat diselesaikan dengan model 2-2-1 yang sama | sekitar 6 baris kode inti |

## Bukti pengerjaan

(screenshot output + pytest + hasil Keras di notebook/Colab)

## Jawaban refleksi

1. Jelaskan dengan kata-katamu sendiri apa yang dihitung `backward()`.
2. Efek jumlah neuron hidden 2 → 4 terhadap kecepatan konvergensi: hidden 4 lebih cepat pada eksperimen ini.
3. Apa pengaruh seed/inisialisasi bobot terhadap hasil training? Mengapa
   inisialisasi nol total akan gagal?

### Jawaban

1. `backward()` menghitung seberapa besar setiap bobot dan bias berkontribusi terhadap error. Nilai gradien itu dipakai untuk mengubah parameter agar loss turun.
2. Pada eksperimen seed 7, hidden 2 konvergen pada epoch 712, sedangkan hidden 4 pada epoch 551. Jadi hidden 4 lebih cepat karena kapasitas model lebih besar.
3. Seed menentukan nilai awal bobot, sehingga jalur training dan kecepatan konvergensi bisa berbeda. Pada seed 3, 7, dan 11 model tetap konvergen, tetapi loss akhir berbeda. Inisialisasi nol total gagal karena neuron hidden menjadi identik dan belajar pola yang sama.
