# Submission Pertemuan 07

- **NIM**    : 105841110523
- **Nama**   : Muhammad Zaky Farhan
- **Kelas**  : 6AI-B

## Lembar Kerja

| |Wh| | Norma "gradien" setelah 50 langkah | Kesimpulan |
|-------|--------------------------------------|------------|
| 0.5 | 3.928e-25 | Vanishing sangat kuat |
| 1.0 | 4.422e-10 | Vanishing |
| 1.5 | 2.820e-01 | Menyusut pelan, masih lebih stabil |

## Bukti pengerjaan

(screenshot output + pytest + plot hidden state / vanishing gradient)

## Jawaban refleksi

1. Apa isi hidden state $h_t$ secara intuitif?
2. Hubungan |Wh| dengan vanishing/exploding gradient: semakin besar faktor `|Wh|`, gradien makin lambat menyusut dan bisa meledak jika terlalu besar.
3. Tiga gerbang LSTM dan fungsinya:
   - Forget gate: membuang informasi lama yang tidak dibutuhkan.
   - Input gate: menyimpan informasi baru yang penting.
   - Output gate: mengatur informasi yang keluar sebagai hidden state.

### Jawaban

1. Hidden state `h_t` berisi ringkasan informasi dari input saat ini dan input sebelumnya. Secara intuitif, hidden state adalah memori sementara RNN.
2. Semakin kecil `|Wh|`, gradien makin cepat menyusut sehingga terjadi vanishing gradient. Jika faktor perkalian terlalu besar, gradien bisa membesar terus dan menjadi exploding gradient.
3. Tiga gerbang LSTM dan fungsinya:
   - Forget gate: menentukan informasi lama mana yang dibuang.
   - Input gate: menentukan informasi baru mana yang disimpan.
   - Output gate: menentukan bagian memori yang dikeluarkan sebagai hidden state.
