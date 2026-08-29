# Submission Pertemuan 06

- **NIM**    : 105841110523
- **Nama**   : Muhammad Zaky Farhan
- **Kelas**  : 6AI-B

## Lembar Kerja

| Kernel | Efek visual yang diamati |
|--------|--------------------------|
| Sobel X | Menonjolkan tepi vertikal, terutama perubahan gelap-terang dari kiri ke kanan. |
| Sobel Y | Menonjolkan tepi horizontal, terutama perubahan gelap-terang dari atas ke bawah. |
| Blur 3×3 | Menghaluskan citra karena setiap piksel diganti rata-rata area sekitar. |
| Sharpen | Mempertegas detail dan tepi citra. |
| Kernel buatanku | Kernel emboss `[[ -2, -1, 0], [-1, 1, 1], [0, 1, 2 ]]` memberi efek timbul dengan menonjolkan arah diagonal. |

## Bukti pengerjaan

(screenshot hasil konvolusi + pytest)

## Jawaban refleksi

1. Perhitungan ukuran feature map 28×28 → Conv3×3(8) → Pool2 → Conv3×3(16) → Pool2:
   `28x28 -> 26x26x8 -> 13x13x8 -> 11x11x16 -> 5x5x16`.
2. Mengapa CNN lebih hemat parameter dibanding MLP untuk citra?
3. Apa fungsi max pooling selain memperkecil ukuran?

### Jawaban

1. Ukuran feature map: input `28x28`; setelah Conv `3x3` valid menjadi `26x26x8`; setelah Pool `2x2` menjadi `13x13x8`; setelah Conv `3x3` valid dengan 16 filter menjadi `11x11x16`; setelah Pool `2x2` menjadi `5x5x16`.
2. CNN lebih hemat parameter karena kernel yang sama dipakai berulang di banyak lokasi citra. MLP membutuhkan bobot terpisah untuk hampir semua hubungan piksel ke neuron.
3. Max pooling juga membantu mengambil fitur paling kuat pada suatu area dan membuat model lebih tahan terhadap pergeseran kecil pada citra.
