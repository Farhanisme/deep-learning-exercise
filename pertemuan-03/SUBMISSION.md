# Submission Pertemuan 03

- **NIM**    : 105841110523
- **Nama**   : Muhammad Zaky Farhan
- **Kelas**  : 6AI-B

## Lembar Kerja

| Gerbang | Akurasi perceptron | Epoch sampai konvergen |
|---------|--------------------|-----------------------|
| AND | 100% | 3 |
| OR | 100% | 3 |
| NAND | 100% | 5 |
| XOR | 50% | tidak konvergen sampai 50 epoch |

## Bukti pengerjaan

(screenshot output + pytest + visualisasi decision boundary)

## Jawaban refleksi

1. Mengapa XOR tidak bisa diselesaikan perceptron tunggal? (sertakan sketsa)
2. Apa peran hidden layer pada MLP sehingga XOR terselesaikan?
3. Bobot akhir perceptron NAND-mu: w = `[-0.2, -0.1]`, b = `0.2`

### Jawaban

1. XOR tidak bisa diselesaikan perceptron tunggal karena titik kelas 1 berada di dua sudut berlawanan, sedangkan titik kelas 0 juga berada di dua sudut berlawanan. Satu garis lurus tidak bisa memisahkan pola seperti ini.
2. Hidden layer membuat MLP bisa membentuk batas keputusan non-linear. Dengan beberapa neuron tersembunyi, model dapat menggabungkan beberapa garis keputusan untuk menyelesaikan XOR.
3. Bobot akhir perceptron NAND: `w = [-0.2, -0.1]`, `b = 0.2`.
