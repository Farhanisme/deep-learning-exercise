# Submission Pertemuan 02

- **NIM**    : 105841110523
- **Nama**   : Muhammad Zaky Farhan
- **Kelas**  : 6AI-B

## Lembar Kerja

(angka berasal dari output resmi `src/regresi.py`; notebook memakai fungsi `latih()` yang sama untuk bukti plot/tabel)
| Learning rate | Loss akhir | Konvergen? | w, b akhir |
|---------------|-----------|------------|------------|
| 0.001 | 0.813787 | Ya | w = 2.7686, b = 1.2183 |
| 0.01 | 0.086298 | Ya | w = 3.0038, b = 1.9960 |
| 0.1 | 0.086298 | Ya | w = 3.0038, b = 1.9961 |
| 1.0 | nan | Tidak | w = nan, b = nan |

## Bukti pengerjaan

(screenshot output + plot kurva loss)

## Jawaban refleksi

1. Apa yang terjadi saat learning rate terlalu besar? Jelaskan dengan kurva loss.
2. Berapa epoch dihemat oleh early stopping-mu?
3. Mengapa gradien dihitung dari turunan loss, bukan dari loss-nya langsung?

### Jawaban

1. Saat learning rate terlalu besar, update parameter meloncat terlalu jauh sehingga loss tidak turun stabil dan bisa divergen. Pada eksperimen, `lr=1.0` menghasilkan loss `nan`.
2. Dengan `lr=0.05` dan batas maksimal 500 epoch, early stopping berhenti pada epoch ke-101 sehingga menghemat 399 epoch.
3. Gradien dihitung dari turunan loss karena turunan menunjukkan arah dan besar perubahan parameter yang membuat loss naik atau turun.

Data `y = -2x + 5` juga berhasil dipelajari dengan hasil `w = -2.0000`, `b = 4.9999`, dan loss akhir sangat kecil.
