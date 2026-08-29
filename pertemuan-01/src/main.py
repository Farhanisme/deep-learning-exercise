"""Pertemuan 1 — Operasi tensor dasar dengan NumPy."""
import numpy as np


def buat_matriks(baris: int, kolom: int, nilai: float) -> np.ndarray:
    """Matriks (baris x kolom) berisi konstanta `nilai`."""
    # TODO: gunakan np.full
    return np.full((baris, kolom), nilai)
    # raise NotImplementedError


def kali_matriks_manual(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Perkalian matriks dengan tiga loop bersarang (TANPA np.dot / @)."""
    # TODO: C[i][j] = sum(A[i][k] * B[k][j] for k in ...)
    if A.shape[1] != B.shape[0]:
        raise ValueError("Jumlah kolom A harus sama dengan jumlah baris B")

    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            total = 0.0
            for k in range(A.shape[1]):
                total += A[i, k] * B[k, j]
            C[i, j] = total
    return C
    # raise NotImplementedError


def kali_matriks_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Perkalian matriks dengan np.dot."""
    # TODO: implementasikan
    return np.dot(A, B)
    # raise NotImplementedError


def normalisasi_minmax(x: np.ndarray) -> np.ndarray:
    """Skala nilai array ke [0, 1]: (x - min) / (max - min)."""
    # TODO: implementasikan
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        return np.zeros_like(x, dtype=float)
    return (x - x_min) / (x_max - x_min)
    # raise NotImplementedError


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU element-wise: max(0, x)."""
    # TODO: gunakan np.maximum
    return np.maximum(0, x)
    # raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    A = rng.random((4, 3))
    B = rng.random((3, 5))

    manual = kali_matriks_manual(A, B)
    cepat = kali_matriks_numpy(A, B)
    print("Hasil identik:", np.allclose(manual, cepat))

    x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    print("relu    :", relu(x))
    print("minmax  :", normalisasi_minmax(x))
