"""Pertemuan 6 — Konvolusi 2D, ReLU, dan max pooling dari nol (NumPy)."""
import numpy as np

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_Y = SOBEL_X.T
BLUR = np.ones((3, 3)) / 9.0
SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)


def konvolusi2d(citra: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Konvolusi 2D mode 'valid' (tanpa padding, stride 1).

    Output shape: (H-k+1, W-k+1).
    Petunjuk: dua loop for atas posisi output; tiap sel =
    np.sum(citra[i:i+k, j:j+k] * kernel).
    """
    # TODO: implementasikan
    h, w = citra.shape
    k_h, k_w = kernel.shape
    out = np.zeros((h - k_h + 1, w - k_w + 1), dtype=float)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            area = citra[i:i + k_h, j:j + k_w]
            out[i, j] = np.sum(area * kernel)

    return out
    # raise NotImplementedError


def relu(x: np.ndarray) -> np.ndarray:
    """max(0, x) element-wise."""
    # TODO: implementasikan
    return np.maximum(0, x)
    # raise NotImplementedError


def max_pooling(citra: np.ndarray, ukuran: int = 2) -> np.ndarray:
    """Max pooling non-overlapping. Sisa baris/kolom yang tak habis dibagi
    `ukuran` boleh dibuang (floor division)."""
    # TODO: implementasikan
    h_out = citra.shape[0] // ukuran
    w_out = citra.shape[1] // ukuran
    out = np.zeros((h_out, w_out), dtype=float)

    for i in range(h_out):
        for j in range(w_out):
            blok = citra[i * ukuran:(i + 1) * ukuran, j * ukuran:(j + 1) * ukuran]
            out[i, j] = np.max(blok)

    return out
    # raise NotImplementedError


def hitung_ukuran_output(h: int, w: int, k: int, pool: int) -> tuple:
    """Ukuran setelah konvolusi valid kxk lalu pooling: ((h-k+1)//pool, (w-k+1)//pool)."""
    # TODO: implementasikan
    return ((h - k + 1) // pool, (w - k + 1) // pool)
    # raise NotImplementedError


if __name__ == "__main__":
    # Citra mainan: garis vertikal terang di tengah
    citra = np.zeros((8, 8))
    citra[:, 4] = 1.0

    tepi = konvolusi2d(citra, SOBEL_X)
    print("Deteksi tepi (Sobel X):")
    print(np.round(tepi, 1))

    hasil = max_pooling(relu(tepi))
    print(f"\nSetelah ReLU + pool 2x2: shape {hasil.shape}")
    print(np.round(hasil, 1))
    print(f"\nUkuran output 28x28, k=3, pool=2: {hitung_ukuran_output(28, 28, 3, 2)}")
