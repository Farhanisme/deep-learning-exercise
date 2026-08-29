"""Pertemuan 7 — Sel RNN sederhana & demo vanishing gradient (NumPy)."""
import numpy as np


def buat_dataset_window(deret: np.ndarray, w: int):
    """Sliding window: X[i] = deret[i:i+w], y[i] = deret[i+w].

    Return (X, y) dengan X.shape == (n-w, w).
    """
    # TODO: implementasikan
    X = []
    y = []
    for i in range(len(deret) - w):
        X.append(deret[i:i + w])
        y.append(deret[i + w])
    return np.array(X), np.array(y)
    # raise NotImplementedError


def rnn_cell(x_t: float, h_prev: np.ndarray, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Satu langkah RNN: h_t = tanh(Wx * x_t + Wh @ h_prev + b).

    x_t skalar, h_prev/Wx/b vektor (n_hidden,), Wh matriks (n_hidden, n_hidden).
    """
    # TODO: implementasikan
    return np.tanh(Wx * x_t + Wh @ h_prev + b)
    # raise NotImplementedError


def rnn_forward(deret: np.ndarray, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Proses seluruh sekuens dari h0 = nol.

    Return array semua hidden state, shape (len(deret), n_hidden).
    """
    # TODO: loop rnn_cell atas deret
    h = np.zeros_like(Wx, dtype=float)
    semua_h = []
    for x_t in deret:
        h = rnn_cell(float(x_t), h, Wx, Wh, b)
        semua_h.append(h.copy())
    return np.array(semua_h)
    # raise NotImplementedError


def demo_vanishing(skala_wh: float, langkah: int = 50) -> float:
    """Simulasi menyusut/meledaknya gradien.

    Mulai g = 1.0; ulangi `langkah` kali: g = g * skala_wh * 0.65
    (0.65 ~ nilai tipikal turunan tanh). Return g akhir.
    """
    # TODO: implementasikan
    g = 1.0
    for _ in range(langkah):
        g = g * skala_wh * 0.65
    return float(g)
    # raise NotImplementedError


if __name__ == "__main__":
    t = np.linspace(0, 4 * np.pi, 100)
    sinus = np.sin(t)
    X, y = buat_dataset_window(sinus, w=5)
    print(f"Dataset window: X {X.shape}, y {y.shape}")

    rng = np.random.default_rng(0)
    n_hidden = 4
    Wx = rng.normal(0, 0.5, n_hidden)
    Wh = rng.normal(0, 0.5, (n_hidden, n_hidden))
    b = np.zeros(n_hidden)
    H = rnn_forward(sinus[:20], Wx, Wh, b)
    print(f"Hidden states: {H.shape}, semua di [-1, 1]: {np.all(np.abs(H) <= 1)}")

    print("\nDemo vanishing/exploding gradient (50 langkah):")
    for s in [0.5, 1.0, 1.5]:
        print(f"  |Wh|={s}: g = {demo_vanishing(s):.3e}")
