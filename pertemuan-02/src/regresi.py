"""Pertemuan 2 — Regresi linear + gradient descent dari nol (NumPy saja)."""
import numpy as np


def buat_data(n: int = 100, seed: int = 42):
    """Dataset sintetis y = 3x + 2 + noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, n)
    y = 3.0 * X + 2.0 + rng.normal(0, 0.3, n)
    return X, y


def prediksi(X: np.ndarray, w: float, b: float) -> np.ndarray:
    """y_hat = w*X + b."""
    # TODO: implementasikan
    return w * X + b
    # raise NotImplementedError


def mse(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error."""
    # TODO: implementasikan
    return float(np.mean((y_pred - y) ** 2))
    # raise NotImplementedError


def hitung_gradien(X: np.ndarray, y: np.ndarray, w: float, b: float):
    """Return (dw, db).

    dw = (2/n) * sum((y_hat - y) * X)
    db = (2/n) * sum(y_hat - y)
    """
    # TODO: implementasikan
    y_hat = prediksi(X, w, b)
    error = y_hat - y
    n = len(X)
    dw = (2 / n) * np.sum(error * X)
    db = (2 / n) * np.sum(error)
    return float(dw), float(db)
    # raise NotImplementedError


def latih(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 500):
    """Gradient descent. Mulai dari w=0, b=0.

    Return (w, b, riwayat_loss) — riwayat_loss berisi loss tiap epoch.
    """
    # TODO: loop epochs kali: hitung gradien, update w & b, catat loss
    w = 0.0
    b = 0.0
    riwayat_loss = []

    for _ in range(epochs):
        y_hat = prediksi(X, w, b)
        loss = mse(y_hat, y)
        riwayat_loss.append(loss)

        dw, db = hitung_gradien(X, y, w, b)
        w -= lr * dw
        b -= lr * db

        if len(riwayat_loss) > 1 and abs(riwayat_loss[-2] - riwayat_loss[-1]) < 1e-9:
            break

    return w, b, riwayat_loss
    # raise NotImplementedError


if __name__ == "__main__":
    X, y = buat_data()
    w, b, riwayat = latih(X, y, lr=0.05, epochs=300)
    print(f"w={w:.3f} (target 3.0) | b={b:.3f} (target 2.0)")
    print(f"loss awal={riwayat[0]:.4f} -> akhir={riwayat[-1]:.4f}")

    print("\nTabel eksperimen learning rate:")
    print(f"{'Learning rate':>13s} {'Loss akhir':>12s} {'Konvergen?':>11s} {'w akhir':>10s} {'b akhir':>10s}")
    print("-" * 62)
    for lr in [0.001, 0.01, 0.1, 1.0]:
        with np.errstate(over="ignore", invalid="ignore"):
            w_lr, b_lr, riwayat_lr = latih(X, y, lr=lr, epochs=500)
        loss_akhir = riwayat_lr[-1]
        konvergen = np.isfinite(loss_akhir)
        loss_teks = "nan" if not konvergen else f"{loss_akhir:.6f}"
        w_teks = "nan" if not np.isfinite(w_lr) else f"{w_lr:.4f}"
        b_teks = "nan" if not np.isfinite(b_lr) else f"{b_lr:.4f}"
        print(f"{lr:13.3g} {loss_teks:>12s} {('Ya' if konvergen else 'Tidak'):>11s} {w_teks:>10s} {b_teks:>10s}")
