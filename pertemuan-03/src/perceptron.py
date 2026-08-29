"""Pertemuan 3 — Perceptron & forward pass MLP (NumPy saja)."""
import numpy as np


def step(z):
    """Step function: 1 jika z >= 0, selain itu 0 (dukungan skalar & array)."""
    # TODO: gunakan np.where(z >= 0, 1, 0)
    return np.where(z >= 0, 1, 0)
    # raise NotImplementedError


def sigmoid(z):
    """Sigmoid: 1 / (1 + exp(-z))."""
    # TODO: implementasikan
    return 1 / (1 + np.exp(-z))
    # raise NotImplementedError


class Perceptron:
    def __init__(self, n_input: int):
        self.w = np.zeros(n_input)
        self.b = 0.0

    def prediksi(self, x: np.ndarray) -> int:
        """step(w . x + b) untuk satu sampel."""
        # TODO: implementasikan
        return int(step(np.dot(self.w, x) + self.b))
        # raise NotImplementedError

    def latih(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 20) -> int:
        """Aturan belajar perceptron per-sampel.

        Untuk tiap epoch dan tiap sampel:
            err = y_i - prediksi(x_i)
            w += lr * err * x_i ;  b += lr * err
        Return: nomor epoch saat konvergen (semua benar), atau `epochs` jika tidak.
        """
        # TODO: implementasikan
        for epoch in range(1, epochs + 1):
            for x_i, y_i in zip(X, y):
                y_pred = self.prediksi(x_i)
                err = y_i - y_pred
                self.w += lr * err * x_i
                self.b += lr * err

            benar = [self.prediksi(x_i) == y_i for x_i, y_i in zip(X, y)]
            if all(benar):
                return epoch

        return epochs
        # raise NotImplementedError


def forward_mlp(x, W1, b1, W2, b2):
    """Forward pass MLP 1 hidden layer, aktivasi sigmoid di kedua layer.

    h = sigmoid(W1 @ x + b1);  y = sigmoid(W2 @ h + b2)
    Return skalar float (output neuron tunggal).
    """
    # TODO: implementasikan
    h = sigmoid(W1 @ x + b1)
    y = sigmoid(W2 @ h + b2)
    return float(y)
    # raise NotImplementedError


# Bobot MLP siap-pakai yang menyelesaikan XOR (untuk eksplorasi)
XOR_W1 = np.array([[20.0, 20.0], [-20.0, -20.0]])
XOR_B1 = np.array([-10.0, 30.0])
XOR_W2 = np.array([20.0, 20.0])
XOR_B2 = -30.0


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    gerbang = {
        "AND": np.array([0, 0, 0, 1]),
        "OR": np.array([0, 1, 1, 1]),
        "NAND": np.array([1, 1, 1, 0]),
        "XOR": np.array([0, 1, 1, 0]),
    }

    print("Tabel perceptron gerbang logika:")
    print(f"{'Gerbang':<8s} {'Akurasi':>8s} {'Epoch sampai konvergen':>28s}")
    print("-" * 48)
    for nama, y in gerbang.items():
        p = Perceptron(2)
        ep = p.latih(X, y, epochs=50)
        akurasi = np.mean([p.prediksi(x) == t for x, t in zip(X, y)])
        epoch_teks = str(ep) if akurasi == 1.0 else f"tidak konvergen sampai {ep} epoch"
        print(f"{nama:<8s} {akurasi:>8.0%} {epoch_teks:>28s}")
        if nama == "NAND":
            print(f"Bobot akhir NAND: w = {p.w.round(1).tolist()}, b = {p.b:.1f}")

    print("\nMLP untuk XOR:")
    for x, t in zip(X, gerbang["XOR"]):
        out = forward_mlp(x, XOR_W1, XOR_B1, XOR_W2, XOR_B2)
        print(f"  {x} -> {out:.3f} (target {t})")
