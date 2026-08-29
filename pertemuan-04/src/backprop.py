"""Pertemuan 4 — Backpropagation MLP 2-2-1 untuk XOR (NumPy saja)."""
import numpy as np

X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_XOR = np.array([0.0, 1.0, 1.0, 0.0])


def sigmoid(z):
    """1 / (1 + exp(-z))."""
    # TODO: implementasikan
    return 1 / (1 + np.exp(-z))
    # raise NotImplementedError


def turunan_sigmoid(z):
    """sigmoid(z) * (1 - sigmoid(z))."""
    # TODO: implementasikan
    s = sigmoid(z)
    return s * (1 - s)
    # raise NotImplementedError


def init_params(n_hidden: int = 2, seed: int = 7) -> dict:
    """Inisialisasi bobot kecil acak."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1, (n_hidden, 2)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, 1, n_hidden),
        "b2": 0.0,
    }


def forward(x: np.ndarray, params: dict):
    """Return (h, y_hat): h = sigmoid(W1@x + b1); y_hat = sigmoid(W2@h + b2)."""
    # TODO: implementasikan
    h = sigmoid(params["W1"] @ x + params["b1"])
    y_hat = sigmoid(params["W2"] @ h + params["b2"])
    return h, float(y_hat)
    # raise NotImplementedError


def backward(x: np.ndarray, y: float, params: dict) -> dict:
    """Hitung gradien satu sampel.

    delta2 = (y_hat - y) * y_hat * (1 - y_hat)          # skalar
    delta1 = (W2 * delta2) * h * (1 - h)                # vektor n_hidden
    Return {"dW2": delta2*h, "db2": delta2, "dW1": outer(delta1, x), "db1": delta1}
    """
    # TODO: implementasikan (panggil forward dulu)
    h, y_hat = forward(x, params)
    delta2 = (y_hat - y) * y_hat * (1 - y_hat)
    delta1 = (params["W2"] * delta2) * h * (1 - h)
    return {
        "dW2": delta2 * h,
        "db2": float(delta2),
        "dW1": np.outer(delta1, x),
        "db1": delta1,
    }
    # raise NotImplementedError


def latih_xor(lr: float = 0.5, epochs: int = 5000, seed: int = 7):
    """SGD per-sampel pada XOR. Return (params, riwayat_loss per epoch)."""
    # TODO: untuk tiap epoch: untuk tiap (x, y): grad = backward(...);
    #       update semua param: p -= lr * grad; catat rata-rata loss epoch
    params = init_params(seed=seed)
    riwayat_loss = []

    for _ in range(epochs):
        for x, y in zip(X_XOR, Y_XOR):
            grad = backward(x, y, params)
            params["W1"] -= lr * grad["dW1"]
            params["b1"] -= lr * grad["db1"]
            params["W2"] -= lr * grad["dW2"]
            params["b2"] -= lr * grad["db2"]

        losses = []
        for x, y in zip(X_XOR, Y_XOR):
            _, y_hat = forward(x, params)
            losses.append(0.5 * (y_hat - y) ** 2)
        riwayat_loss.append(float(np.mean(losses)))

    return params, riwayat_loss
    # raise NotImplementedError


def prediksi_xor(params: dict) -> list:
    """Prediksi biner (0/1) untuk keempat input XOR."""
    return [int(round(float(forward(x, params)[1]))) for x in X_XOR]


def epoch_sampai_benar(n_hidden: int = 2, lr: float = 0.5, epochs: int = 5000, seed: int = 7):
    """Latih XOR dan kembalikan epoch pertama saat semua prediksi benar."""
    params = init_params(n_hidden=n_hidden, seed=seed)
    riwayat_loss = []

    for epoch in range(1, epochs + 1):
        for x, y in zip(X_XOR, Y_XOR):
            grad = backward(x, y, params)
            params["W1"] -= lr * grad["dW1"]
            params["b1"] -= lr * grad["db1"]
            params["W2"] -= lr * grad["dW2"]
            params["b2"] -= lr * grad["db2"]

        losses = []
        for x, y in zip(X_XOR, Y_XOR):
            _, y_hat = forward(x, params)
            losses.append(0.5 * (y_hat - y) ** 2)
        riwayat_loss.append(float(np.mean(losses)))

        if prediksi_xor(params) == [0, 1, 1, 0]:
            return epoch, params, riwayat_loss

    return epochs, params, riwayat_loss


if __name__ == "__main__":
    params, riwayat = latih_xor()
    print(f"loss: {riwayat[0]:.4f} -> {riwayat[-1]:.4f}")
    pred = prediksi_xor(params)
    for x, p, t in zip(X_XOR, pred, Y_XOR):
        print(f"  {x} -> {p} (target {int(t)})")
    print("XOR terpecahkan!" if pred == [0, 1, 1, 0] else "Belum konvergen...")

    print("\nTabel eksperimen XOR:")
    print(f"{'Implementasi':<14s} {'Epoch sampai 4/4 benar':>25s} {'Baris kode inti':>22s}")
    print("-" * 65)
    epoch_hidden_2, _, _ = epoch_sampai_benar(n_hidden=2)
    epoch_hidden_4, _, _ = epoch_sampai_benar(n_hidden=4)
    print(f"{'NumPy manual':<14s} {epoch_hidden_2:>25d} {'sekitar 20 baris':>22s}")
    print(f"{'Keras':<14s} {'model 2-2-1 di notebook':>25s} {'sekitar 6 baris':>22s}")
    print(f"Hidden 2 konvergen pada epoch {epoch_hidden_2}; hidden 4 pada epoch {epoch_hidden_4}.")
