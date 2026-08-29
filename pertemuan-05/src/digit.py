"""Pertemuan 5 — Softmax, cross-entropy, dan MLP untuk klasifikasi digit."""
import time
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax stabil numerik: exp(z - max(z)) / sum(...)."""
    # TODO: implementasikan
    z_stabil = z - np.max(z)
    exp_z = np.exp(z_stabil)
    return exp_z / np.sum(exp_z)
    # raise NotImplementedError


def cross_entropy(prob: np.ndarray, label: int) -> float:
    """-log(prob[label]); clip prob minimal 1e-12 agar tak log(0)."""
    # TODO: implementasikan (np.clip)
    prob_aman = np.clip(prob, 1e-12, 1.0)
    return float(-np.log(prob_aman[label]))
    # raise NotImplementedError


def siapkan_digits():
    """Load digits, normalisasi piksel /16.0, split 75/25 (random_state=42).

    Return (X_train, X_test, y_train, y_test).
    """
    # TODO: digits = load_digits(); X = digits.data / 16.0
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    ukuran_uji = round(len(X) * 0.25)
    return train_test_split(X, y, test_size=ukuran_uji, random_state=42, stratify=y)
    # raise NotImplementedError


def latih_mlp(X_train, y_train) -> MLPClassifier:
    """MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)."""
    # TODO: implementasikan
    model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model
    # raise NotImplementedError


def evaluasi(model, X_test, y_test) -> float:
    """Akurasi pada data uji."""
    # TODO: implementasikan
    return float(model.score(X_test, y_test))
    # raise NotImplementedError


if __name__ == "__main__":
    z = np.array([2.0, 1.0, 0.1])
    p = softmax(z)
    print(f"softmax({z}) = {p.round(3)} (jumlah={p.sum():.3f})")
    print(f"cross_entropy benar : {cross_entropy(p, 0):.3f}")
    print(f"cross_entropy salah : {cross_entropy(p, 2):.3f}")

    X_train, X_test, y_train, y_test = siapkan_digits()
    model = latih_mlp(X_train, y_train)
    print(f"Akurasi test: {evaluasi(model, X_test, y_test):.2%}")

    print("\nTabel eksperimen hidden layer:")
    print(f"{'Hidden layer':<14s} {'Akurasi test':>12s} {'Waktu latih (s)':>18s}")
    print("-" * 48)
    for hidden_layer in [(16,), (32,), (64, 32)]:
        mulai = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer,
                max_iter=500,
                random_state=42,
            )
            model.fit(X_train, y_train)
        durasi = time.perf_counter() - mulai
        print(f"{str(hidden_layer):<14s} {evaluasi(model, X_test, y_test):>12.2%} {durasi:>18.3f}")
