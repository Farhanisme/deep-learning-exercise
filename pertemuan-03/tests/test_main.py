import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perceptron import (
    XOR_B1,
    XOR_B2,
    XOR_W1,
    XOR_W2,
    Perceptron,
    forward_mlp,
    sigmoid,
    step,
)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def test_step():
    assert step(0.5) == 1 and step(-0.5) == 0 and step(0.0) == 1


def test_sigmoid():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(10.0) > 0.99 and sigmoid(-10.0) < 0.01


def _akurasi(p, y):
    return np.mean([p.prediksi(x) == t for x, t in zip(X, y)])


def test_perceptron_and_or():
    for y in [np.array([0, 0, 0, 1]), np.array([0, 1, 1, 1])]:
        p = Perceptron(2)
        p.latih(X, y, lr=0.1, epochs=20)
        assert _akurasi(p, y) == 1.0


def test_perceptron_gagal_xor():
    y = np.array([0, 1, 1, 0])
    p = Perceptron(2)
    p.latih(X, y, lr=0.1, epochs=50)
    assert _akurasi(p, y) < 1.0  # XOR tidak linearly separable


def test_forward_mlp_xor():
    target = [0, 1, 1, 0]
    for x, t in zip(X, target):
        out = forward_mlp(x, XOR_W1, XOR_B1, XOR_W2, XOR_B2)
        assert round(out) == t
