import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backprop import (
    backward,
    forward,
    init_params,
    latih_xor,
    prediksi_xor,
    sigmoid,
    turunan_sigmoid,
)


def test_sigmoid_dan_turunan():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert turunan_sigmoid(0.0) == pytest.approx(0.25)


def test_forward_shape():
    params = init_params()
    h, y_hat = forward(np.array([1.0, 0.0]), params)
    assert h.shape == (2,)
    assert np.isscalar(y_hat) or getattr(y_hat, "shape", ()) == ()


def test_backward_numerik():
    """Cek gradien analitik vs numerik (finite difference)."""
    params = init_params(seed=3)
    x, y = np.array([1.0, 0.0]), 1.0
    grad = backward(x, y, params)

    eps = 1e-6

    def loss(p):
        _, y_hat = forward(x, p)
        return 0.5 * (y_hat - y) ** 2

    p_plus = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in params.items()}
    p_plus["b2"] = params["b2"] + eps
    numerik = (loss(p_plus) - loss(params)) / eps
    assert grad["db2"] == pytest.approx(numerik, abs=1e-4)


def test_latih_xor_konvergen():
    params, riwayat = latih_xor(lr=0.5, epochs=5000, seed=7)
    assert riwayat[-1] < riwayat[0]
    assert prediksi_xor(params) == [0, 1, 1, 0]
