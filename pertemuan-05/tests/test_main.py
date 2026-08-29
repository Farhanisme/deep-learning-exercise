import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digit import cross_entropy, evaluasi, latih_mlp, siapkan_digits, softmax


def test_softmax_berjumlah_satu():
    p = softmax(np.array([2.0, 1.0, 0.1]))
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]


def test_softmax_stabil_numerik():
    p = softmax(np.array([1000.0, 999.0]))  # tanpa trik max -> overflow
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0)


def test_cross_entropy():
    prob = np.array([0.7, 0.2, 0.1])
    assert cross_entropy(prob, 0) == pytest.approx(-np.log(0.7))
    assert cross_entropy(prob, 2) > cross_entropy(prob, 0)


def test_siapkan_digits():
    X_train, X_test, y_train, y_test = siapkan_digits()
    assert X_train.shape[1] == 64
    assert X_train.max() <= 1.0
    assert len(X_test) == round((len(X_train) + len(X_test)) * 0.25)


def test_mlp_akurasi():
    X_train, X_test, y_train, y_test = siapkan_digits()
    model = latih_mlp(X_train, y_train)
    assert evaluasi(model, X_test, y_test) > 0.9
