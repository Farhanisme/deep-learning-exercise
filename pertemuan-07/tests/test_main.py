import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnn import buat_dataset_window, demo_vanishing, rnn_cell, rnn_forward


def test_dataset_window():
    deret = np.arange(10, dtype=float)
    X, y = buat_dataset_window(deret, w=3)
    assert X.shape == (7, 3)
    assert np.array_equal(X[0], np.array([0.0, 1.0, 2.0]))
    assert y[0] == 3.0
    assert y[-1] == 9.0


def test_rnn_cell():
    n = 3
    h = rnn_cell(1.0, np.zeros(n), np.ones(n), np.zeros((n, n)), np.zeros(n))
    assert h.shape == (n,)
    assert np.allclose(h, np.tanh(1.0))
    assert np.all(np.abs(h) <= 1.0)


def test_rnn_forward():
    rng = np.random.default_rng(0)
    n = 4
    H = rnn_forward(np.sin(np.linspace(0, 6, 20)), rng.normal(0, 0.5, n), rng.normal(0, 0.5, (n, n)), np.zeros(n))
    assert H.shape == (20, n)
    assert np.all(np.abs(H) <= 1.0)


def test_demo_vanishing():
    kecil = demo_vanishing(0.5, 50)   # (0.5*0.65)^50 -> nyaris nol
    sedang = demo_vanishing(1.5, 50)  # (1.5*0.65)^50 ~ 0.28 -> menyusut pelan
    assert kecil < 1e-10              # vanishing
    assert sedang > kecil
    assert demo_vanishing(2.0, 50) > 1e5  # (2*0.65)^50 -> exploding
