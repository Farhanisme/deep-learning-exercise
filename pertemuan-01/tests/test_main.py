import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import (
    buat_matriks,
    kali_matriks_manual,
    kali_matriks_numpy,
    normalisasi_minmax,
    relu,
)


def test_buat_matriks():
    m = buat_matriks(2, 3, 7.0)
    assert m.shape == (2, 3)
    assert np.all(m == 7.0)


def test_kali_matriks_sama_dengan_numpy():
    rng = np.random.default_rng(0)
    A, B = rng.random((3, 4)), rng.random((4, 2))
    assert np.allclose(kali_matriks_manual(A, B), A @ B)
    assert np.allclose(kali_matriks_numpy(A, B), A @ B)


def test_normalisasi_minmax():
    x = np.array([2.0, 4.0, 6.0])
    hasil = normalisasi_minmax(x)
    assert hasil.min() == 0.0 and hasil.max() == 1.0
    assert np.allclose(hasil, [0.0, 0.5, 1.0])


def test_relu():
    x = np.array([-3.0, 0.0, 2.5])
    assert np.allclose(relu(x), [0.0, 0.0, 2.5])
