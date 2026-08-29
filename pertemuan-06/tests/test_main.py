import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnn import SOBEL_X, hitung_ukuran_output, konvolusi2d, max_pooling, relu


def test_konvolusi_shape():
    out = konvolusi2d(np.ones((8, 8)), np.ones((3, 3)))
    assert out.shape == (6, 6)


def test_konvolusi_nilai():
    citra = np.arange(9, dtype=float).reshape(3, 3)
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1.0  # identitas -> ambil piksel tengah
    out = konvolusi2d(citra, kernel)
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(4.0)


def test_konvolusi_deteksi_tepi():
    citra = np.zeros((5, 5))
    citra[:, 3:] = 1.0  # tepi vertikal
    out = konvolusi2d(citra, SOBEL_X)
    assert np.abs(out).max() > 0  # tepi terdeteksi


def test_relu():
    x = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(relu(x), np.array([0.0, 0.0, 3.0]))


def test_max_pooling():
    citra = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=float)
    out = max_pooling(citra, 2)
    assert out.shape == (2, 2)
    assert np.array_equal(out, np.array([[6, 8], [14, 16]], dtype=float))


def test_hitung_ukuran_output():
    assert hitung_ukuran_output(28, 28, 3, 2) == (13, 13)
    assert hitung_ukuran_output(8, 8, 3, 2) == (3, 3)
