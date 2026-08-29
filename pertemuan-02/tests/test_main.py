import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regresi import buat_data, hitung_gradien, latih, mse, prediksi


def test_prediksi():
    X = np.array([1.0, 2.0])
    assert np.allclose(prediksi(X, 2.0, 1.0), [3.0, 5.0])


def test_mse():
    assert mse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(2.0)


def test_gradien_nol_di_solusi():
    # di parameter sempurna tanpa noise, gradien ~ 0
    X = np.array([1.0, 2.0, 3.0])
    y = 3.0 * X + 2.0
    dw, db = hitung_gradien(X, y, 3.0, 2.0)
    assert abs(dw) < 1e-9 and abs(db) < 1e-9


def test_latih_konvergen():
    X, y = buat_data()
    w, b, riwayat = latih(X, y, lr=0.05, epochs=500)
    assert w == pytest.approx(3.0, abs=0.2)
    assert b == pytest.approx(2.0, abs=0.2)
    assert riwayat[-1] < riwayat[0]  # loss menurun
