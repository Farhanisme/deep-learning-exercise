"""Cek lingkungan deep learning. Jalankan: python src/cek_env.py"""
import sys

print(f"Python     : {sys.version.split()[0]}")

try:
    import numpy as np
    print(f"NumPy      : {np.__version__}")
except ImportError:
    print("NumPy      : BELUM TERPASANG")

try:
    import matplotlib
    print(f"Matplotlib : {matplotlib.__version__}")
except ImportError:
    print("Matplotlib : BELUM TERPASANG")

try:
    import tensorflow as tf
    print(f"TensorFlow : {tf.__version__} (dipakai mulai pertemuan 4)")
except ImportError:
    print("TensorFlow : belum terpasang — tidak masalah untuk pertemuan 1-3")
