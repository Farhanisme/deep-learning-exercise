"""Tes struktural mini project — memastikan kerangka proyek lengkap."""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import proyek  # noqa: E402


def test_fungsi_wajib_ada():
    for nama in ["muat_data", "siapkan_data", "latih_baseline", "latih_model", "evaluasi"]:
        assert hasattr(proyek, nama), f"fungsi {nama} tidak ditemukan di src/proyek.py"
        assert callable(getattr(proyek, nama))


def test_dokumen_wajib_ada():
    for f in ["PROPOSAL.md", "LAPORAN.md", "SUBMISSION.md", "README.md"]:
        assert (ROOT / f).exists(), f"{f} tidak ditemukan"


def test_proposal_terisi():
    isi = (ROOT / "PROPOSAL.md").read_text(encoding="utf-8")
    assert "YOUR_NIM" not in isi, "PROPOSAL.md masih berisi placeholder YOUR_NIM"


def test_evaluasi_return_dict():
    sig = inspect.signature(proyek.evaluasi)
    assert len(sig.parameters) == 3
