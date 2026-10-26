"""Validate notebooks after they are generated."""

from pathlib import Path

import nbformat


def test_notebooks_validate():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    nb = nbformat.read(root / "VN_ALPR_Colab.ipynb", as_version=4)
    nbformat.validate(nb)
    source = "\n".join(cell.source for cell in nb.cells)
    assert "ROBOFLOW_API_KEY=" not in source
    assert "files.download" in source
    assert "DET_BATCH_SIZE" in source
    assert "OCR_BATCH_SIZE" in source
    assert "host_latency" in source
    assert "LATENCY_ITERATIONS" in source
    assert "START = False" not in source
    assert "https://github.com/nqMinh-GR/NCS.git" in source
    assert not (root / "01_train_export_benchmark.ipynb").exists()
