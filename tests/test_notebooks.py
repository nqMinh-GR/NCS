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
    assert "faster-coco-eval>=1.6.6" in source
    assert "rfdetr[train]==1.11.2" in source
    assert "paddlepaddle-gpu==3.3.0" in source
    assert "virtualenv" in source
    assert "fast-plate-ocr[train]==1.1.0" in source
    assert "tensorflow==2.21.0" in source
    assert "keras_onnxruntime" in source
    assert 'probe_env["MPLBACKEND"] = "Agg"' in source
    assert "highest_listed" in source
    assert "OCR_BATCH_SIZE" in source
    assert "host_latency" in source
    assert "LATENCY_ITERATIONS" in source
    assert "START = False" not in source
    assert "https://github.com/nqMinh-GR/NCS.git" in source
    assert not (root / "01_train_export_benchmark.ipynb").exists()
