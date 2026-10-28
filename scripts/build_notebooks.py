#!/usr/bin/env python3
"""Write the single Colab notebook. Explanations are Vietnamese; code is English."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/nqMinh-GR/NCS.git"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def notebook(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "include_colab_link": True,
        },
    }
    return nb


def write(path: Path, cells: list) -> None:
    nb = notebook(cells)
    nbf.validate(nb)
    path.write_text(nbf.writes(nb), encoding="utf-8")
    print(path)


def colab_notebook() -> list:
    return [
        md(f"""
# Biển số Việt Nam — một notebook Colab

Mở notebook này từ GitHub rồi bấm **Runtime → Run all**. Không có ô `START`. Mọi ô phía dưới đều chạy.

[Open in Colab](https://colab.research.google.com/github/nqMinh-GR/NCS/blob/main/notebooks/VN_ALPR_Colab.ipynb)

Repo: `{REPO}`

Việc notebook làm trên Colab: clone code, cài stack, tải dữ liệu public, fine-tune, đo latency ONNX trên GPU Colab, zip và tải về.
Latency Jetson không lấy từ Colab. `host_latency.csv` là latency máy train. Ô edge vẫn ghi `pending_on_device`.

Trước khi bấm Run all, sửa ô **tham số data / train** nếu muốn đổi batch size, epoch, learning rate, số worker hoặc số ảnh.
"""),
        md("""
## 1. Lấy code

Colab chỉ mở file notebook, không kéo theo cả repo. Ô này clone repo nếu chưa có `configs/project.yaml`.
"""),
        code(f"""
import os, sys, subprocess
from pathlib import Path

REPO = "{REPO}"
ROOT = Path("/content/NCS")
if (ROOT / ".git").exists():
    subprocess.check_call(["git", "-C", str(ROOT), "pull", "--ff-only"])
elif (Path.cwd() / "configs" / "project.yaml").exists():
    ROOT = Path.cwd()
else:
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(ROOT)])
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
for name in list(sys.modules):
    if name == "vn_alpr" or name.startswith("vn_alpr."):
        del sys.modules[name]
try:
    rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    rev = "unknown"
print("ROOT", ROOT.resolve(), "rev", rev)
"""),
        md("""
## 2. Tham số data và train

Sửa các biến này rồi Run all. Chúng được ghi vào lệnh train, không chỉ in ra.

- `DET_BATCH_SIZE`: batch detector (RT-DETRv4, D-FINE, RF-DETR).
- `OCR_BATCH_SIZE`: batch OCR (PP-OCRv6, FastPlateOCR).
- `GRAD_ACCUM_STEPS`: tích lũy gradient của RF-DETR. Để `1` nếu GPU còn đủ bộ nhớ cho batch ở trên.
- `NUM_WORKERS`: số worker đọc ảnh.
- `DET_EPOCHS`, `OCR_EPOCHS`: số epoch. Profile `smoke` không còn ghi đè hai số này sau ô này.
- `LEARNING_RATE`: `None` giữ learning rate upstream. Đặt số dương để ghi đè.
- `DET_INPUT_SIZE`: chỉ đổi độ phân giải RF-DETR khi `FORCE_DET_INPUT = True`. RT-DETRv4 và D-FINE vẫn resize 640 trong config upstream. RF-DETR mặc định dùng đúng native (Nano 384, Small 512, Medium 576).
- `MAX_IMAGES`: `None` dùng hết ảnh sau khi lọc. Đặt số nguyên để cắt tập cho lần chạy ngắn hơn.
- `PROFILE`: `screen` là shortlist. `full` thêm biến thể. `smoke` vẫn không lấy ảnh synthetic làm điểm benchmark.
- `LATENCY_WARMUP`, `LATENCY_ITERATIONS`, `LATENCY_SECONDS`, `LATENCY_REPEATS`: `None` dùng mức trong `configs/project.yaml` (screen: 50 vòng warmup, 500 vòng đo, ít nhất 30 giây, lặp 3 lần). Hạ các số này nếu Colab sắp hết giờ. Batch đo latency là 1, rồi 4 và 8 nếu graph ONNX cho phép.
"""),
        code("""
PROFILE = "screen"  # smoke | screen | full
SEED = 42

DET_BATCH_SIZE = 4
OCR_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
NUM_WORKERS = 2
DET_EPOCHS = 20
OCR_EPOCHS = 10
LEARNING_RATE = None
DET_INPUT_SIZE = 640
FORCE_DET_INPUT = False
MAX_IMAGES = None

LATENCY_WARMUP = None
LATENCY_ITERATIONS = None
LATENCY_SECONDS = None
LATENCY_REPEATS = None

PADDLE_CUDA = "cu126"  # None = wheel mặc định. Nếu index GPU lỗi, ô cài đặt dùng wheel CPU.
"""),
        md("""
## 3. Cài môi trường

PyTorch dùng sẵn trên Colab, kèm RF-DETR. Ô này cũng cài dependency của RT-DETRv4 và D-FINE, trong đó có `faster-coco-eval`. Thiếu package đó thì `train.py` chết ngay lúc import, trước khi GPU nhận model, nên không có đợt tràn VRAM.
Không cài lại `torch` hay `torchvision` từ pip, để giữ bản CUDA của Colab. Paddle và Keras nằm ở hai venv riêng. Colab thường không có `ensurepip`, nên ô này dùng `virtualenv` khi `venv` tạo thất bại. Paddle GPU được ghim `paddlepaddle-gpu==3.3.0` từ index `cu126` vì bản đó có wheel Python 3.13. Keras được ghim `tensorflow==2.21.0`, vì bản đó có wheel Python 3.13. Venv Keras còn cài `onnxruntime==1.31.0` bản CPU: `fast-plate-ocr` import `LicensePlateRecognizer` ngay khi nạp package và báo lỗi nếu thiếu ONNX Runtime. Bản CPU tránh đụng thư viện CUDA của TensorFlow trong cùng venv. Torch 2.x trên Colab gọi `import onnxscript` trong `torch.onnx.export`; thiếu package đó thì export RT-DETRv4/D-FINE chết sau khi train xong. `onnxsim` là bước simplify mặc định của script export upstream. Paddle hoặc Keras cài thất bại thì ô này dừng, không bỏ qua stack đó.
"""),
        code("""
import os, subprocess, sys
from pathlib import Path

def pip_install(python, packages, index=None):
    cmd = [python, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *packages]
    if index:
        cmd.extend(["-i", index])
    print(" ".join(cmd[:8]), "...")
    completed = subprocess.run(cmd, check=False, text=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)

pip_install(sys.executable, [
    "PyYAML", "Pillow", "numpy", "roboflow",
    "rfdetr[train]==1.11.2", "onnx==1.23.2", "onnxscript", "onnxsim",
    "faster-coco-eval>=1.6.6", "tensorboard", "scipy", "calflops", "transformers", "loguru",
    "lmdb", "pyclipper", "rapidfuzz", "regex",
])

ort_package = "onnxruntime==1.31.0"
try:
    import torch
    if torch.cuda.is_available():
        ort_package = "onnxruntime-gpu==1.31.0"
except Exception:
    pass
try:
    pip_install(sys.executable, [ort_package])
    print("onnxruntime", ort_package)
except subprocess.CalledProcessError:
    if ort_package != "onnxruntime==1.31.0":
        print("onnxruntime-gpu failed; using CPU onnxruntime")
        try:
            pip_install(sys.executable, ["onnxruntime==1.31.0"])
        except subprocess.CalledProcessError as exc:
            print("install_failed", "onnxruntime", exc.returncode)
    else:
        print("install_failed", "onnxruntime")

def make_venv(path):
    py = path / "bin" / "python"
    if py.exists():
        return py
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(path)])
    except subprocess.CalledProcessError:
        print("stdlib venv failed; using virtualenv")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "virtualenv"])
        subprocess.check_call([sys.executable, "-m", "virtualenv", str(path)])
    if not py.exists():
        raise RuntimeError(f"venv has no interpreter at {py}")
    subprocess.check_call([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    return py

venv_root = Path("/content/vn_alpr_venvs") if Path("/content").exists() else ROOT / "venvs"
venv_root.mkdir(parents=True, exist_ok=True)
# PaddleOCR v3.7.0 tools/train.py imports these. paddleocr itself is the inference package.
PADDLE_TRAIN = [
    "shapely", "scikit-image", "pyclipper", "lmdb", "tqdm", "numpy", "rapidfuzz",
    "opencv-contrib-python", "cython", "Pillow", "pyyaml", "requests",
    "albumentations", "albucore", "packaging", "paddleocr==3.7.0",
]
def install_keras(py):
    pip_install(str(py), ["fast-plate-ocr==1.1.0"])
    try:
        pip_install(str(py), ["tensorflow==2.21.0"])
    except subprocess.CalledProcessError:
        print("tensorflow GPU wheel failed; installing tensorflow-cpu==2.21.0")
        pip_install(str(py), ["tensorflow-cpu==2.21.0"])
    try:
        pip_install(str(py), ["fast-plate-ocr[train]==1.1.0"])
    except subprocess.CalledProcessError:
        print("keras train extra failed; installing the Linux train packages without coremltools")
        pip_install(str(py), [
            "keras>=3.10,<4", "albumentations", "click", "pandas",
            "pydantic>=2.5.2,<3", "scikit-learn", "tensorboard",
            "matplotlib", "tf2onnx", "onnxslim>=0.1.56",
        ])
    # Package import loads LicensePlateRecognizer, which requires onnxruntime.
    pip_install(str(py), ["onnxruntime==1.31.0"])
    print("keras_onnxruntime", "onnxruntime==1.31.0")

stacks = {
    "paddle": PADDLE_TRAIN,
    "keras": [],
}
PYTHON_BINS = {"torch": sys.executable}
failed = []
for name, packages in stacks.items():
    try:
        py = make_venv(venv_root / name)
        if name == "paddle" and PADDLE_CUDA:
            gpu_index = f"https://www.paddlepaddle.org.cn/packages/stable/{PADDLE_CUDA}/"
            try:
                pip_install(str(py), ["paddlepaddle-gpu==3.3.0"], gpu_index)
            except subprocess.CalledProcessError:
                print("paddle GPU wheel failed; installing the official CPU wheel")
                pip_install(str(py), ["paddlepaddle==3.3.0"], "https://www.paddlepaddle.org.cn/packages/stable/cpu/")
        if name == "keras":
            install_keras(py)
        else:
            pip_install(str(py), packages)
        probe = "import paddle, paddleocr" if name == "paddle" else "import fast_plate_ocr, keras, tensorflow; import fast_plate_ocr.cli.train"
        probe_env = None
        if name == "keras":
            # Colab sets MPLBACKEND to matplotlib_inline, which this venv does not have.
            probe_env = dict(os.environ)
            probe_env["MPLBACKEND"] = "Agg"
            probe_env["KERAS_BACKEND"] = "tensorflow"
        checked = subprocess.run([str(py), "-c", probe], check=False, text=True, capture_output=True, env=probe_env)
        if checked.returncode != 0:
            tail = (checked.stderr or checked.stdout or "import failed").strip()[-500:]
            print("install_failed", name, tail)
            failed.append(name)
            continue
        PYTHON_BINS[name] = str(py)
        print("installed", name, py)
    except subprocess.CalledProcessError as exc:
        print("install_failed", name, exc.returncode)
        failed.append(name)
print(PYTHON_BINS)
if failed:
    raise RuntimeError("required OCR stacks were not installed: " + ", ".join(failed))
"""),
        md("""
## 4. Credential

Train không tải Roboflow khi ô này in `roboflow_key_set False`. Ảnh lỗi là secret đang tên `ROBOFLC`, trong khi code tìm `ROBOFLOW_API_KEY`, nên key không được đọc và bước train không có bộ biển Việt Nam.

Sửa tên secret thành đúng `ROBOFLOW_API_KEY`, bật công tắc Notebook access, rồi chạy lại ô này và các ô dữ liệu/train bên dưới. Ô này không in giá trị key.
Không có key thì notebook vẫn tải bộ Hugging Face public và vẫn gọi train trên bộ đó.
"""),
        code("""
import os

# Tên trong panel Secrets phải khớp một dòng dưới đây. Ảnh đang để ROBOFLC.
ROBOFLOW_SECRET_NAMES = ("ROBOFLOW_API_KEY", "ROBOFLOW_KEY", "ROBOFLOW", "ROBOFLC")

def read_secret(name):
    if os.environ.get(name):
        return os.environ.get(name)
    try:
        from google.colab import userdata
        value = userdata.get(name)
    except Exception as exc:
        return exc
    return value or None

found_name = None
last_error = None
for secret_name in ROBOFLOW_SECRET_NAMES:
    loaded = read_secret(secret_name)
    if isinstance(loaded, str) and loaded.strip():
        os.environ["ROBOFLOW_API_KEY"] = loaded.strip()
        found_name = secret_name
        break
    if isinstance(loaded, Exception):
        last_error = type(loaded).__name__

print("roboflow_key_set", bool(found_name), "secret_name", found_name or "missing")
if not found_name:
    print("Không thấy key. Đặt tên secret là ROBOFLOW_API_KEY, bật Notebook access, rồi chạy lại ô này.")
    if last_error:
        print("colab_error", last_error)
kaggle_user = read_secret("KAGGLE_USERNAME")
kaggle_key = read_secret("KAGGLE_KEY")
if isinstance(kaggle_user, str) and isinstance(kaggle_key, str):
    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"] = kaggle_key
    print("kaggle_set", True)
else:
    print("kaggle_set", False)
"""),
        md("## 5. Áp tham số và xem máy"),
        code("""
import torch
from vn_alpr.config import apply_profile, apply_train_overrides, load_project, train_hparams
from vn_alpr.hardware import write_probe

overrides = {
    "seed": SEED,
    "det_batch_size": DET_BATCH_SIZE,
    "ocr_batch_size": OCR_BATCH_SIZE,
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "num_workers": NUM_WORKERS,
    "det_epochs": DET_EPOCHS,
    "ocr_epochs": OCR_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "det_input_size": DET_INPUT_SIZE,
    "force_det_input": FORCE_DET_INPUT,
    "max_images": MAX_IMAGES,
    "latency_warmup": LATENCY_WARMUP,
    "latency_iterations": LATENCY_ITERATIONS,
    "latency_seconds": LATENCY_SECONDS,
    "latency_repeats": LATENCY_REPEATS,
    "use_gpu": bool(torch.cuda.is_available()),
    "python_bins": PYTHON_BINS,
}
cfg = apply_train_overrides(apply_profile(load_project(), PROFILE), overrides)
hp = train_hparams(cfg)
print("profile", cfg["profile"], "gpu", hp["use_gpu"])
print("det batch/epochs", hp["det_batch_size"], hp["det_epochs"])
print("ocr batch/epochs", hp["ocr_batch_size"], hp["ocr_epochs"])
print("grad_accum", hp["grad_accum_steps"], "workers", hp["num_workers"], "lr", hp["learning_rate"])
print("max_images", hp["max_images"], "force_det_input", hp["force_det_input"], "det_input", hp["det_input_size"])
hw = write_probe(ROOT / cfg["run_root"])
print(hw["role"], hw["machine"], "on_target", hw["benchmark_on_target"])
"""),
        md("""
## 6. Tải dữ liệu và khóa split

Bộ Hugging Face `keremberke/license-plate-object-detection` là dữ liệu nhiều nước, vai trò phụ. Nếu không có bộ biển Việt Nam kèm theo, manifest ghi `mixed_auxiliary_not_vn_leaderboard`.
Roboflow tải khi có key. API liệt kê một version thì dùng version đó. API liệt kê nhiều version thì tải **số cao nhất trong danh sách API trả về** và ghi `version_choice=highest_listed`. Không bịa một số version chưa xuất hiện trong danh sách.
Bộ có chữ hoặc ký tự được tải trước bộ detection. Một class một ký tự được ghép thành chuỗi. Một class đã là chuỗi biển, có cả chữ và số, được giữ nguyên. Không suy ra chữ từ ảnh detection.
Ảnh synthetic không được đưa vào split này. Không có chữ cho cả train và val thì ô này dừng.
"""),
        code("""
import json
from vn_alpr.data.download import env_api_key
from vn_alpr.data.ingest import prepare_training_data

try:
    data_report = prepare_training_data(
        cfg,
        ROOT / "data" / "raw",
        ROOT / cfg["run_root"] / "canonical",
        api_key=env_api_key(),
    )
except Exception as exc:
    print("data_failed", type(exc).__name__, exc)
    data_report = {"status": "failed", "downloads": [], "counts": {}}
print(json.dumps({k: v for k, v in data_report.items() if k != "downloads"}, indent=2, ensure_ascii=False))
for row in data_report.get("downloads", []):
    print(row.get("dataset_id"), row.get("status"), row.get("version"), row.get("version_choice", ""), row.get("reason", ""))
if data_report.get("status") != "prepared" or not data_report.get("ocr"):
    raise RuntimeError("OCR không có chữ biển cho cả train và val. Xem status tải ở trên. Không suy diễn transcription.")
"""),
        md("""
## 7. Fine-tune và ghi leaderboard

Detector COCO được fine-tune trên split vừa khóa. OCR train trên cùng chữ biển: PP-OCRv6, CCT, và SVTRv2 từ đầu bằng code Apache-2.0. Weight SVTR trên Drive không được tải.
Model nào đã có `train.log`, trạng thái `ok`, và file ONNX (hoặc inference Paddle) thì không train lại. Model train rồi nhưng chưa export thì chỉ export lại. Model còn lại phải train. Ô này dừng nếu còn model không `ok`. Không có số accuracy bịa.
"""),
        code("""
from vn_alpr.pipeline import run_profile

try:
    report = run_profile(PROFILE, "train", start=True, overrides=overrides)
except Exception as exc:
    print("train_failed", type(exc).__name__, exc)
    report = {"status": "failed", "train_executed": False, "edge_executed": False, "rows": []}
def log_tail(reason, limit=40):
    from pathlib import Path
    path = Path(str(reason))
    if path.suffix != ".log" or not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("---", path, "---")
    print("\\n".join(lines[-limit:]))

print(report["status"], "train_executed", report["train_executed"], "edge_executed", report["edge_executed"])
pending = []
for row in report["rows"]:
    print(row["model_id"], row["status"], row["batch"], row["reason"])
    if row["status"] != "ok":
        pending.append(row["model_id"] + "=" + row["status"])
        log_tail(row["reason"])
print("--- host latency (not Jetson) ---")
for row in report.get("host_latency") or []:
    print(row["model_id"], row["status"], row.get("backend"), "batch", row.get("batch"), "p50_ms", row.get("p50_ms"), "p95_ms", row.get("p95_ms"))
if report.get("status") != "implemented" or pending:
    raise RuntimeError("models that did not train: " + ", ".join(pending or ["run_failed"]))
"""),
        md("""
## 8. Latency máy train

Ô train đã đo mọi file ONNX trong thư mục chạy: warmup, rồi lặp đến khi đủ số vòng và số giây. Có CUDA thì dùng `CUDAExecutionProvider`, không có thì CPU, và backend ghi đúng provider đã chạy.
Batch 1 là số chính. Batch 4 và 8 chỉ chạy khi dimension batch của ONNX không bị cố định. Hết bộ nhớ thì dòng đó là `oom`, không ghi latency 0.
`host_latency.csv` là latency Colab. Không copy các số này sang bảng Jetson.
"""),
        code("""
import json
from pathlib import Path
table = Path(cfg["run_root"]) / "host_latency.csv"
print(table.read_text(encoding="utf-8") if table.exists() else "chưa có host_latency.csv")
payload = Path(cfg["run_root"]) / "host_latency.json"
if payload.exists():
    rows = json.loads(payload.read_text(encoding="utf-8"))
    for row in rows:
        print(row.get("model_id"), row.get("status"), row.get("backend"), row.get("batch"), row.get("p50_ms"), row.get("p95_ms"), row.get("throughput"))
"""),
        md("""
## 9. Báo cáo

Khuyến nghị chỉ dùng dòng accuracy đã đo. Latency máy train nằm ở mục trên và trong `recommendation.md`.
"""),
        code("""
import subprocess, sys
subprocess.run([sys.executable, "scripts/evaluate.py"], check=False)
subprocess.run([sys.executable, "scripts/merge_reports.py"], check=False)
rec = ROOT / cfg["run_root"] / "recommendation.md"
print(rec.read_text(encoding="utf-8") if rec.exists() else "chưa có recommendation")
"""),
        md("""
## 10. Edge

Colab không phải Jetson. Ô này không build TensorRT và không ghi latency 0. Latency đã đo ở trên là của GPU Colab.
"""),
        code("""
import json
from vn_alpr.edge import build_edge_bundle
from vn_alpr.edge.tensorrt import preflight

run_dir = ROOT / cfg["run_root"]
bundle = build_edge_bundle(run_dir, run_dir / "edge_bundle", list(run_dir.rglob("*.onnx")))
edge = preflight()
payload = {
    "benchmark_on_target": False,
    "status": "pending_on_device",
    "reason": edge["reason"],
    "bundle": bundle["reason"],
    "rows": [],
}
(run_dir / "edge_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(payload["status"], payload["reason"])
"""),
        md("""
## 11. Zip và tải về

Zip bỏ raw data, checkpoint nặng, teacher và secret. File nhỏ hơn ngưỡng được tải qua trình duyệt.
"""),
        code("""
from vn_alpr.packaging import build_results_zip

info = build_results_zip(ROOT / cfg["run_root"], ROOT / "vn_alpr_results.zip")
print(info)
limit = cfg["packaging"]["colab_download_limit_mb"] * 1024 * 1024

def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

if in_colab() and info["bytes"] <= limit:
    from google.colab import files
    files.download(str(ROOT / "vn_alpr_results.zip"))
elif in_colab():
    drive = Path("/content/drive/MyDrive")
    if drive.exists():
        target = drive / "vn_alpr_results.zip"
        target.write_bytes((ROOT / "vn_alpr_results.zip").read_bytes())
        print("zip lớn, đã copy sang", target)
    else:
        print("zip lớn hơn giới hạn download. Lấy file tại", info["zip"])
else:
    print("máy local, zip nằm tại", info["zip"])
"""),
    ]


def colab_int8_notebook() -> list:
    return [
        md(f"""
# Biển số Việt Nam — test GPU Colab INT8

Notebook này **không train**. Nó lấy `vn_alpr_results.zip` trên Google Drive (cùng cấu trúc unzip: `canonical/`, `cct_s_v2/`, `rtdetrv4_s/`, …), quantize ONNX sang INT8, đo latency GPU, và chấm OCR trên `canonical/val.csv` + `canonical/ocr_images`.

Zip train **không chứa ảnh detection gốc** (`raw` bị loại khi pack). AP detector không bịa từ crop OCR. Latency INT8 của detector vẫn chạy.

[Open in Colab](https://colab.research.google.com/github/nqMinh-GR/NCS/blob/main/notebooks/VN_ALPR_Colab_INT8.ipynb)

Repo: `{REPO}`
"""),
        md("""
## 1. Lấy code
"""),
        code(f"""
import os, sys, subprocess
from pathlib import Path

REPO = "{REPO}"
ROOT = Path("/content/NCS")
if (ROOT / ".git").exists():
    subprocess.check_call(["git", "-C", str(ROOT), "pull", "--ff-only"])
elif (Path.cwd() / "configs" / "project.yaml").exists():
    ROOT = Path.cwd()
else:
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(ROOT)])
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
for name in list(sys.modules):
    if name == "vn_alpr" or name.startswith("vn_alpr."):
        del sys.modules[name]
try:
    rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    rev = "unknown"
print("ROOT", ROOT.resolve(), "rev", rev)
"""),
        md("""
## 2. Tham số

- `RESULTS_ZIP`: để `None` thì lần lượt tìm `MyDrive/vn_alpr/vn_alpr_results.zip`, `MyDrive/NCS/vn_alpr_results.zip`, `MyDrive/vn_alpr_results.zip`.
- `MAX_OCR_IMAGES`: `None` chấm hết val. Đặt số nguyên nếu muốn chạy ngắn.
- `CALIB_SAMPLES`: ảnh train dùng cho PTQ INT8 (luôn lấy từ split train trong zip).
- Latency mặc định ngắn hơn notebook train. Tăng nếu muốn đo lâu hơn.
"""),
        code("""
from pathlib import Path

RESULTS_ZIP = None  # Path hoặc None = tìm trên Drive
EXTRACT_DIR = Path("/content/vn_alpr_results")
OUT_DIR = Path("/content/vn_alpr_int8")
SPLIT = "val"
MAX_OCR_IMAGES = None
CALIB_SAMPLES = 256

LATENCY_WARMUP = 20
LATENCY_ITERATIONS = 100
LATENCY_SECONDS = 10
LATENCY_REPEATS = 1
"""),
        md("""
## 3. Cài ONNX Runtime GPU
"""),
        code("""
import subprocess, sys

def pip_install(packages):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *packages]
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode == 0

pip_install(["PyYAML", "Pillow", "numpy", "onnx"])
if not pip_install(["onnxruntime-gpu"]):
    print("onnxruntime-gpu missing, falling back to onnxruntime")
    pip_install(["onnxruntime"])
import onnxruntime as ort
print("providers", ort.get_available_providers())
"""),
        md("""
## 4. Mount Drive và giải nén zip
"""),
        code("""
from pathlib import Path
from google.colab import drive
from vn_alpr.eval_results import extract_results_zip, find_results_zip

drive.mount("/content/drive")
zip_path = Path(RESULTS_ZIP) if RESULTS_ZIP else find_results_zip()
print("zip", zip_path, "bytes", zip_path.stat().st_size)
RESULTS_DIR = extract_results_zip(zip_path, EXTRACT_DIR)
print("results", RESULTS_DIR)
print("has val.csv", (RESULTS_DIR / "canonical" / "val.csv").is_file())
print("ocr_images", sum(1 for _ in (RESULTS_DIR / "canonical" / "ocr_images").glob("*.jpg")))
"""),
        md("""
## 5. Quantize INT8, chấm OCR, đo latency GPU
"""),
        code("""
import json
from vn_alpr.config import apply_train_overrides, load_project
from vn_alpr.eval_results import discover_onnx, run_int8_eval
from vn_alpr.hardware import write_probe

overrides = {
    "latency_warmup": LATENCY_WARMUP,
    "latency_iterations": LATENCY_ITERATIONS,
    "latency_seconds": LATENCY_SECONDS,
    "latency_repeats": LATENCY_REPEATS,
}
cfg = apply_train_overrides(load_project(), overrides)
cfg["quantization"]["calibration_samples"] = int(CALIB_SAMPLES)
print("onnx models", [(m["model_id"], m["kind"]) for m in discover_onnx(RESULTS_DIR)])
hardware = write_probe(OUT_DIR)
report = run_int8_eval(
    RESULTS_DIR,
    OUT_DIR,
    cfg,
    hardware,
    split=SPLIT,
    max_images=MAX_OCR_IMAGES,
    prefer_cuda=True,
)
print("status", report["status"], "ocr_images", report["ocr_images"])
print("detection", report["detection"]["status"], report["detection"]["reason"])
for model_id, payload in report["ocr"].items():
    fp32 = payload.get("fp32") or {}
    int8 = payload.get("int8") or {}
    print(model_id, "fp32_exact", fp32.get("exact_normalized"), "cer", fp32.get("cer"),
          "int8_exact", int8.get("exact_normalized"), "gate", payload.get("gate"))
(OUT_DIR / "int8_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
"""),
        md("""
## 6. Bảng kết quả
"""),
        code("""
from pathlib import Path

for name in ("ocr_int8_leaderboard.csv", "fp32_latency.csv", "int8_latency.csv", "int8_quantization.csv", "recommendation.md"):
    path = OUT_DIR / name
    print("=" * 8, name)
    print(path.read_text(encoding="utf-8")[:4000] if path.exists() else "missing")
"""),
        md("""
## 7. Copy báo cáo sang Drive
"""),
        code("""
import shutil
from pathlib import Path

dest = Path("/content/drive/MyDrive/vn_alpr/int8_eval")
dest.mkdir(parents=True, exist_ok=True)
copied = []
for path in sorted(OUT_DIR.rglob("*")):
    if not path.is_file():
        continue
    if path.suffix.lower() in {".onnx"} and path.stat().st_size > 80 * 1024 * 1024:
        continue
    target = dest / path.relative_to(OUT_DIR)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    copied.append(str(target.relative_to(dest)))
print("copied", len(copied), "files to", dest)
print("leaderboard", dest / "ocr_int8_leaderboard.csv")
"""),
    ]


def main() -> None:
    dest = ROOT / "notebooks"
    dest.mkdir(exist_ok=True)
    write(dest / "VN_ALPR_Colab.ipynb", colab_notebook())
    write(dest / "VN_ALPR_Colab_INT8.ipynb", colab_int8_notebook())
    for stale in ("01_train_export_benchmark.ipynb", "02_jetson_quantize_benchmark.ipynb"):
        path = dest / stale
        if path.exists():
            path.unlink()
            print("removed", path)


if __name__ == "__main__":
    main()
