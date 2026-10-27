"""Upstream command lines. These functions do not launch training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vn_alpr.models import strip_yaml_key_block, zero_distill_loss

RFDETR_TRAIN = '''"""Generated RF-DETR fine-tune. Uses the rfdetr package, not Ultralytics."""
import json
import sys
from pathlib import Path

def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    from rfdetr import {class_name}
    model = {class_name}()
    kwargs = dict(
        dataset_dir=spec["dataset_dir"],
        epochs=spec["epochs"],
        batch_size=spec["batch_size"],
        grad_accum_steps=spec["grad_accum_steps"],
        device=spec["device"],
        resolution=spec["resolution"],
        output_dir=spec["output_dir"],
        resume=spec.get("resume") or None,
    )
    if spec.get("learning_rate") is not None:
        kwargs["lr"] = spec["learning_rate"]
    model.train(**kwargs)
    onnx_dir = Path(spec["onnx_dir"])
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model.export(
        output_dir=str(onnx_dir),
        opset_version=spec["opset"],
        format="onnx",
        batch_size=1,
        dynamic_batch=False,
        fp16=False,
        quantization=None,
        shape=(spec["resolution"], spec["resolution"]),
    )
    print(json.dumps({{"status": "ok", "onnx_dir": str(onnx_dir)}}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def student_yaml(official_text: str, epochs: int, epoch_key: str = "epoches") -> str:
    text = strip_yaml_key_block(official_text, "teacher_model")
    text = zero_distill_loss(text)
    if "teacher_model:" in text:
        raise RuntimeError("teacher_model block was not removed")
    text += f"\n{epoch_key}: {epochs}\n"
    # COCO category id 1 is the only plate class. The upstream MSCOCO table maps
    # that id to label 0, which is the only valid index when num_classes is 1.
    text += "remap_mscoco_category: True\nnum_classes: 1\n"
    # The cached sine embedding is a CPU attribute, not a buffer, so an eval
    # forward adds it to a CUDA feature. null rebuilds it on the feature device.
    text += "eval_spatial_size: null\n"
    return text


def _value(hparams: dict[str, Any] | None, key: str, default: Any) -> Any:
    if not hparams or hparams.get(key) is None:
        return default
    return hparams[key]


def interpreter(hparams: dict[str, Any] | None, stack: str) -> str:
    bins = (hparams or {}).get("python_bins") or {}
    return str(bins.get(stack) or "python")


def _bind_python(argv: list[str], python: str) -> list[str]:
    argv = list(argv)
    if not argv:
        return argv
    if argv[0] == "python":
        argv[0] = python
    elif argv[0] == "fast-plate-ocr" and python != "python":
        argv[0] = str(Path(python).with_name("fast-plate-ocr"))
    return argv


def _yaml_updates(pairs: list[tuple[str, Any]]) -> list[str]:
    updates = []
    for key, value in pairs:
        if value is None:
            continue
        updates.append(f"{key}={value}")
    return updates


def epoch_key_for(model: dict[str, Any]) -> str:
    """RT-DETRv4 reads epoches. D-FINE reads epochs."""
    if model.get("family") == "dfine":
        return "epochs"
    return "epoches"


def student_config_path(run_dir: Path, model: dict[str, Any]) -> Path:
    """Write the fine-tune file beside the upstream YAML.

    load_config joins __include__ paths to the config file's directory.
    A copy in the run root makes ../dfine and ../dataset point outside the clone.
    """
    return run_dir / "upstream" / Path(model["config"]).with_name("vn_alpr_finetune.yml")


def rtdetr_family_commands(
    model: dict[str, Any],
    repo: Path,
    checkpoint: Path,
    run_dir: Path,
    epochs: int,
    seed: int,
    hparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del checkpoint
    batch = int(_value(hparams, "det_batch_size", 4))
    workers = int(_value(hparams, "num_workers", 2))
    learning_rate = _value(hparams, "learning_rate", None)
    epoch_key = epoch_key_for(model)
    config_path = student_config_path(run_dir, model)
    updates = _yaml_updates(
        [
            ("train_dataloader.total_batch_size", batch),
            ("val_dataloader.total_batch_size", batch),
            ("train_dataloader.num_workers", workers),
            ("val_dataloader.num_workers", workers),
            ("train_dataloader.dataset.img_folder", (hparams or {}).get("train_images")),
            ("train_dataloader.dataset.ann_file", (hparams or {}).get("train_ann")),
            ("val_dataloader.dataset.img_folder", (hparams or {}).get("val_images")),
            ("val_dataloader.dataset.ann_file", (hparams or {}).get("val_ann")),
            ("optimizer.lr", learning_rate),
            (epoch_key, epochs),
            ("num_classes", 1),
            ("remap_mscoco_category", True),
            ("eval_spatial_size", "null"),
        ]
    )
    train = [
        "python",
        "train.py",
        "-c",
        str(config_path),
        "--use-amp",
        "--seed",
        str(seed),
        "-t",
        str(run_dir / "checkpoint.pth"),
        "-u",
        *updates,
    ]
    export = [
        "python",
        "tools/deployment/export_onnx.py",
        "--check",
        "-c",
        str(config_path),
        "-r",
        str(run_dir / "best.pth"),
    ]
    python = interpreter(hparams, model.get("stack") or "torch")
    return {
        "cwd": str(repo),
        "commit": model.get("git_commit"),
        "train_argv": _bind_python(train, python),
        "export_argv": _bind_python(export, python),
        "epochs": epochs,
        "batch_size": batch,
        "config": model["config"],
        "distillation": False,
    }


def rfdetr_script(model: dict[str, Any]) -> str:
    class_name = model["python_class"]
    if class_name not in {"RFDETRNano", "RFDETRSmall", "RFDETRMedium"}:
        raise ValueError(f"{class_name} is not an Apache-2.0 detection class in this lab")
    script = RFDETR_TRAIN.format(class_name=class_name)
    if "import ultralytics" in script or "from ultralytics" in script:
        raise RuntimeError("RF-DETR must not be launched through Ultralytics")
    return script


def dfine_commands(
    model: dict[str, Any],
    checkpoint: Path,
    run_dir: Path,
    epochs: int,
    seed: int,
    hparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = model["checkpoint_url"]
    if "obj365" in url or "obj2coco" in url:
        raise ValueError("Objects365 checkpoints are not the default")
    return rtdetr_family_commands(model, Path("."), checkpoint, run_dir, epochs, seed, hparams)


def paddle_commands(model: dict[str, Any], run_dir: Path, epochs: int, hparams: dict[str, Any] | None = None) -> dict[str, Any]:
    batch = int(_value(hparams, "ocr_batch_size", 16))
    workers = int(_value(hparams, "num_workers", 2))
    use_gpu = "true" if _value(hparams, "use_gpu", False) else "false"
    learning_rate = _value(hparams, "learning_rate", None)
    lists = (hparams or {}).get("paddle_lists") or {}
    options = [
        f"Global.epoch_num={epochs}",
        f"Global.save_model_dir={run_dir / 'paddle'}",
        f"Global.use_gpu={use_gpu}",
        f"Train.loader.batch_size_per_card={batch}",
        f"Train.sampler.first_bs={batch}",
        f"Eval.loader.batch_size_per_card={batch}",
        f"Train.loader.num_workers={workers}",
        f"Eval.loader.num_workers={workers}",
        "Global.distributed=False",
        "Train.loader.drop_last=False",
        "Eval.loader.drop_last=False",
    ]
    if learning_rate is not None:
        options.append(f"Optimizer.lr.learning_rate={learning_rate}")
    if lists.get("train_dir"):
        options.append(f"Train.dataset.data_dir={lists['train_dir']}")
        options.append(f"Eval.dataset.data_dir={lists['train_dir']}")
    if lists.get("train_list"):
        options.append(f"Train.dataset.label_file_list=[{lists['train_list']}]")
    if lists.get("val_list"):
        options.append(f"Eval.dataset.label_file_list=[{lists['val_list']}]")
    train = [
        "python",
        "tools/train.py",
        "-c",
        model["config"],
        "-o",
        *options,
    ]
    export = [
        "python",
        "tools/export_model.py",
        "-c",
        model["config"],
        "-o",
        f"Global.pretrained_model={run_dir / 'paddle' / 'best'}",
        f"Global.save_inference_dir={run_dir / 'inference'}",
    ]
    python = interpreter(hparams, "paddle")
    infer = [
        python,
        "-c",
        (
            "from paddleocr import TextRecognition; "
            f"m=TextRecognition(model_name={model['variant']!r}); "
            "print('ready', m)"
        ),
    ]
    return {
        "train_argv": _bind_python(train, python),
        "export_argv": _bind_python(export, python),
        "zero_shot_argv": infer,
        "decoder": "CTCLabelDecode",
        "batch_size": batch,
    }


def cct_commands(model: dict[str, Any], run_dir: Path, epochs: int, hparams: dict[str, Any] | None = None) -> dict[str, Any]:
    if model.get("not_ctc") is not True:
        raise ValueError("CCT must not be trained with CTC")
    batch = int(_value(hparams, "ocr_batch_size", 16))
    workers = int(_value(hparams, "num_workers", 2))
    csv_paths = (hparams or {}).get("ocr_csv") or {}
    train = [
        "fast-plate-ocr",
        "train",
        "--model-config-file",
        str(run_dir / "model_config.yaml"),
        "--plate-config-file",
        str(run_dir / "plate_config.yaml"),
        "--annotations",
        str(csv_paths.get("train") or (run_dir / "train.csv")),
        "--val-annotations",
        str(csv_paths.get("val") or (run_dir / "val.csv")),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch),
        "--workers",
        str(workers),
        "--output-dir",
        str(run_dir / "cct"),
        "--weights-path",
        str(run_dir / "pretrained.keras"),
        "--seed",
        str(_value(hparams, "seed", 42)),
    ]
    learning_rate = _value(hparams, "learning_rate", None)
    if learning_rate is not None:
        train.extend(["--lr", str(learning_rate)])
    export = [
        "fast-plate-ocr",
        "export",
        "--model",
        str(run_dir / "cct" / "best.keras"),
        "--plate-config-file",
        str(run_dir / "plate_config.yaml"),
        "--format",
        "onnx",
    ]
    return {
        "train_argv": train,
        "export_argv": export,
        "env": {"KERAS_BACKEND": "tensorflow"},
        "decoder": "per_slot_classification",
        "batch_size": batch,
    }


SVTR_TRAIN = '''"""Train SVTRv2 from the Apache-2.0 architecture. Pretrained Drive weights are not loaded."""
import sys
from pathlib import Path

import yaml


def main() -> int:
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    source = repo / sys.argv[1]
    dest = Path(sys.argv[2])
    spec = yaml.safe_load(Path(sys.argv[3]).read_text(encoding="utf-8"))
    train_list = spec.get("train_list")
    val_list = spec.get("val_list")
    if not train_list or not val_list:
        raise SystemExit("OCR label lists are missing")
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    for split, list_key in (("Train", "train_list"), ("Eval", "val_list")):
        dataset = cfg[split]["dataset"]
        dataset["name"] = "SimpleDataSet"
        dataset["data_dir"] = spec["data_dir"]
        dataset["label_file_list"] = [spec[list_key]]
        dataset.pop("data_dir_list", None)
        cfg[split].pop("sampler", None)
        loader = cfg[split]["loader"]
        loader["batch_size_per_card"] = int(spec["batch"])
        loader["num_workers"] = int(spec["workers"])
        loader["drop_last"] = False
    cfg["Global"]["epoch_num"] = int(spec["epochs"])
    cfg["Global"]["device"] = "gpu" if spec.get("use_gpu") else "cpu"
    cfg["Global"]["use_amp"] = bool(spec.get("use_gpu"))
    cfg["Global"]["pretrained_model"] = None
    cfg["Global"]["checkpoints"] = None
    cfg["Global"]["use_tensorboard"] = False
    cfg["Global"]["output_dir"] = spec["output_dir"]
    if spec.get("learning_rate") is not None:
        cfg["Optimizer"]["lr"] = float(spec["learning_rate"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    from tools.engine.config import Config
    from tools.engine.trainer import Trainer

    Trainer(Config(str(dest)), mode="train_eval", task="rec").train()
    print("svtr_trained", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def svtr_commands(
    model: dict[str, Any],
    run_dir: Path,
    epochs: int,
    hparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lists = (hparams or {}).get("paddle_lists") or {}
    batch = int(_value(hparams, "ocr_batch_size", 16))
    workers = int(_value(hparams, "num_workers", 2))
    job = {
        "train_list": lists.get("train_list"),
        "val_list": lists.get("val_list"),
        "data_dir": lists.get("train_dir") or ".",
        "batch": batch,
        "workers": workers,
        "epochs": epochs,
        "use_gpu": bool(_value(hparams, "use_gpu", False)),
        "learning_rate": _value(hparams, "learning_rate", None),
        "output_dir": str(run_dir / "svtr"),
    }
    script_path = run_dir / "train_svtr.py"
    job_path = run_dir / "svtr_job.json"
    config_out = run_dir / "svtr.yml"
    script_path.write_text(SVTR_TRAIN, encoding="utf-8")
    job_path.write_text(json.dumps(job), encoding="utf-8")
    python = interpreter(hparams, "torch")
    train = [python, str(script_path), model["config"], str(config_out), str(job_path)]
    export = [
        python,
        "tools/export_rec.py",
        "-c",
        str(config_out),
        "--type",
        "onnx",
        "-o",
        f"Global.output_dir={run_dir / 'svtr'}",
    ]
    return {
        "train_argv": train,
        "export_argv": export,
        "script": SVTR_TRAIN,
        "decoder": "CTC",
        "manual_checkpoint": True,
        "batch_size": batch,
    }


def commands_for(
    model: dict[str, Any],
    run_dir: Path,
    epochs: int,
    seed: int = 42,
    hparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = model["family"]
    run_dir.mkdir(parents=True, exist_ok=True)
    hparams = dict(hparams or {})
    hparams.setdefault("seed", seed)
    if family == "rtdetrv4":
        return rtdetr_family_commands(model, Path(model["official_repo"]), Path("checkpoint.pth"), run_dir, epochs, seed, hparams)
    if family == "dfine":
        return dfine_commands(model, Path("checkpoint.pth"), run_dir, epochs, seed, hparams)
    if family == "deim":
        spec = rtdetr_family_commands(model, Path("."), Path("checkpoint.pth"), run_dir, epochs, seed, hparams)
        spec["note"] = "optional DEIM recipe"
        return spec
    if family == "rtdetrv2":
        return {
            "train_argv": _bind_python(["python", "tools/train.py", "-c", model["config"]], interpreter(hparams, "torch")),
            "export_argv": ["python", model["export_script"], "-c", model["config"]],
            "note": "optional baseline; checkpoint URL not pinned",
        }
    if family == "rfdetr":
        script = rfdetr_script(model)
        path = run_dir / "train_rfdetr.py"
        path.write_text(script, encoding="utf-8")
        python = interpreter(hparams, "torch")
        return {
            "train_argv": [python, str(path), str(run_dir / "job.json")],
            "script": script,
            "resolution": model["native_input"] if not hparams.get("force_det_input") else int(hparams["det_input_size"]),
            "class_name": model["python_class"],
            "batch_size": int(_value(hparams, "det_batch_size", 4)),
        }
    if family == "ppocr":
        return paddle_commands(model, run_dir, epochs, hparams)
    if family == "fast_plate_ocr":
        spec = cct_commands(model, run_dir, epochs, hparams)
        spec["train_argv"] = _bind_python(spec["train_argv"], interpreter(hparams, "keras"))
        spec["export_argv"] = _bind_python(spec["export_argv"], interpreter(hparams, "keras"))
        return spec
    if family == "openocr_svtrv2":
        return svtr_commands(model, run_dir, epochs, hparams)
    if family == "deimv2":
        raise ValueError("DEIMv2 is restricted and has no train command")
    raise KeyError(family)
