"""Upstream command lines. These functions do not launch training."""

from __future__ import annotations

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
    text += "remap_mscoco_category: False\nnum_classes: 1\n"
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
            ("remap_mscoco_category", False),
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


def svtr_commands(model: dict[str, Any], repo: Path, checkpoint: Path | None, epochs: int) -> dict[str, Any]:
    config = repo / model["config"]
    opt = [f"Global.epoch_num={epochs}"]
    if checkpoint:
        opt.append(f"Global.pretrained_model={checkpoint}")
    train = ["python", "tools/train_rec.py", "-c", str(config), "-o", *opt]
    export = ["python", "tools/export_rec.py", "-c", str(config), "--type", "onnx", "-o", *opt]
    return {"train_argv": train, "export_argv": export, "decoder": "CTC", "manual_checkpoint": checkpoint is None}


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
        return svtr_commands(model, Path("OpenOCR"), None, epochs)
    if family == "deimv2":
        raise ValueError("DEIMv2 is restricted and has no train command")
    raise KeyError(family)
