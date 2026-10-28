"""Launch an upstream train command when the stack and a real dataset are both present."""

from __future__ import annotations

import codecs
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from vn_alpr.models.commands import commands_for, epoch_key_for, student_config_path, student_yaml


def find_layout(run_dir: Path) -> dict[str, Any] | None:
    for manifest_path in run_dir.rglob("split_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("self_test"):
            continue
        image_dirs = manifest.get("image_dirs") or {}
        coco = manifest.get("coco") or {}
        if not image_dirs.get("train") or not coco.get("train"):
            continue
        return {
            "train_images": image_dirs["train"],
            "val_images": image_dirs.get("val") or image_dirs["train"],
            "train_ann": coco["train"],
            "val_ann": coco.get("val") or coco["train"],
            "rfdetr_dir": manifest.get("rfdetr_dir"),
            "ocr_csv": manifest.get("ocr_csv"),
            "paddle_lists": manifest.get("paddle_lists"),
            "score_scope": manifest.get("score_scope"),
        }
    return None


def blocker(stack_ready: bool, layout: dict[str, Any] | None, profile: str) -> str | None:
    if not stack_ready:
        return "dependency_missing"
    if layout is None:
        if profile == "smoke":
            return "smoke_self_test_only"
        return "dataset_missing"
    return None


def _clone(repo_url: str, commit: str, dest: Path) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        if shutil.which("git") is None:
            return {"status": "dependency_missing", "reason": "git is not on PATH"}
        cloned = subprocess.run(["git", "clone", repo_url, str(dest)], check=False, capture_output=True, text=True)
        if cloned.returncode != 0:
            return {"status": "failed", "reason": (cloned.stderr or "")[-500:]}
    checked = subprocess.run(["git", "-C", str(dest), "checkout", commit], check=False, capture_output=True, text=True)
    if checked.returncode != 0:
        return {"status": "failed", "reason": (checked.stderr or "")[-500:]}
    return {"status": "ok", "path": str(dest)}


ONNX_EXPORT_PACKAGES = ("onnxscript", "onnxsim")


def ensure_torch_onnx_deps(python: str) -> list[str]:
    """Current torch.onnx.export imports onnxscript before it writes a graph."""
    missing = []
    for name in ONNX_EXPORT_PACKAGES:
        probe = subprocess.run([python, "-c", f"import {name}"], check=False, capture_output=True, text=True)
        if probe.returncode != 0:
            missing.append(name)
    if not missing:
        return []
    print(f"installing ONNX export deps: {' '.join(missing)}", flush=True)
    installed = subprocess.run(
        [python, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *missing],
        check=False,
        capture_output=True,
        text=True,
    )
    if installed.returncode != 0:
        tail = (installed.stderr or installed.stdout or "pip failed").strip()[-400:]
        print(f"onnx_export_deps_failed {tail}", flush=True)
    return missing


def launch_upstream(
    argv: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    exe = argv[0]
    if shutil.which(exe) is None and exe not in {"python", "python3"} and not Path(exe).is_file():
        return {"status": "dependency_missing", "argv": argv, "reason": f"{exe} not on PATH"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if env:
        run_env.update({str(key): str(value) for key, value in env.items()})
    run_env["PYTHONUNBUFFERED"] = "1"
    print(f"--- start {log_path.parent.name} ---", flush=True)
    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=run_env,
        bufsize=0,
    )
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as handle:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if not text:
                continue
            handle.write(text)
            handle.flush()
            print(text, end="", flush=True)
        tail = decoder.decode(b"", final=True)
        if tail:
            handle.write(tail)
            handle.flush()
            print(tail, end="", flush=True)
    returncode = process.wait()
    print(f"--- end {log_path.parent.name} exit={returncode} ---", flush=True)
    return {
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "log": str(log_path),
        "accuracy": None,
    }


def _latest(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    found = [path for path in root.rglob(pattern) if path.is_file()]
    if not found:
        return None
    return sorted(found, key=lambda path: path.stat().st_mtime)[-1]


def has_export_artifact(work: Path) -> bool:
    if any(path.is_file() for path in work.rglob("*.onnx")):
        return True
    infer = work / "inference"
    return infer.is_dir() and any(path.is_file() for path in infer.rglob("*"))


def _replace_flag(argv: list[str], flag: str, value: str) -> list[str]:
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = value
        return argv
    prefix = flag if flag.endswith("=") else f"{flag}="
    for index, item in enumerate(argv):
        if item.startswith(prefix):
            argv[index] = f"{prefix}{value}"
            return argv
    if flag.startswith("Global."):
        argv.append(f"{flag}={value}")
        return argv
    argv.extend([flag, value])
    return argv


def bind_export_checkpoint(model: dict[str, Any], work: Path, export_argv: list[str]) -> tuple[list[str], str | None]:
    """Point export at the file the trainer wrote, not a guessed name."""
    family = model.get("family")
    argv = list(export_argv)
    if family in {"rtdetrv4", "dfine", "deim"}:
        found = _latest(work, "best.pth") or _latest(work, "last.pth")
        if found is None:
            return [], "no detector checkpoint was written"
        return _replace_flag(argv, "-r", str(found)), None
    if family == "ppocr":
        found = _latest(work / "paddle", "best_accuracy.pdparams") or _latest(work / "paddle", "*.pdparams")
        if found is None:
            return [], "no Paddle checkpoint was written"
        argv = [item for item in argv if not item.startswith("Global.pretrained_model=")]
        return _replace_flag(argv, "Global.checkpoints", str(found.with_suffix(""))), None
    if family == "fast_plate_ocr":
        found = _latest(work / "cct", "best.keras")
        if found is None:
            return [], "best.keras was not written"
        return _replace_flag(argv, "--model", str(found)), None
    if family == "openocr_svtrv2":
        found = _latest(work / "svtr", "*.pth")
        if found is None:
            return [], "no SVTR checkpoint was written"
        return _replace_flag(argv, "Global.checkpoints", str(found)), None
    return argv, None


def _fetch_checkpoint(model: dict[str, Any], dest: Path) -> dict[str, Any] | None:
    if dest.exists() and dest.stat().st_size > 0:
        return None
    from vn_alpr.data.download import download_gdrive_binary, fetch_url, looks_like_html

    if model.get("gdrive_id"):
        result = download_gdrive_binary(model["gdrive_id"], dest)
        if result["status"] != "downloaded":
            return {"status": "failed", "reason": result.get("reason"), "accuracy": None}
        return None
    url = model.get("checkpoint_url") or ""
    if not url.startswith("https://") or url.endswith("/") or "drive.google.com/file" in url:
        return None
    try:
        fetch_url(url, dest, timeout=180)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": f"checkpoint download failed: {exc}", "accuracy": None}
    if looks_like_html(dest.read_bytes()[:64]):
        dest.unlink(missing_ok=True)
        return {
            "status": "failed",
            "reason": "checkpoint URL returned HTML. The file was not saved.",
            "accuracy": None,
        }
    return None


def attempt(
    model: dict[str, Any],
    run_dir: Path,
    epochs: int,
    seed: int,
    profile: str,
    hparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the upstream command, or return a specific blocker. Never invent metrics."""
    hparams = dict(hparams or {})
    layout = find_layout(run_dir)
    reason = blocker(True, layout, profile)
    if reason == "dataset_missing":
        return {"status": reason, "reason": "no canonical train split outside the synthetic self-test"}
    if reason == "smoke_self_test_only":
        return {"status": reason, "reason": "smoke synthetic images are not used as a training score"}
    assert layout is not None
    lists = layout.get("paddle_lists") or {}
    if model.get("task") == "ocr" and not lists.get("train_list"):
        return {"status": "ocr_labels_missing", "reason": "split has no plate text for train and val", "accuracy": None}
    if model.get("family") == "fast_plate_ocr" and not (layout.get("ocr_csv") or {}).get("train"):
        return {
            "status": "failed",
            "reason": "plate text does not fit the CCT alphabet and 10 slots",
            "accuracy": None,
        }
    hparams.update(layout)
    work = run_dir / model["model_id"]
    work.mkdir(parents=True, exist_ok=True)
    clone_families = {"rtdetrv4", "dfine", "deim", "openocr_svtrv2", "ppocr"}
    if model["family"] in clone_families:
        repo = work / "upstream"
        cloned = _clone(model["official_repo"], model["git_commit"], repo)
        if cloned["status"] != "ok":
            return cloned
        if model["family"] in {"rtdetrv4", "dfine", "deim"}:
            text = student_yaml(
                (repo / model["config"]).read_text(encoding="utf-8"),
                epochs,
                epoch_key=epoch_key_for(model),
            )
            dest = student_config_path(work, model)
            dest.write_text(text, encoding="utf-8")
            failed = _fetch_checkpoint(model, work / "checkpoint.pth")
            if failed:
                return failed
    weights_missing = False
    if model["family"] == "fast_plate_ocr":
        for name, url_key in (
            ("pretrained.keras", "checkpoint_url"),
            ("model_config.yaml", "model_config_url"),
            ("plate_config.yaml", "plate_config_url"),
        ):
            failed = _fetch_checkpoint({**model, "checkpoint_url": model.get(url_key), "gdrive_id": None}, work / name)
            if failed and name == "pretrained.keras":
                weights_missing = True
                continue
            if failed:
                return failed
    if model["family"] == "rfdetr":
        if not hparams.get("rfdetr_dir"):
            return {"status": "dataset_missing", "reason": "RF-DETR layout was not materialized", "accuracy": None}
        resolution = model["native_input"]
        if hparams.get("force_det_input"):
            resolution = int(hparams["det_input_size"])
        job = {
            "dataset_dir": hparams["rfdetr_dir"],
            "epochs": epochs,
            "batch_size": int(hparams.get("det_batch_size") or 4),
            "grad_accum_steps": int(hparams.get("grad_accum_steps") or 1),
            "device": "cuda" if hparams.get("use_gpu") else "cpu",
            "resolution": resolution,
            "learning_rate": hparams.get("learning_rate"),
            "output_dir": str(work / "out"),
            "onnx_dir": str(work / "onnx"),
            "opset": 17,
            "resume": None,
        }
        (work / "job.json").write_text(json.dumps(job), encoding="utf-8")
    spec = commands_for(model, work, epochs, seed, hparams)
    argv = spec.get("train_argv")
    if not argv:
        return {"status": "failed", "reason": "no train command"}
    if weights_missing and "--weights-path" in argv:
        index = argv.index("--weights-path")
        del argv[index : index + 2]
    cwd = work / "upstream" if model["family"] in clone_families else work
    if model["family"] in {"rtdetrv4", "dfine", "deim", "rfdetr"}:
        ensure_torch_onnx_deps(str(argv[0]) if Path(str(argv[0])).is_file() else "python")
    if hparams.get("skip_train"):
        launched = {"status": "ok", "reason": "already trained", "accuracy": None}
    else:
        launched = launch_upstream(argv, cwd, work / "train.log", env=spec.get("env"))
    export_argv = list(spec.get("export_argv") or [])
    if launched["status"] == "ok" and export_argv:
        export_argv, export_error = bind_export_checkpoint(model, work, export_argv)
        if export_error:
            launched["export_status"] = "failed"
            launched["export_reason"] = export_error
        else:
            exported = launch_upstream(export_argv, cwd, work / "export.log", env=spec.get("env"))
            launched["export_status"] = exported["status"]
            launched["export_log"] = exported.get("log")
            if exported.get("reason"):
                launched["export_reason"] = exported["reason"]
    return launched
