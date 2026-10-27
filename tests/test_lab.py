"""Self-tests for geometry, text, splits, metrics, licenses, and commands."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from vn_alpr.chars import order_character_boxes
from vn_alpr.config import ConfigError, apply_train_overrides, load_project, validate_project
from vn_alpr.data import catalog_plan, load_datasets
from vn_alpr.data.convert import build_detection_coco, class_role, yolo_line_to_xywh
from vn_alpr.data.download import DownloadError, assert_archive, looks_like_html, safe_extract_zip
from vn_alpr.data.prepare import prepare_locked_split
from vn_alpr.data.synthetic import make_synthetic_detection
from vn_alpr.evaluation import evaluate_chain, evaluate_detection, evaluate_ocr
from vn_alpr.geometry import letterbox, letterbox_xywh_to_original, xywh_to_letterbox
from vn_alpr.models import by_id, load_models, selection_reason, strip_yaml_key_block
from vn_alpr.models.commands import commands_for, rfdetr_script, student_config_path, student_yaml
from vn_alpr.packaging import build_results_zip
from vn_alpr.quantization.precision import quality_gate, summarize_precision
from vn_alpr.secrets import find_secrets, redact
from vn_alpr.splits import assign_splits, calibration_ids, leakage
from vn_alpr.textnorm import normalize_plate, unknown_characters


def test_letterbox_roundtrip():
    params = letterbox(320, 80, 640)
    original = (20.0, 10.0, 100.0, 40.0)
    forward = xywh_to_letterbox(original, params)
    back = letterbox_xywh_to_original(forward, params)
    for left, right in zip(original, back):
        assert left == pytest.approx(right)


def test_normalize_does_not_rewrite_confusion_pairs():
    assert normalize_plate("51a-123.45") == "51A12345"
    assert normalize_plate("Ｂ8") == "B8"
    assert normalize_plate("o0") == "O0"
    assert normalize_plate("i1") == "I1"
    assert unknown_characters("51AĐ", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") == ["Đ"]


def test_character_rows_top_then_left():
    boxes = [
        {"x": 30, "y": 40, "w": 10, "h": 12, "char": "2"},
        {"x": 10, "y": 40, "w": 10, "h": 12, "char": "1"},
        {"x": 30, "y": 8, "w": 10, "h": 12, "char": "B"},
        {"x": 10, "y": 8, "w": 10, "h": 12, "char": "A"},
    ]
    result = order_character_boxes(boxes)
    assert result["status"] == "assembled"
    assert result["lines"] == ["AB", "12"]
    assert result["label_origin"] == "derived_character_boxes"
    gapped = [
        {"x": 0, "y": 0, "w": 10, "h": 10, "char": "A"},
        {"x": 80, "y": 0, "w": 10, "h": 10, "char": "B"},
    ]
    assert order_character_boxes(gapped)["status"] == "ambiguous"


def test_split_has_no_leakage_and_calibration_stays_in_train():
    groups = {f"g{i}": [f"s{i}a", f"s{i}b"] for i in range(10)}
    groups["g_test"] = ["official", "copy"]
    assignment = assign_splits(groups, seed=42, official_test_ids=["official"])
    assert leakage(groups, assignment) == []
    assert assignment["g_test"] == "test"
    train = [sample for gid, samples in groups.items() for sample in samples if assignment[gid] == "train"]
    test = [sample for gid, samples in groups.items() for sample in samples if assignment[gid] == "test"]
    calib = calibration_ids(train, 3, 42)
    assert set(calib).isdisjoint(test)
    assert "copy" not in train


def test_detection_denominator_and_chain():
    images = [
        {
            "source": "a",
            "gt": [
                {"bbox": [0, 0, 10, 10], "layout": "single_line"},
                {"bbox": [40, 0, 10, 10], "layout": "single_line"},
            ],
            "preds": [
                {"bbox": [0, 0, 10, 10], "score": 0.9},
                {"bbox": [80, 80, 10, 10], "score": 0.9},
                {"bbox": [0, 0, 0, 10], "score": 0.99},
            ],
        },
        {"source": "a", "gt": [{"bbox": [0, 0, 10, 10]}], "preds": []},
    ]
    metrics = evaluate_detection(images)
    assert metrics["gt_plates"] == 3
    assert metrics["tp"] == 1
    assert metrics["fn"] == 2
    assert metrics["fp"] == 2
    chain = evaluate_chain(
        [
            {
                "gt": [{"bbox": [0, 0, 10, 10], "text": "51A12345"}, {"bbox": [40, 0, 10, 10], "text": "30A99999"}],
                "preds": [
                    {"bbox": [0, 0, 10, 10], "score": 0.9, "text": "51A-123.45"},
                    {"bbox": [40, 0, 10, 10], "score": 0.8, "text": "30A00000"},
                ],
            },
            {"gt": [{"bbox": [0, 0, 10, 10], "text": "29A11111"}], "preds": []},
        ]
    )
    assert chain["gt_plates"] == 3
    assert chain["chain_exact"] == pytest.approx(1 / 3)
    assert chain["ocr_exact_given_detection"] == pytest.approx(0.5)


def test_ocr_empty_is_wrong_and_unreadable_is_preset():
    metrics = evaluate_ocr(
        [
            {"gt_raw": "51A12345", "pred_raw": "", "readable": True, "layout": "single_line", "source": "a"},
            {"gt_raw": "bad", "pred_raw": "bad", "readable": False, "layout": "single_line", "source": "a"},
        ]
    )
    assert metrics["n"] == 1
    assert metrics["exact_normalized"] == 0
    assert metrics["empty_predictions"] == 1
    assert metrics["excluded_unreadable_or_ambiguous"] == 1


def test_int8_label_requires_int8_layers():
    summary = summarize_precision("int8", ["FP16", "FP16"])
    assert summary["status"] == "not_int8"
    assert summary["actual_precision_summary"] != "INT8"
    gate = quality_gate(0.90, 0.898, 0.5)
    assert gate["passed"] is True
    assert quality_gate(0.90, 0.80, 0.5)["passed"] is False


def test_teacher_block_removed_and_commands_match_upstream():
    text = """
output_dir: ./x
teacher_model:
  type: DINOv3TeacherModel
  dinov3_weights_path: pretrain/teacher.pth
optimizer:
  lr: 0.0004
loss_distill: 5
"""
    cleaned = student_yaml(text, 3)
    assert "teacher_model:" not in cleaned
    assert "dinov3" not in cleaned
    assert "loss_distill: 0" in cleaned
    assert "epoches: 3" in cleaned
    assert "epochs: 4" in student_yaml(text, 4, epoch_key="epochs")

    include_root = Path("/tmp/vn_alpr_include")
    work = include_root / "rtdetrv4_s"
    included = work / "upstream" / "configs" / "dfine" / "dfine_hgnetv2_s_coco.yml"
    included.parent.mkdir(parents=True, exist_ok=True)
    included.write_text("num_classes: 80\n", encoding="utf-8")
    model = {"family": "rtdetrv4", "config": "configs/rtv4/rtv4_hgnetv2_s_coco.yml"}
    dest = student_config_path(work, model)
    resolved = (dest.parent / "../dfine/dfine_hgnetv2_s_coco.yml").resolve()
    assert resolved == included.resolve()
    assert not (work / "../dfine/dfine_hgnetv2_s_coco.yml").resolve().exists()
    assert "teacher" not in strip_yaml_key_block(text, "teacher_model")

    root = Path("/tmp/vn_alpr_cmd")
    models = by_id(load_models())
    rf = commands_for(models["rfdetr_small"], root / "rf", 1)
    assert "import ultralytics" not in rf["script"]
    assert "RFDETRSmall" in rf["script"]
    with pytest.raises(ValueError):
        rfdetr_script({"python_class": "RFDETRXLarge"})
    dfine = commands_for(models["dfine_n"], root / "df", 1)
    assert "obj365" not in " ".join(dfine["train_argv"])
    assert "epochs=1" in dfine["train_argv"]
    assert "epoches=1" not in dfine["train_argv"]
    assert "remap_mscoco_category=True" in dfine["train_argv"]
    assert "eval_spatial_size=null" in dfine["train_argv"]
    assert "num_classes=1" in dfine["train_argv"]
    assert dfine["train_argv"][dfine["train_argv"].index("-c") + 1].endswith(
        "upstream/configs/dfine/vn_alpr_finetune.yml"
    )
    rtdetr = commands_for(models["rtdetrv4_s"], root / "rt", 2)
    assert "epoches=2" in rtdetr["train_argv"]
    assert rtdetr["train_argv"][rtdetr["train_argv"].index("-c") + 1].endswith(
        "upstream/configs/rtv4/vn_alpr_finetune.yml"
    )
    assert models["dfine_n"]["checkpoint_url"].endswith("dfine_n_coco.pth")
    paddle = commands_for(models["ppocrv6_tiny"], root / "pp", 1)
    assert "PP-OCRv6_tiny_rec.yml" in " ".join(paddle["train_argv"])
    assert "TextRecognition" in " ".join(paddle["zero_shot_argv"])
    cct = commands_for(models["cct_s_v2"], root / "cct", 1, hparams={"ocr_batch_size": 8, "num_workers": 1, "learning_rate": 0.0003})
    assert cct["decoder"] == "per_slot_classification"
    assert cct["train_argv"][cct["train_argv"].index("--batch-size") + 1] == "8"
    assert "--lr" in cct["train_argv"]
    assert "train_dataloader.total_batch_size=2" in " ".join(
        commands_for(models["dfine_n"], root / "df2", 1, hparams={"det_batch_size": 2, "num_workers": 0})["train_argv"]
    )
    assert "--format" in cct["export_argv"] and "onnx" in cct["export_argv"]
    svtr = commands_for(
        models["svtrv2_t"],
        root / "sv",
        1,
        hparams={"paddle_lists": {"train_dir": "/data", "train_list": "/data/train.txt", "val_list": "/data/val.txt"}, "ocr_batch_size": 8, "use_gpu": True},
    )
    assert "train_svtr.py" in " ".join(svtr["train_argv"])
    assert "svtrv2_tiny_rctc.yml" in " ".join(svtr["train_argv"])
    assert "SimpleDataSet" in svtr["script"]
    assert "drive.google.com" not in svtr["script"]
    assert '["pretrained_model"] = None' in svtr["script"]
    assert svtr["manual_checkpoint"] is True


def test_commercial_filter_and_config_schema():
    cfg = load_project()
    models = by_id(load_models())
    assert selection_reason(models["deimv2"], cfg)[0] is False
    assert selection_reason(models["rfdetr_xlarge"], cfg)[0] is False
    selected, why = selection_reason(models["svtrv2_t"], cfg)
    assert selected is True
    assert why == "train_from_scratch_no_pretrained_weights"
    assert selection_reason(models["rfdetr_small"], cfg)[0] is True
    assert selection_reason(models["ppocrv6_tiny"], cfg)[0] is True
    plan = {row["dataset_id"]: row for row in catalog_plan(True)}
    assert plan["winter_vn"]["selected_for_default_download"] is False
    assert plan["ufpr_alpr"]["selected_for_default_download"] is False
    assert plan["hf_text_candidate"]["selected_for_default_download"] is False
    assert plan["rf_vn_duke"]["version"] is None
    bad = dict(cfg)
    bad["run_mode"] = "banana"
    with pytest.raises(ConfigError):
        validate_project(bad)
    assert any(item["dataset_id"] == "rf_user_vn" for item in load_datasets())


def test_class_map_and_yolo():
    assert class_role("images") == "artifact"
    assert class_role("0") == "artifact"
    assert class_role("license_plate") == "plate"
    assert class_role("license-plate") == "plate"
    assert class_role("A") == "character"
    parsed = yolo_line_to_xywh("0 0.5 0.5 0.2 0.2", 100, 50)
    assert parsed[0] == 0
    assert parsed[1][2] == pytest.approx(20)
    document, rejected = build_detection_coco(
        [
            {
                "sample_id": "a",
                "file_name": "a.jpg",
                "width": 100,
                "height": 50,
                "boxes": [
                    {"name": "license_plate", "bbox": [1, 1, 10, 10]},
                    {"name": "labels", "bbox": [0, 0, 2, 2]},
                ],
            }
        ]
    )
    assert len(document["annotations"]) == 1
    assert rejected[0]["reason"] == "artifact"
    crop, _ = build_detection_coco(
        [
            {
                "sample_id": "crop",
                "file_name": "crop.jpg",
                "width": 40,
                "height": 16,
                "boxes": [{"name": "A", "bbox": [1, 2, 3, 8]}],
                "char_boxes": [{"char": "A", "x": 1, "y": 2, "w": 3, "h": 8}],
            }
        ]
    )
    assert crop["images"] == []


def test_synthetic_prepare_is_marked_self_test(tmp_path: Path):
    samples = make_synthetic_detection(tmp_path)
    samples[0]["file_sha256"] = "same"
    samples[1]["file_sha256"] = "same"
    samples[0]["pixel_sha256"] = "pix"
    samples[1]["pixel_sha256"] = "pix"
    samples[0]["ahash"] = 1
    samples[1]["ahash"] = 1
    manifest = prepare_locked_split(samples, tmp_path / "out", calibration_n=1, self_test=True)
    assert manifest["self_test"] is True
    groups = manifest["groups"]
    assert any(set(members) >= {"syn_0", "syn_1"} for members in groups.values())
    assert leakage(groups, manifest["assignment"]) == []
    ocr = (tmp_path / "out" / "ocr.jsonl").read_text(encoding="utf-8")
    assert "self_test" in ocr


def test_zip_rejects_secrets_and_skips_env(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.md").write_text("ok", encoding="utf-8")
    (run / ".env").write_text("ROBOFLOW_API_KEY=abc", encoding="utf-8")
    archive = build_results_zip(run, tmp_path / "out.zip")
    with zipfile.ZipFile(archive["zip"]) as zf:
        names = zf.namelist()
    assert "report.md" in names
    assert ".env" not in names
    secret = tmp_path / "leak"
    secret.mkdir()
    (secret / "notes.txt").write_text("api_key=supersecret", encoding="utf-8")
    with pytest.raises(RuntimeError):
        build_results_zip(secret, tmp_path / "no.zip")
    assert find_secrets("token: abcdefgh")
    assert not find_secrets("github-token: ${{ secrets.ISSUE }}")
    assert not find_secrets("cls_token : Float(1, 1, 384, strides=[384, 384, 1], device=cpu)")
    assert "Float(1," in redact("cls_token : Float(1, 1, 384)")
    assert "supersecret" not in redact("api_key=supersecret")
    upstream = run / "dfine_n" / "upstream" / ".github" / "workflows"
    upstream.mkdir(parents=True)
    (upstream / "is.yml").write_text("github-token: ${{ secrets.ISSUE }}\n", encoding="utf-8")
    archive = build_results_zip(run, tmp_path / "with_upstream.zip")
    with zipfile.ZipFile(archive["zip"]) as zf:
        names = zf.namelist()
    assert "report.md" in names
    assert all("upstream" not in name and ".github" not in name for name in names)


def test_zip_slip_and_html(tmp_path: Path):
    assert looks_like_html(b"<!DOCTYPE html><html>")
    html = tmp_path / "page.zip"
    html.write_bytes(b"<!DOCTYPE html>")
    with pytest.raises(DownloadError):
        assert_archive(html)
    good = tmp_path / "ok.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("../evil.txt", "nope")
    with pytest.raises(DownloadError):
        safe_extract_zip(good, tmp_path / "dest")


def test_training_blocker_does_not_use_synthetic_scores():
    from vn_alpr.models.runner import blocker

    assert blocker(False, Path("/tmp"), "screen") == "dependency_missing"
    assert blocker(True, None, "smoke") == "smoke_self_test_only"
    assert blocker(True, None, "screen") == "dataset_missing"
    assert blocker(True, {"train_images": "/tmp", "train_ann": "/tmp/a.json"}, "screen") is None
    cfg = load_project()
    assert cfg["edge"]["change_power_mode_automatically"] is False
    assert cfg["quantization"]["calibration_split"] == "train"
    text = Path("configs/project.yaml")
    if not text.exists():
        text = Path(__file__).resolve().parents[1] / "configs" / "project.yaml"
    loaded = yaml.safe_load(text.read_text(encoding="utf-8"))
    assert loaded["project"] == "vn_alpr_model_benchmark"
    json.dumps(cfg)


def test_train_overrides_reject_bad_batch():
    cfg = apply_train_overrides(load_project(), {"det_batch_size": 8, "ocr_epochs": 3, "learning_rate": 0.0002, "max_images": 12})
    assert cfg["training"]["det_batch_size"] == 8
    assert cfg["training"]["ocr_max_epochs"] == 3
    assert cfg["training"]["learning_rate"] == 0.0002
    assert cfg["training"]["max_images"] == 12
    with pytest.raises(ConfigError):
        apply_train_overrides(load_project(), {"det_batch_size": 0})
    with pytest.raises(ConfigError):
        apply_train_overrides(load_project(), {"det_input_size": 100})


def test_coco_tree_materializes_rfdetr_layout(tmp_path: Path):
    from PIL import Image

    from vn_alpr.data.ingest import limit_samples, materialize_images, samples_from_coco_tree

    root = tmp_path / "coco" / "train"
    root.mkdir(parents=True)
    Image.new("RGB", (32, 16), (255, 255, 255)).save(root / "a.jpg")
    Image.new("RGB", (32, 16), (0, 0, 0)).save(root / "b.jpg")
    document = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 32, "height": 16},
            {"id": 2, "file_name": "b.jpg", "width": 32, "height": 16},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 8]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 10, 8], "plate_text": "51A12345"},
        ],
        "categories": [{"id": 1, "name": "license_plate"}],
    }
    (root / "_annotations.coco.json").write_text(json.dumps(document), encoding="utf-8")
    samples = samples_from_coco_tree(tmp_path / "coco", {"dataset_id": "toy", "role": "primary", "country": "VN", "source_family_id": "toy"})
    assert len(samples) == 2
    assert any(sample.get("text_raw") == "51A12345" for sample in samples)
    chars_root = tmp_path / "chars" / "train"
    chars_root.mkdir(parents=True)
    Image.new("RGB", (40, 16), (255, 255, 255)).save(chars_root / "c.jpg")
    char_doc = {
        "images": [{"id": 1, "file_name": "c.jpg", "width": 40, "height": 16}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 4, 10]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [8, 2, 4, 10]},
        ],
        "categories": [{"id": 1, "name": "5"}, {"id": 2, "name": "1"}],
    }
    (chars_root / "_annotations.coco.json").write_text(json.dumps(char_doc), encoding="utf-8")
    char_samples = samples_from_coco_tree(
        tmp_path / "chars",
        {"dataset_id": "chars", "role": "primary", "country": "VN", "source_family_id": "chars"},
    )
    assert char_samples[0]["char_boxes"][0]["char"] == "5"
    assert char_samples[0]["char_boxes"][1]["char"] == "1"
    from vn_alpr.data.ingest import _balance_ocr_splits, _character, _plate_string, choose_listed_version

    assert _character("char_A") == "A"
    assert _plate_string("51A12345") == "51A12345"
    assert _plate_string("motorcycle") is None
    assert choose_listed_version([1, 3, 2], None) == (3, "highest_listed")
    assert choose_listed_version([4], None) == (4, "only_listed")
    assert choose_listed_version([1, 2], 1) == (1, "pinned")
    held_train, held_val = _balance_ocr_splits([{"text": "A"}, {"text": "B"}], [])
    assert held_train and held_val
    manifest = prepare_locked_split(samples + char_samples, tmp_path / "canonical", seed=1, calibration_n=1)
    capped, scope = limit_samples(samples, {"auxiliary": {"max_fraction_of_train": 0.2}, "training": {"max_images": 1}})
    assert scope == "includes_primary"
    assert len(capped) == 1
    ocr_lines = (tmp_path / "canonical" / "ocr.jsonl").read_text(encoding="utf-8")
    assert "51A12345" in ocr_lines
    assert '"text_raw": "51"' in ocr_lines or '"text_normalized": "51"' in ocr_lines
    image_dirs = materialize_images(samples, manifest, tmp_path / "rfdetr")
    linked = []
    for split in ("train", "val", "test"):
        linked.extend(Path(image_dirs[split]).iterdir())
    assert any(path.is_symlink() for path in linked)
    assert (tmp_path / "canonical" / "instances_train.json").exists()


def test_launch_passes_keras_backend(tmp_path: Path):
    import sys

    from vn_alpr.models.runner import launch_upstream

    script = tmp_path / "show_env.py"
    script.write_text("import os\nprint(os.environ.get('KERAS_BACKEND', ''))\n", encoding="utf-8")
    result = launch_upstream([sys.executable, str(script)], tmp_path, tmp_path / "log.txt", env={"KERAS_BACKEND": "tensorflow"})
    assert result["status"] == "ok"
    assert "tensorflow" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_host_latency_timer_and_missing_onnx(tmp_path: Path):
    from vn_alpr.benchmark.host import concrete_shape, measure_host_latency, time_calls
    from vn_alpr.reporting import host_latency_note, write_table

    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    samples = time_calls(fn, warmup=2, iterations=4, min_seconds=0)
    assert calls["n"] == 6
    assert len(samples) == 4
    assert concrete_shape(["N", 3, "H", "W"], 4) == [4, 3, 640, 640]
    rows = measure_host_latency(tmp_path, load_project(), {"machine": "x86_64", "nvidia_smi": None, "nvpmodel": None})
    assert rows[0]["status"] == "no_artifact"
    assert rows[0]["p50_ms"] is None
    assert rows[0]["role"] == "TRAIN_HOST"
    assert rows[0]["benchmark_on_target"] is False
    note = host_latency_note(rows)
    assert "0.00 ms" not in note
    tuned = apply_train_overrides(load_project(), {"latency_iterations": 20, "latency_seconds": 5, "latency_repeats": 1})
    assert tuned["benchmark"]["min_timed_iterations"] == 20
    assert tuned["benchmark"]["min_timed_seconds"] == 5
    write_table(
        tmp_path / "host_latency.csv",
        [
            {
                "model_id": "m",
                "status": "measured",
                "role": "TRAIN_HOST",
                "backend": "CPUExecutionProvider",
                "batch": 1,
                "p50_ms": 3.25,
                "p95_ms": 4.0,
                "p99_ms": 5.0,
                "throughput": 300.0,
                "hardware_fingerprint": "cpu",
            },
            {"model_id": "none", "status": "no_artifact", "role": "TRAIN_HOST", "p50_ms": 0, "batch": 1},
        ],
    )
    csv_text = (tmp_path / "host_latency.csv").read_text(encoding="utf-8")
    assert "3.25" in csv_text
    assert "no_artifact" in csv_text
    no_artifact_line = [line for line in csv_text.splitlines() if "no_artifact" in line][0]
    assert ",0," not in no_artifact_line


def test_onnx_latency_uses_real_samples(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    from vn_alpr.benchmark.host import measure_host_latency

    value = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 8, 8])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [value],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 3, 8, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    work = tmp_path / "toy_det"
    work.mkdir()
    onnx.save(model, work / "toy.onnx")
    cfg = apply_train_overrides(
        load_project(),
        {"latency_warmup": 1, "latency_iterations": 4, "latency_seconds": 0, "latency_repeats": 1},
    )
    cfg["benchmark"]["extra_batches_if_fit"] = []
    rows = measure_host_latency(tmp_path, cfg, {"machine": "x86_64", "nvidia_smi": None, "nvpmodel": None})
    assert rows[0]["status"] == "measured"
    assert rows[0]["p50_ms"] > 0
    assert rows[0]["benchmark_on_target"] is False
    assert rows[0]["backend"] in {"CPUExecutionProvider", "CUDAExecutionProvider"}
    assert rows[0]["n"] == 4


def test_already_trained_keeps_only_ok_logs(tmp_path: Path):
    from vn_alpr.pipeline import _already_trained

    run = tmp_path / "runs"
    (run / "rfdetr_small").mkdir(parents=True)
    (run / "ppocrv6_tiny").mkdir()
    (run / "rfdetr_small" / "train.log").write_text("done", encoding="utf-8")
    (run / "ppocrv6_tiny" / "train.log").write_text("missing labels", encoding="utf-8")
    report = {
        "rows": [
            {"model_id": "rfdetr_small", "status": "ok"},
            {"model_id": "ppocrv6_tiny", "status": "ocr_labels_missing"},
            {"model_id": "cct_s_v2", "status": "ok"},
        ]
    }
    (run / "execution_report.json").write_text(json.dumps(report), encoding="utf-8")
    found = _already_trained(run)
    assert set(found) == {"rfdetr_small"}
