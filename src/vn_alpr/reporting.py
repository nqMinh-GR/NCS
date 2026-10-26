"""Leaderboards contain measured rows only. Missing Jetson numbers stay null."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    "run_id",
    "model_id",
    "checkpoint_hash",
    "dataset_split_hash",
    "protocol",
    "backend",
    "requested_precision",
    "actual_precision_summary",
    "input_shape",
    "batch",
    "role",
    "hardware_fingerprint",
    "power_mode",
    "accuracy",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "throughput",
    "peak_memory",
    "temperature_c",
    "power_w",
    "status",
    "reason",
    "log_path",
]


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            full = {key: row.get(key) for key in COLUMNS}
            if full["status"] in {"pending_on_device", "dependency_missing", "failed", "skipped", "not_int8", "no_artifact", "oom"}:
                for key in ("p50_ms", "p95_ms", "p99_ms", "throughput", "power_w", "temperature_c"):
                    if row.get("role") != "EDGE_TARGET" or full["status"] == "pending_on_device":
                        if key in ("p50_ms", "p95_ms", "p99_ms", "throughput", "power_w", "temperature_c"):
                            full[key] = None
            writer.writerow(full)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def recommend(rows: list[dict[str, Any]]) -> str:
    measured = [
        row
        for row in rows
        if row.get("accuracy") is not None and row.get("status") == "measured" and row.get("self_test") is not True
    ]
    edge = [row for row in measured if row.get("p50_ms") is not None and row.get("benchmark_on_target") is True]
    lines = ["# Khuyến nghị", ""]
    if not measured:
        lines.append("Chưa có accuracy đo trên tập val/test thật. Không chọn model từ bảng COCO hay latency máy chủ.")
        lines.append("Latency, công suất và nhiệt độ Jetson là null với status pending_on_device cho đến khi box chạy notebook edge.")
        return "\n".join(lines) + "\n"
    best = max(measured, key=lambda row: float(row["accuracy"]))
    lines.append(f"Accuracy cao nhất trong các dòng đã đo: `{best['model_id']}` ({best['accuracy']}).")
    if not edge:
        lines.append("Chưa có latency trên Jetson, nên chưa có cặp cân bằng edge.")
    else:
        ranked = sorted(edge, key=lambda row: (-float(row["accuracy"]), float(row["p50_ms"])))
        lines.append(f"Cân bằng accuracy/latency trên box: `{ranked[0]['model_id']}`.")
    lines.append("INT8 chỉ được chọn khi layer report có INT8 và gate trên validation đạt.")
    return "\n".join(lines) + "\n"


def host_latency_note(rows: list[dict[str, Any]]) -> str:
    measured = [row for row in rows if row.get("status") == "measured" and row.get("p50_ms") is not None]
    lines = ["", "## Latency trên máy train", ""]
    if not measured:
        lines.append("Chưa có file ONNX để đo trên máy train. Latency để trống, không ghi 0.")
        lines.append("Latency Jetson vẫn là pending_on_device.")
        return "\n".join(lines) + "\n"
    lines.append("Số dưới đây đo trên TRAIN_HOST. Không dùng các số này làm latency Jetson.")
    for row in sorted(measured, key=lambda item: (str(item.get("model_id")), int(item.get("batch") or 0))):
        lines.append(
            f"- `{row['model_id']}` batch {row['batch']} {row.get('backend')} trên {row.get('hardware_fingerprint')}: "
            f"p50 {float(row['p50_ms']):.2f} ms, p95 {float(row['p95_ms']):.2f} ms, "
            f"throughput {float(row['throughput']):.2f} ảnh/s."
        )
    return "\n".join(lines) + "\n"


def write_summary(path: Path, rows: list[dict[str, Any]], host_rows: list[dict[str, Any]] | None = None) -> None:
    text = recommend(rows)
    if host_rows is not None:
        text = text.rstrip() + "\n" + host_latency_note(host_rows)
    path.write_text(text, encoding="utf-8")
    html_rows = list(rows) + list(host_rows or [])
    html = ["<html><body><h1>VN ALPR summary</h1><table border='1'>"]
    html.append("<tr>" + "".join(f"<th>{key}</th>" for key in ("model_id", "status", "accuracy", "p50_ms")) + "</tr>")
    for row in html_rows:
        html.append(
            "<tr>"
            + "".join(f"<td>{row.get(key)}</td>" for key in ("model_id", "status", "accuracy", "p50_ms"))
            + "</tr>"
        )
    html.append("</table></body></html>")
    path.with_suffix(".html").write_text("\n".join(html), encoding="utf-8")
