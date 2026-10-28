#!/usr/bin/env python3
"""Evaluate a packaged vn_alpr_results tree with INT8 PTQ on this machine."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.config import apply_train_overrides, load_project
from vn_alpr.eval_results import extract_results_zip, run_int8_eval
from vn_alpr.hardware import write_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="unzipped vn_alpr_results directory or zip")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--calib", type=int, default=256)
    args = parser.parse_args()
    src = Path(args.results)
    if src.is_file() and src.suffix.lower() == ".zip":
        results = extract_results_zip(src, Path(args.out).parent / "vn_alpr_results")
    else:
        results = src
    cfg = apply_train_overrides(
        load_project(),
        {"latency_warmup": 5, "latency_iterations": 20, "latency_seconds": 0, "latency_repeats": 1},
    )
    cfg["quantization"]["calibration_samples"] = int(args.calib)
    out = Path(args.out)
    hardware = write_probe(out)
    report = run_int8_eval(results, out, cfg, hardware, split=args.split, max_images=args.max_images)
    print(report["status"], "ocr_images", report["ocr_images"], "models", len(report["models"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
