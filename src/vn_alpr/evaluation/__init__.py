"""Evaluation package."""

from vn_alpr.evaluation.combined import evaluate_chain
from vn_alpr.evaluation.detection import evaluate_detection
from vn_alpr.evaluation.ocr import evaluate_ocr

__all__ = ["evaluate_chain", "evaluate_detection", "evaluate_ocr"]
