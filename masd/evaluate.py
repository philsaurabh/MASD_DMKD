"""One-call evaluation producing the metrics of paper Table 5."""
from __future__ import annotations

import time

import torch

from .metrics import cka_similarity, classification_report, count_gflops, expected_calibration_error


@torch.no_grad()
def evaluate(model, data, num_classes: int, teacher=None, flops: bool = True) -> dict:
    model.eval()
    t0 = time.time()
    log_probs = model(data.x, data.edge_index)
    if log_probs.is_cuda:
        torch.cuda.synchronize()
    out = {"inference_seconds": time.time() - t0}
    out.update(classification_report(log_probs, data.y, data.test_mask))
    out["ece"] = expected_calibration_error(log_probs, data.y, data.test_mask, num_classes)
    if teacher is not None:
        out["cka_vs_teacher"] = cka_similarity(model, teacher, data.x, data.edge_index)
    if flops:
        out["gflops"] = count_gflops(model, data.x, data.edge_index)
    return out
