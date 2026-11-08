"""Model optimisation study: reduced precision (FP16), dynamic INT8 quantisation, global magnitude pruning.

All variants classify the SAME pre-computed hybrid crops from unseen test images, so accuracy differences
come only from the model change. Size = serialised state_dict (and its gzip size, where pruning shows up),
latency = median single-crop time incl. resize + host/device transfer, throughput = batch of 32.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from config import CLASSIFIER_CKPT, DEFAULT_DATA_ROOT, DEFAULT_DETECTOR, FINETUNED_DETECTOR, RESULTS, ensure_dirs
from data import load_bgr, load_dataset
from deep_learning import FineGrainedClassifier, VehicleDetector, sync
from hybrid import VehiclePipeline
from metrics import time_fn, topk_acc
from stages import Spec, compute_boxes, resolve


def sizes(model: nn.Module) -> dict:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    raw = buf.getvalue()
    return {"size_mb": round(len(raw) / 1e6, 2), "gzip_size_mb": round(len(gzip.compress(raw, 3)) / 1e6, 2)}


def prune_model(model: nn.Module, amount: float) -> tuple[nn.Module, float]:
    m = copy.deepcopy(model).float()
    params = [(mod, "weight") for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, n in params:
        prune.remove(mod, n)
    z = sum(int((mod.weight == 0).sum()) for mod, _ in params)
    t = sum(mod.weight.numel() for mod, _ in params)
    return m, z / t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=1500, help="test images used for accuracy / timing")
    ap.add_argument("--detector", default=None)
    ap.add_argument("--no-detector-bench", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    ensure_dirs()
    data = load_dataset(args.data_root)
    test = data.test[:args.limit] if args.limit else data.test
    pipe = VehiclePipeline.load(detector_weights=args.detector)
    cfg = pipe.cfg
    boxes = compute_boxes(pipe, test, "opt", args.n_jobs, detector_tag=str(pipe.det.weights))
    y = np.array([r.label for r in test])
    crops = [pipe.make_crop(load_bgr(r.path), resolve(b, "yolo_ref_cls"), cfg.pad, cfg.use_illum) for r, b in zip(test, boxes)]
    cuda = torch.cuda.is_available()

    base_cpu = FineGrainedClassifier(CLASSIFIER_CKPT, device="cpu", half=False)
    variants: list[tuple[str, FineGrainedClassifier | None, str]] = []
    dev0 = "cuda" if cuda else "cpu"
    variants.append((f"FP32 ({dev0})", FineGrainedClassifier(CLASSIFIER_CKPT, device=dev0, half=False), dev0))
    if cuda:
        variants.append(("FP16 (cuda)", FineGrainedClassifier(CLASSIFIER_CKPT, device="cuda", half=True), "cuda"))
        variants.append(("FP32 (cpu)", base_cpu, "cpu"))
    note = {}
    try:
        torch.backends.quantized.engine = torch.backends.quantized.engine or "fbgemm"
        q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(base_cpu.model).float(), {nn.Linear}, dtype=torch.qint8)
        variants.append(("INT8 dynamic, Linear layers (cpu)", FineGrainedClassifier(CLASSIFIER_CKPT, device="cpu", half=False, model=q), "cpu"))
    except Exception as e:                                   # noqa: BLE001
        note["int8"] = f"dynamic quantisation unavailable: {e}"
    for amt in (0.3, 0.5):
        pm, sp = prune_model(base_cpu.model, amt)
        note[f"sparsity_{amt}"] = sp
        variants.append((f"Pruned {int(amt * 100)}% L1-global ({dev0})",
                         FineGrainedClassifier(CLASSIFIER_CKPT, device=dev0, half=False, model=pm), dev0))
    if cuda:
        pm, _ = prune_model(base_cpu.model, 0.3)
        variants.append(("Pruned 30% + FP16 (cuda)", FineGrainedClassifier(CLASSIFIER_CKPT, device="cuda", half=True, model=pm), "cuda"))

    rows = []
    ref = None
    for name, clf, dev in variants:
        p = clf.probs(crops, tta=False)
        acc = topk_acc(p, y)
        sy = (lambda d=clf.device: sync(d))
        one = time_fn(lambda: clf.logits(crops[:1]), warmup=10, iters=args.iters, sync=sy)
        b32 = time_fn(lambda: clf.logits(crops[:32], batch=32), warmup=1, iters=max(5, args.iters // 3), sync=sy)
        row = {"variant": name, **sizes(clf.model), **acc, "latency_ms_bs1": one["median_ms"],
               "throughput_img_s_bs32": 32000.0 / b32["median_ms"]}
        ref = ref or row
        row["top1_delta_vs_fp32"] = row["top1"] - ref["top1"]
        row["speedup_vs_fp32_bs1"] = ref["latency_ms_bs1"] / row["latency_ms_bs1"]
        rows.append(row)
        print(row, flush=True)

    det_rows = []
    if not args.no_detector_bench and pipe.det is not None and pipe.det.model is not None:
        imgs = [load_bgr(r.path) for r in test[:min(40, len(test))]]
        w = pipe.det.weights
        for name, kw in (("FP32", {}), ("FP16", {"quantize": 16}), ("FP16 (legacy flag)", {"half": True})):
            if name == "FP16 (legacy flag)" and any(r["precision"] == "FP16" and r["ok"] for r in det_rows):
                continue
            try:
                d = VehicleDetector(w, pred_kwargs=kw)
                if kw and not cuda:
                    raise RuntimeError("FP16 inference needs a CUDA device")
                t = time_fn(lambda: d.detect(imgs[0]), warmup=3, iters=args.iters, sync=lambda: sync(d.device))
                det_rows.append({"precision": name.split()[0], "ok": True, "median_ms": t["median_ms"],
                                 "weights_mb": round(os.path.getsize(str(w)) / 1e6, 2) if os.path.exists(str(w)) else None})
            except Exception as e:                           # noqa: BLE001
                det_rows.append({"precision": name.split()[0], "ok": False, "error": str(e)[:120]})

    out = {"n_images": len(test), "device": "cuda" if cuda else "cpu", "classifier": rows, "detector": det_rows, "notes": note,
           "torch": torch.__version__}
    (RESULTS / "optimization.json").write_text(json.dumps(out, indent=1))
    print("saved", RESULTS / "optimization.json")


if __name__ == "__main__":
    main()
