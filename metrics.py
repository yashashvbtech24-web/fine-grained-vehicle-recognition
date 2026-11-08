"""Evaluation metrics: classification, localisation (IoU), detection (AP/mAP), calibration, efficiency."""
from __future__ import annotations

import time

import numpy as np

from classical import iou_xyxy
from data import ClassTable


# ----------------------------------------------------------------------------- classification
def topk_acc(probs: np.ndarray, labels: np.ndarray, ks=(1, 5)) -> dict:
    order = np.argsort(-probs, axis=1)[:, :max(ks)]
    return {f"top{k}": float((order[:, :k] == labels[:, None]).any(1).mean()) for k in ks}


def level_acc(probs: np.ndarray, labels: np.ndarray, table: ClassTable) -> dict:
    """Accuracy at the three granularities requested by the task."""
    pred = probs.argmax(1)
    exact = pred == labels
    mm = table.makemodel_idx[pred] == table.makemodel_idx[labels]
    mk = table.make_idx[pred] == table.make_idx[labels]
    return {"exact_acc": float(exact.mean()), "make_model_acc": float(mm.mean()), "make_acc": float(mk.mean()),
            "year_only_error_rate": float((mm & ~exact).mean())}


def classification_report(probs, labels, table) -> dict:
    return {**topk_acc(probs, labels), **level_acc(probs, labels, table)}


# ----------------------------------------------------------------------------- localisation / detection
def iou_array(pred_boxes, gt_boxes) -> np.ndarray:
    return np.array([iou_xyxy(p, g) if p is not None else 0.0 for p, g in zip(pred_boxes, gt_boxes)])


def localization_summary(ious: np.ndarray) -> dict:
    return {"mean_iou": float(ious.mean()), "median_iou": float(np.median(ious)),
            "iou_ge_0.5": float((ious >= 0.5).mean()), "iou_ge_0.75": float((ious >= 0.75).mean()),
            "iou_ge_0.9": float((ious >= 0.9).mean())}


def average_precision(preds: list[list[tuple]], gts: list, iou_thr: float) -> float:
    """Single-class AP (COCO 101-point interpolation). preds[i] = [(box, conf), ...]; gts[i] = one GT box
    (Stanford Cars has exactly one annotated car per image)."""
    recs = []
    for i, ps in enumerate(preds):
        for box, conf in ps:
            recs.append((conf, i, box))
    recs.sort(key=lambda r: -r[0])
    matched = np.zeros(len(gts), bool)
    tp = np.zeros(len(recs))
    for k, (conf, i, box) in enumerate(recs):
        if not matched[i] and iou_xyxy(box, gts[i]) >= iou_thr:
            matched[i] = True
            tp[k] = 1
    ctp = np.cumsum(tp)
    rec = ctp / max(1, len(gts))
    prec = ctp / (np.arange(len(recs)) + 1)
    for k in range(len(prec) - 2, -1, -1):
        prec[k] = max(prec[k], prec[k + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(rec, r, side="left")
        ap += prec[idx] if idx < len(prec) else 0.0
    return float(ap / 101)


def detection_summary(preds, gts) -> dict:
    thrs = np.arange(0.5, 0.96, 0.05)
    aps = [average_precision(preds, gts, t) for t in thrs]
    return {"AP50": aps[0], "AP75": aps[5], "mAP50_95": float(np.mean(aps))}


# ----------------------------------------------------------------------------- calibration / confidence
def nll(probs, labels) -> float:
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def reliability_bins(conf, correct, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "conf": float(conf[m].mean()) if m.any() else None,
                    "acc": float(correct[m].mean()) if m.any() else None})
    return out


def fit_temperature_np(probs_or_logits: np.ndarray, labels: np.ndarray, is_logits: bool = True) -> float:
    from scipy.optimize import minimize_scalar
    lg = probs_or_logits if is_logits else np.log(probs_or_logits + 1e-9)

    def f(logT):
        z = lg / np.exp(logT)
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(labels)), labels].mean()
    return float(np.exp(minimize_scalar(f, bounds=(-3, 3), method="bounded").x))


def selective_table(conf: np.ndarray, correct: np.ndarray, thresholds=None) -> list[dict]:
    thresholds = np.round(np.arange(0.0, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append({"threshold": float(t), "coverage": float(m.mean()),
                     "accepted_acc": float(correct[m].mean()) if m.any() else None,
                     "rejected_frac": float(1 - m.mean())})
    return rows


def risk_coverage(conf: np.ndarray, correct: np.ndarray):
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    cov = (np.arange(len(c)) + 1) / len(c)
    acc = np.cumsum(c) / (np.arange(len(c)) + 1)
    return cov, acc


def find_threshold(conf: np.ndarray, correct: np.ndarray, target: float = 0.9, min_cov: float = 0.1) -> float:
    """Smallest threshold whose accepted-set accuracy reaches `target` while keeping >= min_cov coverage."""
    best = None
    for t in np.unique(np.round(conf, 3)):
        m = conf >= t
        if m.mean() < min_cov:
            break
        if correct[m].mean() >= target:
            best = float(t)
            break
    if best is None:                       # unreachable target: choose the highest-accuracy point with enough coverage
        ts = np.unique(np.round(conf, 3))
        scored = [(correct[conf >= t].mean(), t) for t in ts if (conf >= t).mean() >= min_cov]
        best = float(max(scored)[1]) if scored else 0.5
    return best


# ----------------------------------------------------------------------------- efficiency
def time_fn(fn, warmup: int = 3, iters: int = 20, sync=None) -> dict:
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t) * 1000)
    return {"mean_ms": float(np.mean(ts)), "median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}
