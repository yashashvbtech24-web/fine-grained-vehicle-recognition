"""Main evaluation on the UNSEEN test split: classical vs deep learning vs hybrid.

Produces work/results/evaluation.json, per-image prediction CSVs and figures covering:
  Top-1 / Top-5, make / make+model / exact accuracy, IoU, AP / mAP, calibration (ECE) and confidence
  thresholds, model size and inference time, viewpoint proxy, top confusions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CLASSICAL_CKPT, CLASSIFIER_CKPT, DEFAULT_DATA_ROOT, DEFAULT_DETECTOR, FIGS, FINETUNED_DETECTOR, RESULTS, ensure_dirs
from data import load_bgr, load_dataset
from hybrid import VehiclePipeline, decide
from metrics import (classification_report, detection_summary, ece, iou_array, localization_summary, nll,
                     reliability_bins, risk_coverage, selective_table)
from stages import ProbStore, Spec, compute_boxes, hybrid_specs, hybrid_probs, resolve


def conf_block(p, y, table, cfg):
    conf, correct = p.max(1), p.argmax(1) == y
    cov, acc = risk_coverage(conf, correct)
    idx = np.linspace(0, len(cov) - 1, 101).astype(int)
    dec = [decide(q, table, cfg.tau_exact, cfg.tau_make) for q in p]
    names = np.array([d[0] for d in dec])
    mk_ok = table.make_idx[p.argmax(1)] == table.make_idx[y]
    stats = {}
    for k in ("accept", "make_only", "reject"):
        m = names == k
        stats[k] = {"fraction": float(m.mean()), "exact_acc": float(correct[m].mean()) if m.any() else None,
                    "make_acc": float(mk_ok[m].mean()) if m.any() else None}
    return {"ece": ece(conf, correct), "nll": nll(p, y), "mean_confidence": float(conf.mean()),
            "reliability": reliability_bins(conf, correct), "threshold_table": selective_table(conf, correct),
            "risk_coverage": {"coverage": cov[idx].tolist(), "accuracy": acc[idx].tolist()},
            "decision_stats": stats}


def detect_only(det, recs):
    out = []
    for r in recs:
        im = load_bgr(r.path)
        out.append([(d.box, d.conf) for d in det.detect(im)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N test images (debug)")
    ap.add_argument("--detector", default=None)
    ap.add_argument("--timing-n", type=int, default=60)
    ap.add_argument("--compare-detectors", action="store_true", help="also evaluate the zero-shot COCO detector")
    args = ap.parse_args()
    ensure_dirs()

    data = load_dataset(args.data_root)
    test = data.test[:args.limit] if args.limit else data.test
    pipe = VehiclePipeline.load(detector_weights=args.detector)
    cfg, table = pipe.cfg, pipe.table
    print("hybrid config:", asdict(cfg))
    boxes = compute_boxes(pipe, test, "test", args.n_jobs, detector_tag=str(pipe.det.weights))
    store = ProbStore(pipe, test, boxes, args.n_jobs)
    y = store.labels
    gts = [b["gt"] for b in boxes]
    res: dict = {"dataset": data.info, "n_test": len(test), "config": asdict(cfg), "detector": str(pipe.det.weights),
                 "classifier": pipe.clf.meta["arch"], "device": str(pipe.clf.device)}

    # ---------------------------------------------------------------- localisation / detection
    loc = {}
    if all(g is not None for g in gts):
        def add(name, pred_boxes, preds):
            row = {**localization_summary(iou_array(pred_boxes, gts)), **detection_summary(preds, gts)}
            loc[name] = row
        add("Classical (edge + contour + GrabCut)", [b["classical"] for b in boxes], [[(b["classical"], 1.0)] for b in boxes])
        yb = [b["yolo"][0] if b["yolo"] else None for b in boxes]
        add("YOLO detector", yb, [b["yolo_all"] for b in boxes])
        loc["YOLO detector"]["detection_rate"] = float(np.mean([b is not None for b in yb]))
        add("YOLO + GrabCut refinement", [b["yolo_ref"] or (b["yolo"][0] if b["yolo"] else None) for b in boxes],
            [[(b["yolo_ref"], b["yolo"][1])] if b["yolo"] and b["yolo_ref"] else [] for b in boxes])
        add("Hybrid localiser (YOLO + refine + classical fallback)", [resolve(b, "yolo_ref_cls") for b in boxes],
            [[(resolve(b, "yolo_ref_cls"), b["yolo"][1] if b["yolo"] else 0.3)] for b in boxes])
        if args.compare_detectors and Path(FINETUNED_DETECTOR).exists():
            from deep_learning import VehicleDetector
            other = VehicleDetector(DEFAULT_DETECTOR if "finetuned" in str(pipe.det.weights) else str(FINETUNED_DETECTOR))
            preds = detect_only(other, test)
            tag = "YOLO zero-shot COCO" if "finetuned" in str(pipe.det.weights) else "YOLO fine-tuned"
            ob = [max(p, key=lambda t: t[1])[0] if p else None for p in preds]
            loc[tag] = {**localization_summary(iou_array(ob, gts)), **detection_summary(preds, gts)}
    res["localization"] = loc

    # ---------------------------------------------------------------- classification
    s_cl = Spec("classical", cfg.pad, False)
    s_dl = Spec("yolo_full", cfg.pad, False, False)
    s_dl_raw = Spec("yolo_full", cfg.pad, False, False, temp=1.0)
    s_gt = Spec("gt", cfg.pad, False, False)
    h_dl, h_cl = hybrid_specs(cfg)
    store.prefetch_dl([s_dl, s_dl_raw, s_gt] + h_dl)
    store.prefetch_cl([s_cl, s_gt] + h_cl)
    P = {"classical": store.cl(s_cl) if pipe.cclf is not None else None, "dl": store.dl(s_dl)}
    P["hybrid"], trig = hybrid_probs(cfg, store)
    P = {k: v for k, v in P.items() if v is not None}
    ub = {"dl_gt_box": store.dl(s_gt)}
    if pipe.cclf is not None:
        ub["classical_gt_box"] = store.cl(s_gt)
    res["second_pass_triggered_fraction"] = float(trig.mean())
    res["classification"] = {k: classification_report(v, y, table) for k, v in P.items()}
    res["upper_bounds_gt_box"] = {k: classification_report(v, y, table) for k, v in ub.items()}
    res["confidence"] = {k: conf_block(v, y, table, cfg) for k, v in P.items()}
    raw = store.dl(s_dl_raw)
    res["calibration_effect_dl"] = {"ece_T=1": ece(raw.max(1), raw.argmax(1) == y), "nll_T=1": nll(raw, y),
                                    "ece_calibrated": res["confidence"]["dl"]["ece"], "nll_calibrated": res["confidence"]["dl"]["nll"],
                                    "temperature": pipe.clf.temperature}

    # ---------------------------------------------------------------- viewpoint proxy (GT box aspect ratio)
    if all(g is not None for g in gts):
        ar = np.array([(g[2] - g[0]) / max(1, g[3] - g[1]) for g in gts])
        q = np.quantile(ar, [1 / 3, 2 / 3])
        bucket = np.digitize(ar, q)
        names = ["narrow boxes (front/rear/oblique views)", "medium", "wide boxes (side views)"]
        res["viewpoint_proxy"] = {names[b]: {"n": int((bucket == b).sum()), **{k: float((v.argmax(1) == y)[bucket == b].mean()) for k, v in P.items()}} for b in range(3)}

    # ---------------------------------------------------------------- confusions (hybrid)
    pred = P["hybrid"].argmax(1)
    cats = Counter()
    pairs = Counter()
    for t, p_ in zip(y, pred):
        if t == p_:
            continue
        if table.makemodel_idx[t] == table.makemodel_idx[p_]:
            cats["same make & model, different year"] += 1
        elif table.make_idx[t] == table.make_idx[p_]:
            cats["same make, different model"] += 1
        else:
            cats["different make"] += 1
        pairs[(int(t), int(p_))] += 1
    res["error_breakdown_hybrid"] = dict(cats)
    res["top_confusions_hybrid"] = [{"true": table.names[a], "pred": table.names[b], "count": c} for (a, b), c in pairs.most_common(15)]
    pc = np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(len(table))])
    order = np.argsort(np.nan_to_num(pc, nan=2))
    res["worst_classes_hybrid"] = [{"class": table.names[i], "acc": float(pc[i])} for i in order[:10] if not np.isnan(pc[i])]
    res["best_classes_hybrid"] = [{"class": table.names[i], "acc": float(pc[i])} for i in order[::-1][:5] if not np.isnan(pc[i])]

    # ---------------------------------------------------------------- per-image CSVs
    for k, v in P.items():
        with open(RESULTS / f"predictions_{k}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "pred", "p1", "correct", "decision", "iou", "top5"])
            for i, r in enumerate(test):
                p = v[i]
                src = {"classical": "classical", "dl": "yolo_full", "hybrid": "yolo_ref_cls"}[k]
                bx = resolve(boxes[i], src)
                iou = iou_array([bx], [gts[i]])[0] if gts[i] else ""
                dec = decide(p, table, cfg.tau_exact, cfg.tau_make)[0]
                w.writerow([r.path, int(y[i]), int(p.argmax()), f"{p.max():.4f}", int(p.argmax() == y[i]), dec,
                            f"{iou:.3f}" if iou != "" else "", " ".join(map(str, np.argsort(-p)[:5]))])
    np.save(RESULTS / "test_probs_hybrid.npy", P["hybrid"].astype(np.float16))

    # ---------------------------------------------------------------- efficiency
    def mb(p):
        return round(Path(p).stat().st_size / 1e6, 2) if Path(p).exists() else None
    n_par = sum(p.numel() for p in pipe.clf.model.parameters())
    eff = {"classifier_params_M": round(n_par / 1e6, 2), "classifier_file_mb": mb(CLASSIFIER_CKPT),
           "detector_file_mb": mb(pipe.det.weights) if Path(str(pipe.det.weights)).exists() else None,
           "classical_model_file_mb": mb(CLASSICAL_CKPT), "python": platform.python_version()}
    try:
        import torch
        eff["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only"
    except Exception:
        pass
    n = min(args.timing_n, len(test))
    stage = {}
    if n:
        imgs = [load_bgr(r.path) for r in test[:n + 3]]
        for mode in P:
            for im in imgs[:3]:
                pipe.run(im, mode, keep_crop=False)
            rows = [pipe.run(im, mode, keep_crop=False).timings for im in imgs[3:3 + n]]
            stage[mode] = {k: {"mean": float(np.mean([r[k] for r in rows])), "median": float(np.median([r[k] for r in rows]))} for k in rows[0]}
            stage[mode]["fps"] = 1000.0 / stage[mode]["total_ms"]["mean"]
    eff["latency_per_image_ms"] = stage
    res["efficiency"] = eff

    (RESULTS / "evaluation.json").write_text(json.dumps(res, indent=1))

    # ---------------------------------------------------------------- figures
    names = {"classical": "Classical", "dl": "Deep learning", "hybrid": "Hybrid"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(3)
    for j, (m, lab) in enumerate([("top1", "Top-1"), ("top5", "Top-5"), ("make_acc", "Make acc.")]):
        ax[0].bar(x + j * 0.27, [res["classification"][k][m] * 100 for k in P], 0.27, label=lab)
    ax[0].set_xticks(x + 0.27)
    ax[0].set_xticklabels([names[k] for k in P])
    ax[0].set_ylabel("%")
    ax[0].set_title("Classification on unseen test images")
    ax[0].legend()
    for k in P:
        c = res["confidence"][k]["risk_coverage"]
        ax[1].plot(np.array(c["coverage"]) * 100, np.array(c["accuracy"]) * 100, label=names[k])
    ax[1].set_xlabel("coverage (% of images answered)")
    ax[1].set_ylabel("accuracy on answered images (%)")
    ax[1].set_title("Risk-coverage (confidence thresholding)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "comparison.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    rb = [b for b in res["confidence"]["hybrid"]["reliability"] if b["n"]]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot([b["conf"] for b in rb], [b["acc"] for b in rb], "o-")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Hybrid reliability (ECE {res['confidence']['hybrid']['ece']:.3f})")
    fig.tight_layout()
    fig.savefig(FIGS / "reliability_hybrid.png", dpi=140)
    plt.close(fig)

    print(json.dumps({"classification": res["classification"], "localization": res["localization"]}, indent=1))
    print("saved", RESULTS / "evaluation.json")


if __name__ == "__main__":
    main()
