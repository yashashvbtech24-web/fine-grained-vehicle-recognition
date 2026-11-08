"""Tune the hybrid pipeline's free parameters on the VALIDATION split only (never on test):

  * fusion weight alpha  (0 = classical expert ignored)
  * context padding
  * confidence thresholds tau_exact / tau_make for low-confidence handling
Writes work/models/hybrid_config.json and work/results/tuning.json.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from config import DEFAULT_DATA_ROOT, RESULTS, ensure_dirs
from data import load_dataset
from hybrid import HybridConfig, VehiclePipeline, decide
from metrics import find_threshold, nll
from stages import ProbStore, compute_boxes, hybrid_probs, hybrid_specs, variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--target-precision", type=float, default=0.90)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--detector", default=None)
    args = ap.parse_args()
    ensure_dirs()
    data = load_dataset(args.data_root, limit_per_class=args.limit_per_class)
    pipe = VehiclePipeline.load(detector_weights=args.detector)
    base = HybridConfig()
    boxes = compute_boxes(pipe, data.val, "val", args.n_jobs, detector_tag=str(pipe.det.weights))
    store = ProbStore(pipe, data.val, boxes, args.n_jobs)
    y = store.labels

    alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0] if pipe.cclf is not None else [0.0]
    pads = [0.0, 0.03, 0.06, 0.1]
    cfgs = [variant(base, fusion_alpha=a, pad=p, second_pass=False, use_fusion=a > 0) for p in pads for a in alphas]
    for c in cfgs:
        d, cl = hybrid_specs(c)
        store.prefetch_dl(d)
        store.prefetch_cl(cl)
    grid = []
    for c in cfgs:
        p, _ = hybrid_probs(c, store)
        grid.append({"pad": c.pad, "alpha": c.fusion_alpha, "val_nll": nll(p, y), "val_top1": float((p.argmax(1) == y).mean())})
    best = min(grid, key=lambda g: (round(g["val_nll"], 4), -g["val_top1"]))
    cfg = variant(base, pad=best["pad"], fusion_alpha=best["alpha"], use_fusion=best["alpha"] > 0)
    print("best pad/alpha on validation:", best)

    # thresholds: tau_exact on exact accuracy of the accepted set, tau_make on make-level accuracy of the rest
    d, cl = hybrid_specs(cfg)
    store.prefetch_dl(d)
    store.prefetch_cl(cl)
    p0, _ = hybrid_probs(variant(cfg, second_pass=False), store)
    tau_e = find_threshold(p0.max(1), p0.argmax(1) == y, args.target_precision)
    cfg = variant(cfg, tau_exact=float(tau_e))
    p, trig = hybrid_probs(cfg, store)
    conf, correct = p.max(1), p.argmax(1) == y
    tau_e = find_threshold(conf, correct, args.target_precision)
    cfg = variant(cfg, tau_exact=float(tau_e))
    rest = conf < cfg.tau_exact
    if rest.sum() >= 10:
        mp = pipe.table.make_probs(p[rest]).max(1)
        mk_ok = pipe.table.make_idx[p[rest].argmax(1)] == pipe.table.make_idx[y[rest]]
        tau_m = find_threshold(mp, mk_ok, args.target_precision, min_cov=0.2)
    else:
        tau_m = cfg.tau_make
    cfg = variant(cfg, tau_make=float(max(tau_m, 0.3)))
    cfg.save()
    (RESULTS / "tuning.json").write_text(json.dumps({"grid": grid, "best": best, "chosen": asdict(cfg),
                                                     "target_precision": args.target_precision,
                                                     "n_val": len(y)}, indent=1))
    dec = [decide(q, pipe.table, cfg.tau_exact, cfg.tau_make)[0] for q in p]
    print("chosen config:", asdict(cfg))
    print({k: dec.count(k) for k in ("accept", "make_only", "reject")}, "on validation")


if __name__ == "__main__":
    main()
