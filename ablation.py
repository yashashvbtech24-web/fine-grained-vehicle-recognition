"""Ablation analysis on the unseen test split.

 1. Cumulative: deep-learning baseline -> + each classical/hybrid component, one at a time
 2. Leave-one-out from the full hybrid
 3. Box-source ablation (GT box / full image / classical / YOLO / YOLO+refine / fallback combos)
 4. Padding sweep and fusion-weight sweep
 5. Classical feature groups (HOG / colour / LBP / geometry) - which cues carry information
 6. Low-confidence policy (reject-only vs accept + make-level fallback)

Every number comes with a paired-bootstrap 95 % CI of the difference to the reference so that
differences within noise are not over-interpreted.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classical import GROUPS, ClassicalClassifier
from config import DEFAULT_DATA_ROOT, FIGS, RESULTS, ensure_dirs
from data import load_dataset
from hybrid import HybridConfig, VehiclePipeline, decide
from metrics import topk_acc
from stages import ProbStore, Spec, compute_boxes, hybrid_probs, hybrid_specs, variant
from train_classical import cached_features


def boot_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    """95 % CI of mean(a) - mean(b) for paired 0/1 vectors."""
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--detector", default=None)
    args = ap.parse_args()
    ensure_dirs()
    data = load_dataset(args.data_root)
    test = data.test[:args.limit] if args.limit else data.test
    pipe = VehiclePipeline.load(detector_weights=args.detector)
    full = pipe.cfg
    table = pipe.table
    boxes = compute_boxes(pipe, test, "test", args.n_jobs, detector_tag=str(pipe.det.weights))
    store = ProbStore(pipe, test, boxes, args.n_jobs)
    y = store.labels

    off = variant(full, use_refine=False, classical_fallback=False, use_illum=False, use_tta=False,
                  use_fusion=False, second_pass=False)
    steps = [
        ("DL baseline (YOLO box -> CNN)", off),
        ("+ classical-localiser fallback", variant(off, classical_fallback=True)),
        ("+ GrabCut box refinement", variant(off, classical_fallback=True, use_refine=True)),
        ("+ illumination normalisation", variant(off, classical_fallback=True, use_refine=True, use_illum=True)),
        ("+ flip test-time augmentation", variant(off, classical_fallback=True, use_refine=True, use_illum=True, use_tta=True)),
        (f"+ classical feature fusion (alpha={full.fusion_alpha:g})",
         variant(off, classical_fallback=True, use_refine=True, use_illum=True, use_tta=True, use_fusion=full.fusion_alpha > 0)),
        ("+ low-confidence second pass (= full hybrid)", full),
    ]
    loo = [("full hybrid", full),
           ("- classical fallback", variant(full, classical_fallback=False)),
           ("- GrabCut refinement", variant(full, use_refine=False)),
           ("- illumination normalisation", variant(full, use_illum=False)),
           ("- flip TTA", variant(full, use_tta=False)),
           ("- classical fusion", variant(full, use_fusion=False)),
           ("- second pass", variant(full, second_pass=False))]
    pad_cfgs = [(f"pad={p:g}", variant(full, pad=p)) for p in (0.0, 0.03, 0.06, 0.10, 0.15)]
    alpha_cfgs = [(f"alpha={a:g}", variant(full, fusion_alpha=a, use_fusion=a > 0)) for a in (0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)] \
        if pipe.cclf is not None else []
    src_specs = [("GT box (upper bound)", "gt"), ("full image (no localisation)", "full"),
                 ("classical localiser only", "classical"), ("YOLO (full-image fallback)", "yolo_full"),
                 ("YOLO + classical fallback", "yolo_cls"), ("YOLO + GrabCut + classical fallback", "yolo_ref_cls")]

    all_cfgs = [c for _, c in steps + loo + pad_cfgs + alpha_cfgs]
    d, cl = [], []
    for c in all_cfgs:
        a, b = hybrid_specs(c)
        d += a
        cl += b
    d += [Spec(s, full.pad, False, False) for _, s in src_specs]
    store.prefetch_dl(d)
    store.prefetch_cl(cl)

    def evalcfg(c):
        p, tr = hybrid_probs(c, store)
        return p, {**topk_acc(p, y), "second_pass_fraction": float(tr.mean())}

    res: dict = {"n_test": len(test), "full_config": asdict(full)}
    ref_p, _ = evalcfg(full)
    ref_ok = ref_p.argmax(1) == y

    def table_rows(items, ref_ok_vec, key_prev=False):
        rows, prev = [], None
        for name, c in items:
            p, m = evalcfg(c)
            ok = p.argmax(1) == y
            row = {"variant": name, **m}
            cmp = prev if key_prev and prev is not None else ref_ok_vec
            row["delta_top1_vs_" + ("previous" if key_prev and prev is not None else "reference")] = float(ok.mean() - cmp.mean())
            row["delta_ci95"] = boot_ci(ok, cmp)
            rows.append(row)
            prev = ok
        return rows

    base_ok = evalcfg(off)[0].argmax(1) == y
    res["cumulative"] = table_rows(steps, base_ok, key_prev=True)
    for (_, c), row in zip(steps, res["cumulative"]):          # every step also vs. the DL baseline (not only vs. the previous step)
        ok = evalcfg(c)[0].argmax(1) == y
        row["delta_top1_vs_baseline"] = float(ok.mean() - base_ok.mean())
        row["delta_ci95_vs_baseline"] = boot_ci(ok, base_ok)
    res["leave_one_out"] = table_rows(loo, ref_ok)
    res["padding_sweep"] = table_rows(pad_cfgs, ref_ok)
    res["alpha_sweep"] = table_rows(alpha_cfgs, ref_ok) if alpha_cfgs else []

    # box-source ablation (everything else minimal: no illumination, no TTA, no fusion)
    rows = []
    base_dl = store.dl(Spec("yolo_full", full.pad, False, False)).argmax(1) == y
    for name, s in src_specs:
        p = store.dl(Spec(s, full.pad, False, False))
        ok = p.argmax(1) == y
        rows.append({"box_source": name, **topk_acc(p, y), "delta_top1_vs_yolo_full": float(ok.mean() - base_dl.mean()),
                     "delta_ci95": boot_ci(ok, base_dl)})
    res["box_source"] = rows

    # classical feature groups (GT-box crops to isolate the information in each cue)
    if pipe.cclf is not None:
        ctr = json.loads((RESULTS / "classical_train.json").read_text()) if (RESULTS / "classical_train.json").exists() else {}
        Xtr = cached_features("train", data.train, args.n_jobs)
        ytr = np.array([r.label for r in data.train])
        Xte = cached_features("test", test, args.n_jobs)
        sl = lambda *g: np.r_[tuple(np.arange(*GROUPS[k]) for k in g)]
        groups = [("HOG only", ("hog",)), ("colour only", ("color",)), ("LBP texture only", ("lbp",)),
                  ("geometry only", ("geometry",)), ("HOG + colour", ("hog", "color")),
                  ("HOG + colour + LBP", ("hog", "color", "lbp")), ("all four groups", ("hog", "color", "lbp", "geometry"))]
        out = []
        for name, g in groups:
            clf = ClassicalClassifier(len(table), n_pca=ctr.get("n_pca", 384), C=ctr.get("best_C", 0.5), columns=sl(*g)).fit(Xtr, ytr)
            p = clf.predict_proba(Xte)
            out.append({"features": name, "dims": int(len(sl(*g))), **topk_acc(p, y)})
            print(out[-1], flush=True)
        res["classical_feature_groups"] = out

    # low-confidence policy
    p, _ = hybrid_probs(full, store)
    dec = np.array([decide(q, table, full.tau_exact, full.tau_make)[0] for q in p])
    ex_ok = p.argmax(1) == y
    mk_ok = table.make_idx[p.argmax(1)] == table.make_idx[y]
    acc_m, mo_m = dec == "accept", dec == "make_only"
    A_ans = acc_m
    B_ans = acc_m | mo_m
    A_wrong = (acc_m & ~ex_ok).sum()
    B_wrong = (acc_m & ~ex_ok).sum() + (mo_m & ~mk_ok).sum()
    res["low_confidence_policy"] = {
        "no_threshold": {"answer_rate": 1.0, "error_rate_among_answers": float(1 - ex_ok.mean())},
        "accept_or_reject": {"answer_rate": float(A_ans.mean()), "error_rate_among_answers": float(A_wrong / max(1, A_ans.sum()))},
        "accept_or_make_level_or_reject": {"answer_rate": float(B_ans.mean()), "error_rate_among_answers": float(B_wrong / max(1, B_ans.sum()))},
        "note": "make_only answers are judged at make level, accept answers at exact make+model+year level"}

    (RESULTS / "ablation.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    c = res["cumulative"]
    ax[0].barh([r["variant"] for r in c][::-1], [r["top1"] * 100 for r in c][::-1], color="#3b6ea5")
    lo = min(r["top1"] for r in c) * 100
    ax[0].set_xlim(max(0, lo - 3), min(100, max(r["top1"] for r in c) * 100 + 2))
    ax[0].set_xlabel("Top-1 accuracy (%)")
    ax[0].set_title("Cumulative ablation")
    if res.get("classical_feature_groups"):
        g = res["classical_feature_groups"]
        ax[1].barh([r["features"] for r in g][::-1], [r["top1"] * 100 for r in g][::-1], color="#a56b3b")
        ax[1].set_xlabel("Top-1 accuracy (%)")
        ax[1].set_title("Classical feature groups")
    fig.tight_layout()
    fig.savefig(FIGS / "ablation.png", dpi=140)
    print(json.dumps({k: res[k] for k in ("cumulative", "leave_one_out")}, indent=1)[:3500])
    print("saved", RESULTS / "ablation.json")


if __name__ == "__main__":
    main()
