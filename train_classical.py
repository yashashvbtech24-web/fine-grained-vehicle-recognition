"""Fit the classical (HOG + colour + LBP + geometry -> PCA -> logistic regression) classifier.

Features for train/val/test crops (GT boxes + padding) are cached in work/cache so ablations are fast.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from classical import ClassicalClassifier, extract_many
from config import CACHE, CLASSICAL_CKPT, CROP_PAD, DEFAULT_DATA_ROOT, RESULTS, ensure_dirs
from data import load_dataset


def cached_features(name: str, recs, n_jobs: int, pad: float = CROP_PAD, use_gt: bool = True):
    f = CACHE / f"feats_{name}_{len(recs)}_{pad}.npy"
    if f.exists():
        return np.load(f)
    X = extract_many([(r.path, r.box if use_gt else None, pad, False) for r in recs], n_jobs=n_jobs, desc=f"features[{name}]")
    np.save(f, X)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--n-pca", type=int, default=384)
    args = ap.parse_args()
    ensure_dirs()
    data = load_dataset(args.data_root, limit_per_class=args.limit_per_class)
    Xtr, ytr = cached_features("train", data.train, args.n_jobs), np.array([r.label for r in data.train])
    Xva, yva = cached_features("val", data.val, args.n_jobs), np.array([r.label for r in data.val])

    best = (-1, None, None)
    for C in (0.05, 0.2, 1.0):                       # tiny grid, selected on the validation split
        clf = ClassicalClassifier(len(data.classes), n_pca=args.n_pca, C=C).fit(Xtr, ytr)
        p = clf.predict_proba(Xva)
        top1 = float((p.argmax(1) == yva).mean())
        top5 = float(np.mean([y in np.argsort(-q)[:5] for q, y in zip(p, yva)]))
        print(f"C={C}: val top1 {top1:.4f} top5 {top5:.4f}")
        if top1 > best[0]:
            best = (top1, C, clf)
    top1, C, clf = best
    clf.save(CLASSICAL_CKPT)
    import json
    (RESULTS / "classical_train.json").write_text(json.dumps({"best_C": C, "val_top1": top1, "n_pca": args.n_pca}, indent=1))
    print("saved", CLASSICAL_CKPT)


if __name__ == "__main__":
    main()
