"""Failure analysis of the hybrid system on the unseen test split.

Errors are grouped into measurable categories. For every category the script reports how many errors it
explains and - crucially - the error RATE among images having the attribute versus the overall rate, so
the stated cause is backed by evidence rather than by anecdote. Galleries (image, GT vs prediction,
Grad-CAM of the predicted class) are written to work/results/figures/failure_*.jpg.
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap

import cv2
import numpy as np
from PIL import Image

from config import DEFAULT_DATA_ROOT, FIGS, RESULTS, ensure_dirs
from data import load_bgr, load_dataset
from deep_learning import gradcam
from hybrid import VehiclePipeline

CAUSES = {
    "same_model_other_year": "Consecutive model years / trims of one model differ only in small details (bumper, grille, lamp "
                             "shape, wheels). At 256 px those cues are a few pixels wide and the visual evidence is genuinely ambiguous.",
    "same_make_other_model": "Models of one manufacturer share a design language (grille, headlight signature) and body style; "
                             "the network matched the family but not the specific model.",
    "confident_cross_make": "High-confidence confusion between different makes: look-alike body styles/colours, or label noise. "
                            "Indicates over-confidence, which temperature scaling only partly corrects.",
    "poor_localisation": "The vehicle box missed part of the car or covered a different vehicle, so the crop lost "
                         "discriminative parts or contained distractors.",
    "difficult_illumination": "Very dark / washed-out images: colour and fine edge cues are lost or noisy; classical normalisation "
                              "recovers some but not all of the information.",
    "small_vehicle": "The vehicle covers a small image area, so after cropping and resizing few pixels carry the "
                     "discriminative detail.",
    "extreme_viewpoint": "Rare, strongly oblique or head-on/rear views (extreme box aspect ratios) are under-represented in training.",
}


def banner(img, lines, h=46):
    out = np.full((img.shape[0] + h, img.shape[1], 3), 24, np.uint8)
    out[:img.shape[0]] = img
    for i, ln in enumerate(lines[:2]):
        cv2.putText(out, ln[:46], (5, img.shape[0] + 17 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--per-category", type=int, default=4)
    ap.add_argument("--detector", default=None)
    args = ap.parse_args()
    ensure_dirs()
    data = load_dataset(args.data_root)
    table = data.table
    rows = list(csv.DictReader(open(RESULTS / "predictions_hybrid.csv")))
    test = {r.path: r for r in data.test}
    n = len(rows)
    lab = np.array([int(r["label"]) for r in rows])
    pred = np.array([int(r["pred"]) for r in rows])
    p1 = np.array([float(r["p1"]) for r in rows])
    iou = np.array([float(r["iou"]) if r["iou"] != "" else np.nan for r in rows])
    wrong = lab != pred

    lum = np.zeros(n)
    area = np.full(n, np.nan)
    ar = np.full(n, np.nan)
    for i, r in enumerate(rows):
        buf = np.fromfile(r["path"], np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_4)
        lum[i] = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].mean() / 255.0
        rec = test[r["path"]]
        if rec.box:
            with Image.open(r["path"]) as pil:
                w, h = pil.size
            area[i] = (rec.box[2] - rec.box[0]) * (rec.box[3] - rec.box[1]) / (w * h)
            ar[i] = (rec.box[2] - rec.box[0]) / max(1, rec.box[3] - rec.box[1])

    same_mm = table.makemodel_idx[lab] == table.makemodel_idx[pred]
    same_make = table.make_idx[lab] == table.make_idx[pred]
    tau = json.loads((RESULTS / "evaluation.json").read_text())["config"]["tau_exact"]
    arq = np.nanquantile(ar, [0.05, 0.95]) if np.isfinite(ar).any() else (0, 1e9)
    cats = {
        "same_model_other_year": (wrong & same_mm, "errors where make+model are right but the year/trim is wrong", np.ones(n, bool)),
        "same_make_other_model": (wrong & same_make & ~same_mm, "errors within the right make but wrong model", np.ones(n, bool)),
        "confident_cross_make": (wrong & ~same_make & (p1 >= tau), f"wrong make with confidence >= tau_exact ({tau:.2f})", np.ones(n, bool)),
        "poor_localisation": (wrong & (iou < 0.5), "errors with box IoU < 0.5", iou < 0.5),
        "difficult_illumination": (wrong & ((lum < 0.30) | (lum > 0.75)), "errors on dark (L<0.30) or bright (L>0.75) images", (lum < 0.30) | (lum > 0.75)),
        "small_vehicle": (wrong & (area < 0.08), "errors where the car covers < 8 % of the image", area < 0.08),
        "extreme_viewpoint": (wrong & ((ar < arq[0]) | (ar > arq[1])), "errors with extreme box aspect ratio (outer 5 % tails)", (ar < arq[0]) | (ar > arq[1])),
    }
    overall_err = float(wrong.mean())
    out = {"n_test": n, "overall_error_rate": overall_err, "categories": []}
    pipe = VehiclePipeline.load(detector_weights=args.detector)

    for key, (mask, desc, group) in cats.items():
        if mask.sum() == 0:
            continue
        group = group & np.isfinite(group.astype(float))
        entry = {"key": key, "description": desc, "n_errors": int(mask.sum()), "share_of_all_errors": float(mask.sum() / max(1, wrong.sum())),
                 "error_rate_in_group": float(wrong[group].mean()) if group.sum() >= 5 and key not in cats_only_errors else None,
                 "group_size": int(group.sum()), "overall_error_rate": overall_err, "likely_cause": CAUSES[key], "examples": []}
        idxs = np.where(mask)[0]
        idxs = idxs[np.argsort(-p1[idxs])][:args.per_category]          # most confident errors first
        tiles = []
        for i in idxs:
            r = rows[i]
            img = load_bgr(r["path"])
            pr = pipe.run(img, "hybrid")
            gt_box = test[r["path"]].box
            vis = img.copy()
            th = max(2, img.shape[1] // 250)
            if gt_box:
                cv2.rectangle(vis, gt_box[:2], gt_box[2:], (0, 200, 0), th)
            cv2.rectangle(vis, pr.box[:2], pr.box[2:], (0, 0, 230), th)
            big = cv2.resize(vis, (400, 300))
            cam = gradcam(pipe.clf, pr.extra["crop"], int(pred[i])) if pipe.clf is not None else None
            crop = cv2.cvtColor(pr.extra["crop"], cv2.COLOR_RGB2BGR)
            if cam is not None:
                heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
                crop = cv2.addWeighted(crop, 0.55, heat, 0.45, 0)
            crop = cv2.resize(crop, (400, 300))
            t = np.vstack([banner(big, [f"GT : {table.names[lab[i]]}", f"Pred: {table.names[pred[i]]} {p1[i]:.0%}"]), crop])
            tiles.append(t)
            entry["examples"].append({"path": r["path"], "gt": table.names[lab[i]], "pred": table.names[pred[i]], "confidence": float(p1[i]),
                                      "iou": None if np.isnan(iou[i]) else float(iou[i]), "mean_luminance": float(lum[i]),
                                      "vehicle_area_fraction": None if np.isnan(area[i]) else float(area[i])})
        if tiles:
            g = f"failure_{key}.jpg"
            cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(FIGS / g))
            entry["gallery"] = g
        out["categories"].append(entry)
        print(f"{key:26s} errors={entry['n_errors']:4d}  group_err_rate={entry['error_rate_in_group']}")
    (RESULTS / "failure_analysis.json").write_text(json.dumps(out, indent=1))
    print("saved", RESULTS / "failure_analysis.json")


cats_only_errors = {"same_model_other_year", "same_make_other_model", "confident_cross_make"}   # defined by the error itself

if __name__ == "__main__":
    main()
