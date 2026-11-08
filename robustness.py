"""Robustness evaluation under controlled degradations (illumination, blur / noise / compression,
resolution, partial occlusion / truncation, rotation / perspective as a viewpoint proxy).

Every condition is applied to the WHOLE test image (so detection is stressed too) and all three
approaches are run through the real pipeline (`VehiclePipeline.run`). Stanford Cars has no viewpoint
annotation, so viewpoint is probed with synthetic rotation / perspective warps; the aspect-ratio
based breakdown in evaluate.py is the natural-viewpoint companion.
"""
from __future__ import annotations

import argparse
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from classical import iou_xyxy
from config import DEFAULT_DATA_ROOT, FIGS, RESULTS, SEED, ensure_dirs
from data import load_bgr, load_dataset
from hybrid import VehiclePipeline, draw_prediction


# --------------------------------------------------------------------------- degradations
def _gamma(img, g, gain=1.0):
    x = (img.astype(np.float32) / 255.0) ** g * gain
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def _noise(img, sigma, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def low_light(img, box, rng):
    return _noise(_gamma(img, 2.2, 0.55), 6, rng), box


def very_low_light(img, box, rng):
    return _noise(_gamma(img, 3.0, 0.35), 9, rng), box


def overexposed(img, box, rng):
    return _gamma(img, 0.45, 1.15), box


def low_contrast(img, box, rng):
    return np.clip(img.astype(np.float32) * 0.4 + 90, 0, 255).astype(np.uint8), box


def blur(img, box, rng):
    s = 0.008 * max(img.shape[:2])
    return cv2.GaussianBlur(img, (0, 0), s), box


def motion_blur(img, box, rng):
    k = max(9, int(0.03 * max(img.shape[:2])) | 1)
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, ker), box


def gauss_noise(img, box, rng):
    return _noise(img, 25, rng), box


def jpeg10(img, box, rng):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 10])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), box


def _scale(img, box, f):
    h, w = img.shape[:2]
    out = cv2.resize(img, (max(16, int(w * f)), max(16, int(h * f))), interpolation=cv2.INTER_AREA)
    return out, (tuple(v * f for v in box) if box else None)


def res_half(img, box, rng):
    return _scale(img, box, 0.5)


def res_quarter(img, box, rng):
    return _scale(img, box, 0.25)


def _occlude(img, box, rng, frac):
    if box is None:
        return img, box
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    ar = rng.uniform(0.6, 1.6)
    ow = min(bw, np.sqrt(frac * bw * bh * ar))
    oh = min(bh, frac * bw * bh / max(ow, 1))
    ox = rng.uniform(x1, x2 - ow)
    oy = rng.uniform(y1, y2 - oh)
    out = img.copy()
    patch = rng.integers(60, 190, (1, 1, 3)) + rng.normal(0, 12, (int(oh), int(ow), 3))
    out[int(oy):int(oy) + int(oh), int(ox):int(ox) + int(ow)] = np.clip(patch, 0, 255).astype(np.uint8)
    return out, box


def occ25(img, box, rng):
    return _occlude(img, box, rng, 0.25)


def occ50(img, box, rng):
    return _occlude(img, box, rng, 0.50)


def truncated(img, box, rng):
    """Car cut by the image border: remove 25 % of the car width from one side."""
    if box is None:
        return img, box
    x1, y1, x2, y2 = [int(v) for v in box]
    cut = int(0.25 * (x2 - x1))
    if rng.random() < 0.5:
        return img[:, x1 + cut:].copy(), (0, y1, x2 - x1 - cut, y2)
    return img[:, :x2 - cut].copy(), (x1, y1, x2 - cut, y2)


def rotate(img, box, rng):
    h, w = img.shape[:2]
    ang = rng.choice([-15, 15])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE), None


def perspective(img, box, rng):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = rng.uniform(-0.10, 0.10, (4, 2)) * np.float32([w, h])
    M = cv2.getPerspectiveTransform(src, src + j.astype(np.float32))
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE), None


CONDITIONS = [("clean", lambda i, b, r: (i, b)), ("low light", low_light), ("very low light", very_low_light),
              ("overexposed", overexposed), ("low contrast", low_contrast), ("gaussian blur", blur),
              ("motion blur", motion_blur), ("gaussian noise", gauss_noise), ("JPEG q=10", jpeg10),
              ("resolution x0.5", res_half), ("resolution x0.25", res_quarter),
              ("occlusion 25%", occ25), ("occlusion 50%", occ50), ("truncated 25%", truncated),
              ("rotation +/-15", rotate), ("perspective warp", perspective)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--limit", type=int, default=500, help="random test subset size (all conditions x all methods)")
    ap.add_argument("--detector", default=None)
    args = ap.parse_args()
    ensure_dirs()
    data = load_dataset(args.data_root)
    pipe = VehiclePipeline.load(detector_weights=args.detector)
    rng0 = np.random.default_rng(SEED)
    idx = rng0.permutation(len(data.test))[:args.limit]
    recs = [data.test[i] for i in sorted(idx)]
    table = pipe.table
    modes = [m for m in ("classical", "dl", "hybrid") if (m != "classical" or pipe.cclf is not None)]

    res = {"n_images": len(recs), "seed": SEED, "conditions": {}}
    example = None
    for ci, (cname, fn) in enumerate(CONDITIONS):
        stats = {m: {"top1": [], "top5": [], "make": [], "iou": [], "conf": []} for m in modes}
        hy_dec = []
        for k, r in enumerate(tqdm(recs, desc=cname)):
            img = load_bgr(r.path)
            rng = np.random.default_rng(SEED * 1000 + ci * 100003 + k)
            img2, box2 = fn(img, r.box, rng)
            preds = pipe.run_all(img2)
            for m in modes:
                p = preds[m]
                order = [t["class_id"] for t in p.top]
                stats[m]["top1"].append(order[0] == r.label)
                stats[m]["top5"].append(r.label in order)
                stats[m]["make"].append(table.make_idx[order[0]] == table.make_idx[r.label])
                stats[m]["conf"].append(p.top[0]["p"])
                stats[m]["iou"].append(iou_xyxy(p.box, box2) if box2 is not None else np.nan)
            hy_dec.append((preds["hybrid"].decision, order[0] == r.label))
            if k == 0 and cname in ("clean", "very low light", "occlusion 50%", "resolution x0.25", "perspective warp", "overexposed"):
                example = example or []
                example.append((cname, draw_prediction(img2, preds["hybrid"])))
        cond = {}
        for m in modes:
            s = stats[m]
            cond[m] = {"top1": float(np.mean(s["top1"])), "top5": float(np.mean(s["top5"])), "make_acc": float(np.mean(s["make"])),
                       "mean_confidence": float(np.mean(s["conf"])),
                       "mean_iou": None if np.isnan(s["iou"]).all() else float(np.nanmean(s["iou"]))}
        acc = [ok for d, ok in hy_dec if d == "accept"]
        cond["hybrid_decisions"] = {"accept_rate": float(np.mean([d == "accept" for d, _ in hy_dec])),
                                    "accepted_acc": float(np.mean(acc)) if acc else None,
                                    "reject_rate": float(np.mean([d == "reject" for d, _ in hy_dec]))}
        res["conditions"][cname] = cond
        print(cname, {m: round(cond[m]["top1"], 3) for m in modes}, flush=True)
        (RESULTS / "robustness.json").write_text(json.dumps(res, indent=1))

    names = list(res["conditions"])
    fig, ax = plt.subplots(figsize=(13, 4.6))
    w = 0.27
    for j, m in enumerate(modes):
        ax.bar(np.arange(len(names)) + j * w, [res["conditions"][n][m]["top1"] * 100 for n in names], w,
               label={"classical": "Classical", "dl": "Deep learning", "hybrid": "Hybrid"}[m])
    ax.set_xticks(np.arange(len(names)) + w)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"Robustness (n={len(recs)} test images per condition)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "robustness.png", dpi=140)
    plt.close(fig)

    if example:
        tiles = [cv2.resize(im, (360, 270)) for _, im in example]
        for (n, _), t in zip(example, tiles):
            cv2.putText(t, n, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(t, n, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
        if rows:
            cv2.imencode(".jpg", np.vstack(rows))[1].tofile(str(FIGS / "robustness_examples.jpg"))
    print("saved", RESULTS / "robustness.json")


if __name__ == "__main__":
    main()
