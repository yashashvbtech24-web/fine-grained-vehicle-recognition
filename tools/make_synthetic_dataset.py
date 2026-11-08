"""Generate a tiny synthetic dataset in the *official Stanford Cars layout*.

It exists only so the whole pipeline (loader -> train -> evaluate -> app) can be smoke-tested
without the real data. Classes differ by body colour / proportions / stripes. Never report
results from this dataset.
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat

NAMES = [
    "Acme Roadster Coupe 2010", "Acme Roadster Coupe 2011", "Acme Roadster Sedan 2010",
    "Aston Martin V8 Vantage Coupe 2012", "Land Rover Rangey SUV 2012", "Rolls-Royce Phantom Sedan 2012",
    "Zeta Hatch Hatchback 2009", "Zeta Hatch Hatchback 2012", "Boxer Truck Crew Cab 2011",
    "Mercedes-Benz Tiny Convertible 1993", "smart fortwo Convertible 2012", "Orion GT Coupe 2012",
]


def draw_car(rng, k: int, size=(480, 360)):
    W, H = size
    bg = np.zeros((H, W, 3), np.uint8)
    c0, c1 = rng.integers(60, 200, 3), rng.integers(60, 200, 3)
    for y in range(H):
        bg[y] = (c0 * (1 - y / H) + c1 * (y / H)).astype(np.uint8)
    bg = np.clip(bg + rng.normal(0, 6, bg.shape), 0, 255).astype(np.uint8)
    bw = int(rng.integers(int(0.5 * W), int(0.8 * W)))
    bh = int(bw * (0.30 + 0.025 * (k % 4)))
    x1 = int(rng.integers(5, W - bw - 5))
    y1 = int(rng.integers(int(0.25 * H), H - bh - 12))
    hue = (k * 0.083) % 1.0
    col = tuple(int(255 * v) for v in colorsys.hsv_to_rgb(hue, 0.85, 0.9))[::-1]
    body = np.array([[x1, y1 + bh], [x1, y1 + int(bh * 0.55)], [x1 + int(bw * .2), y1 + int(bh * .45)],
                     [x1 + int(bw * .3), y1], [x1 + int(bw * (.55 + .03 * (k % 3))), y1],
                     [x1 + int(bw * .8), y1 + int(bh * .45)], [x1 + bw, y1 + int(bh * .55)], [x1 + bw, y1 + bh]])
    cv2.fillPoly(bg, [body], col)
    cv2.rectangle(bg, (x1 + int(bw * .33), y1 + 4), (x1 + int(bw * .52), y1 + int(bh * .4)), (225, 225, 235), -1)
    for s in range(k % 4):
        cv2.line(bg, (x1 + 4, y1 + int(bh * (.6 + .08 * s))), (x1 + bw - 4, y1 + int(bh * (.6 + .08 * s))), (250, 250, 250), 2)
    r = max(6, bh // 5)
    for cx in (x1 + int(bw * .22), x1 + int(bw * .78)):
        cv2.circle(bg, (cx, y1 + bh), r, (25, 25, 25), -1)
        cv2.circle(bg, (cx, y1 + bh), max(2, r // 2), (170, 170, 170), -1)
    return bg, (x1, y1, x1 + bw, y1 + bh + r)


def build(root: Path, n_train: int, n_test: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    (root / "devkit").mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("test", n_test)):
        d = root / f"cars_{split}"
        d.mkdir(exist_ok=True)
        rows = []
        i = 0
        for k in range(len(NAMES)):
            for _ in range(n):
                i += 1
                img, (x1, y1, x2, y2) = draw_car(rng, k)
                fn = f"{i:05d}.jpg"
                cv2.imwrite(str(d / fn), img)
                H, W = img.shape[:2]
                rows.append((x1, y1, min(x2, W), min(y2, H), k + 1, fn))
        dt = np.dtype([("bbox_x1", "O"), ("bbox_y1", "O"), ("bbox_x2", "O"), ("bbox_y2", "O"), ("class", "O"), ("fname", "O")])
        arr = np.empty((1, len(rows)), dtype=dt)
        for j, r in enumerate(rows):
            for f, v in zip(dt.names, r):
                arr[0, j][f] = np.array([[v]]) if f != "fname" else np.array([v])
        name = "cars_train_annos.mat" if split == "train" else "cars_test_annos_withlabels.mat"
        savemat(str(root / "devkit" / name), {"annotations": arr})
    cn = np.empty((1, len(NAMES)), dtype=object)
    for j, n in enumerate(NAMES):
        cn[0, j] = np.array([n])
    savemat(str(root / "devkit" / "cars_meta.mat"), {"class_names": cn})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/synthetic_cars")
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=10)
    a = ap.parse_args()
    build(Path(a.out), a.n_train, a.n_test)
    print("synthetic dataset written to", a.out)
