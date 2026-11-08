"""Intermediate image-processing results for the application ("show intermediate results")."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from classical import (FEAT_H, FEAT_W, canny_auto, classical_localize, contour_candidates, best_vehicle_contour,
                       normalize_illumination, resize_long, sobel_magnitude)
from data import crop_box


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def hue_histogram_plot(bgr, w=420, h=170) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    hist /= hist.max() + 1e-6
    canvas = np.full((h, w, 3), 250, np.uint8)
    bw = w // 36
    for i, v in enumerate(hist):
        col = cv2.cvtColor(np.uint8([[[i * 5, 220, 230]]]), cv2.COLOR_HSV2RGB)[0, 0].tolist()
        cv2.rectangle(canvas, (i * bw, h - 24), ((i + 1) * bw - 1, int(h - 24 - v * (h - 40))), col, -1)
    cv2.putText(canvas, "hue histogram of vehicle region", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def intermediate_steps(bgr: np.ndarray, box=None, localizer_info: dict | None = None):
    """Return ([(title, rgb_image, caption)], feature_dict).

    `box` is the vehicle box finally used by the pipeline (detector / refined). Whole-image steps show the
    classical localiser's view; crop-level steps (HOG, LBP, colour) use `box`."""
    steps = []
    small, _ = resize_long(bgr, 512)
    steps.append(("1. Input", _rgb(small), f"{bgr.shape[1]}x{bgr.shape[0]} px"))

    norm, ill = normalize_illumination(small)
    cap = (f"mean L={ill['mean_L']:.2f}, std={ill['std_L']:.2f}; " +
           (f"gamma {ill['gamma']:.2f}{' + CLAHE' if ill['clahe'] else ''} applied" if ill["applied"] else "well exposed: unchanged"))
    steps.append(("2. Illumination normalisation", _rgb(norm), cap))

    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    steps.append(("3. Grey-scale", cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "luminance used for edges / texture"))
    edges = canny_auto(gray)
    steps.append(("4. Canny edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), "Otsu-adaptive thresholds"))
    steps.append(("5. Sobel gradient magnitude", cv2.cvtColor(sobel_magnitude(gray), cv2.COLOR_GRAY2RGB), "edge strength"))

    cnts, closed = contour_candidates(edges)
    best, score = best_vehicle_contour(cnts, edges)
    ov = small.copy()
    cv2.drawContours(ov, cnts, -1, (0, 200, 255), 2)
    if best:
        cv2.rectangle(ov, best[:2], best[2:], (0, 0, 255), 3)
    steps.append(("6. Contours + best vehicle candidate", _rgb(ov), f"{len(cnts)} candidate contours; red = best-scoring"))

    cbox, cinfo = classical_localize(bgr, debug=True)
    mask = cinfo.get("mask")
    ov2 = small.copy()
    if mask is not None:
        m = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        ov2[m > 0] = (0.55 * ov2[m > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    s = cinfo.get("scale", 1.0)
    sc = small.shape[1] / bgr.shape[1]
    cv2.rectangle(ov2, (int(cbox[0] * sc), int(cbox[1] * sc)), (int(cbox[2] * sc), int(cbox[3] * sc)), (255, 0, 0), 3)
    steps.append(("7. GrabCut ROI (classical localiser)", _rgb(ov2), f"method: {cinfo['method']}"))

    use = box if box is not None else cbox
    crop = crop_box(bgr, use, 0.0)
    steps.append(("8. Vehicle ROI used for classification", _rgb(crop), f"box {tuple(int(v) for v in use)}"))

    cn, _ = normalize_illumination(crop)
    img = cv2.resize(cn, (FEAT_W, FEAT_H), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, hv = hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True, feature_vector=True)
    hv = cv2.normalize(hv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    steps.append(("9. HOG (shape / gradient orientation)", cv2.cvtColor(cv2.resize(hv, (FEAT_W * 2, FEAT_H * 2)), cv2.COLOR_GRAY2RGB), "9 orientations, 16x16 cells"))
    lbp = local_binary_pattern(g, 8, 1, "uniform")
    lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    steps.append(("10. LBP texture map", cv2.cvtColor(cv2.resize(lbp, (FEAT_W * 2, FEAT_H * 2), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB), "uniform LBP (P=8, R=1)"))
    steps.append(("11. Colour (hue histogram)", hue_histogram_plot(crop), "HSV colour descriptor"))

    # geometric descriptors of the classical contour inside the vehicle ROI
    feats = {"illumination mean L": ill["mean_L"], "illumination std": ill["std_L"], "gamma applied": ill["gamma"],
             "ROI width x height": f"{use[2] - use[0]} x {use[3] - use[1]} px",
             "ROI aspect ratio (w/h)": (use[2] - use[0]) / max(1, use[3] - use[1]),
             "ROI area / image": (use[2] - use[0]) * (use[3] - use[1]) / float(bgr.shape[0] * bgr.shape[1]),
             "edge density (ROI)": float(canny_auto(g).mean() / 255.0)}
    ecn, _ = contour_candidates(canny_auto(g), 0.02)
    if ecn:
        c = max(ecn, key=cv2.contourArea)
        area, per = cv2.contourArea(c), cv2.arcLength(c, True)
        hull = cv2.contourArea(cv2.convexHull(c))
        x, y, bw, bh = cv2.boundingRect(c)
        feats.update({"contour solidity": area / max(hull, 1), "contour extent": area / max(bw * bh, 1),
                      "contour circularity": 4 * np.pi * area / max(per * per, 1), "contours in ROI": len(ecn)})
    return steps, feats
