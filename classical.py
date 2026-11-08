"""Classical computer-vision components (no deep learning).

A. Image preprocessing & normalisation  -> normalize_illumination, resize_long
B. Edge detection & contour analysis     -> canny_auto, contour_candidates, best_vehicle_contour
C. Region-of-interest extraction         -> classical_localize, refine_box (GrabCut)
D. Colour / texture / geometric features -> ClassicalFeatureExtractor (HOG + HSV + LBP + shape)
E. Classical classifier                  -> ClassicalClassifier (PCA + multinomial logistic regression)
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from data import clamp_box, crop_box, load_bgr

FEAT_W, FEAT_H = 192, 128          # canonical crop size for feature extraction (landscape, like cars)


# ============================================================================ A. preprocessing
def resize_long(bgr: np.ndarray, long_side: int = 512):
    h, w = bgr.shape[:2]
    s = long_side / max(h, w)
    if abs(s - 1) < 1e-3:
        return bgr, 1.0
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp), s


def luminance_stats(bgr: np.ndarray) -> tuple[float, float]:
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255.0
    return float(L.mean()), float(L.std())


def normalize_illumination(bgr: np.ndarray, force: bool = False, low: float = 0.35,
                           high: float = 0.68, min_std: float = 0.14):
    """Adaptive illumination normalisation.

    * adaptive gamma pushes mean luminance towards 0.5 (only if the image is too dark/bright)
    * CLAHE on the L channel if global contrast is low or a gamma correction was applied
    * light bilateral denoise for very dark images (gamma amplifies sensor noise)
    Returns (image, info). Well-exposed images pass through unchanged (so it cannot hurt them).
    """
    mean, std = luminance_stats(bgr)
    out, gamma, clahe_on = bgr, 1.0, False
    if force or mean < low or mean > high:
        gamma = float(np.clip(np.log(0.5) / np.log(np.clip(mean, 0.02, 0.98)), 0.3, 2.2))
        lut = (np.power(np.arange(256) / 255.0, gamma) * 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
        if mean < 0.2:
            out = cv2.bilateralFilter(out, 5, 25, 5)
    if force or std < min_std or gamma != 1.0:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        clahe_on = True
    return out, {"mean_L": mean, "std_L": std, "gamma": gamma, "clahe": clahe_on, "applied": out is not bgr}


# ============================================================================ B. edges & contours
def canny_auto(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi = float(np.clip(t, 40, 220))
    return cv2.Canny(g, 0.5 * hi, hi)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def contour_candidates(edges: np.ndarray, min_area_frac: float = 0.04):
    h, w = edges.shape
    k = max(3, int(0.012 * max(h, w))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ker, iterations=2)
    closed = cv2.dilate(closed, ker, iterations=1)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area_frac * h * w]
    return cnts, closed


def best_vehicle_contour(cnts, edges: np.ndarray):
    """Score contours by size, centrality, car-like aspect ratio and edge richness."""
    h, w = edges.shape
    best, best_s = None, -1.0
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh / float(w * h)
        ar = bw / max(bh, 1)
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        centrality = 1 - min(1.0, float(np.hypot(cx - 0.5, cy - 0.55)) / 0.6)
        ar_score = float(np.exp(-((np.log(max(ar, 1e-3)) - np.log(1.9)) ** 2) / (2 * 0.5 ** 2)))
        density = float(edges[y:y + bh, x:x + bw].mean()) / 255.0
        s = 0.35 * min(area / 0.5, 1) + 0.30 * centrality + 0.20 * ar_score + 0.15 * min(density / 0.15, 1)
        if s > best_s:
            best, best_s = (x, y, x + bw, y + bh), s
    return best, best_s


# ============================================================================ C. ROI extraction
def grabcut_mask(bgr: np.ndarray, rect_box, iters: int = 3) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box(rect_box, w, h)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    cv2.setRNGSeed(0)
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8)


def _mask_box(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def iou_xyxy(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def classical_localize(bgr: np.ndarray, max_side: int = 384, use_grabcut: bool = True, debug: bool = False):
    """Detector-free vehicle localisation: illumination norm -> Canny -> morphology -> contour scoring
    -> GrabCut refinement. Returns (box_xyxy in original pixels, info)."""
    H, W = bgr.shape[:2]
    small, s = resize_long(bgr, max_side)
    norm, _ = normalize_illumination(small)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    edges = canny_auto(gray)
    cnts, closed = contour_candidates(edges)
    box, score = best_vehicle_contour(cnts, edges)
    h, w = small.shape[:2]
    method = "contour"
    if box is None:                                   # nothing found: central prior
        box, method = (int(.08 * w), int(.2 * h), int(.92 * w), int(.9 * h)), "center-prior"
    mask = None
    if use_grabcut:
        bw, bh = box[2] - box[0], box[3] - box[1]
        rect = clamp_box((box[0] - .04 * bw, box[1] - .04 * bh, box[2] + .04 * bw, box[3] + .04 * bh), w, h)
        mask = grabcut_mask(small, rect)
        if mask is not None:
            gb = _mask_box(mask)
            if gb and iou_xyxy(gb, box) > 0.45 and (gb[2] - gb[0]) * (gb[3] - gb[1]) > 0.08 * w * h:
                box, method = gb, method + "+grabcut"
    out_box = clamp_box([v / s for v in box], W, H)
    info = {"method": method, "contour_score": float(score), "n_contours": len(cnts)}
    if debug:
        info.update(norm=norm, gray=gray, edges=edges, closed=closed, contours=cnts, mask=mask, scale=s)
    return out_box, info


def refine_box(bgr: np.ndarray, box, margin: float = 0.06, max_side: int = 320):
    """Tighten/adjust a detector box with GrabCut on a down-scaled region. A safety guard keeps the
    original box unless the refined box is consistent with it (IoU >= 0.6, area ratio 0.7-1.2)."""
    H, W = bgr.shape[:2]
    box = clamp_box(box, W, H)
    rx1, ry1, rx2, ry2 = clamp_box((box[0] - 0.12 * (box[2] - box[0]), box[1] - 0.12 * (box[3] - box[1]),
                                    box[2] + 0.12 * (box[2] - box[0]), box[3] + 0.12 * (box[3] - box[1])), W, H)
    region = bgr[ry1:ry2, rx1:rx2]
    small, s = resize_long(region, max_side)
    bx = [(box[0] - rx1) * s, (box[1] - ry1) * s, (box[2] - rx1) * s, (box[3] - ry1) * s]
    bw, bh = bx[2] - bx[0], bx[3] - bx[1]
    rect = (bx[0] - margin * bw, bx[1] - margin * bh, bx[2] + margin * bw, bx[3] + margin * bh)
    mask = grabcut_mask(small, rect)
    if mask is None:
        return box, {"refined": False, "reason": "grabcut-failed"}
    gb = _mask_box(mask)
    if gb is None:
        return box, {"refined": False, "reason": "empty-mask"}
    new = clamp_box([gb[0] / s + rx1, gb[1] / s + ry1, gb[2] / s + rx1, gb[3] / s + ry1], W, H)
    ar = ((new[2] - new[0]) * (new[3] - new[1])) / max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    if iou_xyxy(new, box) < 0.6 or not (0.7 <= ar <= 1.2):
        return box, {"refined": False, "reason": "guard"}
    return new, {"refined": True}


# ============================================================================ D. features
_HOG_N = ((FEAT_W // 16) - 1) * ((FEAT_H // 16) - 1) * 4 * 9
GROUPS = {"hog": (0, _HOG_N), "color": (_HOG_N, _HOG_N + 128 + 12), "lbp": (_HOG_N + 140, _HOG_N + 200),
          "geometry": (_HOG_N + 200, _HOG_N + 200 + 16 + 14 + 3)}
FEAT_DIM = GROUPS["geometry"][1]


class ClassicalFeatureExtractor:
    """HOG (shape/edges) + HSV histogram & Lab statistics (colour) + grid LBP (texture)
    + edge-density grid, contour/Hu-moment and symmetry descriptors (geometry)."""

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop, _ = normalize_illumination(crop_bgr)
        img = cv2.resize(crop, (FEAT_W, FEAT_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f_hog = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                    block_norm="L2-Hys", feature_vector=True).astype(np.float32)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        hist = np.sqrt(hist / (hist.sum() + 1e-6))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32) / 255.0
        f_col = np.concatenate([hist, lab.mean(0), lab.std(0), hsv.reshape(-1, 3).mean(0) / 255.0,
                                hsv.reshape(-1, 3).std(0) / 255.0]).astype(np.float32)

        lbp = local_binary_pattern(gray, 8, 1, "uniform")
        cells = []
        for gy in range(2):
            for gx in range(3):
                c = lbp[gy * FEAT_H // 2:(gy + 1) * FEAT_H // 2, gx * FEAT_W // 3:(gx + 1) * FEAT_W // 3]
                h_, _ = np.histogram(c, bins=10, range=(0, 10))
                cells.append(h_ / (h_.sum() + 1e-6))
        f_lbp = np.concatenate(cells).astype(np.float32)

        edges = canny_auto(gray)
        grid = [edges[i * FEAT_H // 4:(i + 1) * FEAT_H // 4, j * FEAT_W // 4:(j + 1) * FEAT_W // 4].mean() / 255.0
                for i in range(4) for j in range(4)]
        cnts, closed = contour_candidates(edges, min_area_frac=0.02)
        geo = np.zeros(14, np.float32)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area, per = cv2.contourArea(c), cv2.arcLength(c, True)
            x, y, bw, bh = cv2.boundingRect(c)
            hull = cv2.contourArea(cv2.convexHull(c))
            hu = cv2.HuMoments(cv2.moments(c)).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
            geo[:7] = np.clip(hu / 10.0, -3, 3)
            geo[7:] = [area / (FEAT_W * FEAT_H), area / max(bw * bh, 1), area / max(hull, 1),
                       4 * np.pi * area / max(per * per, 1), bw / max(bh, 1) / 3.0, np.log1p(len(cnts)),
                       per / (2 * (FEAT_W + FEAT_H))]
        ef = edges.astype(np.float32).ravel()
        er = edges[:, ::-1].astype(np.float32).ravel()
        sym = float(np.corrcoef(ef, er)[0, 1]) if ef.std() > 0 and er.std() > 0 else 0.0
        top_bot = float(edges[:FEAT_H // 2].mean() / (edges[FEAT_H // 2:].mean() + 1e-3))
        f_geo = np.concatenate([np.array(grid, np.float32), geo,
                                np.array([0.0 if np.isnan(sym) else sym, min(top_bot, 5.0), edges.mean() / 255.0], np.float32)])
        return np.concatenate([f_hog, f_col, f_lbp, f_geo]).astype(np.float32)


_EXTRACTOR = ClassicalFeatureExtractor()


def feat_from_item(path: str, box, pad: float, illum_pre: bool = False) -> np.ndarray:
    """Top-level (picklable) helper for joblib: load image, crop, extract features."""
    img = load_bgr(path)
    if box is None:
        box = (0, 0, img.shape[1], img.shape[0])
    crop = crop_box(img, box, pad)
    if illum_pre:
        crop, _ = normalize_illumination(crop)
    return _EXTRACTOR.extract(crop)


def extract_many(items, n_jobs: int = 1, desc: str = "features") -> np.ndarray:
    """items: list of (path, box, pad, illum_pre)."""
    from joblib import Parallel, delayed
    from tqdm import tqdm
    if n_jobs == 1:
        return np.stack([feat_from_item(*it) for it in tqdm(items, desc=desc)])
    res = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(feat_from_item)(*it) for it in tqdm(items, desc=desc))
    return np.stack(res)


# ============================================================================ E. classifier
class ClassicalClassifier:
    """StandardScaler -> PCA -> multinomial logistic regression (gives class probabilities)."""

    def __init__(self, n_classes: int, n_pca: int = 384, C: float = 0.5, columns: slice | None = None):
        self.n_classes, self.n_pca, self.C, self.columns = n_classes, n_pca, C, columns
        self.pipe = None

    def _sel(self, X):
        return X[:, self.columns] if self.columns is not None else X

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = self._sel(X)
        k = int(min(self.n_pca, X.shape[1], X.shape[0] - 1))
        self.pipe = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                                  LogisticRegression(C=self.C, max_iter=300))
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(self._sel(X))
        p = self.pipe.predict_proba(X)
        full = np.full((p.shape[0], self.n_classes), 1e-9, dtype=np.float64)
        full[:, self.pipe.classes_] = p
        return full / full.sum(1, keepdims=True)

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "ClassicalClassifier":
        import joblib
        return joblib.load(path)
