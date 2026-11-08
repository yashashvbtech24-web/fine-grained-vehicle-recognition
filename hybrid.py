"""Hybrid integration of classical CV and deep learning.

Pipeline (mode="hybrid")
------------------------
 1. YOLO detects the vehicle                         (deep learning)
    - if nothing is detected -> classical edge/contour/GrabCut localiser takes over   (classical fallback)
 2. GrabCut-based box refinement with a safety guard (classical)
 3. Crop with context padding + adaptive illumination normalisation (classical)
 4. CNN classification of the crop, flip test-time augmentation (deep learning)
 5. Late fusion with the classical HOG/colour/LBP/geometry classifier:
        log p = log p_cnn + alpha * log p_classical                 (alpha tuned on validation)
 6. Low-confidence handling:
        p_top1 >= tau_exact              -> ACCEPT      (make + model + year)
        else make-level mass >= tau_make -> MAKE_ONLY   (report the make, list candidate models/years)
        else                             -> REJECT      (ask for a better image, show top-5 suggestions)
    A second pass (multi-crop ensemble) is attempted before a prediction is downgraded.

Other modes: "dl" (YOLO -> crop -> CNN, nothing classical) and "classical" (classical localiser +
classical features only).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from classical import (ClassicalClassifier, ClassicalFeatureExtractor, classical_localize, normalize_illumination,
                       refine_box)
from config import CLASSICAL_CKPT, CLASSIFIER_CKPT, CROP_PAD, FINETUNED_DETECTOR, DEFAULT_DETECTOR, HYBRID_CFG
from data import ClassTable, crop_box, expand_box


@dataclass
class HybridConfig:
    pad: float = CROP_PAD
    use_refine: bool = True
    use_illum: bool = True
    use_tta: bool = True
    use_fusion: bool = True
    fusion_alpha: float = 0.3
    second_pass: bool = True
    classical_fallback: bool = True
    tau_exact: float = 0.5
    tau_make: float = 0.6

    def save(self, path=HYBRID_CFG):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path=HYBRID_CFG) -> "HybridConfig":
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text())
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        return cls()


@dataclass
class Prediction:
    mode: str
    box: tuple[int, int, int, int]
    det_conf: float | None
    probs: np.ndarray
    top: list[dict]                      # top-5 entries: class_id, name, make, model, year, p
    decision: str                        # accept | make_only | reject
    make: str
    make_prob: float
    message: str
    located_by: str
    timings: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def top1(self) -> dict:
        return self.top[0]


def softmax_logp(logp: np.ndarray) -> np.ndarray:
    logp = logp - logp.max(-1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(-1, keepdims=True)


def fuse(p_dl: np.ndarray, p_cl: np.ndarray, alpha: float) -> np.ndarray:
    """Product-of-experts fusion in log space (alpha = weight of the classical expert)."""
    if alpha <= 0:
        return p_dl
    return softmax_logp(np.log(p_dl + 1e-9) + alpha * np.log(p_cl + 1e-9))


def decide(probs: np.ndarray, table: ClassTable, tau_exact: float, tau_make: float):
    """Low-confidence handling. Works on a single (C,) probability vector."""
    mp = table.make_probs(probs)[0]
    mk = int(mp.argmax())
    p1 = float(probs.max())
    if p1 >= tau_exact:
        return "accept", table.make_names[mk], float(mp[mk])
    if mp[mk] >= tau_make:
        return "make_only", table.make_names[mk], float(mp[mk])
    return "reject", table.make_names[mk], float(mp[mk])


def explain(decision: str, top: list[dict], make: str, mp: float) -> str:
    t = top[0]
    if decision == "accept":
        return f"Confident: {t['make']} {t['model']} {t['year']} ({t['p']:.0%})."
    if decision == "make_only":
        return (f"Make is clear ({make}, {mp:.0%}) but model/year are uncertain "
                f"(best guess {t['model']} {t['year']} at {t['p']:.0%}). Candidates are listed below.")
    return f"Low confidence (best guess {t['make']} {t['model']} {t['year']} at {t['p']:.0%}). Try a clearer / closer photo."


class VehiclePipeline:
    def __init__(self, classifier=None, detector=None, classical: ClassicalClassifier | None = None,
                 cfg: HybridConfig | None = None, table: ClassTable | None = None):
        self.clf, self.det, self.cclf = classifier, detector, classical
        self.cfg = cfg or HybridConfig()
        self.table = table or (classifier.table if classifier is not None else None)
        self.extractor = ClassicalFeatureExtractor()

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, device: str | None = None, detector_weights: str | None = None, need_detector: bool = True):
        from deep_learning import FineGrainedClassifier, VehicleDetector
        clf = FineGrainedClassifier(CLASSIFIER_CKPT, device=device) if Path(CLASSIFIER_CKPT).exists() else None
        cc = ClassicalClassifier.load(CLASSICAL_CKPT) if Path(CLASSICAL_CKPT).exists() else None
        det = None
        if need_detector:
            w = detector_weights or (str(FINETUNED_DETECTOR) if Path(FINETUNED_DETECTOR).exists() else DEFAULT_DETECTOR)
            det = VehicleDetector(w, device=device)
        return cls(clf, det, cc, HybridConfig.load())

    # ------------------------------------------------------------------ primitives
    def detect(self, bgr: np.ndarray):
        if self.det is None:
            return None, []
        dets = self.det.detect(bgr)
        return self.det.best(dets, bgr.shape), dets

    def locate(self, bgr: np.ndarray, mode: str, shared: dict | None = None):
        """Returns (box, det_conf, located_by, cost_ms). `shared` caches detector / classical results across
        modes; the cost of every cached stage that a mode uses is still charged to that mode."""
        shared = shared if shared is not None else {}
        h, w = bgr.shape[:2]
        full = (0, 0, w, h)

        def classical_box():
            if "cls_box" not in shared:
                t = time.perf_counter()
                shared["cls_box"] = classical_localize(bgr)[0]
                shared["cls_ms"] = (time.perf_counter() - t) * 1000
            return shared["cls_box"], shared["cls_ms"]

        if mode == "classical":
            b, c = classical_box()
            return b, None, "classical edge/contour/GrabCut", c
        if "det" not in shared:
            t = time.perf_counter()
            shared["det"] = self.detect(bgr)[0]
            shared["det_ms"] = (time.perf_counter() - t) * 1000
        det, cost = shared["det"], shared["det_ms"]
        if det is None:
            if mode == "hybrid" and self.cfg.classical_fallback:
                b, c = classical_box()
                return b, None, "classical fallback (no detection)", cost + c
            return full, None, "full image (no detection)", cost
        box = det.box
        if mode == "hybrid" and self.cfg.use_refine:
            t = time.perf_counter()
            box, info = refine_box(bgr, box)
            cost += (time.perf_counter() - t) * 1000
            return box, det.conf, "YOLO + GrabCut refinement" if info["refined"] else "YOLO", cost
        return box, det.conf, "YOLO", cost

    def make_crop(self, bgr, box, pad, illum: bool) -> np.ndarray:
        c = crop_box(bgr, box, pad)
        if illum:
            c, _ = normalize_illumination(c)
        return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

    def classical_probs(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        X = np.stack([self.extractor.extract(cv2.cvtColor(c, cv2.COLOR_RGB2BGR)) for c in crops_rgb])
        return self.cclf.predict_proba(X)

    # ------------------------------------------------------------------ main entry point
    def run(self, bgr: np.ndarray, mode: str = "hybrid", shared: dict | None = None, keep_crop: bool = True) -> Prediction:
        cfg, table = self.cfg, self.table
        tm: dict = {}
        t0 = time.perf_counter()
        box, det_conf, by, tm["localize_ms"] = self.locate(bgr, mode, shared)
        extra: dict = {}

        t = time.perf_counter()
        if mode == "classical":
            crop = self.make_crop(bgr, box, cfg.pad, False)
            probs = self.classical_probs([crop])[0]
        elif mode == "dl":
            crop = self.make_crop(bgr, box, cfg.pad, False)
            probs = self.clf.probs([crop], tta=False)[0]
        else:
            crop = self.make_crop(bgr, box, cfg.pad, cfg.use_illum)
            probs = self.clf.probs([crop], tta=cfg.use_tta)[0]
            if cfg.use_fusion and self.cclf is not None and cfg.fusion_alpha > 0:
                p_cl = self.classical_probs([crop])[0]
                extra["classical_top1"] = int(p_cl.argmax())
                probs = fuse(probs, p_cl, cfg.fusion_alpha)
            if cfg.second_pass and probs.max() < cfg.tau_exact:
                views = [self.make_crop(bgr, box, p, True) for p in (0.0, cfg.pad * 2.5)]
                vp = self.clf.probs(views, tta=True)
                if cfg.use_fusion and self.cclf is not None and cfg.fusion_alpha > 0:
                    vp = np.stack([fuse(v, self.classical_probs([c])[0], cfg.fusion_alpha) for v, c in zip(vp, views)])
                probs = (probs + vp.sum(0)) / (1 + len(vp))
                extra["second_pass"] = True
        tm["classify_ms"] = (time.perf_counter() - t) * 1000
        tm["total_ms"] = tm["localize_ms"] + tm["classify_ms"]

        order = np.argsort(-probs)[:5]
        top = [{**table.describe(int(i)), "p": float(probs[i])} for i in order]
        decision, make, mp = decide(probs, table, cfg.tau_exact, cfg.tau_make)
        if keep_crop:
            extra["crop"] = crop
        return Prediction(mode, tuple(int(v) for v in box), det_conf, probs, top, decision, make, mp,
                          explain(decision, top, make, mp), by, tm, extra)

    def run_all(self, bgr: np.ndarray) -> dict[str, Prediction]:
        shared: dict = {}
        out = {}
        for m in ("classical", "dl", "hybrid"):
            if m == "classical" and self.cclf is None:
                continue
            if m != "classical" and self.clf is None:
                continue
            out[m] = self.run(bgr, m, shared)
        return out


# ============================================================================ evaluation helpers
def draw_prediction(bgr: np.ndarray, pred: Prediction, color=(0, 200, 0)) -> np.ndarray:
    img = bgr.copy()
    x1, y1, x2, y2 = pred.box
    col = {"accept": (0, 190, 0), "make_only": (0, 165, 255), "reject": (0, 0, 220)}[pred.decision]
    th = max(2, round(min(img.shape[:2]) / 220))
    cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
    t = pred.top1
    label = (f"{t['make']} {t['model']} {t['year']} {t['p']:.0%}" if pred.decision == "accept" else
             f"{pred.make} (model?) {pred.make_prob:.0%}" if pred.decision == "make_only" else f"uncertain {t['p']:.0%}")
    fs = max(0.5, min(img.shape[:2]) / 700)
    (tw, tht), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    ty = max(tht + 6, y1)
    cv2.rectangle(img, (x1, ty - tht - 8), (x1 + tw + 8, ty + 2), col, -1)
    cv2.putText(img, label, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return img
