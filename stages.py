"""Cached, batched evaluation of pipeline stages.

Running the full ``VehiclePipeline.run`` image-by-image for every ablation variant would repeat the
expensive steps (decode, YOLO, GrabCut, CNN) dozens of times. Here every *stage output* is computed once and
cached; every variant is then a cheap numpy combination of cached arrays. The arithmetic is identical
to ``VehiclePipeline.run`` (same crops, same fusion, same second-pass rule).
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, replace

import numpy as np

from classical import classical_localize, feat_from_item, refine_box
from config import CACHE
from data import crop_box, load_bgr
from hybrid import HybridConfig, fuse


@dataclass(frozen=True)
class Spec:
    src: str          # box source: gt | full | classical | yolo_full | yolo_cls | yolo_ref_cls | yolo_ref_full
    pad: float
    illum: bool
    tta: bool = False
    temp: float | None = None      # None -> temperature stored in the checkpoint


def src_for(cfg: HybridConfig) -> str:
    if cfg.use_refine:
        return "yolo_ref_cls" if cfg.classical_fallback else "yolo_ref_full"
    return "yolo_cls" if cfg.classical_fallback else "yolo_full"


# ----------------------------------------------------------------------------- boxes
def _cls_box(path):
    return classical_localize(load_bgr(path))[0]


def _refine(path, box):
    return refine_box(load_bgr(path), box)[0]


def compute_boxes(pipe, recs, name: str, n_jobs: int = 1, detector_tag: str = "det", batch: int = 16) -> list[dict]:
    """Per-image dict: gt, full, classical, yolo=(box,conf)|None, yolo_all=[(box,conf)], yolo_ref."""
    from joblib import Parallel, delayed
    from tqdm import tqdm
    key = hashlib.md5(("|".join(r.path for r in recs) + detector_tag).encode()).hexdigest()[:10]
    f = CACHE / f"boxes_{name}_{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    paths = [r.path for r in recs]
    par = (lambda fn, it: Parallel(n_jobs=n_jobs, batch_size=8)(delayed(fn)(*a) for a in it)) if n_jobs != 1 else \
          (lambda fn, it: [fn(*a) for a in it])
    print(f"[{name}] classical localisation of {len(paths)} images ...", flush=True)
    cls_boxes = par(_cls_box, [(p,) for p in paths])
    out = []
    for i in tqdm(range(0, len(recs), batch), desc=f"YOLO[{name}]"):
        chunk = recs[i:i + batch]
        imgs = [load_bgr(r.path) for r in chunk]
        dets = pipe.det.detect_batch(imgs, batch=batch)
        for r, im, ds, cb in zip(chunk, imgs, dets, cls_boxes[i:i + batch]):
            h, w = im.shape[:2]
            b = pipe.det.best(ds, im.shape)
            out.append({"gt": r.box, "full": (0, 0, w, h), "size": (w, h), "classical": cb,
                        "yolo": (b.box, b.conf) if b else None,
                        "yolo_all": [(d.box, d.conf) for d in ds]})
    refined = par(_refine, [(p, e["yolo"][0]) for p, e in zip(paths, out) if e["yolo"]])
    it = iter(refined)
    for e in out:
        e["yolo_ref"] = next(it) if e["yolo"] else None
    f.write_bytes(pickle.dumps(out))
    return out


def resolve(e: dict, src: str):
    if src in ("gt", "full", "classical"):
        return e[src] if e[src] is not None else e["full"]
    y, yr = e["yolo"], e["yolo_ref"]
    if src == "yolo_full":
        return y[0] if y else e["full"]
    if src == "yolo_cls":
        return y[0] if y else e["classical"]
    if src == "yolo_ref_cls":
        return yr if yr else e["classical"]
    if src == "yolo_ref_full":
        return yr if yr else e["full"]
    raise KeyError(src)


# ----------------------------------------------------------------------------- probabilities
class ProbStore:
    """Memoised class-probability arrays for (box-source, pad, illumination, tta) specs."""

    def __init__(self, pipe, recs, boxes, n_jobs: int = 1, chunk: int = 96):
        self.pipe, self.recs, self.boxes, self.n_jobs, self.chunk = pipe, recs, boxes, n_jobs, chunk
        self.labels = np.array([r.label for r in recs])
        self._dl: dict[Spec, np.ndarray] = {}
        self._cl: dict[tuple, np.ndarray] = {}

    # -- deep learning: one pass over the images for every missing spec
    def prefetch_dl(self, specs):
        from tqdm import tqdm
        need = [s for s in dict.fromkeys(specs) if s not in self._dl]
        if not need:
            return
        C = len(self.pipe.table)
        buf = {s: np.zeros((len(self.recs), C), np.float32) for s in need}
        for i in tqdm(range(0, len(self.recs), self.chunk), desc=f"CNN x{len(need)} specs"):
            idx = range(i, min(i + self.chunk, len(self.recs)))
            imgs = [load_bgr(self.recs[j].path) for j in idx]
            for s in need:
                crops = [self.pipe.make_crop(im, resolve(self.boxes[j], s.src), s.pad, s.illum) for im, j in zip(imgs, idx)]
                buf[s][i:i + len(crops)] = self.pipe.clf.probs(crops, tta=s.tta, temperature=s.temp)
        self._dl.update(buf)

    def dl(self, spec: Spec) -> np.ndarray:
        self.prefetch_dl([spec])
        return self._dl[spec]

    # -- classical features -> probabilities
    def prefetch_cl(self, specs):
        need = [(s.src, s.pad, s.illum) for s in specs]
        need = [k for k in dict.fromkeys(need) if k not in self._cl]
        if not need or self.pipe.cclf is None:
            return
        from joblib import Parallel, delayed
        from tqdm import tqdm
        for k in need:
            src, pad, illum = k
            items = [(r.path, resolve(b, src), pad, illum) for r, b in zip(self.recs, self.boxes)]
            if self.n_jobs == 1:
                X = np.stack([feat_from_item(*a) for a in tqdm(items, desc=f"classical feats {k}")])
            else:
                X = np.stack(Parallel(n_jobs=self.n_jobs, batch_size=16)(delayed(feat_from_item)(*a) for a in items))
            self._cl[k] = self.pipe.cclf.predict_proba(X).astype(np.float32)

    def cl(self, spec: Spec) -> np.ndarray:
        self.prefetch_cl([spec])
        return self._cl[(spec.src, spec.pad, spec.illum)]


# ----------------------------------------------------------------------------- hybrid assembly
def hybrid_specs(cfg: HybridConfig):
    """(dl_specs, classical_specs) needed to evaluate a hybrid configuration."""
    src = src_for(cfg)
    main = Spec(src, cfg.pad, cfg.use_illum, cfg.use_tta)
    dl, cl = [main], []
    if cfg.use_fusion and cfg.fusion_alpha > 0:
        cl.append(main)
    if cfg.second_pass:
        for p in (0.0, cfg.pad * 2.5):
            v = Spec(src, p, True, True)
            dl.append(v)
            if cfg.use_fusion and cfg.fusion_alpha > 0:
                cl.append(v)
    return dl, cl


def hybrid_probs(cfg: HybridConfig, store: ProbStore) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised equivalent of VehiclePipeline.run(mode='hybrid'). Returns (probs, second_pass_mask)."""
    src = src_for(cfg)
    main = Spec(src, cfg.pad, cfg.use_illum, cfg.use_tta)
    use_f = cfg.use_fusion and cfg.fusion_alpha > 0 and store.pipe.cclf is not None
    p = store.dl(main)
    if use_f:
        p = fuse(p, store.cl(main), cfg.fusion_alpha)
    trig = np.zeros(len(p), bool)
    if cfg.second_pass:
        trig = p.max(1) < cfg.tau_exact
        if trig.any():
            acc = p.copy()
            for pd in (0.0, cfg.pad * 2.5):
                v = Spec(src, pd, True, True)
                pv = store.dl(v)
                if use_f:
                    pv = fuse(pv, store.cl(v), cfg.fusion_alpha)
                acc[trig] += pv[trig]
            p = np.where(trig[:, None], acc / 3.0, p)
    return p, trig


def variant(cfg: HybridConfig, **kw) -> HybridConfig:
    return replace(cfg, **kw)
