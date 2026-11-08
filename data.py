"""Stanford Cars dataset access (no torch dependency).

Supported on-disk layouts (auto-detected under --data-root):
  1. Official:  cars_train/, cars_test/, devkit/{cars_meta,cars_train_annos,
                cars_test_annos_withlabels}.mat   (also nested Kaggle copies)
  2. CSV:       anno_train.csv, anno_test.csv, names.csv + cars_train/ cars_test/
  3. Folders:   train/<class name>/*.jpg, test/<class name>/*.jpg  (no boxes -> full image)

If the test annotations carry no labels (the original ``cars_test_annos.mat``), a stratified
20 % hold-out of the training set is used as the "unseen" test set and this is recorded in
``CarsData.info`` so the report can state it.
"""
from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MULTIWORD_MAKES = ["AM General", "Aston Martin", "Land Rover"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- class names
def parse_class_name(name: str) -> tuple[str, str, int | None]:
    """'Aston Martin V8 Vantage Coupe 2012' -> ('Aston Martin', 'V8 Vantage Coupe', 2012)."""
    toks = name.strip().split()
    year = None
    if toks and re.fullmatch(r"(19|20)\d\d", toks[-1]):
        year = int(toks[-1])
        toks = toks[:-1]
    rest = " ".join(toks)
    for mk in MULTIWORD_MAKES:
        if rest.lower().startswith(mk.lower() + " "):
            return mk, rest[len(mk):].strip(), year
    return (toks[0] if toks else ""), " ".join(toks[1:]), year


class ClassTable:
    """Make / model / year lookup for the list of class names."""

    def __init__(self, names: list[str]):
        self.names = list(names)
        parsed = [parse_class_name(n) for n in self.names]
        self.make = [p[0] for p in parsed]
        self.model = [p[1] for p in parsed]
        self.year = [p[2] for p in parsed]
        self.make_names = sorted(set(self.make), key=str.lower)
        m2i = {m: i for i, m in enumerate(self.make_names)}
        self.make_idx = np.array([m2i[m] for m in self.make], dtype=np.int64)
        mm = [f"{a}|{b}" for a, b in zip(self.make, self.model)]
        uniq = sorted(set(mm))
        mm2i = {m: i for i, m in enumerate(uniq)}
        self.makemodel_idx = np.array([mm2i[m] for m in mm], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.names)

    def describe(self, i: int) -> dict:
        return {"class_id": int(i), "name": self.names[i], "make": self.make[i],
                "model": self.model[i], "year": self.year[i]}

    def make_probs(self, probs: np.ndarray) -> np.ndarray:
        """Aggregate class probabilities to make level. probs: (C,) or (N,C)."""
        probs = np.atleast_2d(probs)
        out = np.zeros((probs.shape[0], len(self.make_names)), dtype=np.float64)
        for j in range(probs.shape[1]):
            out[:, self.make_idx[j]] += probs[:, j]
        return out


# --------------------------------------------------------------------------- records
@dataclass
class Rec:
    path: str
    box: tuple[int, int, int, int] | None     # (x1, y1, x2, y2) pixels or None
    label: int                                 # 0-based class id
    split: str = "train"


@dataclass
class CarsData:
    classes: list[str]
    train: list[Rec]
    val: list[Rec]
    test: list[Rec]
    info: dict = field(default_factory=dict)
    table: ClassTable | None = None

    def __post_init__(self):
        if self.table is None:
            self.table = ClassTable(self.classes)


# --------------------------------------------------------------------------- image IO
def load_bgr(path: str | Path) -> np.ndarray:
    """cv2.imread replacement that survives non-ASCII paths (Windows) and greyscale files."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot decode image: {path}")
    return img


def clamp_box(box, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = int(max(0, min(w - 2, math.floor(x1))))
    y1 = int(max(0, min(h - 2, math.floor(y1))))
    x2 = int(max(x1 + 2, min(w, math.ceil(x2))))
    y2 = int(max(y1 + 2, min(h, math.ceil(y2))))
    return x1, y1, x2, y2


def expand_box(box, pad: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return clamp_box((x1 - pad * bw, y1 - pad * bh, x2 + pad * bw, y2 + pad * bh), w, h)


def crop_box(img: np.ndarray, box, pad: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = expand_box(box, pad, w, h)
    return img[y1:y2, x1:x2]


# --------------------------------------------------------------------------- locating files
def _first(root: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]
    return None


def _img_dir(root: Path, name: str) -> Path | None:
    """Directory called `name` that directly contains images (handles nested Kaggle copies)."""
    cands = [p for p in root.rglob(name) if p.is_dir()]
    for c in sorted(cands, key=lambda p: len(p.parts), reverse=True):
        if any(f.suffix.lower() in IMG_EXT for f in list(c.iterdir())[:20]):
            return c
    return None


def _read_mat_annos(path: Path) -> list[dict]:
    from scipy.io import loadmat
    m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    out = []
    for a in np.atleast_1d(m["annotations"]):
        cls = getattr(a, "class", None) if hasattr(a, "class") else None
        out.append({"fname": str(a.fname),
                    "box": (int(a.bbox_x1), int(a.bbox_y1), int(a.bbox_x2), int(a.bbox_y2)),
                    "cls": int(cls) if cls is not None else None})
    return out


def _read_csv_annos(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if not r or not r[0].strip():
                continue
            try:
                rows.append({"fname": r[0].strip(), "box": tuple(int(float(v)) for v in r[1:5]),
                             "cls": int(float(r[5]))})
            except (ValueError, IndexError):
                continue                       # header line
    return rows


def _stratified_split(recs: list[Rec], frac: float, seed: int) -> tuple[list[Rec], list[Rec]]:
    rng = random.Random(seed)
    by: dict[int, list[Rec]] = {}
    for r in recs:
        by.setdefault(r.label, []).append(r)
    keep, held = [], []
    for lab in sorted(by):
        items = by[lab][:]
        rng.shuffle(items)
        k = max(1, int(round(frac * len(items)))) if len(items) > 1 else 0
        held += items[:k]
        keep += items[k:]
    return keep, held


def train_val_split(train: list[Rec], val_frac: float = 0.1, seed: int = 42):
    return _stratified_split(train, val_frac, seed)


def load_dataset(root: str | Path, val_frac: float = 0.1, seed: int = 42,
                 limit_per_class: int = 0) -> CarsData:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")
    info: dict = {"root": str(root)}

    meta = _first(root, ["cars_meta.mat"])
    names_csv = _first(root, ["names.csv"])
    tr_mat = _first(root, ["cars_train_annos.mat"])
    te_mat = _first(root, ["cars_test_annos_withlabels.mat"]) or _first(root, ["cars_test_annos.mat"])
    tr_csv = _first(root, ["anno_train.csv"])
    te_csv = _first(root, ["anno_test.csv"])

    classes: list[str] = []
    train: list[Rec] = []
    test: list[Rec] = []

    if meta is not None or names_csv is not None:
        if meta is not None:
            from scipy.io import loadmat
            cn = loadmat(str(meta), squeeze_me=True)["class_names"]
            classes = [str(c) for c in np.atleast_1d(cn)]
        else:
            with open(names_csv, newline="", encoding="utf-8") as f:
                classes = [r[0].strip() for r in csv.reader(f) if r]
        tr_dir, te_dir = _img_dir(root, "cars_train"), _img_dir(root, "cars_test")
        if tr_mat is not None:
            tr_a = _read_mat_annos(tr_mat)
            te_a = _read_mat_annos(te_mat) if te_mat is not None else []
            info["layout"] = "official .mat"
        elif tr_csv is not None:
            tr_a = _read_csv_annos(tr_csv)
            te_a = _read_csv_annos(te_csv) if te_csv is not None else []
            info["layout"] = "csv"
        else:
            raise FileNotFoundError("found class names but no train annotations (.mat or .csv)")
        if tr_dir is None:
            raise FileNotFoundError("could not find a 'cars_train' image directory")
        for a in tr_a:
            if a["cls"] is None:
                continue
            train.append(Rec(str(tr_dir / a["fname"]), a["box"], a["cls"] - 1, "train"))
        if te_dir is not None:
            for a in te_a:
                if a["cls"] is not None:
                    test.append(Rec(str(te_dir / a["fname"]), a["box"], a["cls"] - 1, "test"))
    else:                                       # folder-per-class layout
        tr_root = next((p for p in [root / "train", root / "car_data" / "train", root / "car_data" / "car_data" / "train"] if p.exists()), None)
        te_root = next((p for p in [root / "test", root / "car_data" / "test", root / "car_data" / "car_data" / "test"] if p.exists()), None)
        if tr_root is None:
            raise FileNotFoundError(f"no recognised Stanford Cars layout under {root}")
        classes = sorted(d.name for d in tr_root.iterdir() if d.is_dir())
        c2i = {c: i for i, c in enumerate(classes)}
        for split, base, out in (("train", tr_root, train), ("test", te_root, test)):
            if base is None:
                continue
            for d in sorted(base.iterdir()):
                if d.is_dir() and d.name in c2i:
                    for f in sorted(d.iterdir()):
                        if f.suffix.lower() in IMG_EXT:
                            out.append(Rec(str(f), None, c2i[d.name], split))
        info["layout"] = "class folders (no bounding boxes -> full-image 'boxes')"
        info["has_boxes"] = False

    info.setdefault("has_boxes", True)
    if not test:
        train, test = _stratified_split(train, 0.2, seed)
        for r in test:
            r.split = "test"
        info["test_source"] = "20% stratified hold-out of the training set (official test labels not found)"
    else:
        info["test_source"] = "official test split"

    train_fit, val = train_val_split(train, val_frac, seed)
    if limit_per_class:
        def cap(rs):
            cnt: dict[int, int] = {}
            out = []
            for r in rs:
                cnt[r.label] = cnt.get(r.label, 0) + 1
                if cnt[r.label] <= limit_per_class:
                    out.append(r)
            return out
        train_fit, val, test = cap(train_fit), cap(val), cap(test)
    info.update(n_classes=len(classes), n_train=len(train_fit), n_val=len(val), n_test=len(test))
    return CarsData(classes=classes, train=train_fit, val=val, test=test, info=info)
