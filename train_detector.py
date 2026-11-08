"""Fine-tune a YOLO detector on the Stanford Cars bounding boxes (single class 'car').

Transfer learning: starts from COCO weights (yolov8s.pt, auto-downloaded) and adapts the box
regression to the dataset's annotation style (tight boxes around the whole car), which usually
improves IoU / mAP over the zero-shot COCO detector.

    python train_detector.py --data-root D:/datasets/stanford_cars --epochs 25
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
from PIL import Image

from config import DEFAULT_DATA_ROOT, DEFAULT_DETECTOR, FINETUNED_DETECTOR, WORK, ensure_dirs
from data import load_dataset, load_bgr


def write_split(recs, out: Path, split: str):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(recs):
        if r.box is None:
            continue
        name = f"{Path(r.path).stem}_{i}"
        dst = out / "images" / split / f"{name}.jpg"
        if not dst.exists():
            if r.path.lower().endswith((".jpg", ".jpeg")):
                shutil.copy2(r.path, dst)
            else:
                cv2.imencode(".jpg", load_bgr(r.path))[1].tofile(str(dst))
        with Image.open(dst) as im:
            w, h = im.size
        x1, y1, x2, y2 = r.box
        (out / "labels" / split / f"{name}.txt").write_text(
            f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--weights", default=DEFAULT_DETECTOR)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-per-class", type=int, default=0)
    args = ap.parse_args()
    ensure_dirs()

    data = load_dataset(args.data_root, limit_per_class=args.limit_per_class)
    ds = WORK / "yolo_dataset"
    if ds.exists():
        shutil.rmtree(ds)
    write_split(data.train, ds, "train")
    write_split(data.val, ds, "val")          # validation split of TRAIN data; test images are never used
    (ds / "data.yaml").write_text(f"path: {ds.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n")

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(data=str(ds / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, project=str((WORK / "yolo_runs").resolve()), name="cars", exist_ok=True,
                patience=8, seed=0, plots=False)
    # ask the trainer where it saved (ultralytics may re-root relative project paths)
    best = Path(getattr(model.trainer, "best", "") or "")
    if not best.exists():
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        best = Path(model.trainer.save_dir) / "weights" / "last.pt"
    shutil.copy2(best, FINETUNED_DETECTOR)
    print("fine-tuned detector saved to", FINETUNED_DETECTOR)


if __name__ == "__main__":
    main()
