"""Central configuration: paths and defaults shared by every module."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("P5_WORK", ROOT / "work"))
MODELS = WORK / "models"
RESULTS = WORK / "results"
if not RESULTS.exists() and (ROOT / "results").exists():           # repository layout (results/ next to the code)
    RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
CACHE = WORK / "cache"

DEFAULT_DATA_ROOT = os.environ.get("P5_DATA", str(ROOT / "data" / "stanford_cars"))

# ---- models ---------------------------------------------------------------
DEFAULT_ARCH = "convnext_tiny.fb_in22k_ft_in1k"   # timm name, ImageNet-22k pre-trained
DEFAULT_IMG_SIZE = 256
DEFAULT_DETECTOR = "yolov8s.pt"                    # COCO pre-trained, auto-downloaded by ultralytics
FINETUNED_DETECTOR = MODELS / "detector_finetuned.pt"
CLASSIFIER_CKPT = MODELS / "classifier.pt"
CLASSICAL_CKPT = MODELS / "classical.joblib"
HYBRID_CFG = MODELS / "hybrid_config.json"

COCO_VEHICLE_IDS = (2, 5, 7)                       # car, bus, truck
CROP_PAD = 0.06                                    # context margin around a box (fraction of box size)
SEED = 42


def assemble_checkpoint() -> None:
    """Hosting helper: GitHub's web uploader refuses files over 25 MB, so the cloud copy of the classifier is stored as
    several parts (classifier.pt.part1, part2, ...). Join them once if classifier.pt itself is not there."""
    try:
        parts = sorted(MODELS.glob("classifier.pt.part*"), key=lambda p: int(p.name.rsplit("part", 1)[1]))
        if parts and not CLASSIFIER_CKPT.exists():
            with open(CLASSIFIER_CKPT, "wb") as out:
                for p in parts:
                    out.write(p.read_bytes())
    except Exception as e:                                           # noqa: BLE001
        print("could not assemble classifier checkpoint:", e)


assemble_checkpoint()


def ensure_dirs() -> None:
    for p in (MODELS, RESULTS, FIGS, CACHE):
        p.mkdir(parents=True, exist_ok=True)
