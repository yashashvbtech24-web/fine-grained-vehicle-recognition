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


def _locate(name: str) -> Path:
    """MODELS/<name> if it exists, otherwise the first copy found anywhere in the repository (tolerates an uploaded
    folder layout that differs from the expected work/models/)."""
    default = MODELS / name
    if default.exists():
        return default
    try:
        return next(iter(ROOT.rglob(name)), default)
    except Exception:                                                # noqa: BLE001
        return default


CLASSICAL_CKPT = _locate("classical.joblib")
HYBRID_CFG = _locate("hybrid_config.json")

COCO_VEHICLE_IDS = (2, 5, 7)                       # car, bus, truck
CROP_PAD = 0.06                                    # context margin around a box (fraction of box size)
SEED = 42


def assemble_checkpoint() -> None:
    """Hosting helper: GitHub's web uploader refuses files over 25 MB, so the cloud copy of the classifier is stored as
    several parts (classifier.pt.part1, part2, ...). Join them once if classifier.pt itself is not there. If the
    repository folder is read-only, the joined file is written to the temp folder instead."""
    global CLASSIFIER_CKPT
    if CLASSIFIER_CKPT.exists():
        return
    try:
        found = list(MODELS.glob("classifier.pt.part*")) or list(ROOT.rglob("classifier.pt.part*"))
        parts = sorted(found, key=lambda p: int(p.name.rsplit("part", 1)[1]))
    except Exception as e:                                           # noqa: BLE001
        print("could not list classifier parts:", e)
        return
    if not parts:
        print("no classifier parts found in", MODELS)
        return
    import tempfile
    for target in (CLASSIFIER_CKPT, Path(tempfile.gettempdir()) / "p5_classifier.pt"):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as out:
                for p in parts:
                    out.write(p.read_bytes())
            CLASSIFIER_CKPT = target
            return
        except Exception as e:                                       # noqa: BLE001
            print(f"could not write {target}: {e}")


assemble_checkpoint()


def ensure_dirs() -> None:
    for p in (MODELS, RESULTS, FIGS, CACHE):
        p.mkdir(parents=True, exist_ok=True)
