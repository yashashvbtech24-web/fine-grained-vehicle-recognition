"""Deep-learning components: YOLO vehicle detector + timm fine-grained classifier (transfer learning)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import COCO_VEHICLE_IDS, DEFAULT_DETECTOR
from data import ClassTable, clamp_box, crop_box, load_bgr


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ============================================================================ detector
@dataclass
class Det:
    box: tuple[int, int, int, int]
    conf: float
    cls: int = 2


class VehicleDetector:
    """Ultralytics YOLO wrapper. With COCO weights only vehicle classes (car/bus/truck) are kept;
    with a detector fine-tuned on Stanford Cars (single class) every detection is kept.

    ``predict_fn(bgr) -> list[Det]`` can be injected (unit tests / alternative detectors)."""

    def __init__(self, weights: str | Path = DEFAULT_DETECTOR, device: str | None = None, conf: float = 0.15,
                 iou: float = 0.5, imgsz: int = 640, predict_fn=None, pred_kwargs: dict | None = None):
        self.weights, self.conf, self.iou, self.imgsz = str(weights), conf, iou, imgsz
        self.device = pick_device(device)
        self.pred_kwargs = dict(pred_kwargs or {})      # e.g. {"quantize": 16} / {"half": True} (version dependent)
        self.predict_fn = predict_fn
        self.model = None
        self.single_class = False
        if predict_fn is None:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
            names = getattr(self.model, "names", {}) or {}
            self.single_class = len(names) <= 2

    def _parse(self, res, shape) -> list[Det]:
        h, w = shape[:2]
        out = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(xyxy, conf, cls):
            if not self.single_class and k not in COCO_VEHICLE_IDS:
                continue
            out.append(Det(clamp_box(b, w, h), float(c), int(k)))
        return sorted(out, key=lambda d: -d.conf)

    def detect(self, bgr: np.ndarray) -> list[Det]:
        if self.predict_fn is not None:
            return self.predict_fn(bgr)
        res = self.model.predict(bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=str(self.device),
                                 verbose=False, **self.pred_kwargs)[0]
        return self._parse(res, bgr.shape)

    def detect_batch(self, imgs: list[np.ndarray], batch: int = 16) -> list[list[Det]]:
        if self.predict_fn is not None:
            return [self.predict_fn(i) for i in imgs]
        out = []
        for i in range(0, len(imgs), batch):
            chunk = imgs[i:i + batch]
            res = self.model.predict(chunk, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=str(self.device),
                                     verbose=False, **self.pred_kwargs)
            out += [self._parse(r, im.shape) for r, im in zip(res, chunk)]
        return out

    @staticmethod
    def best(dets: list[Det], shape) -> Det | None:
        """Main subject = highest confidence x sqrt(area fraction) (the dataset has one dominant car)."""
        if not dets:
            return None
        h, w = shape[:2]
        return max(dets, key=lambda d: d.conf * np.sqrt(max(1e-6, (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) / (w * h))))


# ============================================================================ classifier
def build_classifier(arch: str, num_classes: int, pretrained: bool = True, drop_path: float = 0.1,
                     img_size: int | None = None):
    import timm
    kw: dict = dict(pretrained=pretrained, num_classes=num_classes)
    if img_size and any(k in arch for k in ("vit", "deit", "swin", "beit", "eva")):
        kw["img_size"] = img_size
    try:
        return timm.create_model(arch, drop_path_rate=drop_path, **kw)
    except TypeError:
        return timm.create_model(arch, **kw)


def model_norm(model) -> tuple[list[float], list[float]]:
    import timm
    cfg = timm.data.resolve_data_config({}, model=model)
    return list(cfg["mean"]), list(cfg["std"])


class FineGrainedClassifier:
    """Inference wrapper around a checkpoint written by train.py."""

    def __init__(self, ckpt_path: str | Path, device: str | None = None, half: bool | None = None, model=None):
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.meta = ck["meta"]
        self.classes: list[str] = self.meta["classes"]
        self.table = ClassTable(self.classes)
        self.size = int(self.meta["img_size"])
        self.device = pick_device(device)
        self.half = (self.device.type == "cuda") if half is None else half
        self.temperature = float(ck.get("temperature", 1.0))
        if model is None:
            model = build_classifier(self.meta["arch"], len(self.classes), pretrained=False, img_size=self.size)
            model.load_state_dict(ck["model"])
        self.model = model.eval().to(self.device)
        if self.half:
            self.model = self.model.half()
        self.mean = torch.tensor(self.meta["mean"], device=self.device).view(1, 3, 1, 1) * 255.0
        self.std = torch.tensor(self.meta["std"], device=self.device).view(1, 3, 1, 1) * 255.0

    def _to_tensor(self, crops_rgb: list[np.ndarray]) -> torch.Tensor:
        arr = np.stack([cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_AREA if min(c.shape[:2]) > self.size else cv2.INTER_CUBIC)
                        for c in crops_rgb])
        x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float()
        x = (x - self.mean) / self.std
        return x.half() if self.half else x

    @torch.inference_mode()
    def logits(self, crops_rgb: list[np.ndarray], tta: bool = False, batch: int = 64) -> np.ndarray:
        """Raw logits (N, C). With ``tta`` the logits of the image and its mirror are averaged."""
        out = []
        for i in range(0, len(crops_rgb), batch):
            x = self._to_tensor(crops_rgb[i:i + batch])
            lg = self.model(x).float()
            if tta:
                lg = 0.5 * (lg + self.model(torch.flip(x, dims=[3])).float())
            out.append(lg.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), np.float32)

    def probs(self, crops_rgb: list[np.ndarray], tta: bool = False, temperature: float | None = None) -> np.ndarray:
        T = self.temperature if temperature is None else temperature
        lg = self.logits(crops_rgb, tta) / max(T, 1e-3)
        lg = lg - lg.max(1, keepdims=True)
        e = np.exp(lg)
        return e / e.sum(1, keepdims=True)


def gradcam(clf: FineGrainedClassifier, crop_rgb: np.ndarray, class_idx: int | None = None) -> np.ndarray | None:
    """Grad-CAM heat-map (H,W in [0,1]) for CNN backbones exposing forward_features -> (B,C,H,W)."""
    try:
        m = clf.model.float() if clf.half else clf.model
        x = clf._to_tensor([crop_rgb]).float()
        with torch.enable_grad():
            x.requires_grad_(False)
            feats = m.forward_features(x)
            if feats.ndim != 4:
                return None
            feats.retain_grad()
            logits = m.forward_head(feats)
            c = int(logits.argmax(1)) if class_idx is None else class_idx
            m.zero_grad()
            logits[0, c].backward()
        w = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * feats).sum(1))[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (crop_rgb.shape[1], crop_rgb.shape[0]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    except Exception:                                      # noqa: BLE001  (ViT etc. -> no CAM)
        return None
    finally:
        if clf.half:
            clf.model.half()


# ============================================================================ training data
class RandomGamma:
    def __init__(self, p=0.3, lo=0.5, hi=2.2):
        self.p, self.lo, self.hi = p, lo, hi

    def __call__(self, img):
        import torchvision.transforms.functional as F
        if random.random() < self.p:
            return F.adjust_gamma(img, random.uniform(self.lo, self.hi))
        return img


class CropDataset(Dataset):
    """Vehicle crops from GT boxes. Training uses box jitter + random context padding so the classifier
    is robust to imperfect detector boxes, plus illumination / occlusion / blur augmentation."""

    def __init__(self, records, img_size: int, mean, std, train: bool, pad: float = 0.06, jitter: float = 0.08):
        from torchvision import transforms as T
        self.recs, self.size, self.train, self.pad, self.jitter = records, img_size, train, pad, jitter
        norm = T.Normalize(mean, std)
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop((img_size, img_size), scale=(0.7, 1.0), ratio=(0.8, 1.25), antialias=True),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.RandomRotation(8)], p=0.4),
                T.ColorJitter(0.4, 0.4, 0.3, 0.03),
                RandomGamma(),
                T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.15),
                T.ToTensor(), norm,
                T.RandomErasing(p=0.3, scale=(0.04, 0.25), value="random"),
            ])
        else:
            self.tf = T.Compose([T.Resize((img_size, img_size), antialias=True), T.ToTensor(), norm])

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        from PIL import Image
        r = self.recs[i]
        img = load_bgr(r.path)
        h, w = img.shape[:2]
        box = r.box or (0, 0, w, h)
        pad = self.pad
        if self.train:
            bw, bh = box[2] - box[0], box[3] - box[1]
            j = self.jitter
            box = clamp_box((box[0] + random.uniform(-j, j) * bw, box[1] + random.uniform(-j, j) * bh,
                             box[2] + random.uniform(-j, j) * bw, box[3] + random.uniform(-j, j) * bh), w, h)
            pad = random.uniform(0.0, 0.15)
        crop = cv2.cvtColor(crop_box(img, box, pad), cv2.COLOR_BGR2RGB)
        return self.tf(Image.fromarray(crop)), r.label
