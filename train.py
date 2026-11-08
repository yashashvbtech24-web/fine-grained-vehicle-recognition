"""Fine-tune a pre-trained timm backbone on Stanford Cars crops (196 classes).

    python train.py --data-root D:/datasets/stanford_cars --epochs 20

Select the checkpoint by validation top-1 (10 % stratified split of the training set; the test set is
never touched), then fit a temperature on the validation logits for calibrated confidences.
"""
from __future__ import annotations

import argparse
import csv
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CLASSIFIER_CKPT, DEFAULT_ARCH, DEFAULT_DATA_ROOT, DEFAULT_IMG_SIZE, RESULTS, SEED, ensure_dirs
from data import load_dataset
from deep_learning import CropDataset, build_classifier, model_norm, pick_device


def topk(logits: torch.Tensor, y: torch.Tensor, ks=(1, 5)):
    k = min(max(ks), logits.shape[1])
    pred = logits.topk(k, 1).indices
    ok = pred == y[:, None]
    return [ok[:, :min(kk, k)].any(1).float().sum().item() for kk in ks]


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    all_lg, all_y, loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True).to(memory_format=torch.channels_last), y.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            lg = model(x)
        loss += F.cross_entropy(lg.float(), y, reduction="sum").item()
        all_lg.append(lg.float().cpu())
        all_y.append(y.cpu())
    lg, y = torch.cat(all_lg), torch.cat(all_y)
    t1, t5 = topk(lg, y)
    return dict(loss=loss / len(y), top1=t1 / len(y), top5=t5 / len(y)), lg, y


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.detach().exp().clamp(0.05, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--arch", default=DEFAULT_ARCH)
    ap.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4, help="backbone LR (head uses 10x)")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--out", default=str(CLASSIFIER_CKPT))
    args = ap.parse_args()

    ensure_dirs()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    amp = device.type == "cuda"
    data = load_dataset(args.data_root, limit_per_class=args.limit_per_class)
    print("dataset:", data.info)

    model = build_classifier(args.arch, len(data.classes), pretrained=not args.no_pretrained, img_size=args.img_size)
    mean, std = model_norm(model)
    model = model.to(device).to(memory_format=torch.channels_last)

    tr = CropDataset(data.train, args.img_size, mean, std, train=True)
    va = CropDataset(data.val, args.img_size, mean, std, train=False)
    pw = args.workers > 0
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=amp, drop_last=True, persistent_workers=pw)
    vl = DataLoader(va, 64, shuffle=False, num_workers=args.workers, pin_memory=amp, persistent_workers=pw)

    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = [{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
              {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": args.lr * 10}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total = args.epochs * len(tl)
    warm = max(1, len(tl))
    base = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    log_path = RESULTS / "train_log.csv"
    best, best_state, t0 = -1.0, None, time.time()
    with open(log_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "train_loss", "val_loss", "val_top1", "val_top5", "seconds"])
        step = 0
        for ep in range(1, args.epochs + 1):
            model.train()
            run, n = 0.0, 0
            for x, y in tl:
                scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
                for g, b in zip(opt.param_groups, base):
                    g["lr"] = b * scale
                x, y = x.to(device, non_blocking=True).to(memory_format=torch.channels_last), y.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    loss = F.cross_entropy(model(x).float(), y, label_smoothing=args.label_smoothing)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                run += loss.item() * len(y)
                n += len(y)
                step += 1
            m, _, _ = evaluate(model, vl, device, amp)
            wr.writerow([ep, run / n, m["loss"], m["top1"], m["top5"], int(time.time() - t0)])
            f.flush()
            print(f"epoch {ep:02d}/{args.epochs}  train_loss {run / n:.3f}  val_loss {m['loss']:.3f}  "
                  f"val top1 {m['top1']:.4f}  top5 {m['top5']:.4f}  ({int(time.time() - t0)}s)", flush=True)
            if m["top1"] > best:
                best = m["top1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    m, val_lg, val_y = evaluate(model, vl, device, amp)
    T = fit_temperature(val_lg, val_y)
    print(f"best val top1 {m['top1']:.4f} | fitted temperature {T:.3f}")
    np.save(RESULTS / "val_logits.npy", val_lg.numpy())
    np.save(RESULTS / "val_labels.npy", val_y.numpy())
    torch.save({"model": best_state, "temperature": T,
                "meta": {"arch": args.arch, "img_size": args.img_size, "classes": data.classes, "mean": mean, "std": std,
                         "val_top1": m["top1"], "val_top5": m["top5"], "epochs": args.epochs,
                         "dataset_info": data.info, "pretrained": not args.no_pretrained}}, args.out)
    import json
    (RESULTS / "train_summary.json").write_text(json.dumps({
        "arch": args.arch, "img_size": args.img_size, "epochs": args.epochs, "batch_size": args.batch_size, "lr_backbone": args.lr,
        "lr_head": args.lr * 10, "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing, "optimizer": "AdamW",
        "schedule": "1 epoch linear warm-up + cosine decay", "mixed_precision": amp, "pretrained": not args.no_pretrained,
        "n_train": len(data.train), "n_val": len(data.val), "best_val_top1": m["top1"], "best_val_top5": m["top5"],
        "temperature": T, "train_seconds": int(time.time() - t0), "device": str(device)}, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()
