"""Prepare the model files for free cloud hosting (Streamlit Community Cloud).

Usage:  python tools/make_cloud_models.py <models_dir> <output_dir>

* converts the classifier weights to half precision (112 MB -> about 56 MB; the accuracy is unchanged, see the
  optimisation table in the report),
* splits it into parts of at most 20 MB (GitHub's web uploader accepts files up to 25 MB),
* copies the classical model and the hybrid configuration.
config.py joins the parts again when the app starts.
"""
from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

import torch

PART_BYTES = 20 * 1024 * 1024


def main(models_dir: str, out_dir: str) -> None:
    src, dst = Path(models_dir), Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("classifier.pt*"):
        old.unlink()
    ck = torch.load(src / "classifier.pt", map_location="cpu", weights_only=False)
    ck["model"] = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in ck["model"].items()}
    buf = io.BytesIO()
    torch.save(ck, buf)
    data = buf.getvalue()
    n = 0
    for i in range(0, len(data), PART_BYTES):
        n += 1
        (dst / f"classifier.pt.part{n}").write_bytes(data[i:i + PART_BYTES])
    for name in ("classical.joblib", "hybrid_config.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    print(f"classifier: {len(data) / 1e6:.1f} MB in {n} parts -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
