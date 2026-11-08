"""End-to-end smoke test on a synthetic dataset (CPU, a few minutes).

    python tests/smoke_test.py            # run every step
    python tests/smoke_test.py evaluate   # run only the named steps (earlier artefacts must exist)

The detector is replaced by a mock (classical localiser, with 20 % deliberate misses) so the test needs
no internet / pretrained weights. It checks that every script runs and produces its outputs - it says
nothing about real accuracy.
"""
from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["P5_WORK"] = str(ROOT / "work" / "smoke")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import hashlib

import numpy as np

import deep_learning
from classical import classical_localize


def _mock_predict(bgr):
    h = int(hashlib.md5(bgr[::17, ::17].tobytes()).hexdigest(), 16)
    if h % 5 == 0:
        return []                                   # deliberate miss -> exercises the classical fallback
    box = classical_localize(bgr)[0]
    return [deep_learning.Det(tuple(int(v) for v in box), 0.6 + (h % 40) / 100.0, 2)]


class MockDetector(deep_learning.VehicleDetector):
    def __init__(self, weights="mock-detector", device=None, **kw):
        super().__init__(weights, device, predict_fn=_mock_predict)


deep_learning.VehicleDetector = MockDetector

DATA = str(ROOT / "work" / "smoke_data")
STEPS = {
    "data": lambda: __import__("make_synthetic_dataset").build(Path(DATA), 24, 10),
    "train": ("train.py", ["--data-root", DATA, "--arch", "resnet18", "--img-size", "64", "--epochs", "3",
                           "--batch-size", "16", "--workers", "0", "--no-pretrained"]),
    "train_classical": ("train_classical.py", ["--data-root", DATA, "--n-jobs", "2", "--n-pca", "64"]),
    "tune_hybrid": ("tune_hybrid.py", ["--data-root", DATA, "--n-jobs", "2"]),
    "evaluate": ("evaluate.py", ["--data-root", DATA, "--n-jobs", "2", "--timing-n", "8"]),
    "ablation": ("ablation.py", ["--data-root", DATA, "--n-jobs", "2"]),
    "robustness": ("robustness.py", ["--data-root", DATA, "--limit", "24"]),
    "optimize": ("optimize.py", ["--data-root", DATA, "--n-jobs", "2", "--limit", "60", "--no-detector-bench"]),
    "failure_analysis": ("failure_analysis.py", ["--data-root", DATA]),
    "make_report": ("make_report.py", ["--team", "TEST", "--out", str(ROOT / "work" / "smoke" / "eval_P5_TEST.md")]),
    "app": None,
}


def run(step):
    spec = STEPS[step]
    print(f"\n===== {step} =====", flush=True)
    if step == "data":
        spec()
    elif step == "app":
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        assert not at.exception, at.exception
        print("app rendered without exceptions; widgets:", len(at.sidebar.children) if hasattr(at, "sidebar") else "n/a")
    else:
        script, args = spec
        old = sys.argv
        sys.argv = [script] + args
        try:
            runpy.run_path(str(ROOT / script), run_name="__main__")
        finally:
            sys.argv = old


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STEPS)
    failed = []
    for s in wanted:
        try:
            run(s)
        except SystemExit as e:
            if e.code not in (0, None):
                failed.append(s)
        except Exception:                                # noqa: BLE001
            traceback.print_exc()
            failed.append(s)
            break                                        # later steps depend on earlier ones
    print("\nSMOKE TEST:", "FAILED at " + ", ".join(failed) if failed else "all requested steps passed")
    sys.exit(1 if failed else 0)
