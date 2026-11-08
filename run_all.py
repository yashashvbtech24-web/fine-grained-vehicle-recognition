"""Run the whole pipeline end to end.

    python run_all.py --data-root D:/datasets/stanford_cars            # everything
    python run_all.py --data-root ... --skip train_detector            # skip steps (comma/space separated)
    python run_all.py --data-root ... --only evaluate ablation         # only these steps

Steps run in dependency order; a failing step stops the run (rerun with --only to resume).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["train_detector", "train", "train_classical", "tune_hybrid", "evaluate", "ablation", "robustness",
         "optimize", "failure_analysis", "make_report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--team", default=None, help="team id (default: team_id in team.json)")
    ap.add_argument("--epochs", type=int, default=20, help="classifier epochs")
    ap.add_argument("--det-epochs", type=int, default=25, help="detector fine-tuning epochs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fast", action="store_true", help="time-saving preset: skip detector fine-tuning, 10 epochs, "
                    "robustness on 200 images, optimisation on 500 images")
    ap.add_argument("--print-only", action="store_true", help="print the commands without running them")
    args = ap.parse_args()
    if args.fast:
        args.skip = list(set(args.skip + ["train_detector"]))
        args.epochs = min(args.epochs, 10)
    steps = args.only or [s for s in STEPS if s not in args.skip]
    extra = {"train_detector": ["--epochs", str(args.det_epochs), "--workers", str(args.workers)],
             "train": ["--epochs", str(args.epochs), "--workers", str(args.workers)],
             "make_report": (["--team", args.team] if args.team else [])}
    if args.fast:
        extra["robustness"] = ["--limit", "200"]
        extra["optimize"] = ["--limit", "500"]
    no_root = {"make_report"}
    for s in steps:
        cmd = [sys.executable, str(ROOT / f"{s}.py")] + ([] if s in no_root else ["--data-root", args.data_root]) + extra.get(s, [])
        if args.print_only:
            print(" ".join(cmd))
            continue
        print("\n" + "=" * 78 + f"\n>>> {s}\n" + "=" * 78, flush=True)
        t = time.time()
        rc = subprocess.call(cmd, cwd=str(ROOT))
        print(f"<<< {s} finished in {(time.time() - t) / 60:.1f} min (exit code {rc})", flush=True)
        if rc != 0:
            sys.exit(rc)
    print("\nAll steps done. Report: work/eval_P5_<team id>.md   |   App: streamlit run app.py")


if __name__ == "__main__":
    main()
