"""Streamlit application:  streamlit run app.py

Upload a vehicle photo -> detection box -> make / model / year + confidence -> intermediate image-processing
results -> classical vs deep-learning vs hybrid comparison -> benchmark & analysis results.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from analysis import intermediate_steps
from config import CLASSICAL_CKPT, CLASSIFIER_CKPT, DEFAULT_DATA_ROOT, FIGS, FINETUNED_DETECTOR, RESULTS
from data import load_bgr
from hybrid import HybridConfig, VehiclePipeline, draw_prediction

st.set_page_config(page_title="Fine-grained vehicle recognition", layout="wide")
st.title("Intelligent fine-grained vehicle recognition")
st.caption("YOLO detection + CNN fine-grained classification + classical computer vision (Stanford Cars, 196 classes)")

BADGE = {"accept": ("Confident prediction", "success"), "make_only": ("Make identified - model/year uncertain", "warning"),
         "reject": ("Low confidence - no reliable prediction", "error")}


@st.cache_resource(show_spinner="Loading models ...")
def load_pipeline():
    warn = []
    try:
        pipe = VehiclePipeline.load()
    except Exception as e:                                           # noqa: BLE001  e.g. detector weights not downloadable
        warn.append(f"Detector could not be loaded ({str(e)[:140]}). Running without YOLO: the classical localiser is used instead.")
        pipe = VehiclePipeline.load(need_detector=False)
    return pipe, warn


def jload(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


pipe, warns = load_pipeline()
for w in warns:
    st.warning(w)
if pipe.clf is None:
    from config import MODELS
    st.error(f"No trained classifier found at `{CLASSIFIER_CKPT}`. Run `python train.py` first (see README).")
    st.caption("Files in the models folder: " + (", ".join(sorted(p.name for p in MODELS.glob("*"))) if MODELS.exists() else "(folder missing)"))
    st.stop()

# ----------------------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("Input")
up = sb.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png", "bmp", "webp"])
path_in = sb.text_input("...or path to an image on this machine", "")
sample = None
test_dir = Path(DEFAULT_DATA_ROOT)
cands = sorted(test_dir.rglob("cars_test/*.jpg"))[:60] if test_dir.exists() else []
if cands:
    pick = sb.selectbox("...or a dataset test image", ["-"] + [c.name for c in cands])
    sample = next((c for c in cands if c.name == pick), None)

sb.header("Hybrid settings")
cfg = HybridConfig(**vars(pipe.cfg))
cfg.use_refine = sb.checkbox("GrabCut box refinement", cfg.use_refine)
cfg.use_illum = sb.checkbox("Illumination normalisation", cfg.use_illum)
cfg.use_tta = sb.checkbox("Flip test-time augmentation", cfg.use_tta)
cfg.use_fusion = sb.checkbox("Fuse classical-feature classifier", cfg.use_fusion and pipe.cclf is not None, disabled=pipe.cclf is None)
cfg.fusion_alpha = sb.slider("Fusion weight alpha", 0.0, 1.0, float(cfg.fusion_alpha), 0.05)
cfg.second_pass = sb.checkbox("Second pass for low-confidence images", cfg.second_pass)
sb.header("Low-confidence handling")
cfg.tau_exact = sb.slider("Accept threshold (make+model+year)", 0.05, 0.99, float(min(max(cfg.tau_exact, 0.05), 0.99)), 0.01)
cfg.tau_make = sb.slider("Make-level threshold", 0.05, 0.99, float(min(max(cfg.tau_make, 0.05), 0.99)), 0.01)
pipe.cfg = cfg
sb.caption(f"Classifier: `{pipe.clf.meta['arch']}` on `{pipe.clf.device}`  |  detector: `{Path(str(pipe.det.weights)).name if pipe.det else 'none'}`")

img = None
src_name = ""
if up is not None:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    src_name = up.name
elif path_in.strip():
    try:
        clean_path = path_in.strip().strip('"').strip("'")
        img, src_name = load_bgr(clean_path), clean_path
    except Exception as e:                                           # noqa: BLE001
        st.error(f"Cannot read image: {e}")
elif sample is not None:
    img, src_name = load_bgr(sample), sample.name

tab_rec, tab_steps, tab_cmp, tab_bench, tab_about = st.tabs(
    ["Recognition", "Image-processing steps", "Compare approaches", "Benchmarks & analysis", "About"])

if img is None:
    with tab_rec:
        st.info("Upload a vehicle image in the sidebar to start. Benchmarks from the offline evaluation are in the "
                "'Benchmarks & analysis' tab.")
else:
    preds = pipe.run_all(img)
    hy = preds["hybrid"]
    with tab_rec:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.image(cv2.cvtColor(draw_prediction(img, hy), cv2.COLOR_BGR2RGB), caption=f"{src_name} - hybrid result ({hy.located_by})")
        with c2:
            label, kind = BADGE[hy.decision]
            getattr(st, kind)(label)
            t = hy.top1
            m1, m2, m3 = st.columns(3)
            m1.metric("Make", t["make"] if hy.decision != "reject" else hy.make + " ?")
            m2.metric("Model", t["model"] if hy.decision == "accept" else "uncertain")
            m3.metric("Year", str(t["year"]) if hy.decision == "accept" else "uncertain")
            st.metric("Confidence (exact class)", f"{t['p']:.1%}")
            st.write(hy.message)
            st.caption(f"Make-level confidence: {hy.make_prob:.1%}"
                       + (f" | detector confidence: {hy.det_conf:.2f}" if hy.det_conf is not None else " | no detector box used")
                       + (" | second pass used" if hy.extra.get("second_pass") else ""))
            df = pd.DataFrame([{"Rank": i + 1, "Make": r["make"], "Model": r["model"], "Year": r["year"], "Probability": r["p"]}
                               for i, r in enumerate(hy.top)])
            st.dataframe(df.style.format({"Probability": "{:.1%}"}), hide_index=True)
            st.bar_chart(pd.DataFrame({"p": [r["p"] for r in hy.top]}, index=[f"{r['make']} {r['model']} {r['year']}" for r in hy.top]))
            st.caption(f"Latency: localisation {hy.timings['localize_ms']:.0f} ms + classification {hy.timings['classify_ms']:.0f} ms")

    with tab_steps:
        steps, feats = intermediate_steps(img, hy.box)
        cols = st.columns(3)
        for i, (title, im, cap) in enumerate(steps):
            with cols[i % 3]:
                st.image(im, caption=f"{title} - {cap}")
        st.subheader("Geometric / photometric descriptors")
        st.dataframe(pd.DataFrame({"descriptor": list(feats), "value": [f"{v:.3f}" if isinstance(v, float) else str(v) for v in feats.values()]}),
                     hide_index=True)

    with tab_cmp:
        names = {"classical": "Classical (edges + contours + GrabCut -> HOG/colour/LBP -> logistic regression)",
                 "dl": "Deep learning (YOLO -> CNN)", "hybrid": "Hybrid (YOLO + classical refinement/normalisation + CNN + classical fusion)"}
        cols = st.columns(len(preds))
        rows = []
        for col, (m, p) in zip(cols, preds.items()):
            with col:
                st.image(cv2.cvtColor(draw_prediction(img, p), cv2.COLOR_BGR2RGB), caption=names[m])
            t = p.top1
            rows.append({"Approach": m, "Make": t["make"], "Model": t["model"], "Year": t["year"], "Confidence": t["p"],
                         "Decision": p.decision, "Localisation": p.located_by, "Time (ms)": round(p.timings["total_ms"], 1)})
        st.dataframe(pd.DataFrame(rows).style.format({"Confidence": "{:.1%}"}), hide_index=True)
        ev = jload("evaluation.json")
        if ev:
            st.subheader("Dataset-level comparison on unseen test images")
            c = ev["classification"]
            lat = ev["efficiency"].get("latency_per_image_ms", {})
            st.dataframe(pd.DataFrame([{"Approach": k, "Top-1": v["top1"], "Top-5": v["top5"], "Make acc.": v["make_acc"],
                                        "Make+model acc.": v["make_model_acc"],
                                        "ms / image": lat.get(k, {}).get("total_ms", {}).get("mean")} for k, v in c.items()]
                                      ).style.format({"Top-1": "{:.1%}", "Top-5": "{:.1%}", "Make acc.": "{:.1%}", "Make+model acc.": "{:.1%}", "ms / image": "{:.0f}"}),
                         hide_index=True)

with tab_bench:
    ev, ab, rb, op, fa = (jload(n) for n in ("evaluation.json", "ablation.json", "robustness.json", "optimization.json", "failure_analysis.json"))
    if not any([ev, ab, rb, op, fa]):
        st.info("No benchmark results yet. Run `python evaluate.py`, `ablation.py`, `robustness.py`, `optimize.py`, `failure_analysis.py`.")
    if ev:
        st.subheader("Classification (unseen test split)")
        st.caption(f"{ev['n_test']} images; test source: {ev['dataset'].get('test_source')}")
        st.dataframe(pd.DataFrame(ev["classification"]).T.style.format("{:.3f}"))
        st.subheader("Localisation / detection")
        st.dataframe(pd.DataFrame(ev["localization"]).T.style.format("{:.3f}"))
        if (FIGS / "comparison.png").exists():
            st.image(str(FIGS / "comparison.png"))
        st.subheader("Confidence & calibration")
        st.dataframe(pd.DataFrame({k: {"ECE": v["ece"], "NLL": v["nll"], "mean confidence": v["mean_confidence"]} for k, v in ev["confidence"].items()}).T.style.format("{:.3f}"))
        thr = pd.DataFrame(ev["confidence"]["hybrid"]["threshold_table"])
        st.dataframe(thr.style.format({"coverage": "{:.1%}", "accepted_acc": "{:.1%}", "rejected_frac": "{:.1%}"}), hide_index=True)
    if ab:
        st.subheader("Ablation")
        st.dataframe(pd.DataFrame(ab["cumulative"]).drop(columns=["delta_ci95"]).style.format({"top1": "{:.3f}", "top5": "{:.3f}"}), hide_index=True)
        if (FIGS / "ablation.png").exists():
            st.image(str(FIGS / "ablation.png"))
    if rb:
        st.subheader("Robustness")
        df = pd.DataFrame({c: {m: v["top1"] for m, v in d.items() if m in ("classical", "dl", "hybrid")} for c, d in rb["conditions"].items()}).T
        st.bar_chart(df)
        if (FIGS / "robustness_examples.jpg").exists():
            st.image(str(FIGS / "robustness_examples.jpg"), caption="Hybrid predictions under degradation")
    if op:
        st.subheader("Model optimisation")
        st.dataframe(pd.DataFrame(op["classifier"]), hide_index=True)
    if fa:
        st.subheader("Failure cases")
        for cat in fa["categories"]:
            with st.expander(f"{cat['key']} - {cat['n_errors']} errors ({cat['share_of_all_errors']:.0%} of all errors)"):
                st.write(cat["description"])
                if cat.get("error_rate_in_group") is not None:
                    st.write(f"Error rate in this group: **{cat['error_rate_in_group']:.1%}** vs overall {cat['overall_error_rate']:.1%}")
                st.write(cat["likely_cause"])
                if cat.get("gallery") and (FIGS / cat["gallery"]).exists():
                    st.image(str(FIGS / cat["gallery"]), caption="green = ground truth box, red = system box; bottom row: Grad-CAM of the predicted class")

with tab_about:
    st.markdown("""
**Pipeline** - (1) YOLO detects the vehicle; if it finds nothing a classical edge / contour / GrabCut localiser takes over.
(2) GrabCut refines the box (with a safety guard). (3) The crop is illumination-normalised (adaptive gamma + CLAHE, only when needed).
(4) A fine-tuned CNN classifies the crop (196 make/model/year classes) with flip test-time augmentation.
(5) Its probabilities are fused with a classical HOG + colour + LBP + geometry classifier (product of experts, weight tuned on validation data).
(6) Low-confidence handling: *accept* -> full label; *make only* -> manufacturer known, model/year uncertain; *reject* -> ask for a better image.

Thresholds and fusion weight were tuned on a validation split carved out of the training data - never on the test set.
""")
