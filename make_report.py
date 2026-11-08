"""Generate the written evaluation report (Markdown) from the measured result files.

    python make_report.py [--team <team id>]   ->  work/eval_P5_<team id>.md

Every number is read from work/results/*.json. Interpretation sentences are generated *from* those numbers
(and the paired-bootstrap CIs), so the text cannot claim an improvement that the data does not show.
Places that need human input are marked TODO; the script lists them at the end.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from config import FIGS, RESULTS, ROOT, WORK

NAMES = {"classical": "Classical", "dl": "Deep learning", "hybrid": "Hybrid",
         "dl_gt_box": "Deep learning (GT box)", "classical_gt_box": "Classical (GT box)"}


def J(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def pp(x, d=2):
    return f"{100 * x:+.{d}f} pp"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fig(name, caption, embed):
    p = FIGS / name
    if not p.exists():
        return f"*(figure `{name}` not generated yet)*"
    if embed:
        mime = "image/png" if name.endswith("png") else "image/jpeg"
        return f"![{caption}](data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()})\n\n*{caption}*"
    return f"![{caption}](figures/{name})\n\n*{caption}*"


def verdict(delta, ci):
    """Plain-language reading of a paired difference + its 95 % CI."""
    if ci is None:
        return ""
    if ci[0] > 0:
        return "a statistically reliable gain (95 % CI excludes 0)"
    if ci[1] < 0:
        return "a statistically reliable loss (95 % CI excludes 0)"
    return "not distinguishable from zero (95 % CI includes 0)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=None, help="team id; defaults to team_id in team.json")
    ap.add_argument("--prob", default="P5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--embed-images", action="store_true", help="inline figures as base64 (single-file upload)")
    ap.add_argument("--team-file", default=str(ROOT / "team.json"))
    args = ap.parse_args()
    team = json.loads(Path(args.team_file).read_text()) if Path(args.team_file).exists() else {"members": []}
    args.team = args.team or team.get("team_id") or "TODO_TEAM_ID"
    out_path = Path(args.out) if args.out else WORK / f"eval_{args.prob}_{args.team}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emb = args.embed_images

    ev, ab, rb, op, fa, tn, tr = (J(n) for n in ("evaluation.json", "ablation.json", "robustness.json", "optimization.json",
                                                "failure_analysis.json", "tuning.json", "train_summary.json"))
    L: list[str] = []
    add = L.append

    add(f"# {args.prob} - Intelligent Fine-Grained Vehicle Recognition and Analysis System\n")
    add(f"**Team ID:** {args.team}  |  **Problem ID:** {args.prob}  |  **Dataset:** Stanford Cars (196 classes)\n")
    if team["members"]:
        add(md_table(["Member", "USN", "Contribution"], [[m["name"], m["usn"], m["contribution"]] for m in team["members"]]) + "\n")

    # ------------------------------------------------------------------ 1
    add("## 1. Summary\n")
    if ev:
        c = ev["classification"]
        best = max(c, key=lambda k: c[k]["top1"])
        add(f"The system detects a vehicle with YOLO, refines and normalises the region with classical computer vision, classifies it into "
            f"196 make / model / year classes with a fine-tuned CNN (`{ev['classifier']}`), fuses the result with a classical "
            f"HOG + colour + LBP + geometry classifier and applies confidence-based handling of uncertain predictions. "
            f"On the {ev['n_test']} unseen test images the three approaches reach Top-1 accuracy of "
            + ", ".join(f"**{pct(v['top1'])}** ({NAMES[k]})" for k, v in c.items())
            + f"; Top-5 of " + ", ".join(f"{pct(v['top5'])}" for v in c.values()) + ". "
            f"The best approach in this run is **{NAMES[best]}**.\n")
    else:
        add("*Run `python evaluate.py` first - results are not available yet.*\n")

    # ------------------------------------------------------------------ 2
    add("## 2. Dataset and protocol\n")
    if ev:
        d = ev["dataset"]
        add(md_table(["Item", "Value"], [["Classes", d.get("n_classes")], ["Training images (fit)", d.get("n_train")],
                                          ["Validation images (carved from the official training set, 10 % stratified)", d.get("n_val")],
                                          ["Test images (unseen)", ev["n_test"]], ["Test split source", d.get("test_source")],
                                          ["Annotation layout", d.get("layout")]]) + "\n")
        if "hold-out" in str(d.get("test_source")):
            add("> **Note:** official test labels were not available, so a stratified 20 % hold-out of the training set was used as the unseen test set. "
                "Numbers are therefore not directly comparable with results on the official 8,041-image test split.\n")
    add("Protocol: model selection, temperature calibration, fusion weight, context padding and confidence thresholds were all tuned on the "
        "**validation** split. The test split was used once for the reported numbers. Classes are `Make Model Body-style Year`; make and "
        "year are parsed from the class name. Number-plate recognition, tracking and traffic events are not part of this task "
        "(not annotated in Stanford Cars).\n")

    # ------------------------------------------------------------------ 3
    add("## 3. Method\n")
    add("""```
 image -> YOLO vehicle detector --(no detection)--> classical localiser (Canny -> contours -> GrabCut)
              |
              v
   GrabCut box refinement (safety-guarded)           [classical]
              v
   crop + context padding + adaptive illumination normalisation (gamma + CLAHE)   [classical]
              v
   fine-tuned CNN (flip TTA) ------------------------+
              v                                      v
   classical HOG + colour + LBP + geometry -> PCA -> logistic regression
              \\______ product-of-experts fusion (weight tuned on validation) ______/
                                       v
   low-confidence handling:  ACCEPT (make+model+year) | MAKE-ONLY | REJECT   (+ second-pass multi-crop ensemble)
```
""")
    add(md_table(["Requirement", "Implementation"], [
        ["A. Pre-processing / normalisation", "`classical.normalize_illumination`: adaptive gamma + CLAHE only when exposure/contrast is poor; resize, bilateral denoise for very dark images"],
        ["A. Edge detection & contour analysis", "Otsu-adaptive Canny, Sobel, morphological closing, contour scoring (size, centrality, car-like aspect ratio, edge density)"],
        ["A. Colour & texture features", "HSV histogram + Lab/HSV statistics, grid LBP histograms, HOG"],
        ["A. ROI extraction / geometric analysis", "GrabCut mask & box, Hu moments, solidity, extent, circularity, symmetry, edge-density grid"],
        ["B. CNN / ViT fine-grained classifier", f"timm `{ev['classifier'] if ev else 'convnext_tiny'}`, ImageNet pre-trained, fine-tuned on 196 classes"],
        ["B. YOLO detector, transfer learning", "Ultralytics YOLO (COCO pre-trained), optionally fine-tuned on Stanford Cars boxes (`train_detector.py`)"],
        ["C. Hybrid integration", "Detector box -> classical refinement -> normalised crop -> CNN; classical fallback localiser; late fusion with classical classifier"],
        ["C. Low-confidence handling", "Three-way decision (accept / make-only / reject), thresholds tuned on validation data, second-pass ensemble"],
    ]) + "\n")
    if tr:
        add(f"**Training setup:** {tr['optimizer']}, backbone LR {tr['lr_backbone']:g} / head LR {tr['lr_head']:g}, weight decay {tr['weight_decay']:g}, "
            f"label smoothing {tr['label_smoothing']:g}, {tr['schedule']}, batch {tr['batch_size']}, {tr['epochs']} epochs, "
            f"{'mixed precision, ' if tr['mixed_precision'] else ''}input {tr['img_size']}x{tr['img_size']}, device `{tr['device']}` "
            f"({tr['train_seconds'] // 60} min). Augmentation: random resized crop, flip, small rotation, colour jitter, random gamma, blur, "
            f"random erasing (occlusion) and *box jitter + random padding* so the classifier tolerates imperfect detector boxes. "
            f"Best validation Top-1 {pct(tr['best_val_top1'])}, fitted temperature T = {tr['temperature']:.2f}.\n")
    else:
        add("*TODO: add training hyper-parameters (run `train.py` to generate `train_summary.json`).*\n")
    if tn:
        ch = tn["chosen"]
        add(f"**Hybrid parameters chosen on validation data:** fusion weight alpha = {ch['fusion_alpha']:g}, padding = {ch['pad']:g}, "
            f"accept threshold = {ch['tau_exact']:.2f}, make-level threshold = {ch['tau_make']:.2f} (target accepted-set accuracy "
            f"{pct(tn['target_precision'], 0)}).\n")

    # ------------------------------------------------------------------ 4
    if ev:
        c = ev["classification"]
        add("## 4. Results on unseen test images\n")
        add("### 4.1 Classification (Top-1 / Top-5 and hierarchical accuracy)\n")
        rows = [[NAMES[k], pct(v["top1"]), pct(v["top5"]), pct(v["make_model_acc"]), pct(v["make_acc"]), pct(v["year_only_error_rate"])]
                for k, v in c.items()]
        rows += [[NAMES[k] + " *(upper bound)*", pct(v["top1"]), pct(v["top5"]), pct(v["make_model_acc"]), pct(v["make_acc"]), pct(v["year_only_error_rate"])]
                 for k, v in ev["upper_bounds_gt_box"].items()]
        add(md_table(["Approach", "Top-1 (make+model+year)", "Top-5", "Make+model", "Make", "Year-only errors"], rows) + "\n")
        add("Upper-bound rows use the ground-truth box instead of a predicted one, isolating classification from localisation.\n")
        if "dl" in c and "hybrid" in c:
            dd = c["hybrid"]["top1"] - c["dl"]["top1"]
            last = (ab or {}).get("cumulative", [None])[-1]
            txt = f"Hybrid vs deep-learning-only Top-1: **{pp(dd)}**. "
            if last and "delta_ci95_vs_baseline" in last:
                lo, hi = last["delta_ci95_vs_baseline"]
                txt += (f"Paired bootstrap on the ablation run: {pp(last['delta_top1_vs_baseline'])}, 95 % CI [{100 * lo:+.2f}, {100 * hi:+.2f}] pp - "
                        f"{verdict(last['delta_top1_vs_baseline'], last['delta_ci95_vs_baseline'])}.")
            add(txt + "\n")
            if last and abs(dd - last["delta_top1_vs_baseline"]) > 1e-3:
                add("*TODO: evaluation.json and ablation.json disagree on Hybrid-vs-DL (different test subset or model file). Re-run both on the same checkpoint before submitting.*\n")
        if "classical" in c and "dl" in c:
            add(f"The classical-only pipeline reaches {pct(c['classical']['top1'])} Top-1, i.e. {pct(c['classical']['top1'] / max(1e-9, c['dl']['top1']), 0)} of the deep-learning result: hand-crafted global features cannot resolve "
                "fine-grained differences between 196 look-alike classes, which is why the classical branch is used for localisation, normalisation and as a secondary expert rather than as the main classifier.\n")
        add(fig("comparison.png", "Classification accuracy by approach and risk-coverage curves", emb) + "\n")

        add("### 4.2 Localisation and detection\n")
        rows = [[k, f"{v['mean_iou']:.3f}", f"{v['median_iou']:.3f}", pct(v["iou_ge_0.5"]), pct(v["iou_ge_0.75"]), f"{v['AP50']:.3f}", f"{v['AP75']:.3f}", f"{v['mAP50_95']:.3f}"]
                for k, v in ev["localization"].items()]
        add(md_table(["Localiser", "Mean IoU", "Median IoU", "IoU>=0.5", "IoU>=0.75", "AP@0.5", "AP@0.75", "mAP@[.5:.95]"], rows) + "\n")
        add("Stanford Cars has a single annotated car per image, so detection is evaluated as a single-class problem (mAP = AP of the class 'car'; "
            "COCO 101-point interpolation). Additional cars in the background count as false positives for the raw YOLO row.\n")

        add("### 4.3 Efficiency\n")
        e = ev["efficiency"]
        add(md_table(["Item", "Value"], [["Classifier parameters", f"{e['classifier_params_M']} M"], ["Classifier checkpoint", f"{e['classifier_file_mb']} MB"],
                                         ["Detector weights", f"{e.get('detector_file_mb')} MB"], ["Classical model", f"{e.get('classical_model_file_mb')} MB"],
                                         ["Hardware", e.get("gpu", "n/a")]]) + "\n")
        lat = e.get("latency_per_image_ms", {})
        if lat:
            add(md_table(["Approach", "Localise (ms)", "Classify (ms)", "Total (ms)", "FPS"],
                         [[NAMES[k], f"{v['localize_ms']['mean']:.1f}", f"{v['classify_ms']['mean']:.1f}", f"{v['total_ms']['mean']:.1f}", f"{v['fps']:.1f}"] for k, v in lat.items()]) + "\n")
            add("Latency is measured per image (batch 1, includes image-space classical processing; excludes JPEG decoding).\n")

        # ---------------------------------------------------------------- 5 ablation
    if ab:
        add("## 5. Ablation analysis\n")
        add("### 5.1 Cumulative contribution of each hybrid component\n")
        rows = [[r["variant"], pct(r["top1"]), pct(r["top5"]), pp(r.get("delta_top1_vs_previous", r.get("delta_top1_vs_reference", 0))),
                 f"[{100 * r['delta_ci95'][0]:+.2f}, {100 * r['delta_ci95'][1]:+.2f}]",
                 pp(r.get("delta_top1_vs_baseline", 0)),
                 f"[{100 * r['delta_ci95_vs_baseline'][0]:+.2f}, {100 * r['delta_ci95_vs_baseline'][1]:+.2f}]" if "delta_ci95_vs_baseline" in r else "n/a"]
                for r in ab["cumulative"]]
        add(md_table(["Configuration", "Top-1", "Top-5", "Delta vs previous step", "95 % CI (pp)", "Delta vs DL baseline", "95 % CI (pp)"], rows) + "\n")
        helpful = [r["variant"] for r in ab["cumulative"][1:] if r["delta_ci95"][0] > 0]
        harmful = [r["variant"] for r in ab["cumulative"][1:] if r["delta_ci95"][1] < 0]
        neutral = [r["variant"] for r in ab["cumulative"][1:] if r["delta_ci95"][0] <= 0 <= r["delta_ci95"][1]]
        add("Reading (paired bootstrap, 2000 resamples): "
            + (f"steps with a reliable gain: {'; '.join(helpful)}. " if helpful else "no step shows a statistically reliable gain on clean test images. ")
            + (f"Steps with a reliable loss: {'; '.join(harmful)}. " if harmful else "")
            + (f"Statistically neutral on clean images: {'; '.join(neutral)}." if neutral else "")
            + " Components designed for difficult conditions (illumination normalisation, fallback localisation, second pass) are expected to matter most in the "
              "robustness study (Section 7) rather than on clean images.\n")
        add("### 5.2 Leave-one-component-out (from the full hybrid)\n")
        rows = [[r["variant"], pct(r["top1"]), pp(r["delta_top1_vs_reference"]), f"[{100 * r['delta_ci95'][0]:+.2f}, {100 * r['delta_ci95'][1]:+.2f}]"] for r in ab["leave_one_out"]]
        add(md_table(["Configuration", "Top-1", "Delta vs full", "95 % CI (pp)"], rows) + "\n")
        add("### 5.3 Where the vehicle crop comes from\n")
        rows = [[r["box_source"], pct(r["top1"]), pct(r["top5"]), pp(r["delta_top1_vs_yolo_full"])] for r in ab["box_source"]]
        add(md_table(["Box source (same CNN, no other enhancement)", "Top-1", "Top-5", "Delta vs YOLO"], rows) + "\n")
        add("### 5.4 Padding and fusion weight\n")
        add(md_table(["Padding", "Top-1"], [[r["variant"], pct(r["top1"])] for r in ab["padding_sweep"]]) + "\n")
        if ab["alpha_sweep"]:
            add(md_table(["Fusion weight", "Top-1"], [[r["variant"], pct(r["top1"])] for r in ab["alpha_sweep"]]) + "\n")
        if ab.get("classical_feature_groups"):
            add("### 5.5 Classical feature groups (on ground-truth crops)\n")
            add(md_table(["Features", "Dims", "Top-1", "Top-5"], [[r["features"], r["dims"], pct(r["top1"]), pct(r["top5"])] for r in ab["classical_feature_groups"]]) + "\n")
        add("### 5.6 Low-confidence policy\n")
        lp = ab["low_confidence_policy"]
        add(md_table(["Policy", "Answer rate", "Error rate among answers"],
                     [[k.replace("_", " "), pct(v["answer_rate"]), pct(v["error_rate_among_answers"])] for k, v in lp.items() if isinstance(v, dict)]) + "\n")
        add(f"*{lp['note']}.*\n")
        add(fig("ablation.png", "Cumulative ablation and classical feature-group contribution", emb) + "\n")

    # ---------------------------------------------------------------- 6 fine-grained
    if ev:
        add("## 6. Fine-grained discrimination\n")
        eb = ev.get("error_breakdown_hybrid", {})
        tot = sum(eb.values()) or 1
        add("Hybrid errors by confusion type: " + ", ".join(f"{k}: {v} ({pct(v / tot, 0)})" for k, v in eb.items()) + ".\n")
        add(md_table(["True class", "Predicted class", "Count"], [[r["true"], r["pred"], r["count"]] for r in ev["top_confusions_hybrid"][:10]]) + "\n")
        add("Hardest classes: " + "; ".join(f"{r['class']} ({pct(r['acc'], 0)})" for r in ev["worst_classes_hybrid"][:6]) + ".\n")
        vp = ev.get("viewpoint_proxy")
        if vp:
            add("Natural-viewpoint proxy (tertiles of the ground-truth box aspect ratio; Stanford Cars has no viewpoint labels):\n")
            add(md_table(["Box shape", "n"] + [NAMES[k] for k in ev["classification"]],
                         [[k, v["n"]] + [pct(v[m]) for m in ev["classification"]] for k, v in vp.items()]) + "\n")

    # ---------------------------------------------------------------- 7 robustness
    if rb:
        add("## 7. Robustness\n")
        modes = [m for m in ("classical", "dl", "hybrid") if m in next(iter(rb["conditions"].values()))]
        add(f"Each condition is applied to the whole image of {rb['n_images']} random test images (seed {rb['seed']}); all approaches run end-to-end.\n")
        clean = rb["conditions"].get("clean")
        rows = []
        for cname, d in rb["conditions"].items():
            rows.append([cname] + [pct(d[m]["top1"]) for m in modes] + [pct(d["hybrid"]["mean_confidence"], 0), pct(d["hybrid_decisions"]["accept_rate"], 0)])
        add(md_table(["Condition"] + [NAMES[m] + " Top-1" for m in modes] + ["Hybrid mean conf.", "Hybrid accept rate"], rows) + "\n")
        if clean:
            drops = {c: {m: d[m]["top1"] - clean[m]["top1"] for m in modes} for c, d in rb["conditions"].items() if c != "clean"}
            worst = sorted(drops, key=lambda c: drops[c]["hybrid"])[:3]
            add("Largest hybrid accuracy drops vs clean: " + "; ".join(f"{c} ({pp(drops[c]['hybrid'], 1)})" for c in worst) + ". ")
            hy_gain = [c for c in drops if "dl" in modes and drops[c]["hybrid"] - drops[c]["dl"] > 0.02]
            hy_loss = [c for c in drops if "dl" in modes and drops[c]["hybrid"] - drops[c]["dl"] < -0.02]
            add((f"Hybrid beats DL-only by more than 2 pp under: {', '.join(hy_gain)}. " if hy_gain else "Hybrid does not beat DL-only by more than 2 pp under any condition. ")
                + (f"DL-only beats hybrid by more than 2 pp under: {', '.join(hy_loss)}." if hy_loss else "") + "\n")
        add("With this subset size a difference of a few points is within sampling noise (binomial standard error about 2 pp at n=500). "
            "Viewpoint is probed synthetically (rotation, perspective); occlusion is a random patch covering 25 / 50 % of the car box; 'truncated' removes 25 % of the car width.\n")
        add(fig("robustness.png", "Top-1 accuracy under degradations", emb) + "\n")
        add(fig("robustness_examples.jpg", "Hybrid output under selected degradations", emb) + "\n")

    # ---------------------------------------------------------------- 8 confidence
    if ev:
        add("## 8. Confidence analysis and low-confidence handling\n")
        ce = ev["calibration_effect_dl"]
        add(f"Temperature scaling (T = {ce['temperature']:.2f}, fitted on validation logits) changes the DL model's expected calibration error "
            f"from {ce['ece_T=1']:.3f} to {ce['ece_calibrated']:.3f} and NLL from {ce['nll_T=1']:.3f} to {ce['nll_calibrated']:.3f} on the test set.\n")
        add(md_table(["Approach", "ECE", "NLL", "Mean confidence", "Top-1"],
                     [[NAMES[k], f"{v['ece']:.3f}", f"{v['nll']:.3f}", pct(v["mean_confidence"]), pct(ev["classification"][k]["top1"])] for k, v in ev["confidence"].items()]) + "\n")
        h = ev["confidence"]["hybrid"]
        add("Selective prediction for the hybrid system (coverage = share of images answered):\n")
        add(md_table(["Threshold", "Coverage", "Accuracy on answered", "Rejected"],
                     [[f"{r['threshold']:.2f}", pct(r["coverage"]), pct(r["accepted_acc"]), pct(r["rejected_frac"])] for r in h["threshold_table"] if int(round(r["threshold"] * 100)) % 10 == 0 or r["threshold"] in (0.95,)]) + "\n")
        ds = h["decision_stats"]
        add(f"With the validation-tuned thresholds the hybrid system answers **{pct(ds['accept']['fraction'])}** of test images with a full make/model/year label "
            f"(exact accuracy {pct(ds['accept']['exact_acc'])}), gives a make-only answer for {pct(ds['make_only']['fraction'])} (make accuracy {pct(ds['make_only']['make_acc'])}) "
            f"and rejects {pct(ds['reject']['fraction'])} (exact accuracy of the rejected predictions would have been {pct(ds['reject']['exact_acc'])}).\n")
        add(fig("reliability_hybrid.png", "Reliability diagram of the hybrid system", emb) + "\n")

    # ---------------------------------------------------------------- 9 optimisation
    if op:
        add("## 9. Model optimisation\n")
        rows = [[r["variant"], f"{r['size_mb']}", f"{r['gzip_size_mb']}", pct(r["top1"]), pp(r["top1_delta_vs_fp32"]), f"{r['latency_ms_bs1']:.1f}", f"{r['throughput_img_s_bs32']:.0f}"]
                for r in op["classifier"]]
        add(md_table(["Variant", "Size (MB)", "gzip (MB)", "Top-1", "Delta Top-1", "Latency bs1 (ms)", "Throughput bs32 (img/s)"], rows) + "\n")
        add(f"Evaluated on {op['n_images']} test images, torch {op['torch']}, device {op['device']}. Notes: FP16 halves the weight storage and is typically faster on tensor-core GPUs; "
            "dynamic INT8 quantises only the Linear layers (the MLP blocks of the backbone) and runs on CPU; unstructured pruning zeroes weights - the dense kernels run at the same "
            "speed, so any latency difference between pruned and dense rows is measurement noise, and the benefit shows up only in the compressed (gzip) size. "
            + " ".join(f"{v}" for k, v in op.get("notes", {}).items() if isinstance(v, str)) + "\n")
        if op["detector"]:
            add(md_table(["Detector precision", "Median latency (ms)", "Note"], [[r["precision"], f"{r['median_ms']:.1f}" if r["ok"] else "-", r.get("error", "")] for r in op["detector"]]) + "\n")

    # ---------------------------------------------------------------- 10 failures
    if fa:
        add("## 10. Failure cases\n")
        add(f"Overall hybrid error rate: {pct(fa['overall_error_rate'])}. Categories below overlap (one image can fall into several). "
            "'Error rate in group' compares the error rate among all images that have the attribute with the overall error rate - a ratio well above 1 supports the stated cause.\n")
        for cat in fa["categories"]:
            add(f"### {cat['key'].replace('_', ' ').title()} - {cat['n_errors']} error{'s' if cat['n_errors'] != 1 else ''} ({pct(cat['share_of_all_errors'], 0)} of all errors)\n")
            add(f"*Definition:* {cat['description']}.\n")
            if cat.get("error_rate_in_group") is not None:
                ratio = cat["error_rate_in_group"] / max(1e-9, cat["overall_error_rate"])
                add(f"*Evidence:* error rate among the {cat['group_size']} images with this attribute is **{pct(cat['error_rate_in_group'])}** versus {pct(cat['overall_error_rate'])} overall ({ratio:.1f}x).\n")
            add(f"*Likely cause:* {cat['likely_cause']}\n")
            if cat["examples"]:
                add(md_table(["Ground truth", "Predicted", "Confidence", "Box IoU"], [[e["gt"], e["pred"], pct(e["confidence"], 0), "n/a" if e["iou"] is None else f"{e['iou']:.2f}"] for e in cat["examples"]]) + "\n")
            if cat.get("gallery"):
                add(fig(cat["gallery"], f"{cat['key']}: green box = ground truth, red = system; second row = Grad-CAM of the predicted class", emb) + "\n")
        add("*TODO (team): inspect the galleries and confirm or refine each 'likely cause'. The statistics above are measured; the cause text is a hypothesis supported by them, "
            "and Grad-CAM shows where the network looked.*\n")

    # ---------------------------------------------------------------- 11 app
    add("## 11. Application\n")
    add("`streamlit run app.py` provides: (1) image upload, (2) vehicle detection with bounding box, (3) predicted make / model / year, (4) classification confidence "
        "with a three-level decision and the top-5 candidates, (5) intermediate image-processing results (illumination normalisation, edges, contours, GrabCut ROI, HOG, LBP, "
        "colour histogram, geometric descriptors), (6) side-by-side comparison of the classical, deep-learning and hybrid pipelines, plus a benchmark tab with all offline results. "
        "Hybrid components and thresholds can be toggled live in the sidebar.\n")

    # ---------------------------------------------------------------- 12 limits
    add("## 12. Limitations and honest notes\n")
    add("- Results are from a single training run / seed; differences of well under one point are not meaningful (see bootstrap CIs).\n"
        "- Robustness corruptions are synthetic. Viewpoint has no ground truth in Stanford Cars; rotation / perspective warps and box aspect-ratio buckets are proxies.\n"
        "- Year / trim distinctions in the dataset are sometimes nearly invisible; some residual error is irreducible at this resolution.\n"
        "- Classical features add little to a strong CNN; their main value in this system is localisation fallback, illumination normalisation and an independent second opinion.\n"
        "- Confidence thresholds are tuned for a target accepted-set accuracy on a small validation split and should be re-tuned for another domain.\n"
        "- Number-plate recognition, tracking and traffic-event analysis are out of scope (not supported by the dataset annotations).\n")

    # ---------------------------------------------------------------- 13 reproduce
    add("## 13. Reproducibility\n")
    add("```\npython train_detector.py --data-root <path>      # optional: fine-tune YOLO on Stanford Cars boxes\npython train.py --data-root <path>               # fine-tune the CNN classifier\n"
        "python train_classical.py --data-root <path>      # classical feature classifier\npython tune_hybrid.py --data-root <path>          # fusion weight / padding / thresholds on validation\n"
        "python evaluate.py --data-root <path>             # main test evaluation\npython ablation.py ; python robustness.py ; python optimize.py ; python failure_analysis.py\n"
        "python make_report.py --team <team id>            # regenerate this report\nstreamlit run app.py\n```\n")
    if ev:
        add(f"Seed 42; config used for the reported numbers: `{json.dumps(ev['config'])}`; detector `{Path(ev['detector']).name}`.\n")

    # ---------------------------------------------------------------- 14 checklist
    add("## 14. Requirement checklist\n")
    have = lambda x: "done - " if x else "not run - "
    add(md_table(["Requirement", "Status / where"], [
        ["Distinguish similar makes / models / years", f"{have(ev)}Section 4.1 (make / make+model / exact), Section 6"],
        ["Varying illumination and viewing angles", f"{have(rb)}Section 7 (illumination, rotation, perspective), Section 6 viewpoint proxy"],
        ["Partial occlusion and resolutions", f"{have(rb)}Section 7 (occlusion 25/50 %, truncation, x0.5 / x0.25 resolution)"],
        ["Classical vs deep-learning vs hybrid comparison", f"{have(ev)}Section 4"],
        ["Ablation analysis", f"{have(ab)}Section 5"],
        ["Model optimisation (FP16 / INT8 / pruning)", f"{have(op)}Section 9"],
        ["Evaluation on unseen test images", f"{have(ev)}Section 4 (test split never used for tuning)"],
        ["At least three failure cases with causes", f"{have(fa)}Section 10"],
        ["Top-1 / Top-5, IoU, mAP, size + time, robustness, confidence", f"{have(ev)}Sections 4, 7, 8"],
    ]) + "\n")

    text = "\n".join(L)
    out_path.write_text(text, encoding="utf-8")
    todos = [ln.strip()[:110] for ln in text.splitlines() if "TODO" in ln]
    print(f"report written to {out_path}  ({len(text) // 1000} kB)")
    if todos:
        print(f"\n{len(todos)} item(s) need human input before submission:")
        for t in todos:
            print("  -", t)


if __name__ == "__main__":
    main()
