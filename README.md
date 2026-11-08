# P5 - Intelligent Fine-Grained Vehicle Recognition (Stanford Cars)

Hybrid system: **YOLO detection + fine-tuned CNN classification + classical computer vision**, with
confidence-aware output, a Streamlit app and a measured, auto-generated evaluation report.

```
image -> YOLO ──(miss)──> classical localiser (Canny→contours→GrabCut)
          └> GrabCut refinement -> illumination-normalised crop -> CNN (flip TTA) ─┐
                                                                                    ├─ fusion -> accept / make-only / reject
              classical HOG+colour+LBP+geometry -> PCA -> logistic regression ─────┘
```


## Hosted demo (Streamlit Community Cloud)

The app can be hosted for free straight from this repository. `app.py` is the entry point, `requirements.txt` and
`packages.txt` list what the server installs, and the model files live in `work/models/` (the classifier is stored in
three parts because GitHub's web uploader limits files to 25 MB; `config.py` joins them when the app starts).
To rebuild those files from a trained model: `python tools/make_cloud_models.py <work/models folder> work/models`.


## 1. Setup (Windows / Linux, GPU recommended)

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux: source .venv/bin/activate
# install the CUDA build of PyTorch first (pick your CUDA version at pytorch.org), e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Dataset: Stanford Cars - put it anywhere and pass `--data-root`. Any of these layouts is detected automatically:
* official: `cars_train/`, `cars_test/`, `devkit/` (`cars_meta.mat`, `cars_train_annos.mat`, **`cars_test_annos_withlabels.mat`**)
* Kaggle CSV: `anno_train.csv`, `anno_test.csv`, `names.csv` + image folders
* class folders (`train/<class>/…`) - works, but without boxes IoU/mAP cannot be computed

> The original `cars_test_annos.mat` has **no labels**. Download `cars_test_annos_withlabels.mat` (linked on the dataset page) into `devkit/`.
> Without it the code falls back to a 20 % hold-out of the training set as "unseen test" and says so in the report.

## 2. Run everything (about 1.5-3 h on an RTX 4060, most of it training / robustness)

```bash
python run_all.py --data-root D:/datasets/stanford_cars --team <your team id>
```
or step by step:

| Step | Command | What it does | Rough time |
|---|---|---|---|
| 1 | `python train_detector.py --data-root …` | fine-tune YOLO on Stanford Cars boxes (optional, recommended) | 30-60 min |
| 2 | `python train.py --data-root …` | fine-tune ConvNeXt-T (196 classes), temperature calibration | 25-45 min |
| 3 | `python train_classical.py --data-root …` | HOG+colour+LBP+geometry → PCA → logistic regression | 5-10 min |
| 4 | `python tune_hybrid.py --data-root …` | fusion weight, padding, confidence thresholds - **validation only** | 3-5 min |
| 5 | `python evaluate.py --data-root … [--compare-detectors]` | Top-1/5, IoU, AP/mAP, calibration, size/latency, confusions | 10-15 min |
| 6 | `python ablation.py --data-root …` | cumulative + leave-one-out + box source + feature groups (with bootstrap CIs) | 15-25 min |
| 7 | `python robustness.py --data-root … [--limit 500]` | 16 degradations × 3 approaches | 20-40 min |
| 8 | `python optimize.py --data-root …` | FP16 / dynamic INT8 / pruning: size, accuracy, latency | 5 min |
| 9 | `python failure_analysis.py --data-root …` | error categories with evidence + Grad-CAM galleries | 3 min |
| 10 | `python make_report.py --team <team id> [--embed-images]` | writes `work/eval_P5_<team id>.md` from the measured JSON files | seconds |
| app | `streamlit run app.py` | upload → box → make/model/year → confidence → processing steps → 3-way comparison | |

Everything is written under `work/` (`models/`, `results/`, `cache/`). Delete `work/cache` if you change the detector or the data.

Out-of-memory on the GPU? `python train.py --batch-size 16` (or `--arch efficientnet_b0 --img-size 224`).
Windows data-loader problems? `--workers 0`.

## 3. Report

`make_report.py` fills the report **only from measured results** and generates its interpretation sentences from the numbers and
bootstrap confidence intervals (so it will say "not distinguishable from zero" when that is what the data show).
It lists every `TODO` that needs a human: team id and contributions (edit `team.json`) and the review of the failure-case causes.
If the portal needs a single file, use `--embed-images` to inline the figures.

## 4. Files

| File | Purpose |
|---|---|
| `data.py` | dataset loader (3 layouts), make/model/year parsing, stratified val split |
| `classical.py` | illumination normalisation, Canny/Sobel, contours, GrabCut ROI, HOG/colour/LBP/geometry features, classical classifier |
| `deep_learning.py` | YOLO wrapper, timm classifier, Grad-CAM, training dataset with box-jitter / occlusion augmentation |
| `train.py`, `train_classical.py`, `train_detector.py` | training scripts |
| `hybrid.py` | the hybrid pipeline, fusion, low-confidence decision logic |
| `stages.py`, `metrics.py` | cached/batched stage evaluation, Top-k / IoU / AP / ECE / risk-coverage |
| `tune_hybrid.py`, `evaluate.py`, `ablation.py`, `robustness.py`, `optimize.py`, `failure_analysis.py` | analysis suite |
| `analysis.py`, `app.py` | intermediate-result visualisation and the Streamlit app |
| `make_report.py`, `run_all.py`, `team.json` | report generator, orchestration, team table |
| `tests/smoke_test.py`, `tools/make_synthetic_dataset.py` | no-data smoke test (synthetic cars, mock detector) |

Edit `team.json` (team id, members, contributions) before generating the report.

## 5. Design decisions worth knowing (viva)

* **No test leakage**: classifier checkpoint, temperature, fusion weight α, padding and thresholds are all selected on a 10 % validation split of the *training* set.
* **Train/test box mismatch** is handled by training on jittered, randomly padded GT boxes, so the classifier tolerates detector boxes.
* **Hybrid ≠ "always better"**: each classical component is ablated with paired-bootstrap CIs; a component that does not help shows up as such. α is chosen on validation data and can be 0.
* **Low confidence**: three outcomes - accept (full label), make-only (manufacturer reliable, model/year not), reject. Thresholds target a stated accepted-set accuracy (default 90 %).
* **Optimisation**: FP16 (GPU), dynamic INT8 of Linear layers (CPU), global magnitude pruning. Unstructured pruning does *not* speed up dense kernels; its benefit is the compressed size.
* **Out of scope** (not annotated in the dataset): number plates, tracking, traffic events.

## 6. Verifying the code without the dataset

```bash
python tests/smoke_test.py
```
runs every script on a tiny synthetic dataset with a mock detector (CPU, a few minutes). It checks plumbing only - **never quote synthetic numbers**.
