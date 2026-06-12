# MELD-PostOp Webapp

Browser-based automated segmentation of post-operative resection cavities from 3D T1w brain MRI.  
Runs entirely client-side using ONNX Runtime Web — **no data ever leaves the browser**.

**Current status: proof-of-concept using fold 0 only** (~117 MB model, ~2–5 min inference on CPU).

---

## Quick start

### 1. Get the runtime dependencies (once)

```bash
bash web/setup.sh
```

This downloads ONNX Runtime Web WASM files and `coi-serviceworker.js` into `web/wasm/`.

### 2. Convert the model

Download `MELD-PostOp.zip` (v1.0.0) from Figshare:  
<https://figshare.com/s/c5a3d4a5604b229eeb5f>

Install the Python conversion dependencies:

```bash
pip install nnunetv2 torch onnx onnxruntime
```

Convert fold 0 to ONNX (fold 0 is sufficient for the proof-of-concept):

```bash
python scripts/convert_model.py \
    --model-dir /path/to/MELD-PostOp/model \
    --out web/models/meld_postop.onnx \
    --fold 0
```

This creates `web/models/meld_postop_fold0.onnx` (~117 MB).  
To export all 5 folds (for the full ensemble), omit `--fold 0`.

### 3. Start the dev server

```bash
python serve.py
# → http://localhost:8080/web/
```

Then drop a post-operative T1w NIfTI (`.nii` or `.nii.gz`) onto the app and click **Run segmentation**.  
The resulting mask is displayed as a warm-colored overlay and can be downloaded.

---

## Repository layout

```
web/
├── index.html          App shell (NiiVue viewer + sidebar)
├── main.js             UI logic — file loading, worker messaging, overlay display
├── worker.js           Web Worker — full inference pipeline
├── preprocess.js       Resampling (trilinear / nearest-neighbour) + z-score normalisation
├── nifti.js            NIfTI-1 parser and encoder
├── test.html           In-browser unit tests (open in browser to run)
├── setup.sh            Downloads ONNX Runtime Web + coi-serviceworker
├── models/             ONNX model files (not included — generate with convert_model.py)
│   └── meld_postop_fold0.onnx
└── wasm/               ONNX Runtime WASM files (downloaded by setup.sh)

scripts/
├── convert_model.py          nnU-Net v2 → ONNX conversion + validation
├── create_test_data.py       Generates preprocessing test fixtures
└── create_inference_test_data.py  Generates inference test fixtures (requires onnxruntime)

data/tests/             Reference data for the browser test suite
serve.py                Dev server (Python 3, no dependencies) — sets COOP/COEP headers
```

---

## Inference pipeline

Each step matches the corresponding nnU-Net v2 `predict_single_npy_array()` stage:

| Step | Details |
|------|---------|
| Parse NIfTI | Supports UINT8 / int16 / int32 / float32 / float64; applies `scl_slope`/`scl_inter` |
| Resample | Trilinear to target spacing (1×0.9375×0.9375 mm) |
| Normalise | Z-score on foreground (non-zero) voxels |
| Sliding window | 128×160×112 patch, 50 % overlap, Gaussian importance weighting |
| Softmax + argmax | Class 1 = resection cavity |
| Resample back | Nearest-neighbour to original voxel spacing (endpoint-aligned, matches skimage) |
| Encode NIfTI | Copies original affine; UINT8 output |
| Display | NiiVue warm colormap overlay, 70 % opacity |

---

## Browser requirements

| Requirement | Notes |
|-------------|-------|
| ES modules | Chrome 80 / Firefox 75 / Safari 14 |
| SharedArrayBuffer | Required for multi-threaded WASM — needs COOP/COEP headers (served by `serve.py`) |
| RAM | ≥ 4 GB recommended |
| GPU | Not required — WASM CPU provider only |

For deployment, configure your web server with:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

---

## Running the tests

Open `http://localhost:8080/web/test.html` while the dev server is running.  
The test suite covers NIfTI I/O, resampling, z-score normalisation, ONNX inference (fold 0), and the segmentation back-resampling — all compared against Python/onnxruntime ground truth.

---

## Citation

If you use this tool in research, please cite the MELD-PostOp paper and model:

> *Automated segmentation of resection cavities on postoperative T1w MRI in epilepsy surgery.*  
> MELDProject, 2024. <https://github.com/MELDProject/MELD-PostOp>

---

## Licence

Webapp code: MIT.  
MELD-PostOp model weights: see the licence file supplied with the model download.
