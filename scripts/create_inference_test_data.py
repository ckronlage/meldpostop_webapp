#!/usr/bin/env python3
"""
Create expected inference output for testing the JS ONNX sliding-window pipeline.

Uses onnxruntime-python (not PyTorch) to run fold 0 directly so that the
comparison is strictly: onnxruntime-python vs onnxruntime-web (WASM).
Any difference between these two would indicate a bug in our JS glue code,
not a model weight issue.

The test volume (12×17×21 after preprocessing) is smaller than the patch
(128×160×112) in all axes, so the sliding window collapses to a single padded
patch — exactly one model call.  This makes it straightforward to reproduce
in the browser without worrying about the sliding-window accumulation logic.

Run from project root:
    conda activate meldpostop
    python scripts/create_inference_test_data.py
"""

import json
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize as sk_resize
from pathlib import Path
import onnxruntime as ort

PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = PROJECT_ROOT / "data" / "tests"
MODEL_PATH    = PROJECT_ROOT / "web" / "models" / "meld_postop_fold0.onnx"
TARGET_SPACING = [0.9999988079071045, 0.9375, 0.9375]   # [sz, sy, sx] mm
PATCH_SIZE     = [128, 160, 112]                         # [D, H, W]


# ── Step 1: reproduce preprocessing (same as create_test_data.py test 2) ─────
img     = sitk.ReadImage(str(DATA_DIR / "test_preprocess.nii"))
arr_zyx = sitk.GetArrayFromImage(img).astype(np.float32)   # (12, 16, 20)

dst_dims = [
    round(arr_zyx.shape[0] * 1.0 / TARGET_SPACING[0]),   # 12
    round(arr_zyx.shape[1] * 1.0 / TARGET_SPACING[1]),   # 17
    round(arr_zyx.shape[2] * 1.0 / TARGET_SPACING[2]),   # 21
]

resampled = sk_resize(
    arr_zyx, dst_dims,
    order=1, mode="edge", anti_aliasing=False, preserve_range=True,
).astype(np.float32)

fg     = np.sort(resampled[resampled != 0])
lo     = fg[int(0.005 * len(fg))]
hi     = fg[int(0.995 * len(fg))]
mean   = float(np.clip(fg, lo, hi).mean())
std    = float(np.clip(fg, lo, hi).std()) + 1e-8

normalized = ((np.clip(resampled, lo, hi) - mean) / std).astype(np.float32)
D, H, W = normalized.shape
print(f"Preprocessed shape: {[D, H, W]},  range: [{normalized.min():.3f}, {normalized.max():.3f}]")


# ── Step 2: symmetric zero-padding to patch size ───────────────────────────
def pad_symmetric(data, patch):
    """Zero-pad `data` (D,H,W) to at least `patch` (pD,pH,pW), symmetric."""
    pD, pH, pW = patch
    D, H, W = data.shape
    padZ = max(0, pD - D);  plZ = padZ // 2;  prZ = padZ - plZ
    padY = max(0, pH - H);  plY = padY // 2;  prY = padY - plY
    padX = max(0, pW - W);  plX = padX // 2;  prX = padX - plX
    padded = np.pad(data, [(plZ, prZ), (plY, prY), (plX, prX)],
                    mode="constant", constant_values=0)
    return padded, [plZ, plY, plX], [prZ, prY, prX]

padded, pad_left, pad_right = pad_symmetric(normalized, PATCH_SIZE)
pD, pH, pW = padded.shape
print(f"Padded shape:        {[pD, pH, pW]}")
print(f"Pad left  (z,y,x):   {pad_left}")
print(f"Pad right (z,y,x):   {pad_right}")


# ── Step 3: run fold 0 via onnxruntime-python ──────────────────────────────
print(f"\nLoading model: {MODEL_PATH.name} …")
sess         = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name   = sess.get_inputs()[0].name
output_name  = sess.get_outputs()[0].name

inp = padded[np.newaxis, np.newaxis].astype(np.float32)   # [1, 1, pD, pH, pW]
print(f"Running inference on input shape {list(inp.shape)} …")

raw = sess.run([output_name], {input_name: inp})[0]   # [1, 2, pD, pH, pW]
logits_full = raw[0]                                   # [2, pD, pH, pW]

print(f"Raw logit output shape: {list(logits_full.shape)}")
print(f"  bg     range: [{logits_full[0].min():.4f}, {logits_full[0].max():.4f}]")
print(f"  cavity range: [{logits_full[1].min():.4f}, {logits_full[1].max():.4f}]")


# ── Step 4: crop back to original (pre-pad) coordinates ───────────────────
plZ, plY, plX = pad_left
logits = logits_full[:, plZ:plZ+D, plY:plY+H, plX:plX+W]   # [2, D, H, W]
print(f"\nCropped logit shape (original space): {list(logits.shape)}")
print(f"  bg     range: [{logits[0].min():.4f}, {logits[0].max():.4f}]")
print(f"  cavity range: [{logits[1].min():.4f}, {logits[1].max():.4f}]")

# softmax at one interior voxel for a human-readable sanity check
def softmax2(a, b):
    m = max(a, b)
    ea, eb = np.exp(a - m), np.exp(b - m)
    s = ea + eb
    return float(ea / s), float(eb / s)


# ── Step 5: save expected JSON ─────────────────────────────────────────────
spot_positions = [(0, 0, 0), (5, 8, 10), (D-1, H-1, W-1)]
spot_checks = {}
for (z, y, x) in spot_positions:
    bg_l    = float(logits[0, z, y, x])
    cav_l   = float(logits[1, z, y, x])
    bg_s, cav_s = softmax2(bg_l, cav_l)
    key = f"{z}_{y}_{x}"
    spot_checks[key] = {
        "logit_bg":       bg_l,
        "logit_cavity":   cav_l,
        "softmax_bg":     bg_s,
        "softmax_cavity": cav_s,
    }
    print(f"  [{z},{y},{x}]  logits=({bg_l:.4f}, {cav_l:.4f})  "
          f"p_cavity={cav_s:.4f}")

# ── Step 6: argmax + nearest-neighbour back-resample  ─────────────────────
# Mirrors logitsToSegmentation() in preprocess.js:
#   softmax-argmax in resampled space → NN resample to original space.
#
# For our test data the model always predicts background (all logits strongly
# favour bg), so the argmax mask is all-zero.  We verify this, then also test
# the resampling with a synthetic mask that has a non-trivial pattern.

# --- 6a: argmax of actual inference (should be all background) ---
argmax_resampled = np.argmax(logits, axis=0).astype(np.uint8)   # [D, H, W]
print(f"\nArgmax unique values: {np.unique(argmax_resampled).tolist()}")

orig_shape_zyx = list(arr_zyx.shape)   # [12, 16, 20] — original NIfTI shape
argmax_orig = sk_resize(
    argmax_resampled.astype(float), orig_shape_zyx,
    order=0, mode="edge", anti_aliasing=False, preserve_range=True,
).astype(np.uint8)

# --- 6b: synthetic segmentation with a central box of label 1 ---
# Box centred at (cz, cy, cx) with radii (rz, ry, rx)
cz, cy, cx = D // 2, H // 2, W // 2
rz, ry, rx = 3, 4, 4
seg_src = np.zeros([D, H, W], dtype=np.uint8)
seg_src[max(0, cz - rz):min(D, cz + rz),
        max(0, cy - ry):min(H, cy + ry),
        max(0, cx - rx):min(W, cx + rx)] = 1

seg_back = sk_resize(
    seg_src.astype(float), orig_shape_zyx,
    order=0, mode="edge", anti_aliasing=False, preserve_range=True,
).astype(np.uint8)

print(f"Synthetic seg:  {seg_src.sum()} ones in [{D},{H},{W}] resampled space")
print(f"After NN back-resample: {seg_back.sum()} ones in {orig_shape_zyx} original space")

# Spot checks — pick corners, centre, and a point on the box boundary
def seg_spot(arr, positions):
    return {f"{z}_{y}_{x}": int(arr[z, y, x]) for z, y, x in positions}

seg_positions_resampled = [(0, 0, 0), (cz, cy, cx),
                           (cz - rz, cy - ry, cx - rx),   # box corner
                           (D-1, H-1, W-1)]
seg_positions_orig      = [(0, 0, 0),
                           (orig_shape_zyx[0]//2, orig_shape_zyx[1]//2, orig_shape_zyx[2]//2),
                           (orig_shape_zyx[0]-1, orig_shape_zyx[1]-1, orig_shape_zyx[2]-1)]

print("Synthetic seg spot checks (resampled space):")
for k, v in seg_spot(seg_src, seg_positions_resampled).items():
    print(f"  seg_src[{k}] = {v}")
print("After back-resample (original space):")
for k, v in seg_spot(seg_back, seg_positions_orig).items():
    print(f"  seg_back[{k}] = {v}")

expected = {
    "description": "fold 0 ONNX inference on preprocessed test volume (onnxruntime-python)",
    "preprocessed_shape_zyx": [D, H, W],
    "original_shape_zyx":     orig_shape_zyx,
    "pad_left_zyx":           pad_left,
    "pad_right_zyx":          pad_right,
    "padded_shape_zyx":       [pD, pH, pW],
    "logit_range_full_patch": {
        "bg_min":     float(logits_full[0].min()),
        "bg_max":     float(logits_full[0].max()),
        "cavity_min": float(logits_full[1].min()),
        "cavity_max": float(logits_full[1].max()),
    },
    "spot_checks": spot_checks,
    # ── back-resampling test ──────────────────────────────────────────────
    "seg_back_resampling": {
        "description": "endpoint-aligned NN resample from resampled to original space",
        "box_centre_zyx": [cz, cy, cx],
        "box_radii_zyx":  [rz, ry, rx],
        # src spot checks (in resampled space) — reproduced identically in JS
        "src_spot_checks": seg_spot(seg_src, seg_positions_resampled),
        # expected after back-resampling (ground truth from skimage order=0)
        "back_spot_checks": seg_spot(seg_back, seg_positions_orig),
    },
}

out_path = DATA_DIR / "test_inference_expected.json"
with open(out_path, "w") as f:
    json.dump(expected, f, indent=2)
print(f"\nWrote {out_path}")
