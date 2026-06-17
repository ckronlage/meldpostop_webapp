#!/usr/bin/env python3
"""
MELD-PostOp Python ONNX inference — GPU provider test

Mirrors the JS pipeline (run_inference.mjs / inference.js / preprocess.js) exactly.
--provider flag lets you compare CPU vs CoreML execution providers.

Usage:
    conda activate meldpostop_onnx
    # Single file
    python scripts/run_inference.py --input scan.nii.gz --output mask.nii \
        [--model-dir web/models/] [--folds 0] [--provider auto|cpu|coreml]

    # Batch
    python scripts/run_inference.py --input-dir /data/scans/ --output-dir /data/masks/ \
        [--model-dir web/models/] [--folds 0,1,2,3,4] [--provider auto]

--provider  auto   (default) try CoreML mlprogram → CoreML NeuralNetwork → CPU
            cpu    force CPU only
            coreml force CoreML (exits if unavailable or model fails to load)
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import SimpleITK as sitk
from skimage.transform import resize as sk_resize

# ── Constants (Dataset001_MELDPostOp_v1.0.0 plans.json, 3d_fullres) ──────────
TARGET_SPACING = [0.9999988079071045, 0.9375, 0.9375]  # mm [sz, sy, sx]
PATCH_SIZE     = [128, 160, 112]                         # [D, H, W]
PATCH_OVERLAP  = 0.5
N_CLASSES      = 2
GAUSS_SIGMA    = 0.125  # nnU-Net default


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ── Provider selection ─────────────────────────────────────────────────────────

def load_session(model_path, provider_arg):
    """
    Create an ORT InferenceSession, trying CoreML formats in order before CPU.
    Returns (session, provider_label) where provider_label describes what's active.
    """
    available = ort.get_available_providers()
    log(f"  ORT available providers: {available}")

    if provider_arg == 'cpu':
        sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        log(f"  Active providers: {sess.get_providers()}")
        return sess, 'cpu'

    has_coreml = 'CoreMLExecutionProvider' in available
    if not has_coreml:
        if provider_arg == 'coreml':
            log("ERROR: CoreMLExecutionProvider not in this ORT build.")
            sys.exit(1)
        log("  CoreML not available — using CPU.")
        sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        return sess, 'cpu'

    # Try CoreML ML Program format — supports more 3D ops on M-series (54/77 nodes)
    # but crashes if the model has rank-5 InstanceNormalization (nnU-Net 3d_fullres does)
    try:
        providers = [('CoreMLExecutionProvider', {'ModelFormat': 'MLProgram'}), 'CPUExecutionProvider']
        sess = ort.InferenceSession(str(model_path), providers=providers)
        log(f"  CoreML (MLProgram) loaded. Active: {sess.get_providers()}")
        return sess, 'coreml-mlprogram'
    except Exception as e:
        log(f"  CoreML MLProgram failed: {e}")
        if provider_arg == 'coreml':
            log("ERROR: CoreML MLProgram failed and --provider coreml was requested.")
            sys.exit(1)

    # Try CoreML NeuralNetwork format (legacy — 22/77 nodes on CoreML, rest on CPU)
    try:
        providers = [('CoreMLExecutionProvider', {'ModelFormat': 'NeuralNetwork'}), 'CPUExecutionProvider']
        sess = ort.InferenceSession(str(model_path), providers=providers)
        log(f"  CoreML (NeuralNetwork) loaded. Active: {sess.get_providers()}")
        return sess, 'coreml-neuralnetwork'
    except Exception as e:
        log(f"  CoreML NeuralNetwork failed: {e}")

    # Pure CPU fallback
    log("  Falling back to CPU only.")
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    log(f"  Active providers: {sess.get_providers()}")
    return sess, 'cpu'


# ── Preprocessing (mirrors preprocess.js) ─────────────────────────────────────

def read_nifti(path):
    img = sitk.ReadImage(str(path))
    # GetArrayFromImage returns (z, y, x) order matching SimpleITK / nnU-Net convention
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sp_xyz = img.GetSpacing()
    spacing_zyx = [sp_xyz[2], sp_xyz[1], sp_xyz[0]]
    return arr, spacing_zyx, img


def resample_trilinear(arr, src_spacing, tgt_spacing):
    dst_dims = [
        round(arr.shape[i] * src_spacing[i] / tgt_spacing[i])
        for i in range(3)
    ]
    return sk_resize(
        arr, dst_dims,
        order=1, mode='edge', anti_aliasing=False, preserve_range=True,
    ).astype(np.float32)


def foreground_zscore(arr):
    fg = np.sort(arr[arr != 0])
    lo = fg[int(0.005 * len(fg))]
    hi = fg[int(0.995 * len(fg))]
    clipped_fg = np.clip(fg, lo, hi)
    mean = float(clipped_fg.mean())
    std  = float(clipped_fg.std()) + 1e-8
    return ((np.clip(arr, lo, hi) - mean) / std).astype(np.float32), mean, std, lo, hi


# ── Sliding-window inference (mirrors inference.js) ───────────────────────────

def build_gaussian_map(patch_size):
    pD, pH, pW = patch_size
    z = np.arange(pD, dtype=np.float32)
    y = np.arange(pH, dtype=np.float32)
    x = np.arange(pW, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    dz = (zz - pD / 2) / (pD * GAUSS_SIGMA)
    dy = (yy - pH / 2) / (pH * GAUSS_SIGMA)
    dx = (xx - pW / 2) / (pW * GAUSS_SIGMA)
    return np.exp(-0.5 * (dz**2 + dy**2 + dx**2)).astype(np.float32)


def compute_sliding_steps(img_dims, patch_dims, step_size):
    steps = []
    for img_d, p_d in zip(img_dims, patch_dims):
        if img_d <= p_d:
            steps.append([0])
            continue
        target_step = p_d * step_size
        num_steps = math.ceil((img_d - p_d) / target_step) + 1
        actual_step = (img_d - p_d) / (num_steps - 1)
        steps.append([round(actual_step * k) for k in range(num_steps)])
    return steps


def sliding_window_inference(data, sess, patch_size, overlap):
    """
    data:  float32 [D, H, W], normalised
    Returns: float32 [N_CLASSES, D, H, W] softmax probabilities
    """
    D, H, W = data.shape
    pD, pH, pW = patch_size
    gauss_map = build_gaussian_map(patch_size)        # [pD, pH, pW]

    accum_soft   = np.zeros((N_CLASSES, D, H, W), dtype=np.float32)
    accum_weight = np.zeros((D, H, W), dtype=np.float32)

    steps_z, steps_y, steps_x = compute_sliding_steps([D, H, W], patch_size, overlap)
    origins = [(z0, y0, x0) for z0 in steps_z for y0 in steps_y for x0 in steps_x]
    n_patches = len(origins)
    log(f"    {len(steps_z)}×{len(steps_y)}×{len(steps_x)} = {n_patches} patches")

    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    patch_times = []

    for i, (z0, y0, x0) in enumerate(origins):
        # In-bounds slice in global volume
        gz0 = max(0, z0);  gz1 = min(D, z0 + pD)
        gy0 = max(0, y0);  gy1 = min(H, y0 + pH)
        gx0 = max(0, x0);  gx1 = min(W, x0 + pW)
        # Corresponding slice in patch coords
        lz0 = gz0 - z0;  lz1 = gz1 - z0
        ly0 = gy0 - y0;  ly1 = gy1 - y0
        lx0 = gx0 - x0;  lx1 = gx1 - x0

        patch = np.zeros((pD, pH, pW), dtype=np.float32)
        patch[lz0:lz1, ly0:ly1, lx0:lx1] = data[gz0:gz1, gy0:gy1, gx0:gx1]

        t = time.perf_counter()
        raw = sess.run([output_name], {input_name: patch[np.newaxis, np.newaxis]})[0]
        patch_ms = (time.perf_counter() - t) * 1000
        patch_times.append(patch_ms)

        logits = raw[0]  # [2, pD, pH, pW]

        if i == 0:
            log(f"    Patch 0 logits — bg: [{logits[0].min():.2f}, {logits[0].max():.2f}]  "
                f"cavity: [{logits[1].min():.2f}, {logits[1].max():.2f}]  ({patch_ms:.0f} ms)")

        # Numerically stable softmax over class axis
        max_l = logits.max(axis=0, keepdims=True)
        exp_l = np.exp(logits - max_l)
        soft  = exp_l / exp_l.sum(axis=0, keepdims=True)  # [2, pD, pH, pW]

        # Gaussian-weighted accumulation (in-bounds region only)
        w = gauss_map[lz0:lz1, ly0:ly1, lx0:lx1]
        accum_soft[:, gz0:gz1, gy0:gy1, gx0:gx1] += soft[:, lz0:lz1, ly0:ly1, lx0:lx1] * w
        accum_weight[gz0:gz1, gy0:gy1, gx0:gx1]  += w

        if i % max(1, n_patches // 20) == 0 or i == n_patches - 1:
            pct = round((i + 1) / n_patches * 100)
            print(f"\r    {pct:3d} %  ({patch_ms:.0f} ms/patch)", end='', file=sys.stderr, flush=True)

    print('', file=sys.stderr)
    avg_ms = sum(patch_times) / len(patch_times)
    log(f"    avg {avg_ms:.0f} ms/patch over {n_patches} patches")

    accum_weight = np.maximum(accum_weight, 1e-8)
    return accum_soft / accum_weight[np.newaxis]  # [N_CLASSES, D, H, W]


# ── Postprocessing ─────────────────────────────────────────────────────────────

def logits_to_segmentation(soft_vol, orig_dims):
    mask_resampled = np.argmax(soft_vol, axis=0).astype(np.float32)  # [D, H, W]
    return sk_resize(
        mask_resampled, orig_dims,
        order=0, mode='edge', anti_aliasing=False, preserve_range=True,
    ).astype(np.uint8)


# ── File I/O ───────────────────────────────────────────────────────────────────

def write_nifti(mask, ref_img, output_path):
    out_img = sitk.GetImageFromArray(mask)
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, str(output_path))


# ── Model loading ──────────────────────────────────────────────────────────────

def resolve_model_path(fold_idx, model_dir):
    int8 = Path(model_dir) / f'meld_postop_fold{fold_idx}_int8.onnx'
    fp32 = Path(model_dir) / f'meld_postop_fold{fold_idx}.onnx'
    if int8.exists():
        return int8, True
    return fp32, False


# ── Single-file inference ──────────────────────────────────────────────────────

def run_one(input_path, output_path, sessions):
    t0 = time.perf_counter()
    log(f"\n[{Path(input_path).name}]")

    arr, spacing_zyx, ref_img = read_nifti(input_path)
    orig_dims = arr.shape
    log(f"  dims {list(arr.shape)}, spacing {[f'{s:.3f}' for s in spacing_zyx]} mm [z,y,x]")

    resampled = resample_trilinear(arr, spacing_zyx, TARGET_SPACING)
    log(f"  resampled → {list(resampled.shape)}")

    normalised, mean, std, lo, hi = foreground_zscore(resampled)
    log(f"  normalised: mean={mean:.1f}, std={std:.1f}, clip=[{lo:.1f}, {hi:.1f}]")

    D, H, W = resampled.shape
    ensemble_soft = np.zeros((N_CLASSES, D, H, W), dtype=np.float32)

    for f_idx, (sess, label) in enumerate(sessions):
        log(f"  fold {f_idx + 1}/{len(sessions)} inference ({label})…")
        fold_soft = sliding_window_inference(normalised, sess, PATCH_SIZE, PATCH_OVERLAP)
        ensemble_soft += fold_soft / len(sessions)

    mask_orig = logits_to_segmentation(ensemble_soft, orig_dims)
    log(f"  mask: {int(mask_orig.sum())} positive voxels")

    write_nifti(mask_orig, ref_img, output_path)
    log(f"  → {output_path}  ({time.perf_counter() - t0:.1f} s)")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='MELD-PostOp ONNX inference (Python)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input',     help='Input NIfTI file')
    g.add_argument('--input-dir', help='Input directory (batch)')
    p.add_argument('--output',     help='Output mask file (with --input)')
    p.add_argument('--output-dir', help='Output directory (with --input-dir)')
    p.add_argument('--model-dir', default='web/models/')
    p.add_argument('--folds', default='0', help='Comma-separated fold indices')
    p.add_argument('--provider', default='auto', choices=['auto', 'cpu', 'coreml'])
    return p.parse_args()


def main():
    args = parse_args()

    if args.input and not args.output:
        sys.exit("ERROR: --input requires --output")
    if args.input_dir and not args.output_dir:
        sys.exit("ERROR: --input-dir requires --output-dir")

    fold_list = [int(f) for f in args.folds.split(',')]

    log(f"Loading {len(fold_list)} fold model(s) from {args.model_dir}…")
    sessions = []
    for fold_idx in fold_list:
        model_path, quantized = resolve_model_path(fold_idx, args.model_dir)
        if not model_path.exists():
            sys.exit(f"ERROR: model not found: {model_path}")
        label = 'INT8' if quantized else 'FP32'
        log(f"  Loading fold {fold_idx} — {label}: {model_path.name}")
        sess, provider_label = load_session(model_path, args.provider)
        sessions.append((sess, provider_label))
    log("Models ready.")

    if args.input:
        run_one(args.input, args.output, sessions)
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            f for f in Path(args.input_dir).iterdir()
            if f.suffix in ('.nii',) or f.name.endswith('.nii.gz')
        )
        if not files:
            sys.exit(f"No .nii / .nii.gz files found in {args.input_dir}")
        log(f"Found {len(files)} file(s).")
        for i, in_path in enumerate(files):
            stem = in_path.name.replace('.nii.gz', '').replace('.nii', '')
            log(f"[{i+1}/{len(files)}]")
            run_one(str(in_path), str(out_dir / f"{stem}_mask.nii"), sessions)
        log(f"\nDone. {len(files)} file(s) processed.")


if __name__ == '__main__':
    main()
