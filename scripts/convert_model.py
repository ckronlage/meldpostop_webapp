"""
Convert the MELD-PostOp nnU-Net v2 model to ONNX for browser inference.

Prerequisites
-------------
  pip install nnunetv2 torch onnx onnxruntime

The MELD-PostOp model archive (MELD-PostOp.zip) must already be downloaded
from Figshare (https://figshare.com/s/050bec27f32fcd1fd75e) and the license
file placed at <meld_postop_root>/model/LICENSE.

Usage
-----
  python scripts/convert_model.py \
      --model-dir /path/to/MELD-PostOp/model \
      --out       web/models/meld_postop.onnx \
      [--fold 0]  [--patch 128 128 128]  [--opset 17]
"""

import argparse
import sys
import json
from pathlib import Path

import torch
import numpy as np
import onnx
import onnxruntime as ort


# ──────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="nnU-Net v2 → ONNX converter for MELD-PostOp")
    p.add_argument("--model-dir", required=True,
                   help="Path to the nnU-Net model directory, e.g. "
                        ".../Dataset001_MELDPostOp_v1.0.0/nnUNetTrainer__nnUNetPlans__3d_fullres/")
    p.add_argument("--out", default="web/models/meld_postop.onnx",
                   help="Output path for the ONNX file (default: web/models/meld_postop.onnx)")
    p.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Which fold(s) to export (default: all 5)")
    p.add_argument("--patch", type=int, nargs=3, default=None,
                   metavar=("D", "H", "W"),
                   help="Patch size for ONNX dummy input (default: read from plans.json)")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version (default: 17)")
    p.add_argument("--validate", action="store_true", default=True,
                   help="Compare PyTorch vs ONNX outputs after export")
    p.add_argument("--fp16", action="store_true",
                   help="Export in fp16 (halves model size, ~same accuracy)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# Load nnU-Net v2 network via the stable nnUNetPredictor API
# ──────────────────────────────────────────────────────────────────────────
def load_nnunet_network(model_dir: Path, fold: int):
    """
    Use nnUNetPredictor (stable public API) to initialise the network and
    load the checkpoint weights.  The predictor handles all internal plans
    parsing and architecture construction automatically.
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # nnUNetPredictor expects the trainer/plans/config triple encoded in the
    # directory name: nnUNetTrainer__nnUNetPlans__3d_fullres
    # It also needs the *parent* dataset folder so it can find plans.json.
    # We support both layouts:
    #   <model_dir>  =  .../nnUNetTrainer__nnUNetPlans__3d_fullres/
    #   <model_dir>  =  .../Dataset001_MELDPostOp_v1.0.0/nnUNetTrainer__nnUNetPlans__3d_fullres/
    trainer_dir = model_dir
    dir_name = trainer_dir.name  # e.g. "nnUNetTrainer__nnUNetPlans__3d_fullres"

    parts = dir_name.split("__")
    if len(parts) != 3:
        sys.exit(
            f"Expected directory name like 'nnUNetTrainer__nnUNetPlans__3d_fullres', "
            f"got: {dir_name}"
        )
    trainer_name, plans_name, config_name = parts

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(trainer_dir),
        use_folds=(fold,),
        checkpoint_name="checkpoint_final.pth",
    )

    network = predictor.network
    network.eval()

    # Read metadata for reporting
    plans_path = trainer_dir / "plans.json"
    dataset_path = trainer_dir / "dataset.json"
    with open(plans_path) as f:
        plans = json.load(f)
    with open(dataset_path) as f:
        dataset = json.load(f)

    patch_size = plans["configurations"][config_name].get("patch_size", [128, 128, 128])

    print(f"  Trainer      : {trainer_name}")
    print(f"  Config       : {config_name}")
    print(f"  Patch size   : {patch_size}")

    return network, plans, config_name


# ──────────────────────────────────────────────────────────────────────────
# Export to ONNX
# ──────────────────────────────────────────────────────────────────────────
def export_onnx(network, patch_size, out_path: Path, opset: int, fp16: bool):
    pD, pH, pW = patch_size
    dummy = torch.zeros(1, 1, pD, pH, pW, dtype=torch.float32)

    # Force every submodule (including InstanceNorm) into eval mode so the
    # ONNX exporter captures inference-time behaviour, not training-time.
    network.eval()
    for m in network.modules():
        m.training = False

    if fp16:
        network = network.half()
        dummy = dummy.half()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Exporting to {out_path} …")

    torch.onnx.export(
        network,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch", 2: "depth", 3: "height", 4: "width"},
        },
        do_constant_folding=True,
        dynamo=False,  # force legacy TorchScript exporter (no onnxscript dependency)
    )

    model = onnx.load(str(out_path))
    onnx.checker.check_model(model)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  ONNX file size: {size_mb:.1f} MB")
    return model


# ──────────────────────────────────────────────────────────────────────────
# Validate: compare PyTorch vs ONNX Runtime outputs
# ──────────────────────────────────────────────────────────────────────────
def validate(network, out_path: Path, patch_size, fp16: bool):
    print("  Validating ONNX output against PyTorch …")
    pD, pH, pW = patch_size
    dtype = torch.float16 if fp16 else torch.float32

    # Use a smooth, brain-like signal (not pure noise) so instance norm
    # statistics are stable and PyTorch vs ONNX diffs reflect real behaviour.
    rng = np.random.default_rng(42)
    arr = rng.normal(0.0, 1.0, (1, 1, pD, pH, pW)).astype(np.float32)
    # Smooth with a simple box filter to reduce high-freq noise
    from scipy.ndimage import uniform_filter
    arr = uniform_filter(arr, size=(1, 1, 5, 5, 5)).astype(np.float32)
    dummy = torch.from_numpy(arr).to(dtype)

    network.eval()
    with torch.no_grad():
        pt_out = network(dummy).float().numpy()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy.float().numpy()})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    mean_diff = np.abs(pt_out - ort_out).mean()
    print(f"  Max diff: {max_diff:.2e}  |  Mean diff: {mean_diff:.2e}")
    if max_diff > 1e-2:
        print("  WARNING: discrepancy above 1e-2 – instance norm may still be "
              "in training mode. Check the DeprecationWarning output above.")
    else:
        print("  Validation passed ✓")


# ──────────────────────────────────────────────────────────────────────────
# Print the preprocessing constants to embed in worker.js
# ──────────────────────────────────────────────────────────────────────────
def print_worker_constants(plans: dict, config_name: str):
    cfg     = plans.get("configurations", {}).get(config_name, {})
    spacing = cfg.get("spacing", [1.0, 1.0, 1.0])
    patch   = cfg.get("patch_size", [128, 128, 128])

    print("\n  ── Paste these into web/worker.js ──────────────────────────")
    print(f"  const TARGET_SPACING = {spacing};")
    print(f"  const PATCH_SIZE     = {patch};")
    print("  ────────────────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_path  = Path(args.out)

    folds = args.fold
    # Derive per-fold output paths: web/models/meld_postop_fold0.onnx etc.
    out_stem = out_path.stem  # e.g. "meld_postop"
    out_suffix = out_path.suffix  # ".onnx"

    plans = None
    for fold in folds:
        fold_out = out_path.parent / f"{out_stem}_fold{fold}{out_suffix}"
        print(f"\n── Fold {fold} ({'─' * 50})")

        print(f"[1] Loading network (fold {fold}) from {model_dir}")
        network, plans, config_name = load_nnunet_network(model_dir, fold)

        # Derive patch size from plans unless the user overrode it
        patch = args.patch or plans["configurations"][config_name].get("patch_size", [128, 128, 128])

        print(f"[2] Exporting to ONNX  (patch={patch}, opset={args.opset}, fp16={args.fp16})")
        export_onnx(network, patch, fold_out, args.opset, args.fp16)

        if args.validate:
            print("[3] Validating")
            validate(network, fold_out, patch, args.fp16)

    print("\n[4] Preprocessing constants")
    print_worker_constants(plans, config_name)

    exported = [
        str(out_path.parent / f"{out_stem}_fold{f}{out_suffix}")
        for f in folds
    ]
    print("Done. Exported ONNX files:")
    for p in exported:
        print(f"  {p}\n")


if __name__ == "__main__":
    main()
