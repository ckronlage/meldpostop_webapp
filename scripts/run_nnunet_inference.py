#!/usr/bin/env python3
"""
MELD-PostOp nnunetv2 native inference wrapper

Thin wrapper around nnunetv2's predict_entry_point_modelfolder() — all defaults
(threading, Gaussian weighting, TTA, progress bar) are set by nnunetv2 itself.

Usage:
    conda activate meldpostop_onnx

    # Single file
    python scripts/run_nnunet_inference.py --input scan.nii.gz --output mask.nii.gz \
        [--model-dir data/Dataset001_MELDPostOp_v1.0.0] \
        [--folds 0,1,2,3,4] [--device mps|cpu|cuda] \
        [--checkpoint final|best]

    # Batch
    python scripts/run_nnunet_inference.py \
        --input-dir /data/scans/ --output-dir /data/masks/ \
        [--folds 0,1,2,3,4] [--device mps]
"""

import argparse
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

TRAINER_DIR_NAME  = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
DEFAULT_MODEL_DIR = 'data/Dataset001_MELDPostOp_v1.0.0'


def default_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def symlink_with_channel_suffix(src_files, dst_dir):
    """
    Symlink each file into dst_dir, appending _0000 if not already present.
    Returns list of (case_id, dst_path) pairs.
    """
    cases = []
    for src in src_files:
        src = Path(src).resolve()
        if src.name.endswith('.nii.gz'):
            stem, ext = src.name[:-7], '.nii.gz'
        elif src.name.endswith('.nii'):
            stem, ext = src.name[:-4], '.nii'
        else:
            continue
        case_id = stem[:-5] if stem.endswith('_0000') else stem
        dst = Path(dst_dir) / (f'{case_id}_0000{ext}')
        os.symlink(src, dst)
        cases.append((case_id, dst))
    return cases


def call_nnunet(trainer_dir, tmp_in, tmp_out, fold_list, device, checkpoint):
    """Patch sys.argv and call nnunetv2's own entry point."""
    from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder

    argv = [
        'nnUNetv2_predict_from_modelfolder',
        '-i', str(tmp_in),
        '-o', str(tmp_out),
        '-m', str(trainer_dir),
        '-f', *[str(f) for f in fold_list],
        '-device', device,
        '--disable_tta',
        '-npp', '1',
        '-nps', '1',
        '--verbose'
    ]

    sys.argv = argv
    predict_entry_point_modelfolder()


def collect_input_files(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.name.endswith('.nii') or f.name.endswith('.nii.gz'))


def copy_outputs(cases, tmp_out, output_path, batch):
    """
    Copy nnunetv2's outputs (<case_id>.nii.gz) to the requested destination.
    In single-file mode: copy to output_path exactly.
    In batch mode: copy to output_dir/<case_id>_mask.nii.gz.
    """
    out = Path(output_path)
    if not batch:
        case_id = cases[0][0]
        produced = Path(tmp_out) / f'{case_id}.nii.gz'
        if not produced.exists():
            candidates = list(Path(tmp_out).glob('*.nii*'))
            if not candidates:
                sys.exit("ERROR: nnunetv2 produced no output file")
            produced = candidates[0]
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, out)
        print(f"\n→ {out}", file=sys.stderr)
    else:
        out.mkdir(parents=True, exist_ok=True)
        for case_id, _ in cases:
            produced = Path(tmp_out) / f'{case_id}.nii.gz'
            if produced.exists():
                dest = out / f'{case_id}_mask.nii.gz'
                shutil.copy2(produced, dest)
                print(f"  → {dest}", file=sys.stderr)
            else:
                print(f"  WARNING: no output for {case_id}", file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(description='MELD-PostOp nnunetv2 inference')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input',     help='Input NIfTI file')
    g.add_argument('--input-dir', help='Input directory (batch)')
    p.add_argument('--output',     help='Output mask file (with --input)')
    p.add_argument('--output-dir', help='Output directory (with --input-dir)')
    p.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    p.add_argument('--folds', default='0,1,2,3,4')
    p.add_argument('--device', default=None, choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--checkpoint', default='final', choices=['final', 'best'])
    return p.parse_args()


def main():
    args = parse_args()

    if args.input and not args.output:
        sys.exit("ERROR: --input requires --output")
    if args.input_dir and not args.output_dir:
        sys.exit("ERROR: --input-dir requires --output-dir")

    device     = args.device or default_device()
    fold_list  = [int(f) for f in args.folds.split(',')]
    trainer_dir = Path(args.model_dir) / TRAINER_DIR_NAME
    if not trainer_dir.exists():
        sys.exit(f"ERROR: model directory not found: {trainer_dir}")

    batch       = args.input_dir is not None
    input_src   = args.input_dir if batch else args.input
    output_dst  = args.output_dir if batch else args.output

    files = collect_input_files(input_src)
    if not files:
        sys.exit(f"No .nii / .nii.gz files found in {input_src}")

    with tempfile.TemporaryDirectory() as tmp_in, \
         tempfile.TemporaryDirectory() as tmp_out:

        cases = symlink_with_channel_suffix(files, tmp_in)
        call_nnunet(trainer_dir, tmp_in, tmp_out,
                    fold_list, device, args.checkpoint)
        copy_outputs(cases, tmp_out, output_dst, batch)


if __name__ == '__main__':
    main()
