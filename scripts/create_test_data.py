#!/usr/bin/env python3
"""
Create test NIfTI files and expected JSON for verifying the JS pipeline.

Tests:
  1. parseNifti()      — axis ordering and spacing match SimpleITKIO
  2. resampleTrilinear — endpoint-aligned trilinear matches skimage.transform.resize
  3. foregroundZScore  — percentile-clipped z-score statistics match

Run from the project root:
    conda activate meldpostop
    python scripts/create_test_data.py
"""

import json
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize as sk_resize
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data" / "tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SPACING = [0.9999988079071045, 0.9375, 0.9375]   # [sz, sy, sx] from plans.json


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — parseNifti: axis ordering and spacing
# ══════════════════════════════════════════════════════════════════════════════
# Volume defined in (nz, ny, nx) / SimpleITK order.
# Values: data_zyx[z, y, x] = z*ny*nx + y*nx + x  (linear index in ZYX)
# After writing and reading, SimpleITKIO returns the same array unchanged.

nz, ny, nx = 12, 16, 20
sz, sy, sx = 1.0, 0.9375, 0.9375          # spacing in (z, y, x) order

data_zyx = np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx)

img = sitk.GetImageFromArray(data_zyx)    # shape (nz,ny,nx) → sitk size (nx,ny,nz)
img.SetSpacing((sx, sy, sz))              # SimpleITK spacing: (x, y, z)

nii_path = OUT_DIR / "test_gradient.nii"
sitk.WriteImage(img, str(nii_path))
print(f"Wrote  {nii_path}")

# Verify round-trip through SimpleITKIO
img_r       = sitk.ReadImage(str(nii_path))
arr_zyx     = sitk.GetArrayFromImage(img_r)         # (nz, ny, nx)
spacing_zyx = list(img_r.GetSpacing())[::-1]         # [sz, sy, sx]

assert arr_zyx.shape == (nz, ny, nx) and np.allclose(arr_zyx, data_zyx)

t1_expected = {
    "description": "parseNifti() — axis ordering and spacing match SimpleITKIO",
    "dims_zyx":         list(arr_zyx.shape),          # [12, 16, 20]
    "pixdim_zyx":       spacing_zyx,                  # [1.0, 0.9375, 0.9375]
    "nifti_dims_xyz":   [nx, ny, nz],                 # [20, 16, 12]
    "nifti_pixdim_xyz": list(img_r.GetSpacing()),     # [0.9375, 0.9375, 1.0]
    "spot_checks": {
        "data[0]":    float(arr_zyx[0,  0,  0]),      # 0
        "data[1]":    float(arr_zyx[0,  0,  1]),      # 1
        "data[20]":   float(arr_zyx[0,  1,  0]),      # 20   (y stride = nx)
        "data[320]":  float(arr_zyx[1,  0,  0]),      # 320  (z stride = ny*nx)
        "data[3839]": float(arr_zyx[-1, -1, -1]),     # 3839
    },
}
t1_path = OUT_DIR / "test_gradient_expected.json"
with open(t1_path, "w") as f:
    json.dump(t1_expected, f, indent=2)
print(f"Wrote  {t1_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — PreprocessAdapterFromNpy: resampleTrilinear + foregroundZScore
# ══════════════════════════════════════════════════════════════════════════════
# Use 1 mm isotropic spacing (representative of typical T1w MRI).
# The JS resampleTrilinear uses endpoint-aligned coordinates, matching
# skimage.transform.resize (which nnU-Net's DefaultPreprocessor uses).
# For a linear test function, trilinear = cubic = exact; so Python order=1
# and our JS trilinear must agree to floating-point precision.

nz2, ny2, nx2 = 12, 16, 20
sz2, sy2, sx2 = 1.0, 1.0, 1.0            # 1 mm isotropic

data_zyx2 = np.arange(nz2 * ny2 * nx2, dtype=np.float32).reshape(nz2, ny2, nx2)

img2 = sitk.GetImageFromArray(data_zyx2)
img2.SetSpacing((sx2, sy2, sz2))
preproc_path = OUT_DIR / "test_preprocess.nii"
sitk.WriteImage(img2, str(preproc_path))
print(f"\nWrote  {preproc_path}")

# Destination dimensions (same formula as JS resampleTrilinear)
dst_dims = [
    round(nz2 * sz2 / TARGET_SPACING[0]),   # ≈ 12
    round(ny2 * sy2 / TARGET_SPACING[1]),   # = 17
    round(nx2 * sx2 / TARGET_SPACING[2]),   # = 21
]
print(f"  Resampling {[nz2, ny2, nx2]} → {dst_dims}")

# --- Trilinear resampling ---
# skimage.transform.resize uses endpoint-aligned coordinates and order=1 = trilinear.
# nnU-Net's DefaultPreprocessor calls the same function (with order=3 for real data,
# but for our linear test function all orders give identical results).
resampled = sk_resize(
    data_zyx2, dst_dims,
    order=1, mode="edge", anti_aliasing=False, preserve_range=True,
).astype(np.float32)

print(f"  Resampled shape: {list(resampled.shape)}")

# Manually compute expected values for the spot-check positions using the
# endpoint-aligned formula  src = dst * (sN-1)/(dN-1):
def expected_trilinear(z_d, y_d, x_d):
    """Analytical value for our linear test function after trilinear resample."""
    sz_c = z_d * (nz2 - 1) / (dst_dims[0] - 1) if dst_dims[0] > 1 else 0
    sy_c = y_d * (ny2 - 1) / (dst_dims[1] - 1) if dst_dims[1] > 1 else 0
    sx_c = x_d * (nx2 - 1) / (dst_dims[2] - 1) if dst_dims[2] > 1 else 0
    return sz_c * ny2 * nx2 + sy_c * nx2 + sx_c   # linear function is exact

for pos in [(0, 0, 0), (0, 8, 10), (5, 8, 10), (dst_dims[0]-1, dst_dims[1]-1, dst_dims[2]-1)]:
    analytic = expected_trilinear(*pos)
    python   = float(resampled[pos])
    diff     = abs(analytic - python)
    print(f"  resample{list(pos)}: analytic={analytic:.6f}  python={python:.6f}  diff={diff:.2e}")

# --- Normalization ---
# Matches JS foregroundZScore: percentile clip over non-zero voxels, then z-score all.
fg = np.sort(resampled[resampled != 0])
lo_idx = int(0.005 * len(fg))
hi_idx = int(0.995 * len(fg))
lo = float(fg[lo_idx])
hi = float(fg[hi_idx])

fg_clipped = np.clip(fg, lo, hi)
mean = float(fg_clipped.mean())
std  = float(fg_clipped.std()) + 1e-8

normalized = ((np.clip(resampled, lo, hi) - mean) / std).astype(np.float32)

print(f"  Normalization: lo={lo:.2f}, hi={hi:.2f}, mean={mean:.4f}, std={std:.4f}")

t2_expected = {
    "description": "PreprocessAdapterFromNpy — resampleTrilinear + foregroundZScore",
    "input": {
        "nifti_dims_xyz":   [nx2, ny2, nz2],
        "nifti_spacing_xyz": [sx2, sy2, sz2],
    },
    "target_spacing_zyx": TARGET_SPACING,
    "resampled_dims_zyx": dst_dims,
    # Trilinear spot-checks: for a linear function, trilinear is exact.
    "resampled_spot_checks": {
        "0_0_0":   float(resampled[0, 0, 0]),
        "0_8_10":  float(resampled[0, 8, 10]),
        "5_8_10":  float(resampled[5, 8, 10]),
        "last":    float(resampled[-1, -1, -1]),
    },
    "normalization": {
        "lo":   lo,
        "hi":   hi,
        "mean": mean,
        "std":  float(fg_clipped.std()),   # without 1e-8 epsilon, for clean comparison
    },
    "normalized_spot_checks": {
        "0_0_0":   float(normalized[0, 0, 0]),
        "5_8_10":  float(normalized[5, 8, 10]),
        "last":    float(normalized[-1, -1, -1]),
    },
}

t2_path = OUT_DIR / "test_preprocess_expected.json"
with open(t2_path, "w") as f:
    json.dump(t2_expected, f, indent=2)
print(f"Wrote  {t2_path}")
print(f"\nExpected preprocessing output:\n{json.dumps(t2_expected, indent=2)}")
