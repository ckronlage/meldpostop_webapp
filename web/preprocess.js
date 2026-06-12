/**
 * Preprocessing and postprocessing — nnU-Net pipeline in JS.
 *
 * All arrays use nnU-Net / SimpleITK (z, y, x) axis order,
 * consistent with parseNifti() in nifti.js.
 */

/**
 * Resample a 3D volume to a new voxel spacing using trilinear interpolation.
 *
 * Coordinate convention: endpoint-aligned, matching skimage.transform.resize
 * (which is what nnU-Net's DefaultPreprocessor uses):
 *
 *   src = dst * (sN - 1) / (dN - 1)
 *
 * This maps dst=0 → src=0 and dst=dN-1 → src=sN-1 exactly, and is exact
 * for any linear or polynomial function up to degree 1.
 *
 * data       Float32Array  in (nz, ny, nx) order
 * srcDims    [nz, ny, nx]
 * srcSpacing [sz, sy, sx]  mm
 * dstSpacing [sz', sy', sx']  mm  (TARGET_SPACING from plans.json)
 *
 * Returns { data: Float32Array, dims: [nz', ny', nx'] }
 */
export function resampleTrilinear(data, srcDims, srcSpacing, dstSpacing) {
  const [sD, sH, sW] = srcDims;
  const dstDims = srcDims.map((dim, i) => Math.round(dim * srcSpacing[i] / dstSpacing[i]));
  const [dD, dH, dW] = dstDims;
  const dst = new Float32Array(dD * dH * dW);

  // Scale: src_coord = dst_coord * (sN - 1) / (dN - 1)
  const scaleZ = dD > 1 ? (sD - 1) / (dD - 1) : 0;
  const scaleY = dH > 1 ? (sH - 1) / (dH - 1) : 0;
  const scaleX = dW > 1 ? (sW - 1) / (dW - 1) : 0;

  for (let z = 0; z < dD; z++) {
    const sz = z * scaleZ;
    const z0 = Math.min(Math.floor(sz), sD - 1);
    const z1 = Math.min(z0 + 1, sD - 1);
    const tz = sz - z0;

    for (let y = 0; y < dH; y++) {
      const sy = y * scaleY;
      const y0 = Math.min(Math.floor(sy), sH - 1);
      const y1 = Math.min(y0 + 1, sH - 1);
      const ty = sy - y0;

      // Precompute row offsets to avoid multiplications in the inner loop
      const z0y0 = z0 * sH * sW + y0 * sW;
      const z0y1 = z0 * sH * sW + y1 * sW;
      const z1y0 = z1 * sH * sW + y0 * sW;
      const z1y1 = z1 * sH * sW + y1 * sW;

      for (let x = 0; x < dW; x++) {
        const sx = x * scaleX;
        const x0 = Math.min(Math.floor(sx), sW - 1);
        const x1 = Math.min(x0 + 1, sW - 1);
        const tx = sx - x0;

        dst[z * dH * dW + y * dW + x] =
          (1 - tz) * ((1 - ty) * ((1 - tx) * data[z0y0 + x0] + tx * data[z0y0 + x1]) +
                           ty  * ((1 - tx) * data[z0y1 + x0] + tx * data[z0y1 + x1])) +
               tz  * ((1 - ty) * ((1 - tx) * data[z1y0 + x0] + tx * data[z1y0 + x1]) +
                           ty  * ((1 - tx) * data[z1y1 + x0] + tx * data[z1y1 + x1]));
      }
    }
  }
  return { data: dst, dims: dstDims };
}

/**
 * Foreground z-score normalization (nnU-Net ZScoreNormalization for MRI).
 *
 * Clips foreground (non-zero) voxels to [0.5th, 99.5th] percentile,
 * then normalises ALL voxels by (clipped - mean) / std, where mean and std
 * are computed over foreground only.
 *
 * Returns { data: Float32Array, mean, std, lo, hi }
 */
export function foregroundZScore(data) {
  // Collect into a Float32Array so .sort() operates on float32 bit values,
  // matching numpy.sort() on a float32 array (same bit-level comparison).
  let count = 0;
  for (let i = 0; i < data.length; i++) if (data[i] !== 0) count++;
  if (count === 0) return { data: new Float32Array(data.length), mean: 0, std: 1, lo: 0, hi: 0 };

  const fg = new Float32Array(count);
  let j = 0;
  for (let i = 0; i < data.length; i++) if (data[i] !== 0) fg[j++] = data[i];
  fg.sort(); // TypedArray numeric sort — matches numpy's float32 sort
  const lo = fg[Math.floor(0.005 * fg.length)];
  const hi = fg[Math.floor(0.995 * fg.length)];

  let sum = 0, sum2 = 0;
  for (const v of fg) {
    const c = Math.min(Math.max(v, lo), hi);
    sum  += c;
    sum2 += c * c;
  }
  const mean = sum / fg.length;
  const std  = Math.sqrt(sum2 / fg.length - mean * mean) + 1e-8;

  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const c = Math.min(Math.max(data[i], lo), hi);
    out[i] = (c - mean) / std;
  }
  return { data: out, mean, std, lo, hi };
}

/**
 * Nearest-neighbour resampling with endpoint-aligned coordinates.
 * Matches skimage.transform.resize(order=0, mode='edge') which nnU-Net uses
 * (do_separate_z=False path) when back-resampling segmentation masks to the
 * original image space.
 *
 * Coordinate convention (same as resampleTrilinear):
 *   src = dst * (sN - 1) / (dN - 1)
 *
 * data     Uint8Array  flat [nz, ny, nx]
 * srcDims  [nz, ny, nx]   source (resampled-space) dimensions
 * dstDims  [nz', ny', nx'] destination (original-space) dimensions
 * Returns  Uint8Array  flat [nz', ny', nx']
 */
export function resampleNearestNeighbor(data, srcDims, dstDims) {
  const [sD, sH, sW] = srcDims;
  const [dD, dH, dW] = dstDims;
  const dst = new Uint8Array(dD * dH * dW);

  const scaleZ = dD > 1 ? (sD - 1) / (dD - 1) : 0;
  const scaleY = dH > 1 ? (sH - 1) / (dH - 1) : 0;
  const scaleX = dW > 1 ? (sW - 1) / (dW - 1) : 0;

  for (let z = 0; z < dD; z++) {
    const sz = Math.min(sD - 1, Math.max(0, Math.round(z * scaleZ)));
    for (let y = 0; y < dH; y++) {
      const sy = Math.min(sH - 1, Math.max(0, Math.round(y * scaleY)));
      const srcRow = sz * sH * sW + sy * sW;
      const dstRow = z  * dH * dW + y  * dW;
      for (let x = 0; x < dW; x++) {
        const sx = Math.min(sW - 1, Math.max(0, Math.round(x * scaleX)));
        dst[dstRow + x] = data[srcRow + sx];
      }
    }
  }
  return dst;
}

/**
 * convert_predicted_logits_to_segmentation_with_correct_shape (nnU-Net equivalent).
 *
 * Takes the ensemble-averaged softmax output from sliding-window inference,
 * argmaxes it to a binary mask, then resamples back to original image space
 * using endpoint-aligned nearest-neighbour interpolation.
 *
 * softmax   Float32Array  [nVoxels * nClasses] interleaved as
 *                         [bg0, cav0, bg1, cav1, …]
 * resampledDims  [D, H, W]  dimensions in the resampled (model) space
 * origDims       [D, H, W]  dimensions in the original image space
 * nClasses   number of output classes (default 2)
 *
 * Returns Uint8Array flat [D_orig, H_orig, W_orig] with values 0…nClasses-1
 */
export function logitsToSegmentation(softmax, resampledDims, origDims, nClasses = 2) {
  const n = resampledDims[0] * resampledDims[1] * resampledDims[2];

  // Argmax over class axis
  const maskResampled = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let bestClass = 0, bestVal = softmax[i * nClasses];
    for (let c = 1; c < nClasses; c++) {
      const v = softmax[i * nClasses + c];
      if (v > bestVal) { bestVal = v; bestClass = c; }
    }
    maskResampled[i] = bestClass;
  }

  // Nearest-neighbour back-resample to original space
  return resampleNearestNeighbor(maskResampled, resampledDims, origDims);
}
