/**
 * Pure sliding-window ONNX inference — shared by web worker and Node.js CLI.
 *
 * The caller provides a `runModel` callback that hides the ONNX runtime binding:
 *
 *   runModel(patchData: Float32Array, shape: number[]) → Promise<Float32Array>
 *
 * Both ort.js (browser WASM) and onnxruntime-node expose the same Tensor /
 * InferenceSession JS API, so the callback looks identical in both environments.
 *
 * Constants match Dataset001_MELDPostOp_v1.0.0 plans.json (3d_fullres).
 */

export const TARGET_SPACING = [0.9999988079071045, 0.9375, 0.9375]; // mm [sz, sy, sx]
export const PATCH_SIZE      = [128, 160, 112];                       // [D, H, W]
export const PATCH_OVERLAP   = 0.5;   // 50 % overlap per axis (nnU-Net default)
export const N_CLASSES       = 2;     // background + resection cavity

/**
 * Build the Gaussian importance map used for overlap weighting.
 * Computed once per patch size; pass the result into slidingWindowInference.
 */
export function buildGaussianMap(patchSize) {
  const [pD, pH, pW] = patchSize;
  const sigma = 0.125; // nnU-Net default
  const map = new Float32Array(pD * pH * pW);
  for (let z = 0; z < pD; z++) {
    for (let y = 0; y < pH; y++) {
      for (let x = 0; x < pW; x++) {
        const dz = (z - pD / 2) / (pD * sigma);
        const dy = (y - pH / 2) / (pH * sigma);
        const dx = (x - pW / 2) / (pW * sigma);
        map[z * pH * pW + y * pW + x] = Math.exp(-0.5 * (dz * dz + dy * dy + dx * dx));
      }
    }
  }
  return map;
}

/**
 * Compute patch start positions along one axis.
 * Matches nnU-Net compute_steps_for_sliding_window: first step is always 0,
 * last is always imgD-patchD, every voxel covered by at least one patch.
 */
export function computeSlidingSteps(imgDims, patchDims, stepSize) {
  return imgDims.map((imgD, i) => {
    const pD = patchDims[i];
    if (imgD <= pD) return [0];
    const targetStep = pD * stepSize;
    const numSteps   = Math.ceil((imgD - pD) / targetStep) + 1;
    const maxStep    = imgD - pD;
    const actualStep = maxStep / (numSteps - 1);
    return Array.from({ length: numSteps }, (_, k) => Math.round(actualStep * k));
  });
}

/**
 * Sliding-window inference over a single fold.
 *
 * @param {Float32Array} data        Normalised volume [D*H*W]
 * @param {number[]}     dims        [D, H, W]
 * @param {Function}     runModel    async (Float32Array patch, number[] shape) → Float32Array logits
 * @param {number[]}     patchSize   [pD, pH, pW]
 * @param {number}       overlap     Step fraction (0–1), e.g. 0.25
 * @param {Function}     onProgress  (fraction: number) → void
 * @param {number}       foldIdx     Used only for first-patch diagnostics log
 * @param {Function}     [onLog]     Optional (msg: string) → void for diagnostic messages
 *
 * Returns Float32Array [N_CLASSES * D * H * W] interleaved as
 *   [bg_vox0, cav_vox0, bg_vox1, cav_vox1, …]
 */
export async function slidingWindowInference(
  data, dims, runModel, patchSize, overlap, onProgress, foldIdx = 0, onLog = null,
) {
  const [D, H, W] = dims;
  const [pD, pH, pW] = patchSize;
  const pN = pD * pH * pW;
  const gaussMap = buildGaussianMap(patchSize);

  const accumSoft   = new Float32Array(N_CLASSES * D * H * W);
  const accumWeight = new Float32Array(D * H * W);

  const [stepsZ, stepsY, stepsX] = computeSlidingSteps([D, H, W], patchSize, overlap);
  const origins = [];
  for (const z0 of stepsZ)
    for (const y0 of stepsY)
      for (const x0 of stepsX)
        origins.push([z0, y0, x0]);

  for (let i = 0; i < origins.length; i++) {
    const [z0, y0, x0] = origins[i];

    // Extract patch — zero-pad any out-of-bounds coordinates.
    const patch = new Float32Array(pN);
    for (let z = 0; z < pD; z++) {
      const gz = z + z0; if (gz < 0 || gz >= D) continue;
      for (let y = 0; y < pH; y++) {
        const gy = y + y0; if (gy < 0 || gy >= H) continue;
        const srcRow = gz * H * W + gy * W;
        const dstRow = z  * pH * pW + y * pW;
        for (let x = 0; x < pW; x++) {
          const gx = x + x0; if (gx < 0 || gx >= W) continue;
          patch[dstRow + x] = data[srcRow + gx];
        }
      }
    }

    // Run model: input shape [1, 1, pD, pH, pW]; output layout [2, pD, pH, pW] class-first.
    const logits = await runModel(patch, [1, 1, pD, pH, pW]);

    // Diagnostics on first patch of first fold only.
    if (i === 0 && foldIdx === 0 && onLog) {
      let l0min = Infinity, l0max = -Infinity, l1min = Infinity, l1max = -Infinity;
      for (let k = 0; k < pN; k++) {
        if (logits[k]      < l0min) l0min = logits[k];
        if (logits[k]      > l0max) l0max = logits[k];
        if (logits[pN + k] < l1min) l1min = logits[pN + k];
        if (logits[pN + k] > l1max) l1max = logits[pN + k];
      }
      onLog(`Fold 0 patch 0 logits — bg: [${l0min.toFixed(2)}, ${l0max.toFixed(2)}]  cavity: [${l1min.toFixed(2)}, ${l1max.toFixed(2)}]`);
    }

    // Softmax + Gaussian-weighted accumulation (skip out-of-bounds voxels).
    for (let z = 0; z < pD; z++) {
      const gz = z + z0; if (gz < 0 || gz >= D) continue;
      for (let y = 0; y < pH; y++) {
        const gy = y + y0; if (gy < 0 || gy >= H) continue;
        for (let x = 0; x < pW; x++) {
          const gx = x + x0; if (gx < 0 || gx >= W) continue;
          const li = z * pH * pW + y * pW + x;
          const gi = gz * H * W + gy * W + gx;
          const w  = gaussMap[li];
          const l0 = logits[li];
          const l1 = logits[pN + li];
          const maxL = l0 > l1 ? l0 : l1;
          const e0 = Math.exp(l0 - maxL);
          const e1 = Math.exp(l1 - maxL);
          const inv = w / (e0 + e1);
          accumSoft[gi * N_CLASSES]     += e0 * inv;
          accumSoft[gi * N_CLASSES + 1] += e1 * inv;
          accumWeight[gi]               += w;
        }
      }
    }

    if (onProgress && i % 5 === 0) onProgress((i + 1) / origins.length);
  }

  // Normalise by accumulated weights.
  for (let i = 0; i < D * H * W; i++) {
    const w = accumWeight[i] || 1;
    accumSoft[i * N_CLASSES]     /= w;
    accumSoft[i * N_CLASSES + 1] /= w;
  }

  return accumSoft;
}
