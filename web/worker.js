/**
 * MELD-PostOp Web Worker
 *
 * Pipeline:
 *   1. Parse NIfTI header + voxel data (.nii or .nii.gz)
 *   2. Resample to model target spacing
 *   3. Foreground z-score normalisation
 *   4. Sliding-window ONNX inference (patch 128×160×112, 50 % overlap)
 *      repeated for all 5 folds, softmax outputs averaged (ensemble)
 *   5. Argmax → binary mask
 *   6. Resample mask back to original spacing
 *   7. Encode result as NIfTI-1 and post as stageData
 *
 * Constants below match Dataset001_MELDPostOp_v1.0.0 plans.json (3d_fullres).
 * Update TARGET_SPACING / PATCH_SIZE if you retrain with different settings.
 */

import * as ort from './wasm/ort.webgpu.bundle.min.mjs';
import { decompressIfGzip, parseNifti, encodeNifti } from './nifti.js';
import { resampleTrilinear, foregroundZScore, logitsToSegmentation } from './preprocess.js';

// ── Model constants (from MELD-PostOp plans.json) ──────────────────────────
const MODEL_URLS = [0].map(i => `./models/meld_postop_fold${i}.onnx`); // fold 0 only for now
const TARGET_SPACING = [0.9999988079071045, 0.9375, 0.9375]; // mm, from plans.json
const PATCH_SIZE = [128, 160, 112];
const PATCH_OVERLAP = 0.5; // 50 % overlap per axis
const N_CLASSES = 2; // background + resection cavity

// ── Globals ────────────────────────────────────────────────────────────────
let sessions = [];
const GAUSS_MAP = buildGaussianMap(PATCH_SIZE); // constant — compute once

// ── Messaging helpers ──────────────────────────────────────────────────────
function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}
function log(msg) { post('log', { message: msg }); }
function progress(fraction, text = '') { post('progress', { fraction, text }); }
function error(msg) { post('error', { message: msg }); }

// ── Worker message router ──────────────────────────────────────────────────
self.onmessage = async ({ data }) => {
  try {
    switch (data.type) {
      case 'init':
        await handleInit();
        break;
      case 'run':
        await handleRun(data);
        break;
      default:
        break;
    }
  } catch (e) {
    error(e?.message ?? String(e));
  }
};

// ── Init ───────────────────────────────────────────────────────────────────
async function handleInit() {
  ort.env.wasm.wasmPaths = '/web/wasm/';
  sessions = [];
  for (let i = 0; i < MODEL_URLS.length; i++) {
    log(`Loading fold ${i} / ${MODEL_URLS.length - 1}…`);
    progress((i + 0.5) / MODEL_URLS.length, `Loading model fold ${i}`);
    const sess = await ort.InferenceSession.create(MODEL_URLS[i], {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    sessions.push(sess);
  }
  log(`${sessions.length === 1 ? 'Fold 0 model' : `All ${sessions.length} fold models`} ready.`);
  post("initialized");
}

// ── Run ────────────────────────────────────────────────────────────────────
async function handleRun({ inputData }) {
  if (!sessions.length) { error('Models not initialised. Call init first.'); return; }

  // 1. Parse NIfTI ──────────────────────────────────────────────────────────
  log('Parsing NIfTI…');
  progress(0.02, 'Parsing NIfTI');
  const decompressed = await decompressIfGzip(inputData);
  const nii = parseNifti(decompressed);
  log(`NIfTI: native dims ${nii.origDims.join('×')}, spacing ${nii.pixdim.map(v => v.toFixed(3)).join('×')} mm [z,y,x], dtype ${nii.datatype}, scl slope=${nii.sclSlope} inter=${nii.sclInter}`);

  // 2. Resample to target spacing ───────────────────────────────────────────
  log('Resampling to 1 mm isotropic…');
  progress(0.08, 'Resampling');
  const resampled = resampleTrilinear(nii.data, nii.dims, nii.pixdim, TARGET_SPACING);
  log(`Resampled shape: ${resampled.dims.join('×')}`);

  // 3. Foreground z-score normalisation ─────────────────────────────────────
  log('Normalising intensity…');
  progress(0.12, 'Normalising');
  const norm = foregroundZScore(resampled.data);
  const normalised = norm.data;
  log(`Normalised: mean=${norm.mean.toFixed(1)}, std=${norm.std.toFixed(1)}, clip=[${norm.lo.toFixed(1)}, ${norm.hi.toFixed(1)}]`);

  // 4. Sliding-window inference — ensemble over all 5 folds ─────────────────
  const [D, H, W] = resampled.dims;
  const ensembleSoft = new Float32Array(N_CLASSES * D * H * W);

  for (let f = 0; f < sessions.length; f++) {
    log(`Running inference: fold ${f + 1} / ${sessions.length}…`);
    const foldBase = 0.15 + f * (0.70 / sessions.length);
    const foldSpan = 0.70 / sessions.length;

    const foldSoft = await slidingWindowInference(
      normalised, resampled.dims, sessions[f], PATCH_SIZE, PATCH_OVERLAP,
      (frac) => progress(foldBase + frac * foldSpan,
        `Fold ${f + 1}/${sessions.length} – ${Math.round(frac * 100)} %`),
      f,
    );

    for (let i = 0; i < ensembleSoft.length; i++) {
      ensembleSoft[i] += foldSoft[i] / sessions.length;
    }
  }

  // 5+6. Argmax + nearest-neighbour back-resample to original space ───────────
  log('Generating mask…');
  progress(0.87, 'Argmax + resample');

  // Diagnostic: max cavity probability before argmax
  let maxCavityProb = 0;
  for (let i = 0; i < D * H * W; i++) {
    const p = ensembleSoft[N_CLASSES * i + 1];
    if (p > maxCavityProb) maxCavityProb = p;
  }

  const maskOrig = logitsToSegmentation(ensembleSoft, resampled.dims, nii.dims, N_CLASSES);
  const maskVoxels = maskOrig.reduce((s, v) => s + v, 0);
  log(`Mask: ${maskVoxels} positive voxels, max cavity prob = ${maxCavityProb.toFixed(4)}`);

  // 7. Encode output NIfTI ──────────────────────────────────────────────────
  log('Encoding NIfTI output…');
  progress(0.96, 'Encoding');
  const outNii = encodeNifti(maskOrig, nii.origDims, nii.header);

  // 8. Post result ──────────────────────────────────────────────────────────
  post("stageData", {
    stage: 'segmentation',
    data: outNii,
    filename: 'meld_postop_mask.nii',
    mimeType: 'application/octet-stream',
  });

  progress(1.0, 'Complete');
  post("complete", { stages: ['segmentation'] });
}

// ═══════════════════════════════════════════════════════════════════════════
// Gaussian importance map for sliding-window overlap weighting
// ═══════════════════════════════════════════════════════════════════════════
function buildGaussianMap(patchSize) {
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

// ═══════════════════════════════════════════════════════════════════════════
// Sliding-window inference
// Returns flat Float32Array shape [N_CLASSES, D, H, W] interleaved as
// [class0_vox0, class1_vox0, class0_vox1, class1_vox1, …]
// ═══════════════════════════════════════════════════════════════════════════

// Matches nnU-Net compute_steps_for_sliding_window:
// guarantees the first step is always 0 and the last is always imgD-patchD,
// so every voxel is covered by at least one patch.
function computeSlidingSteps(imgDims, patchDims, stepSize) {
  return imgDims.map((imgD, i) => {
    const pD = patchDims[i];
    if (imgD <= pD) return [0];           // whole axis fits in one patch
    const targetStep = pD * stepSize;
    const numSteps   = Math.ceil((imgD - pD) / targetStep) + 1;
    const maxStep    = imgD - pD;
    const actualStep = maxStep / (numSteps - 1);
    return Array.from({ length: numSteps }, (_, k) => Math.round(actualStep * k));
  });
}

async function slidingWindowInference(data, dims, sess, patchSize, overlap, onProgress, foldIdx = 0) {
  const [D, H, W] = dims;
  const [pD, pH, pW] = patchSize;
  const pN = pD * pH * pW;

  const accumSoft   = new Float32Array(N_CLASSES * D * H * W);
  const accumWeight = new Float32Array(D * H * W);

  const [stepsZ, stepsY, stepsX] = computeSlidingSteps([D, H, W], patchSize, overlap);
  const origins = [];
  for (const z0 of stepsZ)
    for (const y0 of stepsY)
      for (const x0 of stepsX)
        origins.push([z0, y0, x0]);

  const inputName  = sess.inputNames[0];
  const outputName = sess.outputNames[0];

  for (let i = 0; i < origins.length; i++) {
    const [z0, y0, x0] = origins[i];

    // Extract patch — zero-pad any out-of-bounds coordinates (handles
    // volumes smaller than the patch size in any axis).
    const patch = new Float32Array(pN); // initialised to 0
    for (let z = 0; z < pD; z++) {
      const gz = z + z0;  if (gz < 0 || gz >= D) continue;
      for (let y = 0; y < pH; y++) {
        const gy = y + y0;  if (gy < 0 || gy >= H) continue;
        const srcRow = gz * H * W + gy * W;
        const dstRow = z  * pH * pW + y * pW;
        for (let x = 0; x < pW; x++) {
          const gx = x + x0;  if (gx < 0 || gx >= W) continue;
          patch[dstRow + x] = data[srcRow + gx];
        }
      }
    }

    // Run model: input shape [1, 1, pD, pH, pW]
    const tensor = new ort.Tensor('float32', patch, [1, 1, pD, pH, pW]);
    const result = await sess.run({ [inputName]: tensor });
    const logits = result[outputName].data; // layout: [2, pD, pH, pW] class-first

    // Diagnostics on first patch of first fold only
    if (i === 0 && foldIdx === 0) {
      let l0min = Infinity, l0max = -Infinity, l1min = Infinity, l1max = -Infinity;
      for (let k = 0; k < pN; k++) {
        if (logits[k]      < l0min) l0min = logits[k];
        if (logits[k]      > l0max) l0max = logits[k];
        if (logits[pN + k] < l1min) l1min = logits[pN + k];
        if (logits[pN + k] > l1max) l1max = logits[pN + k];
      }
      log(`Fold 0 patch 0 logits — bg: [${l0min.toFixed(2)}, ${l0max.toFixed(2)}]  cavity: [${l1min.toFixed(2)}, ${l1max.toFixed(2)}]`);
    }

    // Softmax + weighted accumulate (skip padded / out-of-bounds voxels)
    for (let z = 0; z < pD; z++) {
      const gz = z + z0;  if (gz < 0 || gz >= D) continue;
      for (let y = 0; y < pH; y++) {
        const gy = y + y0;  if (gy < 0 || gy >= H) continue;
        for (let x = 0; x < pW; x++) {
          const gx = x + x0;  if (gx < 0 || gx >= W) continue;
          const li = z * pH * pW + y * pW + x;
          const gi = gz * H * W + gy * W + gx;
          const w  = GAUSS_MAP[li];
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

    if (i % 5 === 0) {
      onProgress((i + 1) / origins.length);
      await new Promise(r => setTimeout(r, 0));   // yield to avoid blocking
    }
  }

  // Normalise by accumulated weights
  for (let i = 0; i < D * H * W; i++) {
    const w = accumWeight[i] || 1;
    accumSoft[i * N_CLASSES]     /= w;
    accumSoft[i * N_CLASSES + 1] /= w;
  }

  return accumSoft;
}

