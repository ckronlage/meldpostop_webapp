/**
 * MELD-PostOp Web Worker
 *
 * Pipeline:
 *   1. Parse NIfTI header + voxel data (.nii or .nii.gz)
 *   2. Resample to model target spacing
 *   3. Foreground z-score normalisation
 *   4. Sliding-window ONNX inference (patch 128×160×112, 50 % overlap)
 *      repeated for all loaded folds, softmax outputs averaged (ensemble)
 *      Backend: WASM (WebGPU EP lacks 3D ConvTranspose support as of ORT 1.21).
 *   5. Argmax → binary mask
 *   6. Resample mask back to original spacing
 *   7. Encode result as NIfTI-1 and post as stageData
 *
 * Inference constants and sliding-window logic live in inference.js (shared
 * with the Node.js CLI so both environments run identical code).
 */

import * as ort from './wasm/ort.webgpu.bundle.min.mjs';
import { decompressIfGzip, parseNifti, encodeNifti } from './nifti.js';
import { resampleTrilinear, foregroundZScore, logitsToSegmentation } from './preprocess.js';
import { slidingWindowInference, TARGET_SPACING, PATCH_SIZE, PATCH_OVERLAP, N_CLASSES } from './inference.js';

// ── Fold selection ─────────────────────────────────────────────────────────
const FOLD_INDICES = [0]; // fold 0 only for now; change to [0,1,2,3,4] for full ensemble

// ── Globals ────────────────────────────────────────────────────────────────
let sessions = [];

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
  ort.env.wasm.wasmPaths = new URL('./wasm/', import.meta.url).href;
  sessions = [];

  // WebGPU, WebNN, and WebGL EPs all support only 2-D Conv/ConvTranspose.
  // nnU-Net 3d_fullres uses 3-D ConvTranspose throughout → WASM is the only
  // viable backend for this model.
  for (let i = 0; i < FOLD_INDICES.length; i++) {
    const foldIdx = FOLD_INDICES[i];
    log(`Loading fold ${foldIdx} / ${FOLD_INDICES[FOLD_INDICES.length - 1]}…`);
    progress((i + 0.5) / FOLD_INDICES.length, `Loading model fold ${foldIdx}`);
    const url = `./models/meld_postop_fold${foldIdx}.onnx`;
    log(`  Model: ${url.split('/').pop()}`);
    const sess = await ort.InferenceSession.create(url, {
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

  // 4. Sliding-window inference — ensemble over all loaded folds ─────────────
  const [D, H, W] = resampled.dims;
  const ensembleSoft = new Float32Array(N_CLASSES * D * H * W);

  for (let f = 0; f < sessions.length; f++) {
    log(`Running inference: fold ${f + 1} / ${sessions.length}…`);
    const foldBase = 0.15 + f * (0.70 / sessions.length);
    const foldSpan = 0.70 / sessions.length;
    const sess = sessions[f];
    const inputName  = sess.inputNames[0];
    const outputName = sess.outputNames[0];

    const runModel = async (patchData, shape) => {
      const tensor = new ort.Tensor('float32', patchData, shape);
      const result = await sess.run({ [inputName]: tensor });
      return result[outputName].data;
    };

    const foldSoft = await slidingWindowInference(
      normalised, resampled.dims, runModel, PATCH_SIZE, PATCH_OVERLAP,
      (frac) => progress(foldBase + frac * foldSpan,
        `Fold ${f + 1}/${sessions.length} – ${Math.round(frac * 100)} %`),
      f,
      log,
    );

    for (let i = 0; i < ensembleSoft.length; i++) {
      ensembleSoft[i] += foldSoft[i] / sessions.length;
    }
  }

  // 5+6. Argmax + nearest-neighbour back-resample to original space ───────────
  log('Generating mask…');
  progress(0.87, 'Argmax + resample');

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
