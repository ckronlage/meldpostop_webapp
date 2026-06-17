/**
 * MELD-PostOp CLI inference — Node.js
 *
 * Runs the identical inference pipeline as the browser web worker, using the
 * same shared inference.js / preprocess.js / nifti.js modules. The only
 * environment-specific pieces are:
 *   - File I/O: node:fs instead of fetch
 *   - Gzip decompression: node:zlib instead of DecompressionStream
 *   - ONNX runtime: onnxruntime-node instead of ort.js WASM
 *
 * Usage:
 *   # Single file
 *   node scripts/run_inference.mjs --input scan.nii.gz --output mask.nii \
 *     [--model-dir web/models/] [--folds 0] [--threads 4]
 *
 *   # Batch (input directory → output directory)
 *   node scripts/run_inference.mjs --input-dir /data/scans/ --output-dir /data/masks/ \
 *     [--model-dir web/models/] [--folds 0,1,2,3,4] [--threads 4]
 *
 * --folds  Comma-separated fold indices to load (default: 0).
 *          Use 0,1,2,3,4 for the full 5-fold ensemble.
 * --model-dir  Directory containing meld_postop_fold{n}.onnx (and optionally
 *              meld_postop_fold{n}_int8.onnx). Default: web/models/
 */

import * as fs   from 'node:fs';
import * as path from 'node:path';
import * as zlib from 'node:zlib';
import * as ort  from 'onnxruntime-node';

import { slidingWindowInference, TARGET_SPACING, PATCH_SIZE, PATCH_OVERLAP, N_CLASSES }
  from '../web/inference.js';
import { resampleTrilinear, foregroundZScore, logitsToSegmentation }
  from '../web/preprocess.js';
import { parseNifti, encodeNifti }
  from '../web/nifti.js';

// ── CLI argument parsing ───────────────────────────────────────────────────

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2);
      args[key] = argv[i + 1] ?? true;
      i++;
    }
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));

const modelDir  = args['model-dir'] ?? 'web/models/';
const foldList  = (args['folds'] ?? '0').split(',').map(Number);
const numThreads = args['threads'] ? parseInt(args['threads'], 10) : 0; // 0 = ORT default
const inputFile  = args['input']      ?? null;
const outputFile = args['output']     ?? null;
const inputDir   = args['input-dir']  ?? null;
const outputDir  = args['output-dir'] ?? null;

if ((!inputFile && !inputDir) || (inputFile && inputDir)) {
  console.error('Provide exactly one of --input <file> or --input-dir <dir>');
  process.exit(1);
}
if (inputFile && !outputFile) {
  console.error('--input requires --output');
  process.exit(1);
}
if (inputDir && !outputDir) {
  console.error('--input-dir requires --output-dir');
  process.exit(1);
}

// ── Logging helpers ────────────────────────────────────────────────────────

function log(msg) { process.stderr.write(msg + '\n'); }

// ── Model loading ──────────────────────────────────────────────────────────

function resolveModelPath(foldIdx, modelDir) {
  const int8 = path.join(modelDir, `meld_postop_fold${foldIdx}_int8.onnx`);
  const fp32 = path.join(modelDir, `meld_postop_fold${foldIdx}.onnx`);
  if (fs.existsSync(int8)) return { filePath: int8, quantized: true };
  return { filePath: fp32, quantized: false };
}

async function loadSessions(foldIndices, modelDir) {
  const sessions = [];
  for (const foldIdx of foldIndices) {
    const { filePath, quantized } = resolveModelPath(foldIdx, modelDir);
    if (!fs.existsSync(filePath)) {
      throw new Error(`Model file not found: ${filePath}`);
    }
    log(`  Loading fold ${foldIdx} — ${quantized ? 'INT8' : 'FP32'}: ${path.basename(filePath)}`);
    const sessionOptions = {};
    if (numThreads > 0) {
      sessionOptions.intraOpNumThreads = numThreads;
    }
    const sess = await ort.InferenceSession.create(filePath, sessionOptions);
    sessions.push(sess);
  }
  return sessions;
}

// ── NIfTI file I/O ─────────────────────────────────────────────────────────

function readNifti(filePath) {
  let buf = fs.readFileSync(filePath);
  // Decompress .nii.gz (gzip magic bytes: 0x1f 0x8b)
  if (buf[0] === 0x1f && buf[1] === 0x8b) {
    buf = zlib.gunzipSync(buf);
  }
  // Convert Node.js Buffer to ArrayBuffer
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function writeNifti(filePath, arrayBuffer) {
  fs.writeFileSync(filePath, Buffer.from(arrayBuffer));
}

// ── Single-file inference ──────────────────────────────────────────────────

async function runOne(inputPath, outputPath, sessions) {
  const t0 = Date.now();
  log(`\n[${path.basename(inputPath)}]`);

  // 1. Parse NIfTI
  const ab  = readNifti(inputPath);
  const nii = parseNifti(ab);
  log(`  dims ${nii.origDims.join('×')}, spacing ${nii.pixdim.map(v => v.toFixed(3)).join('×')} mm [z,y,x], dtype ${nii.datatype}`);

  // 2. Resample
  const resampled = resampleTrilinear(nii.data, nii.dims, nii.pixdim, TARGET_SPACING);
  log(`  resampled → ${resampled.dims.join('×')}`);

  // 3. Normalise
  const norm      = foregroundZScore(resampled.data);
  const normalised = norm.data;
  log(`  normalised: mean=${norm.mean.toFixed(1)}, std=${norm.std.toFixed(1)}, clip=[${norm.lo.toFixed(1)}, ${norm.hi.toFixed(1)}]`);

  // 4. Sliding-window inference — ensemble over all loaded folds
  const [D, H, W] = resampled.dims;
  const ensembleSoft = new Float32Array(N_CLASSES * D * H * W);

  for (let f = 0; f < sessions.length; f++) {
    log(`  fold ${f + 1}/${sessions.length} inference…`);
    const sess       = sessions[f];
    const inputName  = sess.inputNames[0];
    const outputName = sess.outputNames[0];

    const runModel = async (patchData, shape) => {
      const tensor = new ort.Tensor('float32', patchData, shape);
      const result = await sess.run({ [inputName]: tensor });
      return result[outputName].data;
    };

    const foldSoft = await slidingWindowInference(
      normalised, resampled.dims, runModel, PATCH_SIZE, PATCH_OVERLAP,
      (frac) => process.stderr.write(`\r    ${Math.round(frac * 100)} %  `),
      f,
      log,
    );
    process.stderr.write('\r                \r');

    for (let i = 0; i < ensembleSoft.length; i++) {
      ensembleSoft[i] += foldSoft[i] / sessions.length;
    }
  }

  // 5+6. Argmax + back-resample
  const maskOrig   = logitsToSegmentation(ensembleSoft, resampled.dims, nii.dims, N_CLASSES);
  const maskVoxels = maskOrig.reduce((s, v) => s + v, 0);
  log(`  mask: ${maskVoxels} positive voxels`);

  // 7. Encode and write
  const outNii = encodeNifti(maskOrig, nii.origDims, nii.header);
  writeNifti(outputPath, outNii);
  log(`  → ${outputPath}  (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
}

// ── Main ───────────────────────────────────────────────────────────────────

async function main() {
  log(`Loading ${foldList.length} fold model(s) from ${modelDir}…`);
  const sessions = await loadSessions(foldList, modelDir);
  log(`Models ready.`);

  if (inputFile) {
    await runOne(inputFile, outputFile, sessions);
  } else {
    // Batch mode: process all .nii and .nii.gz files in inputDir
    fs.mkdirSync(outputDir, { recursive: true });
    const files = fs.readdirSync(inputDir)
      .filter(f => f.endsWith('.nii') || f.endsWith('.nii.gz'))
      .sort();

    if (files.length === 0) {
      log(`No .nii / .nii.gz files found in ${inputDir}`);
      process.exit(1);
    }
    log(`Found ${files.length} file(s).`);

    for (let i = 0; i < files.length; i++) {
      const stem   = files[i].replace(/\.nii(\.gz)?$/, '');
      const inPath = path.join(inputDir, files[i]);
      const outPath = path.join(outputDir, `${stem}_mask.nii`);
      log(`[${i + 1}/${files.length}]`);
      await runOne(inPath, outPath, sessions);
    }
    log(`\nDone. ${files.length} file(s) processed.`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
