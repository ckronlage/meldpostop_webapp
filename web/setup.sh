#!/usr/bin/env bash
# Downloads ONNX Runtime Web files and coi-serviceworker into web/
# Run once before launching the app: bash web/setup.sh

set -e
cd "$(dirname "$0")"

ORT_VERSION="1.26.0"
ORT_BASE="https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist"

mkdir -p wasm

echo "Downloading ONNX Runtime Web ${ORT_VERSION}..."
for f in \
  ort.webgpu.bundle.min.mjs \
  ort-wasm-simd-threaded.mjs \
  ort-wasm-simd-threaded.wasm \
  ort-wasm-simd-threaded.asyncify.mjs \
  ort-wasm-simd-threaded.asyncify.wasm \
  ort-wasm-simd-threaded.jsep.mjs \
  ort-wasm-simd-threaded.jsep.wasm
do
  echo "  $f"
  curl -fsSL "${ORT_BASE}/${f}" -o "wasm/${f}"
done

echo "Downloading coi-serviceworker..."
curl -fsSL "https://raw.githubusercontent.com/gzuidhof/coi-serviceworker/master/coi-serviceworker.js" \
  -o "coi-serviceworker.js"

echo "Done. Run: python ../serve.py"
