// NiiVue is loaded via UMD script tag — available as window.niivue
const { Niivue } = window.niivue;

// ── DOM refs ───────────────────────────────────────────────────────────────
const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const fileInfo     = document.getElementById('file-info');
const runBtn       = document.getElementById('run-btn');
const dlBtn        = document.getElementById('dl-btn');
const progressWrap = document.getElementById('progress-wrap');
const progressBar  = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const consoleWrap  = document.getElementById('console-wrap');
const placeholder  = document.getElementById('viewer-placeholder');

// ── State ──────────────────────────────────────────────────────────────────
let loadedFile    = null;   // File object
let baseVolumeURL = null;   // Object URL of the input NIfTI (kept for overlay reload)
let maskObjectURL = null;   // Object URL of the output mask NIfTI

// ── NiiVue ─────────────────────────────────────────────────────────────────
const nv = new Niivue({
  backColor: [0, 0, 0, 1],
  show3Dcrosshair: true,
});
nv.attachToCanvas(document.getElementById('niivue-canvas'));

// ── Console logger ─────────────────────────────────────────────────────────
function log(msg, isError = false) {
  const line = document.createElement('div');
  line.className = 'log-line' + (isError ? ' err' : '');
  const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
  line.textContent = `[${ts}] ${msg}`;
  consoleWrap.appendChild(line);
  consoleWrap.scrollTop = consoleWrap.scrollHeight;
}

function setProgress(fraction, text = '') {
  if (fraction <= 0) {
    progressWrap.style.display = 'none';
    return;
  }
  progressWrap.style.display = 'block';
  progressBar.style.width = `${Math.min(100, fraction * 100).toFixed(1)}%`;
  progressText.textContent = text;
}

// ── File loading ───────────────────────────────────────────────────────────
async function loadFile(file) {
  loadedFile = file;

  // Revoke previous URLs
  if (baseVolumeURL) { URL.revokeObjectURL(baseVolumeURL); baseVolumeURL = null; }
  if (maskObjectURL) { URL.revokeObjectURL(maskObjectURL); maskObjectURL = null; }

  baseVolumeURL = URL.createObjectURL(file);

  fileInfo.style.display = 'block';
  fileInfo.textContent   = `${file.name}  (${(file.size / 1e6).toFixed(1)} MB)`;
  placeholder.style.display = 'none';
  runBtn.disabled = false;
  dlBtn.disabled  = true;

  log(`Loading ${file.name}…`);
  await nv.loadVolumes([{ url: baseVolumeURL, name: file.name }]);
  log('Volume loaded.');
}

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) loadFile(f);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

// ── Worker ─────────────────────────────────────────────────────────────────
let worker = null;

function initWorker() {
  if (worker) worker.terminate();
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

  worker.onmessage = async ({ data }) => {
    switch (data.type) {

      case 'initialized':
        log('Model ready — drop a NIfTI and click Run.');
        runBtn.disabled = !loadedFile;
        break;

      case 'progress':
        setProgress(data.fraction, data.text);
        break;

      case 'log':
        log(data.message);
        break;

      case 'stageData': {
        if (maskObjectURL) URL.revokeObjectURL(maskObjectURL);

        maskObjectURL = URL.createObjectURL(
          new Blob([data.data], { type: 'application/octet-stream' }),
        );

        // Reload base + mask together via loadVolumes (the canonical NiiVue API
        // for multiple volumes; addVolume expects an NVImage object in v0.46).
        // cal_min=0.5 makes value=0 (background) transparent; cal_max=1.0 maps
        // value=1 to the maximum warm colormap entry (fully coloured).
        await nv.loadVolumes([
          { url: baseVolumeURL, name: loadedFile.name },
          {
            url:      maskObjectURL,
            name:     'meld_postop_mask.nii',
            colormap: 'warm',
            opacity:  0.7,
            cal_min:  0.5,
            cal_max:  1.0,
          },
        ]);

        dlBtn.disabled = false;
        log('Segmentation overlay displayed.');
        break;
      }

      case 'complete':
        log('Done.');
        setProgress(1, 'Done');
        setTimeout(() => setProgress(0), 2000);
        runBtn.disabled = false;
        break;

      case 'error':
        log(`Error: ${data.message}`, true);
        setProgress(0);
        runBtn.disabled = false;
        break;
    }
  };

  worker.onerror = e => {
    log(`Worker error: ${e.message}`, true);
    runBtn.disabled = false;
  };

  log('Loading model (fold 0, ~117 MB on first load)…');
  worker.postMessage({ type: 'init' });
}

initWorker();

// ── Run button ─────────────────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
  if (!loadedFile) return;
  runBtn.disabled = true;
  dlBtn.disabled  = true;
  setProgress(0.01, 'Starting…');
  log('Running segmentation…');

  const buf = await loadedFile.arrayBuffer();
  worker.postMessage({ type: 'run', inputData: buf }, [buf]);
});

// ── Download button ────────────────────────────────────────────────────────
dlBtn.addEventListener('click', () => {
  if (!maskObjectURL) return;
  const a = document.createElement('a');
  a.href = maskObjectURL;
  a.download = loadedFile
    ? loadedFile.name.replace(/\.nii(\.gz)?$/, '_meld_postop_mask.nii')
    : 'meld_postop_mask.nii';
  a.click();
});
