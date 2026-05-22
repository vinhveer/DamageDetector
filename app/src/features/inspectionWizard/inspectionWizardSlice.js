import { createSlice } from '@reduxjs/toolkit';

// ── Default prompt groups ───────────────────────────────────────────────────
const DEFAULT_PROMPT_GROUPS = [
  { id: 1, name: 'crack', prompt: 'crack . surface crack . structural crack . hairline crack' },
  { id: 2, name: 'mold', prompt: 'mold . mildew . moss . stain . water damage' },
  { id: 3, name: 'spall', prompt: 'spalling . broken concrete . exposed rebar . delamination' },
];

// ── localStorage migration helper ──────────────────────────────────────────
// New keys (preferred)
const LS_SAM_CHECKPOINT = 'damage-detector.inspection.sam-checkpoint';
const LS_GDINO_CHECKPOINT = 'damage-detector.inspection.gdino-checkpoint';
const LS_DEVICE = 'damage-detector.inspection.device';
const LS_GDINO_CONFIG = 'damage-detector.inspection.gdino-config';
const LS_SAM_CONFIG = 'damage-detector.inspection.sam-config';

// Old keys (fallback)
const OLD_SAM_CHECKPOINT = 'damage-detector.samcrop-checkpoint';
const OLD_GDINO_CHECKPOINT = 'damage-detector.samcrop-gdino-checkpoint';
const OLD_DEVICE = 'damage-detector.samcrop-device';
const OLD_GDINO_CONFIG = 'damage-detector.sam-gdino.gdino';
const OLD_SAM_CONFIG = 'damage-detector.sam-gdino.sam';

/**
 * Reads new key first, falls back to old key, returns parsed JSON or raw string.
 * If both are missing, returns the provided fallback.
 */
function loadMigrated(newKey, oldKey, fallback) {
  try {
    const newVal = localStorage.getItem(newKey);
    if (newVal !== null) {
      try { return JSON.parse(newVal); } catch { return newVal; }
    }
    const oldVal = localStorage.getItem(oldKey);
    if (oldVal !== null) {
      localStorage.setItem(newKey, oldVal);
      try { return JSON.parse(oldVal); } catch { return oldVal; }
    }
    return fallback;
  } catch {
    return fallback;
  }
}

// ── Default sub-state shapes ────────────────────────────────────────────────
const defaultGdino = {
  boxThreshold: 0.16,
  textThreshold: 0.16,
  maxDets: 80,
  tiledThreshold: 512,
  tileScales: ['small', 'medium', 'large'],
  promptGroups: DEFAULT_PROMPT_GROUPS,
  checkpointPath: '',
  device: 'auto',
};

const defaultSam = {
  backend: 'sam',
  modelType: 'vit_h',
  checkpointPath: '',
  loraCheckpointPath: '',
  device: 'auto',
  multimask: false,
  minMaskArea: 0,
  expandBoxPx: 0,
};

const defaultSemantic = {
  enabled: true,
  modelName: 'ViT-B-32',
  pretrained: 'laion2b_s34b_b79k',
  batchSize: 16,
};

const defaultSpatial = {
  enabled: true,
  iouThreshold: 0.5,
  containmentThreshold: 0.8,
};

// ── Initial state ────────────────────────────────────────────────────────────
const initialState = {
  step: 0, // 0..6
  skipCrop: false,
  source: { paths: [], recursive: false },
  crop: {
    activeImagePath: null,
    perImage: {},
    // perImage shape: { [path]: { segMode, points, pointMode, textPrompt, status, skipped, maskB64, rawMaskB64, bbox, cropPreviewB64, cropError } }
    samCheckpoint: loadMigrated(LS_SAM_CHECKPOINT, OLD_SAM_CHECKPOINT, ''),
    gdinoCheckpoint: loadMigrated(LS_GDINO_CHECKPOINT, OLD_GDINO_CHECKPOINT, ''),
    device: loadMigrated(LS_DEVICE, OLD_DEVICE, 'auto'),
    boxThreshold: 0.15,
    textThreshold: 0.15,
    cropPadding: 10,
    transparentBg: true,
    croppedPaths: [],
    cropOutputDir: '',
  },
  gdino: loadMigrated(LS_GDINO_CONFIG, OLD_GDINO_CONFIG, defaultGdino),
  semantic: defaultSemantic,
  spatial: defaultSpatial,
  sam: loadMigrated(LS_SAM_CONFIG, OLD_SAM_CONFIG, defaultSam),
  detection: {
    status: 'idle',
    sessionId: null,
    logs: [],
    runId: null,
    semanticRunId: null,
    dbPath: null,
    boxesByImage: {},
    suspectByImage: {},
  },
  segmentation: {
    status: 'idle',
    sessionId: null,
    logs: [],
    masksByImage: {},
    outputDir: null,
  },
  showBoxOverlay: true,
  showMaskOverlay: true,
  selectedImagePath: null,
  error: null,
};

// ── Default per-image crop entry ────────────────────────────────────────────
function defaultPerImage() {
  return {
    segMode: 'point',
    points: [],
    pointMode: 'positive',
    textPrompt: '',
    status: 'idle',
    skipped: false,
    maskB64: null,
    rawMaskB64: null,
    bbox: null,
    cropPreviewB64: null,
    cropError: null,
  };
}

function removeCroppedOutputForSource(state, sourcePath) {
  const filename = String(sourcePath || '').split(/[\\/]/).pop() || '';
  const dot = filename.lastIndexOf('.');
  const base = (dot > 0 ? filename.slice(0, dot) : filename).replace(/[^a-zA-Z0-9._-]+/g, '_') || 'image';
  const expected = `${base}_crop.png`;
  state.crop.croppedPaths = state.crop.croppedPaths.filter((p) => !String(p).endsWith(`/${expected}`) && !String(p).endsWith(`\\${expected}`));
}

function resetCropResult(state, path) {
  state.crop.perImage[path].status = 'idle';
  state.crop.perImage[path].skipped = false;
  state.crop.perImage[path].maskB64 = null;
  state.crop.perImage[path].rawMaskB64 = null;
  state.crop.perImage[path].bbox = null;
  state.crop.perImage[path].cropPreviewB64 = null;
  state.crop.perImage[path].cropError = null;
  removeCroppedOutputForSource(state, path);
}

// ── Slice ───────────────────────────────────────────────────────────────────
const slice = createSlice({
  name: 'inspectionWizard',
  initialState,
  reducers: {
    // ─── Step ─────────────────────────────────────────────────────────
    goToStep(state, { payload }) {
      state.step = payload;
    },

    // ─── Source ───────────────────────────────────────────────────────
    setSourcePaths(state, { payload }) {
      state.source.paths = payload;
    },
    addSourcePaths(state, { payload }) {
      const set = new Set(state.source.paths);
      payload.forEach((p) => set.add(p));
      state.source.paths = [...set];
    },
    removeSourcePath(state, { payload }) {
      state.source.paths = state.source.paths.filter((p) => p !== payload);
    },
    setRecursive(state, { payload }) {
      state.source.recursive = payload;
    },
    setSkipCrop(state, { payload }) {
      state.skipCrop = payload;
    },

    // ─── Crop per-image ───────────────────────────────────────────────
    setCropActiveImage(state, { payload }) {
      state.crop.activeImagePath = payload;
      if (payload && !state.crop.perImage[payload]) {
        state.crop.perImage[payload] = defaultPerImage();
      }
    },
    setCropSegMode(state, { payload: { path, segMode } }) {
      if (!state.crop.perImage[path]) state.crop.perImage[path] = defaultPerImage();
      state.crop.perImage[path].segMode = segMode;
      resetCropResult(state, path);
    },
    addCropPoint(state, { payload: { path, point } }) {
      if (!state.crop.perImage[path]) state.crop.perImage[path] = defaultPerImage();
      state.crop.perImage[path].points.push(point);
      resetCropResult(state, path);
    },
    removeCropPoint(state, { payload: { path, index } }) {
      if (state.crop.perImage[path]) {
        state.crop.perImage[path].points.splice(index, 1);
        resetCropResult(state, path);
      }
    },
    clearCropPoints(state, { payload: path }) {
      if (state.crop.perImage[path]) {
        state.crop.perImage[path].points = [];
        resetCropResult(state, path);
      }
    },
    setCropTextPrompt(state, { payload: { path, textPrompt } }) {
      if (!state.crop.perImage[path]) state.crop.perImage[path] = defaultPerImage();
      state.crop.perImage[path].textPrompt = textPrompt;
      resetCropResult(state, path);
    },
    setCropPointMode(state, { payload: { path, pointMode } }) {
      if (!state.crop.perImage[path]) state.crop.perImage[path] = defaultPerImage();
      state.crop.perImage[path].pointMode = pointMode;
    },
    setCropRunning(state, { payload: path }) {
      if (!state.crop.perImage[path]) state.crop.perImage[path] = defaultPerImage();
      state.crop.perImage[path].status = 'running';
      state.crop.perImage[path].skipped = false;
      state.crop.perImage[path].maskB64 = null;
      state.crop.perImage[path].rawMaskB64 = null;
      state.crop.perImage[path].bbox = null;
      state.crop.perImage[path].cropPreviewB64 = null;
      state.crop.perImage[path].cropError = null;
    },
    setCropResult(state, { payload: { path, maskB64, rawMaskB64, bbox, cropPreviewB64 } }) {
      if (!state.crop.perImage[path]) state.crop.perImage[path] = defaultPerImage();
      state.crop.perImage[path].status = 'done';
      state.crop.perImage[path].skipped = false;
      state.crop.perImage[path].maskB64 = maskB64 ?? null;
      state.crop.perImage[path].rawMaskB64 = rawMaskB64 ?? null;
      state.crop.perImage[path].bbox = bbox ?? null;
      state.crop.perImage[path].cropPreviewB64 = cropPreviewB64 ?? null;
      state.crop.perImage[path].cropError = null;
    },
    setCropError(state, { payload: { path, error } }) {
      if (!state.crop.perImage[path]) state.crop.perImage[path] = defaultPerImage();
      state.crop.perImage[path].status = 'error';
      state.crop.perImage[path].cropError = error;
    },
    setCropSkipped(state, { payload: { path, skipped } }) {
      if (!state.crop.perImage[path]) state.crop.perImage[path] = defaultPerImage();
      state.crop.perImage[path].skipped = Boolean(skipped);
      if (skipped) {
        state.crop.perImage[path].status = 'idle';
        state.crop.perImage[path].maskB64 = null;
        state.crop.perImage[path].rawMaskB64 = null;
        state.crop.perImage[path].bbox = null;
        state.crop.perImage[path].cropPreviewB64 = null;
        state.crop.perImage[path].cropError = null;
        removeCroppedOutputForSource(state, path);
      }
    },
    resetCropImage(state, { payload: path }) {
      state.crop.perImage[path] = defaultPerImage();
      removeCroppedOutputForSource(state, path);
    },

    // ─── Crop shared config ───────────────────────────────────────────
    setCropPadding(state, { payload }) {
      state.crop.cropPadding = payload;
    },
    setTransparentBg(state, { payload }) {
      state.crop.transparentBg = payload;
    },
    setCropOutputDir(state, { payload }) {
      state.crop.cropOutputDir = payload;
    },
    setCropSamCheckpoint(state, { payload }) {
      state.crop.samCheckpoint = payload;
      localStorage.setItem(LS_SAM_CHECKPOINT, payload);
    },
    setCropGdinoCheckpoint(state, { payload }) {
      state.crop.gdinoCheckpoint = payload;
      localStorage.setItem(LS_GDINO_CHECKPOINT, payload);
    },
    setCropDevice(state, { payload }) {
      state.crop.device = payload;
      localStorage.setItem(LS_DEVICE, payload);
    },

    // ─── Crop output ──────────────────────────────────────────────────
    addCroppedPath(state, { payload }) {
      state.crop.croppedPaths.push(payload);
    },
    setCroppedPaths(state, { payload }) {
      state.crop.croppedPaths = payload;
    },

    // ─── Detection config ─────────────────────────────────────────────
    setGdino(state, { payload }) {
      state.gdino = { ...state.gdino, ...payload };
      localStorage.setItem(LS_GDINO_CONFIG, JSON.stringify(state.gdino));
    },
    setSemantic(state, { payload }) {
      state.semantic = { ...state.semantic, ...payload };
    },
    setSpatial(state, { payload }) {
      state.spatial = { ...state.spatial, ...payload };
    },
    setPromptGroups(state, { payload }) {
      state.gdino.promptGroups = payload;
      localStorage.setItem(LS_GDINO_CONFIG, JSON.stringify(state.gdino));
    },

    // ─── Detection lifecycle ──────────────────────────────────────────
    detectionStarted(state, { payload }) {
      state.detection = {
        status: 'running',
        sessionId: payload,
        logs: [],
        runId: null,
        semanticRunId: null,
        dbPath: null,
        boxesByImage: {},
        suspectByImage: {},
      };
      state.error = null;
    },
    detectionLog(state, { payload: { sessionId, line } }) {
      if (state.detection.sessionId !== sessionId) return;
      state.detection.logs.push(line);
      if (state.detection.logs.length > 500) state.detection.logs.shift();
    },
    detectionResult(state, { payload }) {
      state.detection.status = 'done';
      state.detection.dbPath = payload.db_path ?? null;
      state.detection.runId = payload.run_id ?? null;
      state.detection.semanticRunId = payload.semantic_run_id ?? null;
      state.detection.boxesByImage = payload.boxes_by_image ?? {};
      state.detection.suspectByImage = payload.suspect_by_image ?? {};
      if (Object.keys(state.detection.boxesByImage).length > 0 && !state.selectedImagePath) {
        state.selectedImagePath = Object.keys(state.detection.boxesByImage)[0];
      }
    },
    detectionClosed(state, { payload: { sessionId, code } }) {
      if (state.detection.sessionId !== sessionId) return;
      if (state.detection.status === 'running') {
        state.detection.status = code === '0' ? 'done' : 'error';
        if (code !== '0') state.error = `Detection process exited with code ${code}`;
      }
    },
    detectionFailed(state, { payload }) {
      state.detection.status = 'error';
      state.detection.sessionId = null;
      state.error = payload || 'Detection failed to start';
    },

    // ─── Segmentation config ──────────────────────────────────────────
    setSam(state, { payload }) {
      state.sam = { ...state.sam, ...payload };
      localStorage.setItem(LS_SAM_CONFIG, JSON.stringify(state.sam));
    },

    // ─── Segmentation lifecycle ───────────────────────────────────────
    segmentationStarted(state, { payload }) {
      state.segmentation = {
        status: 'running',
        sessionId: payload,
        logs: [],
        masksByImage: {},
        outputDir: null,
      };
      state.error = null;
    },
    segmentationLog(state, { payload: { sessionId, line } }) {
      if (state.segmentation.sessionId !== sessionId) return;
      state.segmentation.logs.push(line);
      if (state.segmentation.logs.length > 500) state.segmentation.logs.shift();
    },
    segmentationResult(state, { payload }) {
      state.segmentation.status = 'done';
      state.segmentation.masksByImage = payload.masks_by_image ?? {};
      state.segmentation.outputDir = payload.output_dir ?? null;
    },
    segmentationClosed(state, { payload: { sessionId, code } }) {
      if (state.segmentation.sessionId !== sessionId) return;
      if (state.segmentation.status === 'running') {
        state.segmentation.status = code === '0' ? 'done' : 'error';
        if (code !== '0') state.error = `SAM process exited with code ${code}`;
      }
    },
    segmentationFailed(state, { payload }) {
      state.segmentation.status = 'error';
      state.segmentation.sessionId = null;
      state.error = payload || 'SAM failed to start';
    },

    // ─── UI ───────────────────────────────────────────────────────────
    setSelectedImage(state, { payload }) {
      state.selectedImagePath = payload;
    },
    toggleBoxOverlay(state) {
      state.showBoxOverlay = !state.showBoxOverlay;
    },
    toggleMaskOverlay(state) {
      state.showMaskOverlay = !state.showMaskOverlay;
    },
    clearError(state) {
      state.error = null;
    },
  },
});

// ── Actions export ──────────────────────────────────────────────────────────
export const {
  goToStep,
  // Source
  setSourcePaths, addSourcePaths, removeSourcePath, setRecursive, setSkipCrop,
  // Crop per-image
  setCropActiveImage, setCropSegMode, addCropPoint, removeCropPoint, clearCropPoints,
  setCropTextPrompt, setCropPointMode, setCropRunning, setCropResult, setCropError, setCropSkipped, resetCropImage,
  // Crop shared
  setCropPadding, setTransparentBg, setCropOutputDir, setCropSamCheckpoint, setCropGdinoCheckpoint, setCropDevice,
  // Crop output
  addCroppedPath, setCroppedPaths,
  // Detection
  setGdino, setSemantic, setSpatial, setPromptGroups,
  detectionStarted, detectionLog, detectionResult, detectionClosed, detectionFailed,
  // Segmentation
  setSam, segmentationStarted, segmentationLog, segmentationResult, segmentationClosed, segmentationFailed,
  // UI
  setSelectedImage, toggleBoxOverlay, toggleMaskOverlay, clearError,
} = slice.actions;

// ── Selectors ───────────────────────────────────────────────────────────────
export const selectDetectionInputs = (state) => {
  const s = state.inspectionWizard;
  return s.skipCrop ? s.source.paths : s.crop.croppedPaths;
};

// ── Thunks ──────────────────────────────────────────────────────────────────
function venvConfig() {
  try {
    return JSON.parse(localStorage.getItem('damage-detector.workflow-python') || 'null');
  } catch {
    return null;
  }
}

/**
 * Run crop segmentation for a single image (point or text mode).
 */
export const runCropSegmentation = (imagePath) => async (dispatch, getState) => {
  const { crop } = getState().inspectionWizard;
  const imgState = crop.perImage[imagePath];
  if (!imagePath || !crop.samCheckpoint) return;

  const venv = venvConfig();
  const common = {
    samCheckpoint: crop.samCheckpoint,
    device: crop.device,
    venvDir: venv?.venvDir || '',
    useGlobalPython: venv?.useGlobalPython !== false,
  };

  dispatch(setCropRunning(imagePath));
  try {
    let result;
    if (imgState?.segMode === 'text') {
      result = await window.electronAPI.segmentTextSam({
        imagePath,
        textPrompt: imgState.textPrompt,
        gdinoCheckpoint: crop.gdinoCheckpoint,
        boxThreshold: crop.boxThreshold,
        textThreshold: crop.textThreshold,
        ...common,
      });
    } else {
      const points = imgState?.points || [];
      result = await window.electronAPI.segmentPointSam({
        imagePath,
        points: points.map((p) => [p.x, p.y]),
        labels: points.map((p) => p.label),
        ...common,
      });
    }
    if (result.error) {
      dispatch(setCropError({ path: imagePath, error: result.error }));
    } else {
      dispatch(setCropResult({
        path: imagePath,
        maskB64: result.overlay_b64 ?? result.mask_b64 ?? null,
        rawMaskB64: result.mask_b64 ?? null,
        bbox: result.bbox ?? null,
        cropPreviewB64: result.crop_preview_b64 ?? null,
      }));
    }
  } catch (err) {
    dispatch(setCropError({ path: imagePath, error: err?.message || 'Unknown error' }));
  }
};

/**
 * Run GroundingDINO detection using selectDetectionInputs for image paths.
 */
export const runDetection = () => async (dispatch, getState) => {
  const state = getState();
  const { gdino, semantic, spatial, source } = state.inspectionWizard;
  const imagePaths = selectDetectionInputs(state);
  const venv = venvConfig();

  const values = {
    image_paths: imagePaths,
    recursive: source.recursive,
    box_threshold: gdino.boxThreshold,
    text_threshold: gdino.textThreshold,
    max_dets: gdino.maxDets,
    tiled_threshold: gdino.tiledThreshold,
    tile_scales: gdino.tileScales.join(','),
    prompt_groups: JSON.stringify(gdino.promptGroups.map((g) => ({ name: g.name, prompt: g.prompt }))),
    gdino_checkpoint: gdino.checkpointPath,
    device: gdino.device,
    semantic_enabled: semantic.enabled,
    semantic_model: semantic.modelName,
    semantic_pretrained: semantic.pretrained,
    semantic_batch_size: semantic.batchSize,
    spatial_enabled: spatial.enabled,
    spatial_iou: spatial.iouThreshold,
    spatial_containment: spatial.containmentThreshold,
    venv_dir: venv?.venvDir || '',
    use_global_python: venv?.useGlobalPython !== false,
  };

  try {
    const { sessionId, error } = await window.electronAPI.startWorkflow({ workflowId: 'sam_gdino_wizard_detect', values });
    if (error || !sessionId) {
      dispatch(detectionFailed(error || 'Workflow did not return a session id'));
      return;
    }
    dispatch(detectionStarted(sessionId));
  } catch (err) {
    dispatch(detectionFailed(err?.message || 'Failed to start detection'));
  }
};

/**
 * Run SAM segmentation on detection results.
 */
export const runSegmentation = () => async (dispatch, getState) => {
  const { detection, sam } = getState().inspectionWizard;
  const venv = venvConfig();

  const values = {
    db_path: detection.dbPath,
    sam_backend: sam.backend,
    sam_model_type: sam.modelType,
    sam_checkpoint: sam.checkpointPath,
    lora_checkpoint: sam.loraCheckpointPath,
    device: sam.device,
    multimask: sam.multimask,
    min_mask_area: sam.minMaskArea,
    expand_box_px: sam.expandBoxPx,
    venv_dir: venv?.venvDir || '',
    use_global_python: venv?.useGlobalPython !== false,
  };

  try {
    const { sessionId, error } = await window.electronAPI.startWorkflow({ workflowId: 'sam_gdino_wizard_segment', values });
    if (error || !sessionId) {
      dispatch(segmentationFailed(error || 'Workflow did not return a session id'));
      return;
    }
    dispatch(segmentationStarted(sessionId));
  } catch (err) {
    dispatch(segmentationFailed(err?.message || 'Failed to start SAM'));
  }
};

// ── Default reducer export ──────────────────────────────────────────────────
export default slice.reducer;
