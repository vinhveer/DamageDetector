import { createSlice } from '@reduxjs/toolkit';

const CHECKPOINT_KEY = 'damage-detector.segment-checkpoint';
const DEVICE_KEY = 'damage-detector.segment-device';
const GDINO_KEY = 'damage-detector.segment-gdino-checkpoint';
const OUTPUT_DIR_KEY = 'damage-detector.segment-output-dir';
const VENV_CONFIG_KEY = 'damage-detector.workflow-python';

function readVenvConfig() {
  try {
    return JSON.parse(window.localStorage.getItem(VENV_CONFIG_KEY) || 'null');
  } catch {
    return null;
  }
}

const segmentSlice = createSlice({
  name: 'segment',
  initialState: {
    imagePath: null,
    // point mode
    points: [],
    pointMode: 'positive',     // 'positive' | 'negative'
    // text mode
    textPrompt: '',
    gdinoCheckpoint: localStorage.getItem(GDINO_KEY) || '',
    boxThreshold: 0.15,
    textThreshold: 0.15,
    // shared
    segMode: 'point',          // 'point' | 'text'
    status: 'idle',            // 'idle' | 'running' | 'done' | 'error'
    result: null,
    error: null,
    samCheckpoint: localStorage.getItem(CHECKPOINT_KEY) || '',
    device: localStorage.getItem(DEVICE_KEY) || 'auto',
    outputDir: localStorage.getItem(OUTPUT_DIR_KEY) || '',
    showOverlay: false,
    // folder queue
    queue: [],                 // array of image paths
    queueIndex: 0,             // index of current image within queue
    processed: {},             // { [imagePath]: 'done' | 'error' }
    autoAdvance: true,         // jump to next image after a successful run
  },
  reducers: {
    setImagePath(state, action) {
      state.imagePath = action.payload;
      state.points = [];
      state.status = 'idle';
      state.result = null;
      state.error = null;
      state.showOverlay = false;
    },
    setSamCheckpoint(state, action) {
      state.samCheckpoint = action.payload;
      localStorage.setItem(CHECKPOINT_KEY, action.payload);
    },
    setGdinoCheckpoint(state, action) {
      state.gdinoCheckpoint = action.payload;
      localStorage.setItem(GDINO_KEY, action.payload);
    },
    setDevice(state, action) {
      state.device = action.payload;
      localStorage.setItem(DEVICE_KEY, action.payload);
    },
    setOutputDir(state, action) {
      state.outputDir = action.payload;
      localStorage.setItem(OUTPUT_DIR_KEY, action.payload);
    },
    setSegMode(state, action) {
      state.segMode = action.payload;
      state.status = 'idle';
      state.result = null;
      state.error = null;
      state.showOverlay = false;
    },
    setPointMode(state, action) {
      state.pointMode = action.payload;
    },
    setTextPrompt(state, action) {
      state.textPrompt = action.payload;
    },
    setBoxThreshold(state, action) {
      state.boxThreshold = action.payload;
    },
    setTextThreshold(state, action) {
      state.textThreshold = action.payload;
    },
    addPoint(state, action) {
      state.points.push(action.payload);
    },
    removePoint(state, action) {
      state.points.splice(action.payload, 1);
    },
    clearPoints(state) {
      state.points = [];
    },
    setRunning(state) {
      state.status = 'running';
      state.result = null;
      state.error = null;
    },
    setResult(state, action) {
      state.status = 'done';
      state.result = action.payload;
      state.showOverlay = true;
    },
    setError(state, action) {
      state.status = 'error';
      state.error = action.payload;
    },
    toggleOverlay(state) {
      state.showOverlay = !state.showOverlay;
    },
    setQueue(state, action) {
      state.queue = action.payload || [];
      state.queueIndex = 0;
      state.processed = {};
    },
    clearQueue(state) {
      state.queue = [];
      state.queueIndex = 0;
      state.processed = {};
    },
    setQueueIndex(state, action) {
      const idx = action.payload;
      if (idx < 0 || idx >= state.queue.length) return;
      state.queueIndex = idx;
      state.imagePath = state.queue[idx];
      state.points = [];
      state.status = 'idle';
      state.result = null;
      state.error = null;
      state.showOverlay = false;
    },
    markProcessed(state, action) {
      const { path, status } = action.payload;
      if (path) state.processed[path] = status;
    },
    setAutoAdvance(state, action) {
      state.autoAdvance = action.payload;
    },
  },
});

export const {
  setImagePath,
  setSamCheckpoint,
  setGdinoCheckpoint,
  setDevice,
  setOutputDir,
  setSegMode,
  setPointMode,
  setTextPrompt,
  setBoxThreshold,
  setTextThreshold,
  addPoint,
  removePoint,
  clearPoints,
  setRunning,
  setResult,
  setError,
  toggleOverlay,
  setQueue,
  clearQueue,
  setQueueIndex,
  markProcessed,
  setAutoAdvance,
} = segmentSlice.actions;

// Keep old name as alias so nothing else breaks
export const setMode = setPointMode;

export default segmentSlice.reducer;

export const runSegmentation = () => async (dispatch, getState) => {
  const { imagePath, segMode, points, textPrompt, samCheckpoint, gdinoCheckpoint, device, boxThreshold, textThreshold, outputDir } = getState().segment;
  if (!imagePath || !samCheckpoint) return;

  const venv = readVenvConfig();
  const common = { samCheckpoint, device, outputDir: outputDir || '', venvDir: venv?.venvDir || '', useGlobalPython: venv?.useGlobalPython !== false };

  dispatch(setRunning());

  try {
    let result;
    if (segMode === 'text') {
      result = await window.electronAPI.segmentTextSam({
        imagePath,
        textPrompt,
        gdinoCheckpoint,
        boxThreshold,
        textThreshold,
        ...common,
      });
    } else {
      result = await window.electronAPI.segmentPointSam({
        imagePath,
        points: points.map((p) => [p.x, p.y]),
        labels: points.map((p) => p.label),
        ...common,
      });
    }

    if (result.error) {
      dispatch(setError(result.error));
      dispatch(markProcessed({ path: imagePath, status: 'error' }));
    } else {
      dispatch(setResult({
        overlayPath: result.overlay_path,
        maskPath: result.mask_path,
        cutoutPath: result.cutout_path,
        overlayB64: result.overlay_b64,
        cutoutB64: result.cutout_b64,
        score: result.score,
        maskArea: result.mask_area,
        detections: result.detections,
        modelType: result.model_type,
        device: result.device,
      }));
      dispatch(markProcessed({ path: imagePath, status: 'done' }));

      // Auto-advance to the next unprocessed image in the queue
      const { queue, queueIndex, autoAdvance } = getState().segment;
      if (autoAdvance && queue.length > 0 && queueIndex < queue.length - 1) {
        dispatch(setQueueIndex(queueIndex + 1));
      }
    }
  } catch (err) {
    dispatch(setError(err?.message || 'Unknown error'));
    dispatch(markProcessed({ path: imagePath, status: 'error' }));
  }
};
