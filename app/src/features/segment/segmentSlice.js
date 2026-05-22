import { createSlice } from '@reduxjs/toolkit';

const CHECKPOINT_KEY = 'damage-detector.segment-checkpoint';
const DEVICE_KEY = 'damage-detector.segment-device';
const GDINO_KEY = 'damage-detector.segment-gdino-checkpoint';
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
    showOverlay: false,
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
  },
});

export const {
  setImagePath,
  setSamCheckpoint,
  setGdinoCheckpoint,
  setDevice,
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
} = segmentSlice.actions;

// Keep old name as alias so nothing else breaks
export const setMode = setPointMode;

export default segmentSlice.reducer;

export const runSegmentation = () => async (dispatch, getState) => {
  const { imagePath, segMode, points, textPrompt, samCheckpoint, gdinoCheckpoint, device, boxThreshold, textThreshold } = getState().segment;
  if (!imagePath || !samCheckpoint) return;

  const venv = readVenvConfig();
  const common = { samCheckpoint, device, venvDir: venv?.venvDir || '', useGlobalPython: venv?.useGlobalPython !== false };

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
    } else {
      dispatch(setResult({
        overlayPath: result.overlay_path,
        maskPath: result.mask_path,
        overlayB64: result.overlay_b64,
        score: result.score,
        maskArea: result.mask_area,
        detections: result.detections,
        modelType: result.model_type,
        device: result.device,
      }));
    }
  } catch (err) {
    dispatch(setError(err?.message || 'Unknown error'));
  }
};
