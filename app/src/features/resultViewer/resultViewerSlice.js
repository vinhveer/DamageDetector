import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { LABELS, initialPaths } from './utils.js';

const emptyClusters = () => Object.fromEntries(LABELS.map((label) => [label, []]));

const initialState = {
  paths: initialPaths,
  runs: [],
  selectedRunId: '',
  selectedLabel: LABELS[0],
  mode: 'all',
  clustersByLabel: emptyClusters(),
  selectedCluster: null,
  assignments: [],
  imageSize: 220,
  screen: 'connect',
  settingsOpen: false,
  loading: false,
  detailLoading: false,
  error: '',
  initialized: false
};

export const fetchResultViewerDefaults = createAsyncThunk('resultViewer/fetchDefaults', async () => {
  return window.electronAPI.getResultViewerDefaults();
});

export const loadResultViewerRuns = createAsyncThunk('resultViewer/loadRuns', async (_, { getState }) => {
  const { paths } = getState().resultViewer;
  return window.electronAPI.listResultViewerRuns({ featureDbPath: paths.featureDbPath });
});

export const loadResultViewerClusters = createAsyncThunk('resultViewer/loadClusters', async ({ runId, resetSelection = true } = {}, { getState }) => {
  const { paths, selectedRunId, mode } = getState().resultViewer;
  const nextRunId = runId || selectedRunId;
  if (!nextRunId || !paths.featureDbPath) {
    return { clustersByLabel: emptyClusters(), resetSelection };
  }
  const entries = await Promise.all(LABELS.map(async (label) => {
    const result = await window.electronAPI.listResultViewerClusters({ featureDbPath: paths.featureDbPath, runId: nextRunId, labelScope: label, mode });
    return [label, result.clusters || []];
  }));
  return { clustersByLabel: Object.fromEntries(entries), resetSelection };
});

export const openResultViewerCluster = createAsyncThunk('resultViewer/openCluster', async (cluster, { getState }) => {
  const { paths, selectedRunId } = getState().resultViewer;
  const result = await window.electronAPI.listResultViewerAssignments({
    featureDbPath: paths.featureDbPath,
    sourceDbPath: paths.sourceDbPath,
    imageRootPath: paths.imageRootPath,
    runId: selectedRunId,
    clusterKey: cluster.cluster_key
  });
  return { cluster, assignments: result.assignments || [] };
});

export const clearResultViewerClusterFlags = createAsyncThunk('resultViewer/clearClusterFlags', async (_, { dispatch, getState }) => {
  const { paths, selectedRunId, selectedCluster } = getState().resultViewer;
  if (!selectedCluster || !selectedRunId) return;
  await window.electronAPI.clearResultViewerClusterFlags({ featureDbPath: paths.featureDbPath, runId: selectedRunId, clusterKey: selectedCluster.cluster_key });
  await dispatch(loadResultViewerClusters({ runId: selectedRunId, resetSelection: false }));
});

export const clearResultViewerResultFlags = createAsyncThunk('resultViewer/clearResultFlags', async (resultIds, { dispatch, getState }) => {
  const { paths, selectedRunId } = getState().resultViewer;
  if (!selectedRunId || resultIds.length === 0) return { resultIds: [] };
  await window.electronAPI.clearResultViewerResultFlags({ featureDbPath: paths.featureDbPath, runId: selectedRunId, resultIds });
  await dispatch(loadResultViewerClusters({ runId: selectedRunId, resetSelection: false }));
  return { resultIds };
});

const resultViewerSlice = createSlice({
  name: 'resultViewer',
  initialState,
  reducers: {
    updatePath(state, action) {
      const { name, value } = action.payload;
      state.paths[name] = value;
    },
    selectRun(state, action) {
      const runId = action.payload;
      const run = state.runs.find((item) => item.grouping_run_id === runId);
      state.selectedRunId = runId;
      state.selectedCluster = null;
      state.assignments = [];
      state.screen = 'results';
      if (run?.source_db_path) state.paths.sourceDbPath = run.source_db_path || state.paths.sourceDbPath;
    },
    setSelectedLabel(state, action) {
      state.selectedLabel = action.payload;
    },
    setMode(state, action) {
      state.mode = action.payload;
    },
    setImageSize(state, action) {
      state.imageSize = action.payload;
    },
    setScreen(state, action) {
      state.screen = action.payload;
    },
    setSettingsOpen(state, action) {
      state.settingsOpen = action.payload;
    },
    setSelectedCluster(state, action) {
      state.selectedCluster = action.payload;
      if (!action.payload) state.assignments = [];
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchResultViewerDefaults.pending, (state) => {
        state.error = '';
      })
      .addCase(fetchResultViewerDefaults.fulfilled, (state, action) => {
        state.initialized = true;
        state.paths = {
          featureDbPath: action.payload.featureDbPath || '',
          sourceDbPath: action.payload.sourceDbPath || '',
          imageRootPath: action.payload.imageRootPath || ''
        };
      })
      .addCase(fetchResultViewerDefaults.rejected, (state, action) => {
        state.initialized = true;
        state.error = action.error.message || 'Failed to load defaults';
      })
      .addCase(loadResultViewerRuns.pending, (state) => {
        state.loading = true;
        state.error = '';
        state.selectedCluster = null;
        state.assignments = [];
      })
      .addCase(loadResultViewerRuns.fulfilled, (state, action) => {
        const nextRuns = action.payload.runs || [];
        const nextRun = nextRuns[0] || null;
        state.loading = false;
        state.runs = nextRuns;
        state.selectedRunId = nextRun?.grouping_run_id || '';
        if (nextRun?.source_db_path) state.paths.sourceDbPath = nextRun.source_db_path || state.paths.sourceDbPath;
        state.screen = 'results';
      })
      .addCase(loadResultViewerRuns.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to load runs';
        state.runs = [];
        state.selectedRunId = '';
      })
      .addCase(loadResultViewerClusters.pending, (state, action) => {
        state.loading = true;
        state.error = '';
        if (action.meta.arg?.resetSelection !== false) {
          state.selectedCluster = null;
          state.assignments = [];
        }
      })
      .addCase(loadResultViewerClusters.fulfilled, (state, action) => {
        state.loading = false;
        state.clustersByLabel = action.payload.clustersByLabel;
      })
      .addCase(loadResultViewerClusters.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to load clusters';
      })
      .addCase(openResultViewerCluster.pending, (state, action) => {
        state.selectedCluster = action.meta.arg;
        state.screen = 'detail';
        state.detailLoading = true;
        state.error = '';
      })
      .addCase(openResultViewerCluster.fulfilled, (state, action) => {
        state.detailLoading = false;
        state.selectedCluster = action.payload.cluster;
        state.assignments = action.payload.assignments;
      })
      .addCase(openResultViewerCluster.rejected, (state, action) => {
        state.detailLoading = false;
        state.error = action.error.message || 'Failed to load assignments';
        state.assignments = [];
      })
      .addCase(clearResultViewerClusterFlags.fulfilled, (state) => {
        state.assignments = state.assignments.map((row) => ({ ...row, is_outlier: 0, label_suspect: 0 }));
      })
      .addCase(clearResultViewerResultFlags.fulfilled, (state, action) => {
        const ids = new Set((action.payload?.resultIds || []).map(Number));
        state.assignments = state.assignments.map((row) => (ids.has(Number(row.result_id)) ? { ...row, is_outlier: 0, label_suspect: 0 } : row));
      });
  }
});

export const {
  updatePath,
  selectRun,
  setSelectedLabel,
  setMode,
  setImageSize,
  setScreen,
  setSettingsOpen,
  setSelectedCluster
} = resultViewerSlice.actions;

export default resultViewerSlice.reducer;
