import { useState, useEffect, useCallback } from 'react';
import { LABELS, initialPaths } from '../utils.js';

export default function useResultViewer() {
  const [paths, setPaths] = useState(initialPaths);
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [selectedLabel, setSelectedLabel] = useState(LABELS[0]);
  const [mode, setMode] = useState('all');
  const [clustersByLabel, setClustersByLabel] = useState(Object.fromEntries(LABELS.map((label) => [label, []])));
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [assignments, setAssignments] = useState([]);
  const [imageSize, setImageSize] = useState(220);
  const [screen, setScreen] = useState('connect');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState('');

  const selectedRun = runs.find((run) => run.grouping_run_id === selectedRunId) || null;

  const updatePath = useCallback((name, value) => {
    setPaths((current) => ({ ...current, [name]: value }));
  }, []);

  const selectRun = useCallback((runId) => {
    const run = runs.find((item) => item.grouping_run_id === runId);
    setSelectedRunId(runId);
    setSelectedCluster(null);
    setAssignments([]);
    setScreen('results');
    if (run?.source_db_path) {
      setPaths((prev) => ({ ...prev, sourceDbPath: run.source_db_path || prev.sourceDbPath }));
    }
  }, [runs]);

  const loadRuns = useCallback(async () => {
    setLoading(true);
    setError('');
    setSelectedCluster(null);
    setAssignments([]);
    try {
      const result = await window.electronAPI.listResultViewerRuns({ featureDbPath: paths.featureDbPath });
      const nextRuns = result.runs || [];
      const nextRun = nextRuns[0] || null;
      setRuns(nextRuns);
      setSelectedRunId(nextRun?.grouping_run_id || '');
      if (nextRun?.source_db_path) setPaths((current) => ({ ...current, sourceDbPath: nextRun.source_db_path || current.sourceDbPath }));
      setScreen('results');
    } catch (event) {
      setError(String(event.message || event));
      setRuns([]);
      setSelectedRunId('');
    } finally {
      setLoading(false);
    }
  }, [paths.featureDbPath]);

  const loadClusters = useCallback(async (runId = selectedRunId, options = {}) => {
    if (!runId || !paths.featureDbPath) return;
    setLoading(true);
    setError('');
    if (options.resetSelection !== false) {
      setSelectedCluster(null);
      setAssignments([]);
    }
    try {
      const entries = await Promise.all(LABELS.map(async (label) => {
        const result = await window.electronAPI.listResultViewerClusters({ featureDbPath: paths.featureDbPath, runId, labelScope: label, mode });
        return [label, result.clusters || []];
      }));
      setClustersByLabel(Object.fromEntries(entries));
    } catch (event) {
      setError(String(event.message || event));
    } finally {
      setLoading(false);
    }
  }, [paths.featureDbPath, mode, selectedRunId]);

  const openCluster = useCallback(async (cluster) => {
    if (!selectedRunId) return;
    setSelectedCluster(cluster);
    setScreen('detail');
    setDetailLoading(true);
    setError('');
    try {
      const result = await window.electronAPI.listResultViewerAssignments({
        featureDbPath: paths.featureDbPath,
        sourceDbPath: paths.sourceDbPath,
        imageRootPath: paths.imageRootPath,
        runId: selectedRunId,
        clusterKey: cluster.cluster_key
      });
      setAssignments(result.assignments || []);
    } catch (event) {
      setError(String(event.message || event));
      setAssignments([]);
    } finally {
      setDetailLoading(false);
    }
  }, [paths.featureDbPath, paths.sourceDbPath, paths.imageRootPath, selectedRunId]);

  const clearClusterFlags = useCallback(async () => {
    if (!selectedCluster || !selectedRunId) return;
    await window.electronAPI.clearResultViewerClusterFlags({ featureDbPath: paths.featureDbPath, runId: selectedRunId, clusterKey: selectedCluster.cluster_key });
    setAssignments((current) => current.map((row) => ({ ...row, is_outlier: 0, label_suspect: 0 })));
    await loadClusters(selectedRunId, { resetSelection: false });
  }, [selectedCluster, selectedRunId, paths.featureDbPath, loadClusters]);

  const clearResultFlags = useCallback(async (resultIds) => {
    if (!selectedRunId || resultIds.length === 0) return;
    await window.electronAPI.clearResultViewerResultFlags({ featureDbPath: paths.featureDbPath, runId: selectedRunId, resultIds });
    const idSet = new Set(resultIds.map(Number));
    setAssignments((current) => current.map((row) => (idSet.has(Number(row.result_id)) ? { ...row, is_outlier: 0, label_suspect: 0 } : row)));
    await loadClusters(selectedRunId, { resetSelection: false });
  }, [selectedRunId, paths.featureDbPath, loadClusters]);

  useEffect(() => {
    let cancelled = false;
    window.electronAPI.getResultViewerDefaults()
      .then((defaults) => {
        if (!cancelled) {
          setPaths({
            featureDbPath: defaults.featureDbPath || '',
            sourceDbPath: defaults.sourceDbPath || '',
            imageRootPath: defaults.imageRootPath || ''
          });
        }
      })
      .catch((event) => {
        if (!cancelled) setError(String(event.message || event));
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (screen !== 'connect' && selectedRunId) loadClusters(selectedRunId, { resetSelection: screen !== 'detail' });
  }, [mode, paths.featureDbPath, screen, selectedRunId, loadClusters]);

  return {
    paths, runs, selectedRun, selectedRunId, selectedLabel, mode,
    clustersByLabel, selectedCluster, assignments, imageSize,
    screen, settingsOpen, loading, detailLoading, error,
    updatePath, selectRun, loadRuns, loadClusters, openCluster,
    clearClusterFlags, clearResultFlags,
    setSelectedLabel, setScreen, setSettingsOpen, setMode,
    setImageSize, setAssignments, setSelectedCluster
  };
}
