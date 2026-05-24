import { useCallback, useEffect, useRef, useState } from 'react';
import SetupScreen from './components/SetupScreen.jsx';
import ReviewScreen from './components/ReviewScreen.jsx';

const utcNow = () => new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');

export default function ClusterLabeling() {
  const api = typeof window !== 'undefined' ? window.electronAPI : null;

  const [mode, setMode] = useState('setup');
  const [paths, setPaths] = useState({ clusterDbPath: '', sourceDbPath: '', imageRootPath: '', sessionsDir: '' });
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [clusters, setClusters] = useState([]);
  const [labelOptions, setLabelOptions] = useState(['all']);
  const [labelFilter, setLabelFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortBy, setSortBy] = useState('size');
  const [selectedClusterId, setSelectedClusterId] = useState(null);
  const [clusterDetail, setClusterDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState('');
  const [sessions, setSessions] = useState([]);
  const [selectedSessionId, setSelectedSessionId] = useState(null);
  const [sessionData, setSessionData] = useState(null);
  const [saving, setSaving] = useState(false);
  const [lastSavedAt, setLastSavedAt] = useState(null);
  const saveTimerRef = useRef(null);

  useEffect(() => {
    if (!api) return;
    api.getClusterLabelingDefaults()
      .then((defaults) => setPaths((prev) => ({ ...prev, ...defaults })))
      .catch((err) => setError(String(err?.message || err)));
  }, [api]);

  const handlePathChange = (key, value) => setPaths((prev) => ({ ...prev, [key]: value }));

  const loadRuns = useCallback(async () => {
    if (!api) return;
    setLoading(true);
    setError('');
    try {
      const result = await api.listClusterRuns({ clusterDbPath: paths.clusterDbPath });
      const list = result.runs || [];
      setRuns(list);
      if (list.length && !selectedRunId) {
        setSelectedRunId(list[0].cluster_run_id);
      }
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setLoading(false);
    }
  }, [api, paths.clusterDbPath, selectedRunId]);

  // Load clusters when run is selected
  useEffect(() => {
    if (!api || !selectedRunId) {
      setClusters([]);
      return;
    }
    setLoading(true);
    api.listClusters({
      clusterDbPath: paths.clusterDbPath,
      clusterRunId: selectedRunId,
      sortBy,
      filterLabel: labelFilter,
    })
      .then((result) => {
        setClusters(result.clusters || []);
        setLabelOptions(result.labels || ['all']);
      })
      .catch((err) => setError(String(err?.message || err)))
      .finally(() => setLoading(false));
  }, [api, selectedRunId, paths.clusterDbPath, sortBy, labelFilter]);

  // Load sessions for selected run
  useEffect(() => {
    if (!api || !selectedRunId) {
      setSessions([]);
      return;
    }
    api.listClusterSessions({ sessionsDir: paths.sessionsDir, clusterRunId: selectedRunId })
      .then((result) => setSessions(result.sessions || []))
      .catch((err) => setError(String(err?.message || err)));
  }, [api, selectedRunId, paths.sessionsDir]);

  // Load selected session
  useEffect(() => {
    if (!api || !selectedSessionId) {
      setSessionData(null);
      return;
    }
    api.loadClusterSession({ sessionsDir: paths.sessionsDir, sessionId: selectedSessionId })
      .then((data) => setSessionData(data))
      .catch((err) => setError(String(err?.message || err)));
  }, [api, selectedSessionId, paths.sessionsDir]);

  // Load cluster detail
  useEffect(() => {
    if (!api || !selectedRunId || selectedClusterId == null) {
      setClusterDetail(null);
      return;
    }
    setDetailLoading(true);
    api.getClusterMembers({
      clusterDbPath: paths.clusterDbPath,
      sourceDbPath: paths.sourceDbPath,
      clusterRunId: selectedRunId,
      clusterId: Number(selectedClusterId),
      imageRootPath: paths.imageRootPath,
      memberLimit: 48,
    })
      .then((result) => setClusterDetail(result))
      .catch((err) => setError(String(err?.message || err)))
      .finally(() => setDetailLoading(false));
  }, [api, selectedRunId, selectedClusterId, paths.clusterDbPath, paths.sourceDbPath, paths.imageRootPath]);

  const handleCreateSession = useCallback(async (title) => {
    if (!api || !selectedRunId) return;
    try {
      const data = await api.createClusterSession({
        sessionsDir: paths.sessionsDir,
        clusterRunId: selectedRunId,
        title: (title || '').trim() || undefined,
      });
      const list = await api.listClusterSessions({ sessionsDir: paths.sessionsDir, clusterRunId: selectedRunId });
      setSessions(list.sessions || []);
      setSelectedSessionId(data.session_id);
    } catch (err) {
      setError(String(err?.message || err));
    }
  }, [api, selectedRunId, paths.sessionsDir]);

  const handleDeleteSession = useCallback(async (sessionId) => {
    if (!api || !sessionId) return;
    try {
      await api.deleteClusterSession({ sessionsDir: paths.sessionsDir, sessionId });
      const list = await api.listClusterSessions({ sessionsDir: paths.sessionsDir, clusterRunId: selectedRunId });
      setSessions(list.sessions || []);
      if (selectedSessionId === sessionId) {
        setSelectedSessionId(null);
        setSessionData(null);
      }
    } catch (err) {
      setError(String(err?.message || err));
    }
  }, [api, paths.sessionsDir, selectedRunId, selectedSessionId]);

  const scheduleSave = useCallback((nextData) => {
    if (!api || !selectedSessionId) return;
    setSessionData(nextData);
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(async () => {
      setSaving(true);
      try {
        const result = await api.saveClusterSession({
          sessionsDir: paths.sessionsDir,
          sessionId: selectedSessionId,
          payload: nextData,
        });
        setLastSavedAt(result.last_updated_utc);
        setSessionData((prev) => prev
          ? { ...prev, last_updated_utc: result.last_updated_utc, stats: result.stats }
          : prev
        );
      } catch (err) {
        setError(String(err?.message || err));
      } finally {
        setSaving(false);
      }
    }, 400);
  }, [api, paths.sessionsDir, selectedSessionId]);

  const updateClusterEntry = useCallback((partial) => {
    if (!sessionData || selectedClusterId == null) return;
    const key = String(selectedClusterId);
    const prev = sessionData.clusters?.[key] || { status: 'unreviewed', user_label: '', notes: '' };
    const next = {
      ...sessionData,
      clusters: {
        ...(sessionData.clusters || {}),
        [key]: { ...prev, ...partial, labeled_at_utc: utcNow() },
      },
    };
    scheduleSave(next);
  }, [sessionData, selectedClusterId, scheduleSave]);

  if (mode === 'setup') {
    return (
      <div className="h-full overflow-auto bg-[var(--bg)]">
        <SetupScreen
          paths={paths}
          onPathChange={handlePathChange}
          onLoad={loadRuns}
          loading={loading}
          error={error}
          runs={runs}
          selectedRunId={selectedRunId}
          onSelectRun={(id) => {
            setSelectedRunId(id);
            setSelectedSessionId(null);
            setSelectedClusterId(null);
          }}
          sessions={sessions}
          selectedSessionId={selectedSessionId}
          onSelectSession={setSelectedSessionId}
          onCreateSession={handleCreateSession}
          onDeleteSession={handleDeleteSession}
          onStartLabeling={() => {
            if (!selectedRunId || !selectedSessionId) return;
            if (clusters.length > 0 && selectedClusterId == null) {
              setSelectedClusterId(clusters[0].cluster_id);
            }
            setMode('review');
          }}
        />
      </div>
    );
  }

  return (
    <ReviewScreen
      sessionData={sessionData}
      hasSession={Boolean(selectedSessionId)}
      saving={saving}
      lastSavedAt={lastSavedAt}
      totalClusters={clusters.length}
      clusters={clusters}
      labelOptions={labelOptions}
      sortBy={sortBy}
      setSortBy={setSortBy}
      statusFilter={statusFilter}
      setStatusFilter={setStatusFilter}
      labelFilter={labelFilter}
      setLabelFilter={setLabelFilter}
      selectedClusterId={selectedClusterId}
      setSelectedClusterId={setSelectedClusterId}
      clusterDetail={clusterDetail}
      detailLoading={detailLoading}
      onBack={() => setMode('setup')}
      onUpdateEntry={updateClusterEntry}
    />
  );
}
