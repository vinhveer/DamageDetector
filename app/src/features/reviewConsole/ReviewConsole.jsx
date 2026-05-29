import { useCallback, useEffect, useMemo, useState } from 'react';
import { ErrorMessage } from '../../components/ui/index.js';
import SetupScreen from './components/SetupScreen.jsx';
import TopBar from './components/TopBar.jsx';
import BottomBar from './components/BottomBar.jsx';
import StageRail from './components/StageRail.jsx';
import ConfirmDialog from './components/ConfirmDialog.jsx';
import DisagreementStage from './stages/DisagreementStage.jsx';
import PrototypePicksStage from './stages/PrototypePicksStage.jsx';
import CoreClusterStage from './stages/CoreClusterStage.jsx';
import OutlierStage from './stages/OutlierStage.jsx';
import RelabelBatchStage from './stages/RelabelBatchStage.jsx';
import { useReviewSession } from './useReviewSession.js';
import { useKeyboardShortcuts } from './useKeyboardShortcuts.js';

const STORAGE_KEY = 'review-console-paths';

const loadStoredPaths = () => {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
  catch { return {}; }
};

export default function ReviewConsole() {
  const [paths, setPaths] = useState({ resemiDbPath: '', imageRootPath: '', sessionsDir: '' });
  const [screen, setScreen] = useState('setup'); // 'setup' | 'console'
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [reviewer, setReviewer] = useState('');
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [counts, setCounts] = useState({});
  const [stage, setStage] = useState('disagreement');
  const [filters, setFilters] = useState({ label: '', reasonCode: '', reliabilityMin: '', reliabilityMax: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [commitState, setCommitState] = useState(null);

  // ── Load defaults once ──────────────────────────────────────────────────
  useEffect(() => {
    window.electronAPI.getReviewConsoleDefaults().then((defaults) => {
      const stored = loadStoredPaths();
      setPaths({ ...defaults, ...stored });
    });
  }, []);

  const persistPaths = useCallback((next) => {
    setPaths(next);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch { /* ignore */ }
  }, []);

  const onPathChange = useCallback((key, value) => {
    persistPaths({ ...paths, [key]: value });
  }, [paths, persistPaths]);

  const loadRuns = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await window.electronAPI.listReviewRuns({ resemiDbPath: paths.resemiDbPath });
      if (res?.error) throw new Error(res.error);
      setRuns(res.runs || []);
      if (res.runs?.length && !selectedRunId) setSelectedRunId(res.runs[0].run_id);
    } catch (err) {
      setError(err.message || 'Failed to load runs');
      setRuns([]);
    } finally {
      setLoading(false);
    }
  }, [paths.resemiDbPath, selectedRunId]);

  const refreshSessions = useCallback(async () => {
    if (!selectedRunId) return;
    const res = await window.electronAPI.listReviewSessions({ sessionsDir: paths.sessionsDir, runId: selectedRunId });
    setSessions(res?.sessions || []);
  }, [paths.sessionsDir, selectedRunId]);

  useEffect(() => { if (selectedRunId) refreshSessions(); }, [selectedRunId, refreshSessions]);

  const selectedRun = useMemo(() => runs.find((r) => r.run_id === selectedRunId), [runs, selectedRunId]);

  const onCreateSession = useCallback(async () => {
    const created = await window.electronAPI.createReviewSession({
      sessionsDir: paths.sessionsDir,
      runId: selectedRunId,
      reviewer,
      reliabilityRunId: selectedRun?.reliability_run_id,
      decisionPolicyRunId: selectedRun?.decision_policy_run_id,
      prototypeVersionId: selectedRun?.prototype_version_id,
    });
    if (created?.error) { setError(created.error); return; }
    await refreshSessions();
    setActiveSession(created);
    setScreen('console');
  }, [paths.sessionsDir, selectedRunId, reviewer, selectedRun, refreshSessions]);

  const onOpenSession = useCallback(async (sessionId) => {
    const data = await window.electronAPI.loadReviewSession({ sessionsDir: paths.sessionsDir, sessionId });
    if (data?.error) { setError(data.error); return; }
    setActiveSession(data);
    setScreen('console');
  }, [paths.sessionsDir]);

  const onDeleteSession = useCallback(async (sessionId) => {
    await window.electronAPI.deleteReviewSession({ sessionsDir: paths.sessionsDir, sessionId });
    await refreshSessions();
  }, [paths.sessionsDir, refreshSessions]);

  // ── Draft session management ────────────────────────────────────────────
  const reviewSession = useReviewSession({
    session: activeSession,
    sessionsDir: paths.sessionsDir,
    onSessionChange: refreshSessions,
  });

  // ── Queue counts when entering console / changing run ─────────────────────
  useEffect(() => {
    if (screen !== 'console' || !selectedRunId) return;
    window.electronAPI.getReviewQueueCounts({ resemiDbPath: paths.resemiDbPath, runId: selectedRunId })
      .then((res) => setCounts(res?.error ? {} : res))
      .catch(() => setCounts({}));
  }, [screen, selectedRunId, paths.resemiDbPath]);

  const ctx = useMemo(() => ({
    resemiDbPath: paths.resemiDbPath,
    imageRootPath: paths.imageRootPath,
    runId: selectedRunId,
    filters,
  }), [paths.resemiDbPath, paths.imageRootPath, selectedRunId, filters]);

  const onFilterChange = useCallback((key, value) => setFilters((f) => ({ ...f, [key]: value })), []);

  const onCommit = useCallback(() => {
    setCommitState({ confirming: true });
  }, []);

  const doCommit = useCallback(async () => {
    setCommitState({ running: true });
    await reviewSession.flushSave();
    const res = await window.electronAPI.commitReviewSession({
      resemiDbPath: paths.resemiDbPath,
      sessionsDir: paths.sessionsDir,
      sessionId: reviewSession.draft.session_id,
    });
    if (res?.error) { setCommitState({ error: res.error }); return; }
    setCommitState({ done: res });
    const reloaded = await window.electronAPI.loadReviewSession({ sessionsDir: paths.sessionsDir, sessionId: reviewSession.draft.session_id });
    setActiveSession(reloaded);
    await refreshSessions();
  }, [paths.resemiDbPath, paths.sessionsDir, reviewSession, refreshSessions]);

  // Top-level shortcut: Undo works across stages.
  useKeyboardShortcuts({ onUndo: reviewSession.undo }, screen === 'console' && !commitState);

  const deferredCount = useMemo(
    () => Object.values(reviewSession.draft?.decisions || {}).filter((d) => d.action === 'defer').length,
    [reviewSession.draft],
  );

  if (screen === 'setup') {
    return (
      <div className="flex h-full flex-col bg-[var(--bg)]">
        <SetupScreen
          paths={paths}
          onPathChange={onPathChange}
          onLoad={loadRuns}
          loading={loading}
          error={error}
          runs={runs}
          selectedRunId={selectedRunId}
          onSelectRun={setSelectedRunId}
          reviewer={reviewer}
          onReviewerChange={setReviewer}
          sessions={sessions}
          onCreateSession={onCreateSession}
          onOpenSession={onOpenSession}
          onDeleteSession={onDeleteSession}
        />
      </div>
    );
  }

  const stageProps = {
    ctx,
    session: reviewSession.draft,
    applyDecision: reviewSession.applyDecision,
    decisionFor: reviewSession.decisionFor,
    registerShortcuts: true,
  };

  return (
    <div className="flex h-full flex-col bg-[var(--bg)]">
      <TopBar
        run={selectedRun}
        session={reviewSession.draft}
        dirty={reviewSession.dirty}
        saving={reviewSession.saving}
        onBack={() => { reviewSession.flushSave(); setScreen('setup'); }}
      />
      <div className="flex min-h-0 flex-1">
        <StageRail
          stage={stage}
          onSelectStage={setStage}
          counts={counts}
          filters={filters}
          onFilterChange={onFilterChange}
          deferredCount={deferredCount}
        />
        <main className="flex min-w-0 flex-1 flex-col">
          {reviewSession.error && <div className="p-2"><ErrorMessage message={reviewSession.error} /></div>}
          {stage === 'disagreement' && <DisagreementStage {...stageProps} />}
          {stage === 'prototype' && <PrototypePicksStage {...stageProps} />}
          {stage === 'core' && <CoreClusterStage {...stageProps} />}
          {stage === 'outlier' && <OutlierStage {...stageProps} />}
          {stage === 'relabel' && <RelabelBatchStage {...stageProps} />}
        </main>
      </div>
      <BottomBar
        decisionCount={reviewSession.decisionCount}
        canUndo={reviewSession.canUndo}
        committed={reviewSession.committed}
        saving={reviewSession.saving}
        onUndo={reviewSession.undo}
        onCommit={onCommit}
      />

      <ConfirmDialog
        open={Boolean(commitState?.confirming)}
        title="Commit review session?"
        message={`This writes ${reviewSession.decisionCount} decision(s) into the resemi database. Committed sessions cannot be edited.`}
        confirmLabel="Commit"
        onConfirm={() => { setCommitState(null); doCommit(); }}
        onCancel={() => setCommitState(null)}
      />

      {commitState?.running && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 text-[13px] text-[var(--text)]">Committing…</div>
      )}
      {commitState?.error && (
        <ConfirmDialog open title="Commit failed" message={commitState.error} confirmLabel="OK"
          onConfirm={() => setCommitState(null)} onCancel={() => setCommitState(null)} />
      )}
      {commitState?.done && (
        <ConfirmDialog open title="Session committed" tone="primary"
          message={`Wrote ${commitState.done.decision_count} decisions${commitState.done.prototype_version_id ? ` and prototype version ${commitState.done.prototype_version_id}` : ''}.`}
          confirmLabel="OK" onConfirm={() => setCommitState(null)} onCancel={() => setCommitState(null)} />
      )}
    </div>
  );
}
