import useResultViewer from './hooks/useResultViewer.js';
import ConnectView from './components/ConnectView.jsx';
import ResultsView from './components/ResultsView.jsx';
import ClusterDetailView from './components/ClusterDetailView.jsx';
import SettingsDrawer from './components/SettingsDrawer.jsx';

export default function ResultViewer() {
  const {
    paths, runs, selectedRun, selectedRunId, selectedLabel, mode,
    clustersByLabel, selectedCluster, assignments, imageSize,
    screen, settingsOpen, loading, detailLoading, error,
    updatePath, selectRun, loadRuns, openCluster,
    clearClusterFlags, clearResultFlags,
    setSelectedLabel, setScreen, setSettingsOpen, setMode,
    setSelectedCluster,
    setImageSize
  } = useResultViewer();

  return (
    <div className="h-full min-w-0 bg-[var(--docker-bg)] text-[var(--docker-text)] rv-font">
      {screen === 'connect' && (
        <ConnectView
          paths={paths}
          loading={loading}
          error={error}
          onPathChange={updatePath}
          onLoad={loadRuns}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      )}

      {screen === 'results' && (
        <ResultsView
          runs={runs}
          selectedRun={selectedRun}
          selectedRunId={selectedRunId}
          selectedLabel={selectedLabel}
          clustersByLabel={clustersByLabel}
          loading={loading}
          error={error}
          onBack={() => setScreen('connect')}
          onRunChange={selectRun}
          onLabelChange={setSelectedLabel}
          onOpenCluster={openCluster}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      )}

      {screen === 'detail' && selectedCluster && (
        <ClusterDetailView
          cluster={selectedCluster}
          assignments={assignments}
          imageSize={imageSize}
          loading={detailLoading}
          error={error}
          onBack={() => { setSelectedCluster(null); setScreen('results'); }}
          onClearClusterFlags={clearClusterFlags}
          onClearResultFlags={clearResultFlags}
        />
      )}

      <SettingsDrawer
        opened={settingsOpen}
        paths={paths}
        runs={runs}
        runId={selectedRunId}
        mode={mode}
        imageSize={imageSize}
        loading={loading}
        onClose={() => setSettingsOpen(false)}
        onPathChange={updatePath}
        onLoad={loadRuns}
        onRunChange={selectRun}
        onModeChange={setMode}
        onImageSizeChange={setImageSize}
      />
    </div>
  );
}
