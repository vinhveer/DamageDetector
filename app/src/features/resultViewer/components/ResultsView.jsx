import { IconAdjustments, IconChevronLeft } from '@tabler/icons-react';
import { IconButton, Button, SelectControl, EmptyState, ErrorMessage } from '../../../components/ui/index.js';
import PageHeader from './PageHeader.jsx';
import LabelTabs from './LabelTabs.jsx';
import ClusterList from './ClusterList.jsx';
import { formatNumber, shortId } from '../utils.js';

export default function ResultsView({
  runs,
  selectedRun,
  selectedRunId,
  selectedLabel,
  clustersByLabel,
  loading,
  error,
  onBack,
  onRunChange,
  onLabelChange,
  onOpenCluster,
  onOpenSettings
}) {
  const clusters = clustersByLabel[selectedLabel] || [];
  return (
    <div className="rv-enter flex h-full flex-col bg-[var(--bg)] rv-font">
      <PageHeader
        title="Result Viewer"
        subtitle={selectedRun ? `${formatNumber(selectedRun.total_clusters)} clusters · ${shortId(selectedRun.grouping_run_id)}` : undefined}
        left={
          <IconButton label="Back" onClick={onBack}>
            <IconChevronLeft size={16} />
          </IconButton>
        }
        right={
          <>
            <SelectControl
              value={selectedRunId}
              onChange={(e) => onRunChange(e.currentTarget.value)}
              className="w-[260px]"
              disabled={runs.length === 0}
            >
              <option value="">No run selected</option>
              {runs.map((run) => (
                <option key={run.grouping_run_id} value={run.grouping_run_id}>
                  {run.created_at_utc?.slice(0, 16) || 'Unknown'} · {shortId(run.grouping_run_id)}
                </option>
              ))}
            </SelectControl>
            <Button onClick={onOpenSettings}>
              <IconAdjustments size={14} />
              Settings
            </Button>
          </>
        }
      />

      <LabelTabs selectedLabel={selectedLabel} clustersByLabel={clustersByLabel} onChange={onLabelChange} />

      {error && <div className="px-6 py-3"><ErrorMessage error={error} /></div>}

      <main className="min-h-0 flex-1 overflow-hidden bg-[var(--bg)]">
        {loading ? <EmptyState title="Loading…" /> : <ClusterList clusters={clusters} onOpen={onOpenCluster} />}
      </main>
    </div>
  );
}
