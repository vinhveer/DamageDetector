import { IconArrowLeft } from '@tabler/icons-react';
import { IconButton } from '../../../components/ui/index.js';
import ReviewBadge from './ReviewBadge.jsx';
import { formatNum } from '../reviewConstants.js';

const statusTone = (status, dirty) => {
  if (status === 'committed') return 'green';
  if (dirty) return 'amber';
  return 'gray';
};

export default function TopBar({ run, session, dirty, saving, onBack }) {
  const status = String(session?.status || 'draft');
  const label = status === 'committed' ? 'COMMITTED' : dirty ? 'DIRTY' : saving ? 'SAVING' : 'DRAFT';
  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4">
      <IconButton label="Back to setup" onClick={onBack}><IconArrowLeft size={16} /></IconButton>
      <div className="flex min-w-0 items-center gap-2 text-[12px]">
        <span className="font-medium text-[var(--text)]">{run?.run_id}</span>
        <span className="text-[var(--text-muted)]">·</span>
        <span className="text-[var(--text-muted)]">{formatNum(run?.total_detections)} detections</span>
      </div>
      <div className="ml-auto flex items-center gap-3 text-[11px] text-[var(--text-muted)]">
        {run?.reliability_run_id && <span title="Reliability run">rel: {run.reliability_run_id.slice(0, 10)}</span>}
        {run?.decision_policy_run_id && <span title="Decision policy run">policy: {run.decision_policy_run_id.slice(0, 10)}</span>}
        {run?.prototype_version_id && <span title="Prototype version">proto: {run.prototype_version_id.slice(0, 10)}</span>}
        <span className="text-[var(--text)]">{session?.reviewer || 'anon'}</span>
        <ReviewBadge tone={statusTone(status, dirty)}>{label}</ReviewBadge>
      </div>
    </header>
  );
}
