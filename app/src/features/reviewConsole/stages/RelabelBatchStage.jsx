import { useCallback, useEffect, useState } from 'react';
import { Button, EmptyState, ErrorMessage } from '../../../components/ui/index.js';
import ConfirmDialog from '../components/ConfirmDialog.jsx';
import { formatFloat, formatNum } from '../reviewConstants.js';

const LARGE_BATCH = 100;

// Stage 5 — wired but light until the decision policy produces relabel_candidate rows (SPEC §4.5).
export default function RelabelBatchStage({ ctx, session, applyDecision, decisionFor }) {
  const { resemiDbPath, runId } = ctx;
  const [batches, setBatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [confirm, setConfirm] = useState(null);
  const committed = session?.status === 'committed';

  useEffect(() => {
    setLoading(true);
    setError('');
    window.electronAPI.listReviewRelabelBatches({ resemiDbPath, runId })
      .then((res) => { if (res?.error) throw new Error(res.error); setBatches(res.batches || []); })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [resemiDbPath, runId]);

  const approve = useCallback((batch) => {
    applyDecision({
      target_type: 'batch',
      batch_id: batch.batch_id,
      target_id: `batch:${batch.batch_id}`,
      action: 'approve_batch',
      previous_label: batch.from_label,
      new_label: batch.to_label,
      reason_codes: ['relabel_candidate_core_margin'],
      affected_result_ids: [],
      note: `Approve ${batch.count} ${batch.from_label}→${batch.to_label}`,
    });
  }, [applyDecision]);

  const onApprove = useCallback((batch) => {
    if (batch.count > LARGE_BATCH) {
      setConfirm({
        title: 'Approve large relabel batch?',
        message: `This relabels ${formatNum(batch.count)} detections ${batch.from_label} → ${batch.to_label}. Confirm?`,
        confirmLabel: 'Approve',
        onConfirm: () => { approve(batch); setConfirm(null); },
      });
      return;
    }
    approve(batch);
  }, [approve]);

  if (error) return <div className="p-6"><ErrorMessage message={error} /></div>;
  if (!loading && batches.length === 0) {
    return <EmptyState title="No relabel batches">The decision policy has not produced relabel candidates for this run yet.</EmptyState>;
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto p-4">
      <div className="flex flex-col gap-2">
        {batches.map((b) => {
          const drafted = decisionFor(`batch:${b.batch_id}`);
          return (
            <div key={b.batch_id} className="flex items-center gap-4 rounded-[6px] border border-[var(--border-muted)] bg-[var(--surface)] p-3 text-[12px]">
              <div className="flex items-center gap-2 text-[13px] text-[var(--text)]">
                <span>{b.from_label}</span>
                <span className="text-[var(--text-muted)]">→</span>
                <span className="font-medium">{b.to_label}</span>
              </div>
              <span className="tabular-nums text-[var(--text-muted)]">{formatNum(b.count)} items</span>
              <span className="tabular-nums text-[var(--text-muted)]">avg reliab {formatFloat(b.avg_reliability, 2)}</span>
              <div className="ml-auto">
                {drafted
                  ? <span className="text-[var(--success)]">Approved (draft)</span>
                  : <Button variant="primary" disabled={committed} onClick={() => onApprove(b)} className="h-7 px-3">Approve batch</Button>}
              </div>
            </div>
          );
        })}
      </div>
      <ConfirmDialog
        open={Boolean(confirm)}
        title={confirm?.title}
        message={confirm?.message}
        confirmLabel={confirm?.confirmLabel}
        onConfirm={confirm?.onConfirm}
        onCancel={() => setConfirm(null)}
      />
    </div>
  );
}
