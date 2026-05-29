import { useCallback, useEffect, useState } from 'react';
import { IconCheck, IconBan } from '@tabler/icons-react';
import { EmptyState, ErrorMessage } from '../../../components/ui/index.js';
import ResultImage from '../../shared/ResultImage.jsx';
import { cn } from '../../../components/ui/cn.js';
import ReviewBadge from '../components/ReviewBadge.jsx';
import { LABELS, formatFloat, formatNum } from '../reviewConstants.js';

export default function PrototypePicksStage({ ctx, session, applyDecision, decisionFor }) {
  const { resemiDbPath, imageRootPath, runId, filters } = ctx;
  const [label, setLabel] = useState(filters.label || 'crack');
  const [items, setItems] = useState([]);
  const [labelCounts, setLabelCounts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const committed = session?.status === 'committed';

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await window.electronAPI.listReviewPrototypeCandidates({
        resemiDbPath, runId, imageRootPath, label, limit: 200,
      });
      if (res?.error) throw new Error(res.error);
      setItems(res.items || []);
      setLabelCounts(res.label_counts || []);
    } catch (err) {
      setError(err.message || 'Failed to load candidates');
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [resemiDbPath, runId, imageRootPath, label]);

  useEffect(() => { load(); }, [load]);

  const pick = useCallback((item, isReject) => {
    applyDecision({
      target_type: 'prototype',
      result_id: item.result_id,
      action: isReject ? 'add_reject_prototype' : 'add_prototype',
      previous_label: item.final_label,
      new_label: isReject ? 'reject' : item.final_label,
      affected_result_ids: [item.result_id],
    });
  }, [applyDecision]);

  const labelCount = labelCounts.find((l) => l.label === label)?.count ?? 0;
  const picked = items.filter((it) => decisionFor(`prototype:${it.result_id}`)).length;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2 text-[12px]">
        <span className="text-[var(--text-muted)]">Label</span>
        {LABELS.map((l) => (
          <button
            key={l}
            type="button"
            onClick={() => setLabel(l)}
            className={cn('rounded-[5px] px-2 py-1', label === l ? 'bg-[var(--active)] text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]')}
          >{l}</button>
        ))}
        <div className="ml-auto flex items-center gap-3 text-[11px] text-[var(--text-muted)]">
          <span>{formatNum(labelCount)} candidates</span>
          <span>{picked} picked</span>
          {labelCount < 20 && <ReviewBadge tone="amber">&lt;20 prototypes</ReviewBadge>}
        </div>
      </div>

      {error && <div className="p-4"><ErrorMessage message={error} /></div>}
      {!loading && !error && items.length === 0 && (
        <EmptyState title="No candidates">No high-reliability auto-accept crops for this label.</EmptyState>
      )}

      <div
        className="grid min-h-0 flex-1 content-start gap-3 overflow-auto p-4"
        style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))' }}
      >
        {items.map((it) => {
          const d = decisionFor(`prototype:${it.result_id}`);
          const isReject = d?.action === 'add_reject_prototype';
          const isProto = d?.action === 'add_prototype';
          return (
            <div
              key={it.result_id}
              className={cn(
                'grid justify-items-center gap-1.5 rounded-[6px] border bg-[var(--surface)] p-2',
                isProto ? 'border-[var(--success)]' : isReject ? 'border-[var(--danger)]' : 'border-[var(--border-muted)]',
              )}
            >
              <ResultImage row={it} imageSize={130} />
              <div className="flex w-full items-center justify-between text-[11px]">
                <span className="text-[var(--text-muted)]">#{it.result_id}</span>
                <span className="text-[var(--text-muted)]">{formatFloat(it.reliability_score, 2)}</span>
              </div>
              <div className="flex w-full gap-1">
                <button
                  type="button"
                  disabled={committed}
                  onClick={() => pick(it, false)}
                  className={cn('flex h-7 flex-1 items-center justify-center gap-1 rounded-[5px] border text-[11px]',
                    isProto ? 'border-[var(--success)] bg-[var(--success-bg)] text-[var(--success)]' : 'border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)]')}
                >
                  <IconCheck size={13} />Proto
                </button>
                <button
                  type="button"
                  disabled={committed}
                  onClick={() => pick(it, true)}
                  className={cn('flex h-7 flex-1 items-center justify-center gap-1 rounded-[5px] border text-[11px]',
                    isReject ? 'border-[var(--danger)] bg-[var(--danger-bg)] text-[var(--danger)]' : 'border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)]')}
                >
                  <IconBan size={13} />Reject
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
