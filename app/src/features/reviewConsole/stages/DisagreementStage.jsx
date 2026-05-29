import { useCallback, useEffect, useState } from 'react';
import { EmptyState, ErrorMessage } from '../../../components/ui/index.js';
import ResultImage from '../../shared/ResultImage.jsx';
import { cn } from '../../../components/ui/cn.js';
import ItemCanvas from '../components/ItemCanvas.jsx';
import ActionBar from '../components/ActionBar.jsx';
import RightInspector from '../components/RightInspector.jsx';
import ReviewBadge from '../components/ReviewBadge.jsx';
import ConfirmDialog from '../components/ConfirmDialog.jsx';
import { useKeyboardShortcuts } from '../useKeyboardShortcuts.js';
import { LABELS, decisionTone, formatFloat } from '../reviewConstants.js';

const HIGH_SIMILARITY = 0.75;

export default function DisagreementStage({ ctx, session, applyDecision, decisionFor, registerShortcuts }) {
  const { resemiDbPath, imageRootPath, runId, filters } = ctx;
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [evidence, setEvidence] = useState(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [confirm, setConfirm] = useState(null);

  const loadItems = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await window.electronAPI.listReviewDisagreement({
        resemiDbPath, runId, imageRootPath, filters, limit: 300, offset: 0,
      });
      if (res?.error) throw new Error(res.error);
      setItems(res.items || []);
      setTotal(res.total || 0);
      setSelectedIdx(0);
    } catch (err) {
      setError(err.message || 'Failed to load items');
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [resemiDbPath, runId, imageRootPath, filters]);

  useEffect(() => { loadItems(); }, [loadItems]);

  const selected = items[selectedIdx] || null;

  useEffect(() => {
    if (!selected) { setEvidence(null); return undefined; }
    let cancelled = false;
    setEvidenceLoading(true);
    window.electronAPI.getReviewItemEvidence({
      resemiDbPath, runId, resultId: selected.result_id, imageRootPath,
    }).then((res) => {
      if (cancelled) return;
      setEvidence(res?.error ? null : res);
    }).catch(() => { if (!cancelled) setEvidence(null); })
      .finally(() => { if (!cancelled) setEvidenceLoading(false); });
    return () => { cancelled = true; };
  }, [selected, resemiDbPath, runId, imageRootPath]);

  const makeDecision = useCallback((action, newLabel) => {
    if (!selected) return;
    applyDecision({
      target_type: 'result',
      result_id: selected.result_id,
      action,
      previous_label: selected.initial_label,
      new_label: newLabel ?? selected.initial_label,
      previous_decision_type: selected.decision_type,
      new_decision_type: action === 'accept' ? 'manual_accept' : action === 'reject' ? 'manual_reject' : action === 'relabel' ? 'manual_relabel' : selected.decision_type,
      reason_codes: selected.reason_codes,
      affected_result_ids: [selected.result_id],
    });
  }, [selected, applyDecision]);

  const goNext = useCallback(() => setSelectedIdx((i) => Math.min(i + 1, items.length - 1)), [items.length]);
  const goPrev = useCallback(() => setSelectedIdx((i) => Math.max(i - 1, 0)), []);

  const onAccept = useCallback(() => { makeDecision('accept'); goNext(); }, [makeDecision, goNext]);
  const onReject = useCallback(() => {
    if (!selected) return;
    const sim = Math.max(selected.prototype_similarity || 0, selected.nearest_core_similarity || 0);
    if (sim >= HIGH_SIMILARITY) {
      setConfirm({
        title: 'Reject despite high damage similarity?',
        message: `This crop matches a damage prototype/core at ${formatFloat(sim, 2)} similarity. Reject anyway?`,
        tone: 'danger',
        confirmLabel: 'Reject',
        onConfirm: () => { makeDecision('relabel', 'reject'); setConfirm(null); goNext(); },
      });
      return;
    }
    makeDecision('relabel', 'reject');
    goNext();
  }, [selected, makeDecision, goNext]);
  const onRelabel = useCallback((label) => { makeDecision('relabel', label); goNext(); }, [makeDecision, goNext]);
  const onAmbiguous = useCallback(() => { makeDecision('mark_ambiguous'); goNext(); }, [makeDecision, goNext]);
  const onDefer = useCallback(() => { makeDecision('defer'); goNext(); }, [makeDecision, goNext]);
  const onAddPrototype = useCallback(() => {
    if (!selected) return;
    applyDecision({
      target_type: 'prototype',
      result_id: selected.result_id,
      action: 'add_prototype',
      previous_label: selected.initial_label,
      new_label: selected.final_label || selected.initial_label,
      affected_result_ids: [selected.result_id],
    });
  }, [selected, applyDecision]);

  useKeyboardShortcuts({
    onAccept, onReject, onRelabel: () => onRelabel(LABELS.find((l) => l !== selected?.initial_label)),
    onAmbiguous, onDefer, onAddPrototype, onNext: goNext, onPrev: goPrev,
    onPickCandidate: (n) => {
      const candidates = LABELS.filter((l) => l !== selected?.initial_label);
      if (candidates[n - 1]) onRelabel(candidates[n - 1]);
    },
  }, registerShortcuts && !confirm);

  const draftDecision = selected ? decisionFor(`result:${selected.result_id}`) : null;

  if (error) return <div className="p-6"><ErrorMessage message={error} /></div>;
  if (!loading && items.length === 0) {
    return <EmptyState title="Disagreement queue empty">No suspect items match the current filters.</EmptyState>;
  }

  return (
    <div className="flex h-full min-h-0">
      {/* Item strip */}
      <div className="flex w-[150px] shrink-0 flex-col overflow-auto border-r border-[var(--border-muted)] bg-[var(--surface)] p-2">
        <div className="mb-1 px-1 text-[10px] text-[var(--text-muted)]">{items.length} of {total}</div>
        {items.map((it, idx) => {
          const d = decisionFor(`result:${it.result_id}`);
          return (
            <button
              key={it.result_id}
              type="button"
              onClick={() => setSelectedIdx(idx)}
              className={cn(
                'mb-1 grid justify-items-center gap-1 rounded-[5px] border p-1',
                idx === selectedIdx ? 'border-[var(--primary)] bg-[var(--active)]' : 'border-[var(--border-muted)] hover:bg-[var(--hover)]',
              )}
            >
              <ResultImage row={it} imageSize={110} />
              <div className="flex w-full items-center justify-between gap-1 px-0.5 text-[10px]">
                <span className="truncate text-[var(--text-muted)]">#{it.result_id}</span>
                {d ? <ReviewBadge tone="blue">{d.action[0].toUpperCase()}</ReviewBadge>
                  : <ReviewBadge tone={decisionTone(it.decision_type)}>{it.initial_label.slice(0, 3)}</ReviewBadge>}
              </div>
            </button>
          );
        })}
      </div>

      {/* Center: canvas + actions */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 min-h-0">
          <ItemCanvas item={selected} evidence={evidence} />
        </div>
        <ActionBar
          item={selected}
          disabled={!selected || session?.status === 'committed'}
          onAccept={onAccept}
          onRelabel={onRelabel}
          onReject={onReject}
          onAmbiguous={onAmbiguous}
          onDefer={onDefer}
          onAddPrototype={onAddPrototype}
        />
      </div>

      <RightInspector item={selected} evidence={evidence} evidenceLoading={evidenceLoading} draftDecision={draftDecision} />

      <ConfirmDialog
        open={Boolean(confirm)}
        title={confirm?.title}
        message={confirm?.message}
        tone={confirm?.tone}
        confirmLabel={confirm?.confirmLabel}
        onConfirm={confirm?.onConfirm}
        onCancel={() => setConfirm(null)}
      />
    </div>
  );
}
