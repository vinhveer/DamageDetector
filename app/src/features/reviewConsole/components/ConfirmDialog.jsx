import { useEffect } from 'react';
import { Button } from '../../../components/ui/index.js';

// SPEC §6.3 confirmation gate. Enter confirms, Esc cancels.
export default function ConfirmDialog({ open, title, message, confirmLabel = 'Confirm', tone = 'primary', onConfirm, onCancel }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Enter') { e.preventDefault(); onConfirm?.(); }
      else if (e.key === 'Escape') { e.preventDefault(); onCancel?.(); }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [open, onConfirm, onCancel]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-6" onClick={onCancel}>
      <div
        className="w-full max-w-[420px] rounded-[8px] border border-[var(--border)] bg-[var(--surface)] p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="text-[14px] font-semibold text-[var(--text)]">{title}</div>
        <p className="mt-2 text-[12px] leading-relaxed text-[var(--text-muted)]">{message}</p>
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="secondary" onClick={onCancel}>Cancel <span className="ml-1 opacity-60">Esc</span></Button>
          <Button variant={tone === 'danger' ? 'danger' : 'primary'} onClick={onConfirm}>
            {confirmLabel} <span className="ml-1 opacity-60">⏎</span>
          </Button>
        </div>
      </div>
    </div>
  );
}
