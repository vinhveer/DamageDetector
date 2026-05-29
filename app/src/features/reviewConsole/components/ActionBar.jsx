import { Button } from '../../../components/ui/index.js';
import { LABELS } from '../reviewConstants.js';

// Evidence-adjacent action buttons (SPEC §3.2). Relabel offers each label as a quick target.
export default function ActionBar({ item, onAccept, onRelabel, onReject, onAmbiguous, onDefer, onAddPrototype, disabled }) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-[var(--border-muted)] bg-[var(--surface)] px-3 py-2">
      <Button variant="primary" onClick={onAccept} disabled={disabled} className="h-8">Accept <kbd className="ml-1 opacity-60">A</kbd></Button>
      <div className="flex items-center gap-1">
        <span className="text-[11px] text-[var(--text-muted)]">Relabel</span>
        {LABELS.filter((l) => l !== item?.initial_label).map((l, idx) => (
          <Button key={l} variant="secondary" onClick={() => onRelabel(l)} disabled={disabled} className="h-8 px-2 text-[12px]">
            {l} <kbd className="ml-1 opacity-50">{idx + 1}</kbd>
          </Button>
        ))}
      </div>
      <Button variant="danger" onClick={onReject} disabled={disabled} className="h-8">Reject <kbd className="ml-1 opacity-60">X</kbd></Button>
      <Button variant="secondary" onClick={onAmbiguous} disabled={disabled} className="h-8">Ambiguous <kbd className="ml-1 opacity-60">M</kbd></Button>
      <Button variant="ghost" onClick={onDefer} disabled={disabled} className="h-8">Defer <kbd className="ml-1 opacity-60">D</kbd></Button>
      <Button variant="ghost" onClick={onAddPrototype} disabled={disabled} className="h-8">+Proto <kbd className="ml-1 opacity-60">P</kbd></Button>
    </div>
  );
}
