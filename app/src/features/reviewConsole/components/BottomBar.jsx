import { IconArrowBackUp, IconDeviceFloppy } from '@tabler/icons-react';
import { Button } from '../../../components/ui/index.js';
import { SHORTCUTS, formatNum } from '../reviewConstants.js';

export default function BottomBar({ decisionCount, total, canUndo, committed, saving, onUndo, onCommit }) {
  return (
    <footer className="flex h-10 shrink-0 items-center gap-4 border-t border-[var(--border-muted)] bg-[var(--surface)] px-4 text-[11px] text-[var(--text-muted)]">
      <div className="flex items-center gap-2 overflow-hidden">
        {SHORTCUTS.map((s) => (
          <span key={s.key} className="flex shrink-0 items-center gap-1">
            <kbd className="rounded-[3px] border border-[var(--border)] bg-[var(--surface-2)] px-1 text-[10px] text-[var(--text)]">{s.key}</kbd>
            <span>{s.label}</span>
          </span>
        ))}
      </div>
      <div className="ml-auto flex items-center gap-3">
        <span className="tabular-nums">{formatNum(decisionCount)} drafted{total != null ? ` / ${formatNum(total)}` : ''}</span>
        {saving && <span className="text-[var(--text-muted)]">saving…</span>}
        <Button variant="ghost" onClick={onUndo} disabled={!canUndo || committed} className="h-7 px-2">
          <IconArrowBackUp size={14} className="mr-1" />Undo
        </Button>
        <Button variant="primary" onClick={onCommit} disabled={committed || decisionCount === 0} className="h-7 px-3">
          <IconDeviceFloppy size={14} className="mr-1" />{committed ? 'Committed' : 'Commit session'}
        </Button>
      </div>
    </footer>
  );
}
