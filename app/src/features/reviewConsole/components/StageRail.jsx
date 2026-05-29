import { cn } from '../../../components/ui/cn.js';
import { LABELS, STAGES, formatNum } from '../reviewConstants.js';

function CountPill({ value }) {
  return (
    <span className="ml-auto rounded-[4px] bg-[var(--surface-2)] px-1.5 text-[11px] tabular-nums text-[var(--text-muted)]">
      {formatNum(value)}
    </span>
  );
}

export default function StageRail({ stage, onSelectStage, counts, filters, onFilterChange, deferredCount }) {
  return (
    <aside className="flex w-[230px] shrink-0 flex-col gap-3 overflow-auto border-r border-[var(--border-muted)] bg-[var(--surface)] p-3">
      <div>
        <div className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Stages</div>
        <div className="flex flex-col gap-0.5">
          {STAGES.map((s) => {
            const active = stage === s.value;
            return (
              <button
                key={s.value}
                type="button"
                onClick={() => onSelectStage(s.value)}
                className={cn(
                  'flex h-8 w-full items-center gap-2 rounded-[5px] px-2 text-[12px] font-medium',
                  active ? 'bg-[var(--active)] text-[var(--text)]' : 'text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
                )}
              >
                <span className="truncate">{s.label}</span>
                <CountPill value={counts?.[s.countKey]} />
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex items-center justify-between rounded-[5px] border border-[var(--border-muted)] px-2 py-1.5 text-[11px] text-[var(--text-muted)]">
        <span>Deferred (draft)</span>
        <span className="tabular-nums">{formatNum(deferredCount)}</span>
      </div>

      <div className="mt-1 border-t border-[var(--border-muted)] pt-3">
        <div className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Filters</div>
        <div className="flex flex-col gap-2 text-[11px]">
          <label className="grid gap-1">
            <span className="text-[var(--text-muted)]">Label</span>
            <select
              value={filters.label || ''}
              onChange={(e) => onFilterChange('label', e.target.value)}
              className="h-7 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[12px] text-[var(--text)]"
            >
              <option value="">All</option>
              {LABELS.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </label>

          <label className="grid gap-1">
            <span className="text-[var(--text-muted)]">Reason code</span>
            <input
              type="text"
              value={filters.reasonCode || ''}
              placeholder="e.g. model_disagreement"
              onChange={(e) => onFilterChange('reasonCode', e.target.value)}
              className="h-7 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[12px] text-[var(--text)] focus:border-[var(--primary)] focus:outline-none"
            />
          </label>

          <div className="grid grid-cols-2 gap-2">
            <label className="grid gap-1">
              <span className="text-[var(--text-muted)]">Reliab. min</span>
              <input
                type="number" min="0" max="1" step="0.05"
                value={filters.reliabilityMin ?? ''}
                onChange={(e) => onFilterChange('reliabilityMin', e.target.value)}
                className="h-7 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[12px] text-[var(--text)]"
              />
            </label>
            <label className="grid gap-1">
              <span className="text-[var(--text-muted)]">Reliab. max</span>
              <input
                type="number" min="0" max="1" step="0.05"
                value={filters.reliabilityMax ?? ''}
                onChange={(e) => onFilterChange('reliabilityMax', e.target.value)}
                className="h-7 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[12px] text-[var(--text)]"
              />
            </label>
          </div>
        </div>
      </div>
    </aside>
  );
}
