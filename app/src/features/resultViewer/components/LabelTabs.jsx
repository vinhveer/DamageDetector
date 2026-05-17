import { cn } from '../../../components/ui/cn.js';
import { LABELS } from '../utils.js';

export default function LabelTabs({ selectedLabel, clustersByLabel, onChange }) {
  return (
    <div className="flex h-10 items-end gap-5 border-b border-[var(--border-muted)] px-6">
      {LABELS.map((label) => {
        const active = selectedLabel === label;
        const count  = clustersByLabel[label]?.length ?? 0;
        return (
          <button
            key={label}
            type="button"
            onClick={() => onChange(label)}
            data-ui="tab"
            className={cn(
              'flex h-10 items-center gap-1.5 border-b-2 px-0 pb-0.5 text-[13px] font-medium capitalize',
              active
                ? 'border-[var(--primary)] text-[var(--text)]'
                : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
            )}
          >
            {label}
            <span className={cn(
              'rounded-[4px] px-1.5 py-0.5 text-[11px]',
              active
                ? 'bg-[var(--primary-bg)] text-[var(--primary)]'
                : 'bg-[var(--surface-2)] text-[var(--text-muted)]'
            )}>
              {count}
            </span>
          </button>
        );
      })}
    </div>
  );
}
