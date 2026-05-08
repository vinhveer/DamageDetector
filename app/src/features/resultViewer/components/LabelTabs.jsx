import { cn } from '../../../components/ui/cn.js';
import { LABELS } from '../utils.js';

export default function LabelTabs({ selectedLabel, clustersByLabel, onChange }) {
  return (
    <div className="flex h-11 items-end gap-6 bg-[var(--docker-bg)] px-8">
      {LABELS.map((label) => {
        const active = selectedLabel === label;
        const count = clustersByLabel[label]?.length || 0;
        return (
          <button
            key={label}
            type="button"
            onClick={() => onChange(label)}
            className={cn(
              'flex h-10 items-center gap-2 border-b-2 px-0 text-[13px] font-medium capitalize',
              active ? 'border-[var(--docker-blue)] text-[var(--docker-text)]' : 'border-transparent text-[var(--docker-muted)] hover:text-[var(--docker-text)]'
            )}
            data-ui="tab"
          >
            {label}
            <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-600">{count}</span>
          </button>
        );
      })}
    </div>
  );
}