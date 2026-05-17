import { cn } from '../../../components/ui/cn.js';
import { Button, EmptyState } from '../../../components/ui/index.js';
import { formatFloat } from '../utils.js';
import ResultImage from './ResultImage.jsx';

export default function ImageGrid({ groups, imageSize, onOpenImage, onClearFlags }) {
  if (groups.length === 0) return <EmptyState title="No images" />;
  return (
    <div className="grid h-full min-h-0 content-start gap-4 overflow-auto p-6" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${Math.max(190, imageSize)}px, 1fr))` }}>
      {groups.map((group, index) => {
        const row = group[0];
        const flagged = group.some((item) => Number(item.is_outlier) !== 0 || Number(item.label_suspect) !== 0);
        return (
          <div key={`${row?.result_id}-${index}`} data-ui="tile" className={cn('grid justify-items-center gap-2 rounded-[6px] border bg-[var(--surface)] p-2', flagged ? 'border-[var(--danger)]' : 'border-[var(--border-muted)]')}>
            <button type="button" onClick={() => onOpenImage(index)} className="text-left">
              <ResultImage row={row} imageSize={imageSize} />
            </button>
            <div className="flex w-full items-start justify-between gap-2 text-[12px]">
              <div className="min-w-0">
                <div className="truncate font-medium text-[var(--text)]">#{index + 1} · id {row.result_id}</div>
                <div className="truncate text-[var(--text-muted)]">{row.predicted_label} · {formatFloat(row.predicted_probability_pct, 1)}%</div>
              </div>
              {flagged && onClearFlags && <Button variant="ghost" onClick={() => onClearFlags(group.map((item) => item.result_id))} className="h-6 px-1.5 text-[11px] text-[var(--danger)]">Clear</Button>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
