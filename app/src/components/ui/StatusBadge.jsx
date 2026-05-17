import { cn } from './cn.js';

const map = {
  running: 'bg-[var(--primary-bg)] text-[var(--primary)] border-transparent',
  done:    'bg-[var(--success-bg)] text-[var(--success)] border-transparent',
  error:   'bg-[var(--danger-bg)]  text-[var(--danger)]  border-transparent',
};

export default function StatusBadge({ status }) {
  if (!status) return null;
  return (
    <span
      data-ui="badge"
      className={cn(
        'inline-flex h-5 items-center rounded-[4px] border px-1.5 text-[11px] font-medium',
        map[status] ?? 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)]'
      )}
    >
      {status}
    </span>
  );
}
