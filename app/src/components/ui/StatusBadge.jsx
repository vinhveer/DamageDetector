import { cn } from './cn.js';

export default function StatusBadge({ status }) {
  if (!status) return null;
  const map = {
    running: 'border-blue-200 bg-blue-50 text-blue-700',
    done: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    error: 'border-red-200 bg-red-50 text-red-700'
  };
  const className = map[status] || 'border-slate-200 bg-slate-50 text-slate-700';
  return (
    <span data-ui="badge" className={cn('inline-flex h-5 items-center rounded border px-1.5 text-[11px] font-medium', className)}>
      {status}
    </span>
  );
}