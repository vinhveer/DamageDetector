import { cn } from './cn.js';

const tones = {
  neutral: 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)]',
  blue:    'border-transparent bg-[var(--primary-bg)] text-[var(--primary)]',
  red:     'border-transparent bg-[var(--danger-bg)]  text-[var(--danger)]',
  green:   'border-transparent bg-[var(--success-bg)] text-[var(--success)]',
};

export default function Badge({ children, tone = 'neutral', className = '' }) {
  return (
    <span
      data-ui="badge"
      className={cn(
        'inline-flex h-5 items-center rounded-[4px] border px-1.5 text-[11px] font-medium',
        tones[tone],
        className
      )}
    >
      {children}
    </span>
  );
}
