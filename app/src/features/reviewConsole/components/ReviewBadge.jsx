import { cn } from '../../../components/ui/cn.js';

// SPEC §7.3 badges. Color is paired with a text label so it's never the only channel.
const tones = {
  green: 'bg-[var(--success-bg)] text-[var(--success)]',
  amber: 'bg-[var(--warning-bg)] text-[var(--warning)]',
  red: 'bg-[var(--danger-bg)] text-[var(--danger)]',
  blue: 'bg-[var(--primary-bg)] text-[var(--primary)]',
  gray: 'bg-[var(--surface-2)] text-[var(--text-muted)]',
};

export default function ReviewBadge({ children, tone = 'gray', className = '' }) {
  return (
    <span
      className={cn(
        'inline-flex h-5 items-center rounded-[4px] border border-transparent px-1.5 text-[10px] font-semibold uppercase tracking-wide',
        tones[tone] || tones.gray,
        className,
      )}
    >
      {children}
    </span>
  );
}
