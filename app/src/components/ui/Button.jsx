import { cn } from './cn.js';

const variants = {
  primary:   'border-transparent bg-[var(--primary)] text-[var(--bg)] hover:bg-[var(--primary-hover)]',
  secondary: 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text)] hover:border-[var(--text-subtle)] hover:bg-[var(--surface-3)]',
  danger:    'border-transparent bg-[var(--danger)] text-white hover:opacity-90',
  ghost:     'border-transparent bg-transparent text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
};

export default function Button({ variant = 'secondary', className = '', type = 'button', ...props }) {
  return (
    <button
      type={type}
      className={cn(
        'inline-flex h-8 items-center justify-center gap-1.5 rounded-[6px] border px-3 text-[13px] font-medium',
        'disabled:cursor-not-allowed disabled:opacity-40',
        variants[variant],
        className
      )}
      {...props}
    />
  );
}
