import { cn } from './cn.js';

export default function TextInput({ className = '', ...props }) {
  return (
    <input
      className={cn(
        'h-8 min-w-0 rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5',
        'text-[13px] text-[var(--text)] outline-none',
        'placeholder:text-[var(--text-subtle)]',
        'focus:border-[var(--primary)] focus:ring-2 focus:ring-[var(--primary-bg)]',
        className
      )}
      {...props}
    />
  );
}
