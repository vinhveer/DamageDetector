import { cn } from './cn.js';

export default function IconButton({ label, className = '', ...props }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        'inline-flex h-8 w-8 items-center justify-center rounded-md border border-transparent text-[var(--docker-muted)]',
        'hover:border-[var(--docker-border)] hover:bg-white hover:text-[var(--docker-text)]',
        className
      )}
      {...props}
    />
  );
}