import { cn } from './cn.js';

export default function IconButton({ label, className = '', ...props }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        'inline-flex h-7 w-7 items-center justify-center rounded-[5px]',
        'text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
        className
      )}
      {...props}
    />
  );
}
