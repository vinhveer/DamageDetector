import { cn } from './cn.js';

export default function SelectControl({ className = '', children, ...props }) {
  return (
    <select
      className={cn(
        'h-[34px] min-w-0 rounded-md border border-[var(--docker-border)] bg-white px-2.5 text-[13px] text-[var(--docker-text)] outline-none',
        'focus:border-[var(--docker-blue)] focus:ring-2 focus:ring-blue-100',
        className
      )}
      {...props}
    >
      {children}
    </select>
  );
}