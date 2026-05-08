import { cn } from './cn.js';

const variants = {
  primary: 'border-[var(--docker-blue)] bg-[var(--docker-blue)] text-white hover:bg-[var(--docker-blue-hover)]',
  secondary: 'border-[var(--docker-border)] bg-white text-[var(--docker-text)] hover:bg-[var(--docker-hover)]',
  danger: 'border-red-600 bg-red-600 text-white hover:bg-red-700',
  ghost: 'border-transparent bg-transparent text-[var(--docker-text)] hover:bg-[var(--docker-hover)]'
};

export default function Button({ variant = 'secondary', className = '', type = 'button', ...props }) {
  return (
    <button
      type={type}
      className={cn(
        'inline-flex h-8 items-center justify-center gap-1.5 rounded-md border px-3 text-[13px] font-medium',
        'disabled:cursor-not-allowed disabled:opacity-50',
        variants[variant],
        className
      )}
      {...props}
    />
  );
}