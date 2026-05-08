import { cn } from './cn.js';

const tones = {
  neutral: 'border-slate-200 bg-slate-50 text-slate-700',
  blue: 'border-blue-200 bg-blue-50 text-blue-700',
  red: 'border-red-200 bg-red-50 text-red-700',
  green: 'border-emerald-200 bg-emerald-50 text-emerald-700'
};

export default function Badge({ children, tone = 'neutral', className = '' }) {
  return (
    <span data-ui="badge" className={cn('inline-flex h-5 items-center rounded border px-1.5 text-[11px] font-medium', tones[tone], className)}>
      {children}
    </span>
  );
}