export default function PageHeader({ title, subtitle, left, right }) {
  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-[var(--border-muted)] px-6">
      <div className="flex min-w-0 items-center gap-2.5">
        {left}
        <div className="min-w-0">
          <span className="truncate text-[13px] font-medium text-[var(--text)]">{title}</span>
          {subtitle && <span className="ml-2 truncate text-[12px] text-[var(--text-muted)]">{subtitle}</span>}
        </div>
      </div>
      {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
    </header>
  );
}
