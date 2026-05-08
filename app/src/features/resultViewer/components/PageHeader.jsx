export default function PageHeader({ title, subtitle, left, right }) {
  return (
    <header className="flex h-[56px] shrink-0 items-center justify-between bg-[var(--docker-bg)] px-8">
      <div className="flex min-w-0 items-center gap-2.5">
        {left}
        <div className="min-w-0">
          <h1 className="truncate text-[16px] font-semibold leading-5 text-[var(--docker-text)]">{title}</h1>
          {subtitle && <p className="truncate text-[12px] text-[var(--docker-muted)]">{subtitle}</p>}
        </div>
      </div>
      {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
    </header>
  );
}