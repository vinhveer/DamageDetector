export default function EmptyState({ title, children }) {
  return (
    <div className="flex h-full min-h-[240px] items-center justify-center text-center text-[13px] text-[var(--docker-muted)]">
      <div className="grid gap-1">
        <div className="font-medium text-[var(--docker-text)]">{title}</div>
        {children && <div>{children}</div>}
      </div>
    </div>
  );
}