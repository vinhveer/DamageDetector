export default function EmptyState({ title, children }) {
  return (
    <div className="flex h-full min-h-[240px] items-center justify-center text-center">
      <div className="grid gap-1">
        <div className="text-[13px] font-medium text-[var(--text)]">{title}</div>
        {children && <div className="text-[13px] text-[var(--text-muted)]">{children}</div>}
      </div>
    </div>
  );
}
