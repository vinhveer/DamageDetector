export default function Field({ label, children }) {
  return (
    <label className="grid gap-1.5">
      <span className="text-[12px] font-medium text-[var(--text-muted)]">{label}</span>
      {children}
    </label>
  );
}
