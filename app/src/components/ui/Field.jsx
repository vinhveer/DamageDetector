export default function Field({ label, children }) {
  return (
    <label className="grid gap-1.5 text-[12px] font-medium text-[var(--docker-text)]">
      <span>{label}</span>
      {children}
    </label>
  );
}