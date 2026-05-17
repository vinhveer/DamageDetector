import { IconAlertTriangle } from '@tabler/icons-react';

export default function ErrorMessage({ error }) {
  return (
    <div
      data-ui="panel"
      className="flex items-start gap-2 rounded-[6px] border border-[var(--danger-bg)] bg-[var(--danger-bg)] px-3 py-2.5 text-[13px] text-[var(--danger)]"
    >
      <IconAlertTriangle size={15} className="mt-0.5 shrink-0" />
      <span>{error}</span>
    </div>
  );
}
