import { IconAlertTriangle } from '@tabler/icons-react';

export default function ErrorMessage({ error }) {
  return (
    <div data-ui="panel" className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-[13px] text-red-700">
      <IconAlertTriangle size={15} className="mt-0.5 shrink-0" />
      <span>{error}</span>
    </div>
  );
}