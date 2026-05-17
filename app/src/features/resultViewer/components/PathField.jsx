import { IconFolder } from '@tabler/icons-react';
import { Field, TextInput, IconButton } from '../../../components/ui/index.js';

export default function PathField({ label, value, browseMode = 'file', onChange }) {
  const browse = async () => {
    const selected = await window.electronAPI.browsePath(browseMode);
    if (selected) onChange(selected);
  };
  return (
    <Field label={label}>
      <div className="flex min-w-0 gap-1.5">
        <TextInput
          value={value}
          onChange={(e) => onChange(e.currentTarget.value)}
          className="flex-1"
        />
        <IconButton
          label={`Browse ${label}`}
          onClick={browse}
          className="shrink-0 border border-[var(--border)] bg-[var(--surface-2)] hover:bg-[var(--surface-3)]"
        >
          <IconFolder size={14} />
        </IconButton>
      </div>
    </Field>
  );
}
