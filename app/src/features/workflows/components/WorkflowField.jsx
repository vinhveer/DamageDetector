import { Field, SelectControl, TextInput, Button } from '../../../components/ui/index.js';

export default function WorkflowField({ workflowId, input, value, onChange }) {
  const label = input.label || input.name;
  const update = (next) => onChange(input.name, next);

  if (input.type === 'boolean') {
    return (
      <label className="flex items-center gap-2 text-[13px] text-[var(--text)]">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => update(event.currentTarget.checked)}
          className="h-4 w-4 rounded-[3px] border-[var(--border)] text-[var(--primary)]"
        />
        <span>{label}</span>
      </label>
    );
  }

  if (input.type === 'choice') {
    return (
      <Field label={label}>
        <SelectControl value={value === '' || value == null ? '' : String(value)} onChange={(event) => update(event.currentTarget.value)}>
          {(input.choices || []).map((choice) => (
            <option key={`${workflowId}-${input.name}-${choice}`} value={String(choice)}>{String(choice)}</option>
          ))}
        </SelectControl>
      </Field>
    );
  }

  if (input.type === 'integer') {
    return (
      <Field label={label}>
        <TextInput type="number" value={typeof value === 'number' ? value : Number(value || 0)} onChange={(event) => update(Number(event.currentTarget.value || 0))} />
      </Field>
    );
  }

  const isPath = input.type === 'path';
  const browse = async () => {
    const selected = await window.electronAPI.browsePath(input.browse || 'file');
    if (selected) update(selected);
  };

  return (
    <Field label={label}>
      <div className="flex gap-2">
        <TextInput value={String(value ?? '')} onChange={(event) => update(event.currentTarget.value)} className="flex-1" />
        {isPath && (
          <Button onClick={browse} className="shrink-0">Browse</Button>
        )}
      </div>
    </Field>
  );
}
