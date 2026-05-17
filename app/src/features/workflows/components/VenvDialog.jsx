import { Button, TextInput, Field } from '../../../components/ui/index.js';

export default function VenvDialog({ value, onChange, onUseGlobal, onSet }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4">
      <div data-ui="panel" className="rv-enter w-full max-w-[420px] rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-5 shadow-[0_2px_8px_rgba(0,0,0,0.4)]">
        <div className="text-[15px] font-semibold text-[var(--text)]">Python environment</div>
        <div className="mt-2 text-[13px] text-[var(--text-muted)]">Set a virtual environment directory before running workflows.</div>
        <div className="mt-4">
          <Field label="venv directory">
            <TextInput value={value} onChange={(event) => onChange(event.currentTarget.value)} placeholder="/path/to/.venv" autoFocus />
          </Field>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <Button onClick={onUseGlobal}>Use Global</Button>
          <Button variant="primary" onClick={onSet} disabled={!value.trim()}>Set</Button>
        </div>
      </div>
    </div>
  );
}
