import { Button, Field, TextInput } from '../../../components/ui/index.js';

export default function SettingsForm({ saveDir, defaultDir, chooseFolder, resetDefault, updateSaveDir }) {
  return (
    <section className="grid gap-4 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-5">
      <Field label="Save directory">
        <div className="flex gap-2">
          <TextInput value={saveDir} onChange={(event) => updateSaveDir(event.currentTarget.value)} className="flex-1" />
          <Button onClick={chooseFolder} className="shrink-0">Browse</Button>
        </div>
      </Field>

      <div className="flex items-center justify-between gap-3 border-t border-[var(--border-muted)] pt-4">
        <div className="min-w-0 text-[13px] text-[var(--text-muted)]">
          Default: {defaultDir || 'Downloads'}
        </div>
        <Button onClick={resetDefault} disabled={!defaultDir}>Reset to default</Button>
      </div>
    </section>
  );
}
