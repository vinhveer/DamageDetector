import { Button, Field, TextInput } from '../../../components/ui/index.js';

export default function SettingsForm({ saveDir, defaultDir, chooseFolder, resetDefault, updateSaveDir }) {
  return (
    <section className="grid gap-4 rounded-lg border border-[var(--docker-border)] bg-white p-5">
      <Field label="Thư mục lưu">
        <div className="flex gap-2">
          <TextInput value={saveDir} onChange={(event) => updateSaveDir(event.currentTarget.value)} className="flex-1" />
          <Button onClick={chooseFolder} className="shrink-0">Chọn thư mục</Button>
        </div>
      </Field>

      <div className="flex items-center justify-between gap-3 border-t border-[var(--docker-border)] pt-4">
        <div className="min-w-0 text-[13px] text-[var(--docker-muted)]">
          Mặc định: {defaultDir || 'Downloads'}
        </div>
        <Button onClick={resetDefault} disabled={!defaultDir}>Đặt lại mặc định</Button>
      </div>
    </section>
  );
}
