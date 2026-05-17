import { Button, Field, SelectControl } from '../../../components/ui/index.js';
import Drawer from './Drawer.jsx';
import PathField from './PathField.jsx';
import { VIEW_MODES, formatNumber, shortId } from '../utils.js';

export default function SettingsDrawer({
  opened,
  paths,
  runs,
  runId,
  mode,
  imageSize,
  loading,
  onClose,
  onPathChange,
  onLoad,
  onRunChange,
  onModeChange,
  onImageSizeChange
}) {
  return (
    <Drawer opened={opened} title="Settings" onClose={onClose}>
      <div className="grid gap-5">
        <section className="grid gap-3">
          <div className="text-[13px] font-semibold text-[var(--text)]">Connection</div>
          <PathField label="Feature DB" value={paths.featureDbPath} onChange={(value) => onPathChange('featureDbPath', value)} />
          <PathField label="Source DB" value={paths.sourceDbPath} onChange={(value) => onPathChange('sourceDbPath', value)} />
          <PathField label="Image root" browseMode="directory" value={paths.imageRootPath} onChange={(value) => onPathChange('imageRootPath', value)} />
          <Button variant="primary" onClick={onLoad} disabled={loading || !paths.featureDbPath} className="w-fit">
            Load
          </Button>
        </section>

        <section className="grid gap-3 border-t border-[var(--border-muted)] pt-4">
          <div className="text-[13px] font-semibold text-[var(--text)]">Results</div>
          <Field label="Run">
            <SelectControl value={runId} onChange={(event) => onRunChange(event.currentTarget.value)} disabled={runs.length === 0}>
              <option value="">No run selected</option>
              {runs.map((run) => (
                <option key={run.grouping_run_id} value={run.grouping_run_id}>
                  {run.created_at_utc?.slice(0, 16) || 'Unknown'} · {shortId(run.grouping_run_id)} · {formatNumber(run.total_clusters)} groups
                </option>
              ))}
            </SelectControl>
          </Field>
          <Field label="View mode">
            <SelectControl value={mode} onChange={(event) => onModeChange(event.currentTarget.value)}>
              {VIEW_MODES.map((item) => (
                <option key={item.value} value={item.value}>{item.label}</option>
              ))}
            </SelectControl>
          </Field>
        </section>

        <section className="grid gap-4 border-t border-[var(--border-muted)] pt-4">
          <div className="text-[13px] font-semibold text-[var(--text)]">Display</div>
          <label className="grid gap-2 text-[12px] font-medium text-[var(--text)]">
            <span>Image size <span className="font-normal text-[var(--text-muted)]">{imageSize}px</span></span>
            <input type="range" min="120" max="720" step="20" value={imageSize} onChange={(event) => onImageSizeChange(Number(event.currentTarget.value))} />
          </label>
        </section>
      </div>
    </Drawer>
  );
}
