import { IconAdjustments, IconDatabase, IconRefresh } from '@tabler/icons-react';
import { Button, ErrorMessage } from '../../../components/ui/index.js';
import PageHeader from './PageHeader.jsx';
import PathField from './PathField.jsx';

export default function ConnectView({ paths, loading, error, onPathChange, onLoad, onOpenSettings }) {
  return (
    <div className="rv-enter flex h-full flex-col bg-[var(--bg)] rv-font">
      <PageHeader
        title="Result Viewer"
        right={
          <Button onClick={onOpenSettings}>
            <IconAdjustments size={14} />
            Settings
          </Button>
        }
      />

      <main className="flex min-h-0 flex-1 items-start justify-center overflow-auto px-6 py-12">
        <div className="w-full max-w-[560px]">

          {/* Section heading */}
          <div className="mb-5 flex items-center gap-2">
            <IconDatabase size={16} className="text-[var(--text-muted)]" />
            <span className="text-[13px] font-medium text-[var(--text)]">Connect database</span>
          </div>

          {/* Form card */}
          <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface)] p-5">
            <div className="grid gap-4">
              <PathField
                label="Feature DB"
                value={paths.featureDbPath}
                onChange={(v) => onPathChange('featureDbPath', v)}
              />
              <PathField
                label="Source DB"
                value={paths.sourceDbPath}
                onChange={(v) => onPathChange('sourceDbPath', v)}
              />
              <PathField
                label="Image root"
                browseMode="directory"
                value={paths.imageRootPath}
                onChange={(v) => onPathChange('imageRootPath', v)}
              />
            </div>

            {error && <div className="mt-4"><ErrorMessage error={error} /></div>}

            <div className="mt-5 border-t border-[var(--border-muted)] pt-4">
              <Button
                variant="primary"
                onClick={onLoad}
                disabled={loading || !paths.featureDbPath}
              >
                <IconRefresh size={14} />
                {loading ? 'Loading…' : 'Load'}
              </Button>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
