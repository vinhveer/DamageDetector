import { IconAdjustments, IconDatabase, IconRefresh } from '@tabler/icons-react';
import { Button } from '../../../components/ui/index.js';
import { ErrorMessage } from '../../../components/ui/index.js';
import PageHeader from './PageHeader.jsx';
import PathField from './PathField.jsx';

export default function ConnectView({ paths, loading, error, onPathChange, onLoad, onOpenSettings }) {
  return (
    <div className="rv-enter flex h-full flex-col bg-[var(--docker-bg)] rv-font">
      <PageHeader
        title="Result Viewer"
        right={
          <Button onClick={onOpenSettings}>
            <IconAdjustments size={14} />
            Settings
          </Button>
        }
      />
      <main className="flex min-h-0 flex-1 items-start justify-center overflow-auto px-8 py-16">
        <div className="grid w-full max-w-[640px] gap-5">
          <div className="flex items-center gap-2 text-[var(--docker-text)]">
            <IconDatabase size={18} />
            <h2 className="text-[18px] font-semibold">Connect database</h2>
          </div>
          <div className="grid gap-3">
            <PathField label="Feature DB" value={paths.featureDbPath} onChange={(value) => onPathChange('featureDbPath', value)} />
            <PathField label="Source DB" value={paths.sourceDbPath} onChange={(value) => onPathChange('sourceDbPath', value)} />
            <PathField label="Image root" browseMode="directory" value={paths.imageRootPath} onChange={(value) => onPathChange('imageRootPath', value)} />
          </div>
          {error && <ErrorMessage error={error} />}
          <div className="flex items-center gap-2 pt-1">
            <Button variant="primary" onClick={onLoad} disabled={loading || !paths.featureDbPath}>
              <IconRefresh size={14} />
              {loading ? 'Loading' : 'Load'}
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}