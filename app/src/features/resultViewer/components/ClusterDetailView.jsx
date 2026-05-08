import { useEffect, useMemo, useState } from 'react';
import { IconChevronLeft } from '@tabler/icons-react';
import { cn } from '../../../components/ui/cn.js';
import { IconButton, Button, EmptyState, ErrorMessage } from '../../../components/ui/index.js';
import PageHeader from './PageHeader.jsx';
import ImageGrid from './ImageGrid.jsx';
import AssignmentsTable from './AssignmentsTable.jsx';
import ImageLightbox from './ImageLightbox.jsx';
import { formatNumber, groupRowsByImage, DETAIL_VIEWS } from '../utils.js';

export default function ClusterDetailView({ cluster, assignments, imageSize, loading, error, onBack, onClearClusterFlags, onClearResultFlags }) {
  const [view, setView] = useState('grid');
  const [lightboxIndex, setLightboxIndex] = useState(-1);
  const groups = useMemo(() => groupRowsByImage(assignments), [assignments]);

  useEffect(() => {
    setLightboxIndex(-1);
    setView('grid');
  }, [cluster?.cluster_key]);

  return (
    <div className="rv-enter flex h-full flex-col bg-[var(--docker-bg)] rv-font">
      <PageHeader
        title={cluster.cluster_key}
        subtitle={`${formatNumber(assignments.length)} rows · ${formatNumber(groups.length)} images`}
        left={
          <IconButton label="Back" onClick={onBack}>
            <IconChevronLeft size={16} />
          </IconButton>
        }
        right={
          <>
            <div className="flex h-8 overflow-hidden rounded-md border border-[var(--docker-border)] bg-white">
              {DETAIL_VIEWS.map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setView(item)}
                  className={cn('px-3 text-[13px] font-medium capitalize', view === item ? 'bg-[var(--docker-active)] text-[var(--docker-blue)]' : 'text-[var(--docker-muted)] hover:bg-[var(--docker-hover)]')}
                >
                  {item}
                </button>
              ))}
            </div>
            <Button variant="danger" onClick={onClearClusterFlags}>Clear flags</Button>
          </>
        }
      />
      {error && <div className="px-8 py-3"><ErrorMessage error={error} /></div>}
      <main className="min-h-0 flex-1 overflow-hidden bg-white">
        {loading && <EmptyState title="Loading" />}
        {!loading && view === 'grid' && <ImageGrid groups={groups} imageSize={imageSize} onOpenImage={setLightboxIndex} onClearFlags={onClearResultFlags} />}
        {!loading && view === 'table' && <AssignmentsTable rows={assignments} />}
      </main>
      <div className="relative z-[9999] pointer-events-auto">
        <ImageLightbox groups={groups} index={lightboxIndex} onClose={() => setLightboxIndex(-1)} />
      </div>
    </div>
  );
}
