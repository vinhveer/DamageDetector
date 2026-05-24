import { useCallback, useEffect, useMemo, useState } from 'react';
import { IconRefresh, IconStar, IconStarFilled, IconDots, IconArchive, IconPencil, IconCopy } from '@tabler/icons-react';
import { Button, EmptyState, ErrorMessage, SelectControl } from '../../components/ui/index.js';
import PageHeader from '../shared/PageHeader.jsx';
import { cn } from '../../components/ui/cn.js';
import { shortId } from '../shared/viewerUtils.js';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime.js';

dayjs.extend(relativeTime);

function RowMenu({ run, onAction, onClose }) {
  useEffect(() => {
    const close = () => onClose();
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [onClose]);

  const items = [
    { key: 'open', label: 'Open' },
    { key: 'setActive', label: 'Set as active', disabled: !!run.is_active },
    { key: 'rename', label: 'Rename' },
    { key: 'duplicate', label: 'New from this…' },
    run.is_archived
      ? { key: 'unarchive', label: 'Unarchive' }
      : { key: 'archive', label: 'Archive', danger: !!run.is_active },
  ];

  return (
    <div
      className="absolute right-0 top-full z-50 mt-1 min-w-[160px] rounded-[4px] border border-[var(--border)] bg-[var(--surface)] py-1 shadow-md"
      onMouseDown={(e) => e.stopPropagation()}
    >
      {items.map((item) => (
        <button
          key={item.key}
          type="button"
          disabled={item.disabled}
          onMouseDown={(e) => { e.stopPropagation(); onAction(item.key, run); onClose(); }}
          className={cn(
            'block w-full px-3 py-1.5 text-left text-[13px] disabled:opacity-40',
            item.danger ? 'text-[var(--danger)]' : 'text-[var(--text)] hover:bg-[var(--hover)]'
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

export default function VersionsList({ paths, onOpenVersion, onNewVersion }) {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showArchived, setShowArchived] = useState(false);
  const [menuRunId, setMenuRunId] = useState(null);
  const [renaming, setRenaming] = useState(null); // { review_run_id, name }

  const load = useCallback(async () => {
    if (!paths.reviewDbPath) return;
    setLoading(true);
    setError('');
    try {
      const result = await window.electronAPI.listPrototypeVersions({
        reviewDbPath: paths.reviewDbPath,
        include_archived: showArchived
      });
      setRuns(result.runs || []);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }, [paths.reviewDbPath, showArchived]);

  useEffect(() => { load(); }, [load]);

  const parentNameMap = useMemo(() => {
    const map = {};
    for (const r of runs) map[r.review_run_id] = r.display_name || shortId(r.review_run_id);
    return map;
  }, [runs]);

  const handleAction = async (action, run) => {
    if (action === 'open') return onOpenVersion(run.review_run_id);
    if (action === 'duplicate') return onNewVersion(run.review_run_id);
    if (action === 'rename') return setRenaming({ review_run_id: run.review_run_id, name: run.display_name || '' });
    if (action === 'setActive') {
      await window.electronAPI.setPrototypeVersionActive({ reviewDbPath: paths.reviewDbPath, review_run_id: run.review_run_id });
      load();
    }
    if (action === 'archive') {
      if (run.is_active && !confirm('This is the active version. Archiving will deactivate it. Continue?')) return;
      await window.electronAPI.archivePrototypeVersion({ reviewDbPath: paths.reviewDbPath, review_run_id: run.review_run_id });
      load();
    }
    if (action === 'unarchive') {
      await window.electronAPI.unarchivePrototypeVersion({ reviewDbPath: paths.reviewDbPath, review_run_id: run.review_run_id });
      load();
    }
  };

  const submitRename = async () => {
    if (!renaming) return;
    const result = await window.electronAPI.renamePrototypeVersion({
      reviewDbPath: paths.reviewDbPath,
      review_run_id: renaming.review_run_id,
      display_name: renaming.name
    });
    if (result.error) { setError(result.error); return; }
    setRenaming(null);
    load();
  };

  return (
    <div className="rv-enter flex h-full flex-col bg-[var(--bg)] rv-font">
      <PageHeader
        title="Prototype Review"
        subtitle={`${runs.length} versions`}
        right={
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1.5 text-[12px] text-[var(--text-muted)]">
              <input type="checkbox" checked={showArchived} onChange={(e) => setShowArchived(e.target.checked)} className="accent-[var(--primary)]" />
              Show archived
            </label>
            <Button onClick={load}><IconRefresh size={14} /> Refresh</Button>
          </div>
        }
      />
      {error && <div className="px-6 py-2"><ErrorMessage error={error} /></div>}
      <main className="min-h-0 flex-1 overflow-auto">
        {loading && <EmptyState title="Loading" />}
        {!loading && runs.length === 0 && (
          <EmptyState title="No versions yet" />
        )}
        {!loading && runs.length > 0 && (
          <table className="w-full min-w-[900px] border-collapse text-left text-[13px]">
            <thead className="sticky top-0 z-10 border-b border-[var(--border-muted)] bg-[var(--surface)] text-[12px] font-semibold text-[var(--text-muted)]">
              <tr>
                <th className="w-8 px-3 py-3"></th>
                <th className="px-4 py-3">Name</th>
                <th className="w-28 px-4 py-3">Parent</th>
                <th className="w-36 px-4 py-3">Created</th>
                <th className="w-32 px-4 py-3 text-center">Prototypes</th>
                <th className="w-24 px-4 py-3 text-right">Clusters</th>
                <th className="w-24 px-4 py-3 text-right">A / N</th>
                <th className="w-12 px-3 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr
                  key={run.review_run_id}
                  onClick={() => onOpenVersion(run.review_run_id)}
                  className={cn(
                    'cursor-pointer border-b border-[var(--border-muted)] hover:bg-[var(--hover)]',
                    run.is_archived && 'opacity-50'
                  )}
                >
                  <td className="px-3 py-3 text-center">
                    {run.is_active ? <IconStarFilled size={14} className="text-[var(--warning)]" /> : null}
                  </td>
                  <td className="px-4 py-3 font-medium text-[var(--text)]">
                    {renaming?.review_run_id === run.review_run_id ? (
                      <input
                        autoFocus
                        value={renaming.name}
                        onChange={(e) => setRenaming({ ...renaming, name: e.target.value })}
                        onBlur={submitRename}
                        onKeyDown={(e) => { if (e.key === 'Enter') submitRename(); if (e.key === 'Escape') setRenaming(null); }}
                        onClick={(e) => e.stopPropagation()}
                        className="w-full rounded border border-[var(--primary)] bg-transparent px-1 py-0.5 text-[13px] outline-none"
                      />
                    ) : (
                      run.display_name || shortId(run.review_run_id)
                    )}
                  </td>
                  <td className="px-4 py-3 text-[var(--text-muted)]">
                    {run.parent_review_run_id ? (parentNameMap[run.parent_review_run_id] || '—') : '—'}
                  </td>
                  <td className="px-4 py-3 text-[var(--text-muted)]" title={run.created_at_utc}>
                    {dayjs(run.created_at_utc).fromNow()}
                  </td>
                  <td className="px-4 py-3 text-center tabular-nums text-[var(--text)]">
                    {run.prototype_counts?.crack || 0}/{run.prototype_counts?.spall || 0}/{run.prototype_counts?.mold || 0}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums text-[var(--text)]">{run.total_clusters}</td>
                  <td className="px-4 py-3 text-right tabular-nums text-[var(--text)]">
                    {run.auto_accept_clusters} / {run.need_review_clusters}
                  </td>
                  <td className="relative px-3 py-3 text-center" onClick={(e) => e.stopPropagation()}>
                    <button
                      type="button"
                      onClick={() => setMenuRunId(menuRunId === run.review_run_id ? null : run.review_run_id)}
                      className="rounded p-1 text-[var(--text-muted)] hover:bg-[var(--hover)]"
                    >
                      <IconDots size={14} />
                    </button>
                    {menuRunId === run.review_run_id && (
                      <RowMenu run={run} onAction={handleAction} onClose={() => setMenuRunId(null)} />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </main>
    </div>
  );
}
