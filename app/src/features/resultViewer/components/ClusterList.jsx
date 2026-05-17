import { useMemo, useState } from 'react';
import { IconChevronRight } from '@tabler/icons-react';
import { cn } from '../../../components/ui/cn.js';
import { Badge, EmptyState, TextInput } from '../../../components/ui/index.js';
import { LABEL_BADGE_CLASSES, formatNumber, formatFloat } from '../utils.js';

const columns = [
  { key: 'cluster_key', label: 'Cluster', className: 'px-6 py-3.5 text-left' },
  { key: 'major_label', label: 'Label', className: 'w-28 px-4 py-3.5 text-left' },
  { key: 'cluster_size', label: 'Images', className: 'w-28 px-4 py-3.5 text-right' },
  { key: 'purity', label: 'Purity', className: 'w-28 px-4 py-3.5 text-right' },
  { key: 'outlier_count', label: 'Outliers', className: 'w-28 px-4 py-3.5 text-right' },
  { key: 'counts', label: 'Counts', className: 'w-44 px-4 py-3.5 text-right' }
];

const sortValue = (cluster, key) => {
  if (key === 'counts') return Number(cluster.crack_count || 0) + Number(cluster.mold_count || 0) + Number(cluster.spall_count || 0);
  return cluster[key];
};

const compareValues = (a, b) => {
  const aNumber = Number(a);
  const bNumber = Number(b);
  if (Number.isFinite(aNumber) && Number.isFinite(bNumber)) return aNumber - bNumber;
  return String(a ?? '').localeCompare(String(b ?? ''), undefined, { numeric: true, sensitivity: 'base' });
};

export default function ClusterList({ clusters, onOpen }) {
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState({ key: 'cluster_size', direction: 'desc' });

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = needle
      ? clusters.filter((cluster) => [
        cluster.cluster_key,
        cluster.major_label,
        cluster.predicted_label_scope,
        cluster.cluster_id
      ].some((value) => String(value ?? '').toLowerCase().includes(needle)))
      : clusters;
    return [...filtered].sort((a, b) => {
      const result = compareValues(sortValue(a, sort.key), sortValue(b, sort.key));
      return sort.direction === 'asc' ? result : -result;
    });
  }, [clusters, query, sort]);

  const toggleSort = (key) => {
    setSort((current) => ({ key, direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc' }));
  };

  if (clusters.length === 0) {
    return <EmptyState title="No clusters" />;
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      <div className="flex min-h-16 items-center justify-between border-b border-[var(--border-muted)] px-6 py-3">
        <div className="text-[13px] font-medium text-[var(--text-muted)]">{formatNumber(rows.length)} clusters</div>
        <TextInput className="h-9 w-[320px]" placeholder="Filter clusters" value={query} onChange={(event) => setQuery(event.currentTarget.value)} />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
      <table className="w-full min-w-[860px] border-collapse text-left text-[13px]">
        <thead className="sticky top-0 z-10 border-b border-[var(--border-muted)] bg-[var(--surface)] text-[12px] font-semibold text-[var(--text-muted)]">
          <tr>
            {columns.map((column) => (
              <th key={column.key} className={column.className}>
                <button type="button" onClick={() => toggleSort(column.key)} className="inline-flex items-center gap-1 hover:text-[var(--text)]">
                  {column.label}
                  {sort.key === column.key && <span className="text-[10px]">{sort.direction === 'asc' ? '↑' : '↓'}</span>}
                </button>
              </th>
            ))}
            <th className="w-12 px-4 py-3.5" />
          </tr>
        </thead>
        <tbody>
          {rows.map((cluster) => (
            <tr key={cluster.cluster_key} onClick={() => onOpen(cluster)} className="cursor-pointer border-b border-[var(--border-muted)] hover:bg-[var(--hover)]">
              <td className="px-6 py-4 font-medium text-[var(--text)]">{cluster.cluster_key}</td>
              <td className="px-4 py-4">
                <span className={cn('rounded border px-1.5 py-0.5 text-[11px] font-medium', LABEL_BADGE_CLASSES[cluster.major_label] || 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)]')}>
                  {cluster.major_label || '-'}
                </span>
              </td>
              <td className="px-4 py-4 text-right text-[var(--text)]">{formatNumber(cluster.cluster_size)}</td>
              <td className="px-4 py-4 text-right text-[var(--text)]">{formatFloat(cluster.purity)}</td>
              <td className="px-4 py-4 text-right">
                {Number(cluster.outlier_count || 0) > 0 ? <Badge tone="red">{cluster.outlier_count}</Badge> : <span className="text-[var(--text-muted)]">0</span>}
              </td>
              <td className="px-4 py-4 text-right text-[12px] text-[var(--text-muted)]">
                C {cluster.crack_count || 0} · M {cluster.mold_count || 0} · S {cluster.spall_count || 0}
              </td>
              <td className="px-4 py-4 text-[var(--text-muted)]"><IconChevronRight size={15} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && <EmptyState title="No matching clusters" />}
      </div>
    </div>
  );
}
