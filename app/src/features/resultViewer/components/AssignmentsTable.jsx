import { useMemo, useState } from 'react';
import { EmptyState, TextInput } from '../../../components/ui/index.js';
import { formatFloat } from '../utils.js';

const columns = [
  { key: 'result_id', label: 'ID', className: 'px-6 py-3.5 text-left' },
  { key: 'image_rel_path', label: 'Image', className: 'px-4 py-3.5 text-left' },
  { key: 'predicted_label', label: 'Label', className: 'px-4 py-3.5 text-left' },
  { key: 'predicted_probability_pct', label: 'Conf%', className: 'px-4 py-3.5 text-right' },
  { key: 'detector_score', label: 'Score', className: 'px-4 py-3.5 text-right' },
  { key: 'distance_to_center', label: 'Distance', className: 'px-4 py-3.5 text-right' },
  { key: 'suggested_label', label: 'Suggested', className: 'px-4 py-3.5 text-left' }
];

const compareValues = (a, b) => {
  const aNumber = Number(a);
  const bNumber = Number(b);
  if (Number.isFinite(aNumber) && Number.isFinite(bNumber)) return aNumber - bNumber;
  return String(a ?? '').localeCompare(String(b ?? ''), undefined, { numeric: true, sensitivity: 'base' });
};

export default function AssignmentsTable({ rows }) {
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState({ key: 'distance_to_center', direction: 'asc' });

  const visibleRows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = needle
      ? rows.filter((row) => [
        row.result_id,
        row.image_rel_path,
        row.image_path,
        row.predicted_label,
        row.suggested_label
      ].some((value) => String(value ?? '').toLowerCase().includes(needle)))
      : rows;
    return [...filtered].sort((a, b) => {
      const result = compareValues(a[sort.key], b[sort.key]);
      return sort.direction === 'asc' ? result : -result;
    });
  }, [query, rows, sort]);

  const toggleSort = (key) => {
    setSort((current) => ({ key, direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc' }));
  };

  if (rows.length === 0) return <EmptyState title="No rows" />;
  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      <div className="flex min-h-16 items-center justify-between border-b border-[var(--border-muted)] px-6 py-3">
        <div className="text-[13px] font-medium text-[var(--text-muted)]">{visibleRows.length} rows</div>
        <TextInput className="h-9 w-[320px]" placeholder="Filter rows" value={query} onChange={(event) => setQuery(event.currentTarget.value)} />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
      <table className="w-full min-w-[1040px] border-collapse text-left text-[13px]">
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
          </tr>
        </thead>
        <tbody>
          {visibleRows.map((row) => (
            <tr key={row.result_id} className="border-b border-[var(--border-muted)] hover:bg-[var(--hover)]">
              <td className="px-6 py-4 rv-mono text-[12px] text-[var(--text-muted)]">{row.result_id}</td>
              <td className="max-w-[420px] truncate px-4 py-4 text-[var(--text-muted)]">{row.image_rel_path || row.image_path || '-'}</td>
              <td className="px-4 py-4 text-[var(--text)]">{row.predicted_label}</td>
              <td className="px-4 py-4 text-right text-[var(--text)]">{formatFloat(row.predicted_probability_pct, 1)}</td>
              <td className="px-4 py-4 text-right text-[var(--text)]">{formatFloat(row.detector_score)}</td>
              <td className="px-4 py-4 text-right text-[var(--text)]">{formatFloat(row.distance_to_center)}</td>
              <td className="px-4 py-4 text-[var(--text-muted)]">{row.suggested_label || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {visibleRows.length === 0 && <EmptyState title="No matching rows" />}
      </div>
    </div>
  );
}
