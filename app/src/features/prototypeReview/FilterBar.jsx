import { useEffect, useRef, useState } from 'react';
import { IconChevronDown, IconSearch, IconX } from '@tabler/icons-react';
import { cn } from '../../components/ui/cn.js';

function FilterChip({ label, summary, active, children }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const close = (event) => {
      if (!ref.current?.contains(event.target)) setOpen(false);
    };
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className={cn(
          'flex h-7 items-center gap-1 rounded-full border px-3 text-[12px] font-medium',
          active
            ? 'border-[var(--primary)] bg-[var(--primary-bg)] text-[var(--primary)]'
            : 'border-[var(--border)] bg-transparent text-[var(--text-muted)] hover:bg-[var(--hover)]'
        )}
      >
        {label} · {summary}
        <IconChevronDown size={13} />
      </button>
      {open && (
        <div className="absolute left-0 top-8 z-30 min-w-[220px] rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-3 shadow-md">
          {children}
        </div>
      )}
    </div>
  );
}

function Checklist({ options, selected, counts = {}, onToggle, onAll, onNone, onInvert }) {
  return (
    <div className="space-y-2">
      {options.map((option) => (
        <label key={option.value} className="flex items-center gap-2 text-[12px] text-[var(--text)]">
          <input
            type="checkbox"
            checked={selected.includes(option.value)}
            onChange={() => onToggle(option.value)}
            className="accent-[var(--primary)]"
          />
          <span className="flex-1">{option.label}</span>
          <span className="text-[var(--text-muted)]">{counts[option.value] || 0}</span>
        </label>
      ))}
      <div className="flex gap-2 border-t border-[var(--border-muted)] pt-2">
        <button type="button" onClick={onAll} className="text-[12px] text-[var(--primary)]">All</button>
        <button type="button" onClick={onNone} className="text-[12px] text-[var(--primary)]">None</button>
        <button type="button" onClick={onInvert} className="text-[12px] text-[var(--primary)]">Invert</button>
      </div>
    </div>
  );
}

export default function FilterBar({
  activeLabel,
  tabs,
  selections,
  filters,
  sort,
  sortOptions,
  bucketOptions,
  labelOptions,
  bucketCounts,
  labelCounts,
  totalCount,
  visibleCount,
  pickedCount,
  excludedCount,
  onActiveLabelChange,
  onFiltersChange,
  onSortChange,
  onClear,
}) {
  const [searchDraft, setSearchDraft] = useState(filters.search);

  useEffect(() => setSearchDraft(filters.search), [filters.search]);
  useEffect(() => {
    const timer = window.setTimeout(() => onFiltersChange({ search: searchDraft }), 200);
    return () => window.clearTimeout(timer);
  }, [onFiltersChange, searchDraft]);

  const allBuckets = bucketOptions.map((item) => item.value);
  const allLabels = labelOptions.map((item) => item.value);
  const bucketActive = filters.buckets.length !== allBuckets.length;
  const labelActive = filters.labels.length !== allLabels.length;
  const confidenceActive = filters.minTopScore > 0 || filters.minConfidenceGap > 0;
  const anyActive = Boolean(filters.search) || bucketActive || labelActive || filters.picks !== 'all' || confidenceActive;
  const sortLabel = sortOptions.find((item) => item.value === sort)?.label || sort;

  const toggleListValue = (key, value) => {
    const current = filters[key];
    onFiltersChange({
      [key]: current.includes(value) ? current.filter((item) => item !== value) : [...current, value]
    });
  };

  return (
    <div className="border-b border-[var(--border-muted)] bg-[var(--bg)]">
      <div className="flex items-center gap-1 px-6 py-2">
        {tabs.map((label) => (
          <button
            key={label}
            type="button"
            onClick={() => onActiveLabelChange(label)}
            className={cn(
              'rounded-[4px] px-3 py-1.5 text-[13px] font-medium',
              label === 'excluded'
                ? activeLabel === label
                  ? 'bg-[var(--danger-bg)] text-[var(--danger)]'
                  : 'text-[var(--danger)] opacity-70 hover:bg-[var(--danger-bg)]'
                : activeLabel === label
                  ? 'bg-[var(--primary-bg)] text-[var(--primary)]'
                  : 'text-[var(--text-muted)] hover:bg-[var(--hover)]'
            )}
          >
            {label} {selections[label].clusters.size}g+{selections[label].images.size}i
          </button>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2 px-6 pb-2">
        <div className="flex h-7 min-w-[260px] flex-1 items-center gap-2 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5">
          <IconSearch size={14} className="text-[var(--text-muted)]" />
          <input
            value={searchDraft}
            onChange={(event) => setSearchDraft(event.currentTarget.value)}
            placeholder="Search cluster, result, image path"
            className="min-w-0 flex-1 bg-transparent text-[12px] text-[var(--text)] outline-none placeholder:text-[var(--text-muted)]"
          />
        </div>

        <select
          value={sort}
          onChange={(event) => onSortChange(event.currentTarget.value)}
          className="h-7 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[12px] text-[var(--text)] outline-none"
        >
          {sortOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
      </div>

      <div className="flex flex-wrap items-center gap-2 px-6 pb-2">
        <FilterChip label="Bucket" summary={bucketActive ? `${filters.buckets.length}/${allBuckets.length}` : 'all'} active={bucketActive}>
          <Checklist
            options={bucketOptions}
            selected={filters.buckets}
            counts={bucketCounts}
            onToggle={(value) => toggleListValue('buckets', value)}
            onAll={() => onFiltersChange({ buckets: allBuckets })}
            onNone={() => onFiltersChange({ buckets: [] })}
            onInvert={() => onFiltersChange({ buckets: allBuckets.filter((value) => !filters.buckets.includes(value)) })}
          />
        </FilterChip>

        <FilterChip label="Label" summary={labelActive ? filters.labels.join(', ') || 'none' : 'all'} active={labelActive}>
          <Checklist
            options={labelOptions}
            selected={filters.labels}
            counts={labelCounts}
            onToggle={(value) => toggleListValue('labels', value)}
            onAll={() => onFiltersChange({ labels: allLabels })}
            onNone={() => onFiltersChange({ labels: [] })}
            onInvert={() => onFiltersChange({ labels: allLabels.filter((value) => !filters.labels.includes(value)) })}
          />
        </FilterChip>

        <FilterChip label="Picks" summary={filters.picks.replaceAll('_', ' ')} active={filters.picks !== 'all'}>
          {[
            ['all', 'All'],
            ['picked_this', 'Picked this label'],
            ['picked_any', 'Picked any label'],
            ['not_picked', 'Not picked'],
            ['excluded', 'Excluded only'],
          ].map(([value, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => onFiltersChange({ picks: value })}
              className={cn(
                'block w-full rounded-[4px] px-2 py-1.5 text-left text-[12px]',
                filters.picks === value ? 'bg-[var(--primary-bg)] text-[var(--primary)]' : 'text-[var(--text)] hover:bg-[var(--hover)]'
              )}
            >
              {label}
            </button>
          ))}
        </FilterChip>

        <FilterChip
          label="Confidence"
          summary={confidenceActive ? `>=${filters.minTopScore.toFixed(2)} / gap ${filters.minConfidenceGap.toFixed(2)}` : 'any'}
          active={confidenceActive}
        >
          <div className="space-y-3 text-[12px] text-[var(--text)]">
            <label className="block">
              <span className="mb-1 block text-[var(--text-muted)]">Top score &gt;= {filters.minTopScore.toFixed(2)}</span>
              <input type="range" min={0} max={1} step={0.01} value={filters.minTopScore} onChange={(event) => onFiltersChange({ minTopScore: Number(event.currentTarget.value) })} className="w-full" />
            </label>
            <label className="block">
              <span className="mb-1 block text-[var(--text-muted)]">Confidence gap &gt;= {filters.minConfidenceGap.toFixed(2)}</span>
              <input type="range" min={0} max={0.5} step={0.01} value={filters.minConfidenceGap} onChange={(event) => onFiltersChange({ minConfidenceGap: Number(event.currentTarget.value) })} className="w-full" />
            </label>
          </div>
        </FilterChip>

        {anyActive && (
          <button type="button" onClick={onClear} className="flex h-7 items-center gap-1 rounded-[4px] px-2 text-[12px] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]">
            <IconX size={13} /> Clear
          </button>
        )}
      </div>

      <div className="px-6 pb-2 text-[12px] text-[var(--text-muted)]">
        Showing {visibleCount} of {totalCount} clusters · {pickedCount} picked · {excludedCount} excluded · Sort by {sortLabel}
      </div>
    </div>
  );
}
