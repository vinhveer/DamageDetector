import { useEffect, useMemo } from 'react';
import { IconArrowLeft, IconArrowMerge, IconArrowsSplit, IconX } from '@tabler/icons-react';
import BoxThumbnail from './BoxThumbnail.jsx';

const STATUS_META = {
  unreviewed:   { label: 'Chưa duyệt',   color: 'var(--text-muted)', bg: 'var(--surface-2)' },
  approved:     { label: 'Approved',     color: 'var(--success)',    bg: 'rgb(16 185 129 / 0.12)' },
  needs_split:  { label: 'Cần split',    color: 'var(--warning)',    bg: 'rgb(251 191 36 / 0.14)' },
  needs_merge:  { label: 'Cần merge',    color: 'var(--primary)',    bg: 'rgb(99 102 241 / 0.14)' },
  rejected:     { label: 'Reject',       color: 'var(--danger)',     bg: 'rgb(239 68 68 / 0.14)' },
};

const CLASS_META = {
  crack: { label: 'Crack', color: '#fbbf24', bg: 'rgb(251 191 36 / 0.14)', hotkey: '1' },
  mold:  { label: 'Mold',  color: '#10b981', bg: 'rgb(16 185 129 / 0.14)', hotkey: '2' },
  spall: { label: 'Spall', color: '#60a5fa', bg: 'rgb(96 165 250 / 0.14)', hotkey: '3' },
};

const SORT_OPTIONS = [
  { value: 'size', label: 'Size DESC' },
  { value: 'purity', label: 'Purity ASC' },
  { value: 'cluster_id', label: 'Cluster ID' },
];

const fractionPct = (frac) => `${Math.round((frac || 0) * 100)}%`;

const formatSince = (iso) => {
  if (!iso) return '';
  const ts = Date.parse(iso);
  if (!ts) return '';
  const secs = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (secs < 60) return `${secs}s`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m`;
  return `${Math.floor(mins / 60)}h`;
};

function StatusPill({ status }) {
  const meta = STATUS_META[status] || STATUS_META.unreviewed;
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium"
      style={{ color: meta.color, backgroundColor: meta.bg }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: meta.color }} />
      {meta.label}
    </span>
  );
}

function DistributionChips({ distribution, dominant }) {
  const entries = Object.entries(distribution || {}).sort((a, b) => b[1].count - a[1].count);
  if (entries.length === 0) return null;
  return (
    <div className="flex items-center gap-1">
      {entries.map(([label, info]) => {
        const isDominant = label === dominant;
        const color = CLASS_META[label]?.color;
        return (
          <span
            key={label}
            className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium"
            style={{
              color: isDominant ? '#fff' : color || 'var(--text)',
              backgroundColor: isDominant ? color : 'var(--surface-2)',
            }}
          >
            {label} {info.count} ({fractionPct(info.fraction)})
          </span>
        );
      })}
    </div>
  );
}

function TopBar({ session, onBack, totalClusters, saving, lastSavedAt, detail, currentStatus }) {
  const reviewed = Number(session?.stats?.reviewed || 0);
  const pct = totalClusters > 0 ? Math.min(100, (reviewed / totalClusters) * 100) : 0;
  const c = detail?.cluster;

  return (
    <div className="shrink-0 border-b border-[var(--border-muted)] bg-[var(--surface)]">
      <div className="flex items-center gap-3 px-4 py-2.5">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-[6px] border border-[var(--border)] px-3 text-[12px] font-medium text-[var(--text)] hover:bg-[var(--hover)]"
        >
          <IconArrowLeft size={14} />
          Setup
        </button>

        <div className="hidden shrink-0 border-l border-[var(--border-muted)] pl-3 text-[12px] text-[var(--text-muted)] sm:block">
          <span className="font-medium text-[var(--text)]">{session?.title || 'Untitled'}</span>
        </div>

        {c ? (
          <div className="flex min-w-0 flex-1 items-center gap-2.5 border-l border-[var(--border-muted)] pl-3">
            <span className="font-mono text-[14px] font-semibold text-[var(--text)]">
              #{c.cluster_id}
            </span>
            <span className="shrink-0 text-[11px] text-[var(--text-muted)]">
              {c.size} boxes
            </span>
            <div className="min-w-0 flex-1 overflow-hidden">
              <div className="flex flex-wrap items-center gap-1">
                <DistributionChips distribution={c.label_distribution} dominant={c.dominant_label} />
              </div>
            </div>
            <StatusPill status={currentStatus} />
          </div>
        ) : (
          <div className="flex-1 text-[12px] text-[var(--text-muted)]">
            Chọn cluster ở sidebar để bắt đầu.
          </div>
        )}

        <div className="flex shrink-0 items-center gap-2 text-[11px] text-[var(--text-muted)]">
          <span className="font-mono">
            {reviewed}/{totalClusters}
          </span>
          <span className="hidden md:inline">·</span>
          <span className="hidden md:inline-flex items-center gap-2">
            <span title="Approved">✓{Number(session?.stats?.approved || 0)}</span>
            <span title="Split">↯{Number(session?.stats?.needs_split || 0)}</span>
            <span title="Merge">⇆{Number(session?.stats?.needs_merge || 0)}</span>
            <span title="Reject">✗{Number(session?.stats?.rejected || 0)}</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                saving
                  ? 'animate-pulse bg-[var(--warning)]'
                  : lastSavedAt
                    ? 'bg-[var(--success)]'
                    : 'bg-[var(--text-muted)]'
              }`}
            />
            <span className="hidden sm:inline">
              {saving ? 'Saving' : lastSavedAt ? `Saved ${formatSince(lastSavedAt)}` : 'Not saved'}
            </span>
          </span>
        </div>
      </div>
      <div className="h-0.5 w-full bg-[var(--surface-2)]">
        <div className="h-full bg-[var(--primary)] transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function NavSidebar({
  clusters,
  selectedClusterId,
  onSelectCluster,
  sessionEntries,
  sortBy,
  setSortBy,
  statusFilter,
  setStatusFilter,
  labelFilter,
  setLabelFilter,
  labelOptions,
}) {
  return (
    <aside className="flex w-[260px] shrink-0 flex-col border-r border-[var(--border-muted)] bg-[var(--surface)]">
      <div className="shrink-0 border-b border-[var(--border-muted)] p-3">
        <div className="grid grid-cols-3 gap-1.5">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="h-7 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-1 text-[11px] text-[var(--text)]"
          >
            {SORT_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="h-7 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-1 text-[11px] text-[var(--text)]"
          >
            <option value="all">All</option>
            {Object.entries(STATUS_META).map(([key, meta]) => (
              <option key={key} value={key}>{meta.label}</option>
            ))}
          </select>
          <select
            value={labelFilter}
            onChange={(e) => setLabelFilter(e.target.value)}
            className="h-7 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-1 text-[11px] text-[var(--text)]"
          >
            {labelOptions.map((o) => <option key={o} value={o}>{o === 'all' ? 'All labels' : o}</option>)}
          </select>
        </div>
        <div className="mt-2 text-[10px] text-[var(--text-muted)]">
          {clusters.length} cluster{clusters.length === 1 ? '' : 's'}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {clusters.length === 0 ? (
          <div className="p-4 text-[11px] text-[var(--text-muted)]">
            Không có cluster phù hợp với filter.
          </div>
        ) : (
          <div className="divide-y divide-[var(--border-muted)]">
            {clusters.map((c) => {
              const entry = sessionEntries?.[String(c.cluster_id)] || { status: 'unreviewed' };
              const meta = STATUS_META[entry.status] || STATUS_META.unreviewed;
              const isActive = selectedClusterId === c.cluster_id;
              const displayClass = entry.cluster_class || c.dominant_label;
              const dominantFrac = c.label_distribution?.[c.dominant_label]?.fraction ?? 0;
              return (
                <button
                  key={c.cluster_id}
                  type="button"
                  onClick={() => onSelectCluster(c.cluster_id)}
                  className={`relative flex w-full items-center gap-2.5 px-3 py-2.5 text-left transition-colors ${
                    isActive ? 'bg-[var(--active)]' : 'hover:bg-[var(--hover)]'
                  }`}
                >
                  {isActive && (
                    <span className="absolute left-0 top-0 h-full w-0.5 bg-[var(--primary)]" />
                  )}
                  <span
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: meta.color }}
                    title={meta.label}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[12px] font-medium text-[var(--text)]">#{c.cluster_id}</span>
                      <span className="text-[11px] text-[var(--text-muted)]">{c.size}</span>
                    </div>
                    <div className="mt-0.5 flex items-center gap-1.5">
                      <span
                        className="text-[11px] font-medium"
                        style={{ color: CLASS_META[displayClass]?.color || 'var(--text)' }}
                      >
                        {displayClass}
                      </span>
                      <span className="text-[10px] text-[var(--text-muted)]">{fractionPct(dominantFrac)}</span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}

function ClusterCanvas({ detail, loading }) {
  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-[12px] text-[var(--text-muted)]">
        Đang tải cluster…
      </div>
    );
  }
  if (!detail?.cluster) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
        <div className="text-[13px] font-medium text-[var(--text)]">Sẵn sàng</div>
        <div className="text-[12px] text-[var(--text-muted)]">
          Chọn 1 cluster ở sidebar bên trái để bắt đầu review.
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      <section className="px-6 pt-6 pb-4">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-[12px] font-semibold uppercase tracking-wide text-[var(--text)]">
            Representatives
            <span className="ml-2 font-normal text-[var(--text-muted)]">
              ({detail.representatives.length} closest to centroid)
            </span>
          </h3>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {detail.representatives.map((box) => (
            <BoxThumbnail key={box.result_id} box={box} size={220} />
          ))}
        </div>
      </section>

      <section className="border-t border-[var(--border-muted)] px-6 pt-5 pb-8">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-[12px] font-semibold uppercase tracking-wide text-[var(--text)]">
            Members
            <span className="ml-2 font-normal text-[var(--text-muted)]">
              ({detail.members.length} / {detail.total_members})
            </span>
          </h3>
        </div>
        <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-8 xl:grid-cols-10">
          {detail.members.map((box) => (
            <BoxThumbnail key={box.result_id} box={box} size={96} showLabel={false} />
          ))}
        </div>
      </section>
    </div>
  );
}

function ClassActionButton({ classKey, count, dominant, active, onClick, disabled }) {
  const meta = CLASS_META[classKey];
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="group inline-flex h-12 items-center gap-2 rounded-[8px] border px-3 text-[12px] font-semibold transition-all disabled:cursor-not-allowed disabled:opacity-30"
      style={{
        borderColor: active ? meta.color : 'var(--border)',
        backgroundColor: active ? meta.color : dominant ? meta.bg : 'var(--surface)',
        color: active ? '#fff' : meta.color,
        boxShadow: active ? `0 0 0 2px ${meta.color}30` : 'none',
      }}
    >
      <kbd
        className="flex h-6 w-6 items-center justify-center rounded border font-mono text-[11px]"
        style={{
          borderColor: active ? 'rgb(255 255 255 / 0.5)' : 'currentColor',
          backgroundColor: active ? 'rgb(255 255 255 / 0.15)' : 'transparent',
        }}
      >
        {meta.hotkey}
      </kbd>
      <span className="leading-none">No, this is {meta.label}</span>
      <span
        className="ml-1 inline-flex h-5 min-w-[24px] items-center justify-center rounded-full px-1.5 text-[10px] font-medium"
        style={{
          backgroundColor: active ? 'rgb(255 255 255 / 0.15)' : 'var(--surface-2)',
          color: active ? '#fff' : 'var(--text-muted)',
        }}
      >
        {count}
      </span>
      {dominant && !active && (
        <span
          className="ml-1 text-[10px] uppercase tracking-wide opacity-70"
          title="Dominant class trong cluster"
        >
          • dominant
        </span>
      )}
    </button>
  );
}

function IssueActionButton({ label, Icon, hotkey, color, onClick, disabled }) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="inline-flex h-12 items-center gap-2 rounded-[8px] border px-3 text-[12px] font-semibold transition-all hover:bg-current/10 disabled:cursor-not-allowed disabled:opacity-30"
      style={{ borderColor: color, color }}
    >
      <kbd
        className="flex h-6 w-6 items-center justify-center rounded border font-mono text-[11px]"
        style={{ borderColor: 'currentColor' }}
      >
        {hotkey}
      </kbd>
      <Icon size={14} />
      <span className="leading-none">{label}</span>
    </button>
  );
}

function BottomNav({ detail, currentEntry, onConfirmClass, onSetStatus, disabled }) {
  const distribution = detail?.cluster?.label_distribution || {};
  const dominantClass = detail?.cluster?.dominant_label;
  const chosenClass = currentEntry?.cluster_class || null;
  const noCluster = !detail?.cluster;

  return (
    <div className="shrink-0 border-t border-[var(--border-muted)] bg-[var(--surface)]">
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
            Confirm class
          </span>
          {['crack', 'mold', 'spall'].map((classKey) => (
            <ClassActionButton
              key={classKey}
              classKey={classKey}
              count={distribution[classKey]?.count || 0}
              dominant={classKey === dominantClass}
              active={chosenClass === classKey}
              disabled={disabled || noCluster}
              onClick={() => onConfirmClass(classKey)}
            />
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
            Issues
          </span>
          <IssueActionButton
            label="Split"
            Icon={IconArrowsSplit}
            hotkey="S"
            color="var(--warning)"
            disabled={disabled || noCluster}
            onClick={() => onSetStatus('needs_split')}
          />
          <IssueActionButton
            label="Merge"
            Icon={IconArrowMerge}
            hotkey="M"
            color="var(--primary)"
            disabled={disabled || noCluster}
            onClick={() => onSetStatus('needs_merge')}
          />
          <IssueActionButton
            label="Reject"
            Icon={IconX}
            hotkey="R"
            color="var(--danger)"
            disabled={disabled || noCluster}
            onClick={() => onSetStatus('rejected')}
          />
        </div>
      </div>
      <div className="border-t border-[var(--border-muted)] bg-[var(--surface-2)] px-5 py-1.5 text-center font-mono text-[10px] text-[var(--text-muted)]">
        1·2·3 confirm class  ·  S·M·R issues  ·  ↑↓ navigate clusters
      </div>
    </div>
  );
}

export default function ReviewScreen({
  sessionData,
  hasSession,
  saving,
  lastSavedAt,
  totalClusters,
  clusters,
  labelOptions,
  sortBy,
  setSortBy,
  statusFilter,
  setStatusFilter,
  labelFilter,
  setLabelFilter,
  selectedClusterId,
  setSelectedClusterId,
  clusterDetail,
  detailLoading,
  onBack,
  onUpdateEntry,
}) {
  const visibleClusters = useMemo(() => {
    if (!sessionData) return clusters;
    if (statusFilter === 'all') return clusters;
    return clusters.filter((c) => {
      const status = sessionData.clusters?.[String(c.cluster_id)]?.status || 'unreviewed';
      return status === statusFilter;
    });
  }, [clusters, sessionData, statusFilter]);

  const currentEntry = sessionData?.clusters?.[String(selectedClusterId ?? '')] || {
    status: 'unreviewed',
    user_label: '',
    notes: '',
    cluster_class: null,
  };

  const advance = () => {
    const idx = visibleClusters.findIndex((c) => c.cluster_id === selectedClusterId);
    const next = visibleClusters[idx + 1];
    if (next) setSelectedClusterId(next.cluster_id);
  };

  const handleConfirmClass = (classKey) => {
    if (!hasSession) return;
    onUpdateEntry({
      status: 'approved',
      cluster_class: classKey,
      user_label: currentEntry.user_label || classKey,
    });
    advance();
  };

  const handleSetStatus = (status) => {
    if (!hasSession) return;
    onUpdateEntry({ status });
    advance();
  };

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if (selectedClusterId == null) return;
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) return;
      const key = e.key.toLowerCase();
      if (key === '1') { e.preventDefault(); handleConfirmClass('crack'); }
      else if (key === '2') { e.preventDefault(); handleConfirmClass('mold'); }
      else if (key === '3') { e.preventDefault(); handleConfirmClass('spall'); }
      else if (key === 's') { e.preventDefault(); handleSetStatus('needs_split'); }
      else if (key === 'm') { e.preventDefault(); handleSetStatus('needs_merge'); }
      else if (key === 'r') { e.preventDefault(); handleSetStatus('rejected'); }
      else if (e.key === 'ArrowDown') {
        e.preventDefault();
        const idx = visibleClusters.findIndex((c) => c.cluster_id === selectedClusterId);
        const next = visibleClusters[Math.min(idx + 1, visibleClusters.length - 1)];
        if (next) setSelectedClusterId(next.cluster_id);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        const idx = visibleClusters.findIndex((c) => c.cluster_id === selectedClusterId);
        const next = visibleClusters[Math.max(idx - 1, 0)];
        if (next) setSelectedClusterId(next.cluster_id);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedClusterId, visibleClusters, hasSession, currentEntry.user_label]);

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <TopBar
        session={sessionData}
        onBack={onBack}
        totalClusters={totalClusters}
        saving={saving}
        lastSavedAt={lastSavedAt}
        detail={clusterDetail}
        currentStatus={currentEntry.status || 'unreviewed'}
      />
      <div className="flex min-h-0 flex-1">
        <NavSidebar
          clusters={visibleClusters}
          selectedClusterId={selectedClusterId}
          onSelectCluster={setSelectedClusterId}
          sessionEntries={sessionData?.clusters}
          sortBy={sortBy}
          setSortBy={setSortBy}
          statusFilter={statusFilter}
          setStatusFilter={setStatusFilter}
          labelFilter={labelFilter}
          setLabelFilter={setLabelFilter}
          labelOptions={labelOptions}
        />
        <main className="min-w-0 flex-1 overflow-hidden bg-[var(--bg)]">
          <ClusterCanvas detail={clusterDetail} loading={detailLoading} />
        </main>
      </div>
      <BottomNav
        detail={clusterDetail}
        currentEntry={currentEntry}
        onConfirmClass={handleConfirmClass}
        onSetStatus={handleSetStatus}
        disabled={!hasSession}
      />
    </div>
  );
}
