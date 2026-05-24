import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  IconArrowLeft,
  IconCheck,
  IconChecks,
  IconLoader2,
  IconPlus,
  IconTrash,
} from '@tabler/icons-react';
import BoxThumbnail from '../clusterLabeling/components/BoxThumbnail.jsx';
import { cn } from '../../components/ui/cn.js';

const ALL_CLASSES = ['mold', 'spall', 'crack', 'reject'];

const classColor = (c) => (
  c === 'crack' ? 'text-[#fbbf24]'
    : c === 'mold' ? 'text-[#10b981]'
    : c === 'spall' ? 'text-[#60a5fa]'
    : c === 'reject' ? 'text-[#f472b6]'
    : 'text-[var(--text)]'
);

const classBorderHex = (c) => (
  c === 'crack' ? '#fbbf24'
    : c === 'mold' ? '#10b981'
    : c === 'spall' ? '#60a5fa'
    : c === 'reject' ? '#f472b6'
    : ''
);

const formatNum = (n) => Number(n || 0).toLocaleString();

const formatSince = (iso) => {
  if (!iso) return '';
  const ts = Date.parse(iso);
  if (!ts) return '';
  const secs = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  return `${Math.floor(mins / 60)}h ago`;
};

// ── Per-box decision helpers ───────────────────────────────────────────────
const buildKeepDecision = (currentClass) => ({
  action: 'keep',
  target_label: null,
  current_label_at_decision: currentClass,
  source: 'group_action',
  decided_at_utc: new Date().toISOString(),
});

const buildChangeDecision = (currentClass, target) => ({
  action: 'change',
  target_label: target,
  current_label_at_decision: currentClass,
  source: 'group_action',
  decided_at_utc: new Date().toISOString(),
});

// ── Setup screen ───────────────────────────────────────────────────────────
function PathField({ label, value, placeholder, onChange }) {
  return (
    <label className="grid gap-1.5">
      <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">{label}</span>
      <input
        type="text"
        value={value || ''}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="h-9 rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] px-3 font-mono text-[12px] text-[var(--text)] focus:border-[var(--primary)] focus:outline-none"
      />
    </label>
  );
}

function SetupScreen({
  paths, onPathChange, onLoad, loading, error,
  runs, selectedRunId, onSelectRun, classTotals,
  sessions, selectedSessionId, onSelectSession,
  onCreateSession, onDeleteSession, sessionTitle, onSessionTitleChange,
  onStart, canStart,
}) {
  return (
    <div className="mx-auto flex h-full w-full max-w-[920px] flex-col gap-5 overflow-auto p-6">
      <header>
        <h1 className="text-[20px] font-semibold text-[var(--text)]">Step 7 — Label Review</h1>
        <p className="mt-1 text-[12px] text-[var(--text-muted)]">
          Common patterns (CV-MLP confident) auto-accept. Uncommon (suspects) clustered into small groups
          with suggested actions. User chỉ review các group.
        </p>
      </header>

      <section className="grid gap-3 rounded-[10px] border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="text-[13px] font-semibold text-[var(--text)]">1 · Databases</div>
        <PathField
          label="suspect_clusters.sqlite3 (Step 7 output)"
          value={paths.suspectClusterDbPath}
          placeholder="/path/to/suspect_clusters.sqlite3"
          onChange={(v) => onPathChange('suspectClusterDbPath', v)}
        />
        <PathField
          label="damage_scan.sqlite3 (Step 2 — for bbox coords)"
          value={paths.sourceDbPath}
          placeholder="/path/to/damage_scan.sqlite3"
          onChange={(v) => onPathChange('sourceDbPath', v)}
        />
        <PathField
          label="Image root"
          value={paths.imageRootPath}
          placeholder="/path/to/images"
          onChange={(v) => onPathChange('imageRootPath', v)}
        />
        <div className="flex items-center justify-between pt-1">
          <span className="text-[11px] text-[var(--text-muted)]">
            {runs.length > 0 ? `${runs.length} runs loaded` : 'Chưa load runs'}
          </span>
          <button
            type="button"
            onClick={onLoad}
            disabled={loading}
            className="h-8 rounded-[6px] border border-[var(--primary)] bg-[var(--primary)] px-3 text-[12px] font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Load runs'}
          </button>
        </div>
        {error && <div className="text-[12px] text-[var(--danger)]">{error}</div>}
      </section>

      {runs.length > 0 && (
        <section className="grid gap-3 rounded-[10px] border border-[var(--border)] bg-[var(--surface)] p-5">
          <div className="text-[13px] font-semibold text-[var(--text)]">2 · Chọn suspect-cluster run</div>
          <div className="grid max-h-[260px] gap-1 overflow-auto">
            {runs.map((run) => {
              const isActive = run.run_id === selectedRunId;
              return (
                <button
                  key={run.run_id}
                  type="button"
                  onClick={() => onSelectRun(run.run_id)}
                  className={cn(
                    'flex items-center justify-between rounded-[6px] border px-3 py-2 text-left text-[12px]',
                    isActive
                      ? 'border-[var(--primary)] bg-[var(--primary)]/10 text-[var(--text)]'
                      : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)] hover:border-[var(--primary)]'
                  )}
                >
                  <div className="flex flex-col">
                    <span className="font-mono text-[11px] text-[var(--text)]">{run.run_id.slice(0, 12)}</span>
                    <span className="text-[10px] text-[var(--text-muted)]">{run.created_at_utc}</span>
                  </div>
                  <div className="text-right text-[11px]">
                    <div className="text-[var(--text)]">{formatNum(run.total_suspects)} suspects</div>
                    <div className="text-[var(--text-muted)]">{run.n_clusters} clusters</div>
                  </div>
                </button>
              );
            })}
          </div>
          {classTotals.length > 0 && (
            <div className="flex flex-wrap gap-2 pt-1 text-[11px]">
              {classTotals.map((c) => (
                <span key={c.current_label} className="rounded border border-[var(--border)] bg-[var(--surface-2)] px-2 py-0.5 font-mono">
                  <span className={classColor(c.current_label)}>{c.current_label}</span>: {formatNum(c.total_suspects)} ({c.n_clusters} groups)
                </span>
              ))}
            </div>
          )}
        </section>
      )}

      {selectedRunId && (
        <section className="grid gap-3 rounded-[10px] border border-[var(--border)] bg-[var(--surface)] p-5">
          <div className="text-[13px] font-semibold text-[var(--text)]">3 · Session</div>
          <div className="flex gap-2">
            <input
              type="text"
              value={sessionTitle}
              placeholder="Tên session mới..."
              onChange={(e) => onSessionTitleChange(e.target.value)}
              className="h-8 flex-1 rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 text-[12px] text-[var(--text)] focus:border-[var(--primary)] focus:outline-none"
            />
            <button
              type="button"
              onClick={onCreateSession}
              className="flex h-8 items-center gap-1.5 rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] px-3 text-[12px] hover:border-[var(--primary)]"
            >
              <IconPlus size={14} /> Tạo session
            </button>
          </div>
          <div className="grid max-h-[200px] gap-1 overflow-auto">
            {sessions.length === 0 && <div className="text-[12px] text-[var(--text-muted)]">Chưa có session nào.</div>}
            {sessions.map((s) => {
              const isActive = s.session_id === selectedSessionId;
              return (
                <div
                  key={s.session_id}
                  className={cn(
                    'flex items-center justify-between gap-2 rounded-[6px] border px-3 py-1.5 text-[12px]',
                    isActive
                      ? 'border-[var(--primary)] bg-[var(--primary)]/10'
                      : 'border-[var(--border)] bg-[var(--surface-2)] hover:border-[var(--primary)]'
                  )}
                >
                  <button
                    type="button"
                    onClick={() => onSelectSession(s.session_id)}
                    className="flex flex-1 flex-col items-start text-left"
                  >
                    <span className="text-[var(--text)]">{s.title || s.session_id}</span>
                    <span className="text-[10px] text-[var(--text-muted)]">
                      {s.session_id.slice(0, 8)} · reviewed {s.stats?.reviewed || 0} ({formatSince(s.last_updated_utc)})
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onDeleteSession(s.session_id)}
                    className="rounded p-1 text-[var(--text-muted)] hover:bg-[var(--danger)]/20 hover:text-[var(--danger)]"
                    title="Delete session"
                  >
                    <IconTrash size={14} />
                  </button>
                </div>
              );
            })}
          </div>
          <button
            type="button"
            disabled={!canStart}
            onClick={onStart}
            className="mt-2 h-9 rounded-[6px] border border-[var(--primary)] bg-[var(--primary)] px-4 text-[13px] font-medium text-white disabled:opacity-50"
          >
            Bắt đầu review →
          </button>
        </section>
      )}
    </div>
  );
}

// ── Review screen ──────────────────────────────────────────────────────────
function ClassTabs({ classTotals, activeClass, onSelect, decidedByClass }) {
  return (
    <div className="flex items-center gap-1 border-b border-[var(--border-muted)] px-4">
      {ALL_CLASSES.map((c) => {
        const t = classTotals.find((x) => x.current_label === c);
        const total = t?.total_suspects || 0;
        const nGroups = t?.n_clusters || 0;
        const decided = decidedByClass[c] || 0;
        const isActive = c === activeClass;
        if (total === 0) return null;
        return (
          <button
            key={c}
            type="button"
            onClick={() => onSelect(c)}
            className={cn(
              'flex h-9 items-center gap-2 border-b-2 px-3 text-[12px] font-medium',
              isActive
                ? `${classColor(c)} border-current`
                : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
            )}
          >
            <span className="capitalize">{c}</span>
            <span className="text-[10px] font-mono text-[var(--text-muted)]">
              {nGroups} groups · {formatNum(total)} suspects
              {decided > 0 && <span className="text-[var(--success)]"> · {decided} ✓</span>}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function ClusterListRow({ cluster, isActive, decided, onClick }) {
  const dot = decided === cluster.size ? 'bg-[var(--success)]'
    : decided > 0 ? 'bg-[#fbbf24]'
    : 'bg-[var(--border)]';
  const label = cluster.is_noise_cluster ? 'Noise' : `#${cluster.cluster_id}`;
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'flex items-center justify-between gap-2 rounded-[6px] border px-2.5 py-2 text-left text-[12px]',
        isActive
          ? 'border-[var(--primary)] bg-[var(--primary)]/10'
          : 'border-transparent hover:border-[var(--border)] hover:bg-[var(--surface-2)]'
      )}
    >
      <div className="flex items-center gap-2 truncate">
        <span className={cn('h-2 w-2 rounded-full', dot)} />
        <span className="font-mono text-[var(--text)]">{label}</span>
        <span className="text-[var(--text-muted)]">·</span>
        <span className="text-[var(--text-muted)]">{cluster.size}</span>
      </div>
      {!cluster.is_noise_cluster && cluster.dominant_cv_label && (
        <span className={cn('font-mono text-[10px]', classColor(cluster.dominant_cv_label))}>
          →{cluster.dominant_cv_label} {(cluster.dominant_cv_fraction * 100).toFixed(0)}%
        </span>
      )}
    </button>
  );
}

function SuggestionCard({ cluster, decisionForGroup, onApply }) {
  if (!cluster) return null;
  const isNoise = cluster.is_noise_cluster;
  const current = cluster.current_label;
  const suggestedTarget = cluster.suggested_target_label;
  const suggestedAction = cluster.suggested_action;
  const fraction = (cluster.dominant_cv_fraction * 100).toFixed(0);

  const choices = [];
  if (!isNoise && suggestedAction === 'change' && suggestedTarget) {
    choices.push({
      key: 'suggested',
      primary: true,
      label: `Accept: change all → ${suggestedTarget}`,
      sublabel: `Classifier nói ${fraction}% nhóm này là ${suggestedTarget}`,
      target: suggestedTarget,
      color: classBorderHex(suggestedTarget),
    });
  }
  choices.push({ key: 'keep', label: `Keep all as ${current}`, target: '__keep__', color: '#10b981' });
  for (const t of ALL_CLASSES) {
    if (t === current) continue;
    if (suggestedAction === 'change' && t === suggestedTarget) continue; // already as primary
    choices.push({ key: `to-${t}`, label: `→ ${t}`, target: t, color: classBorderHex(t) });
  }

  return (
    <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface-2)] p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-[12px] text-[var(--text-muted)]">
          {isNoise
            ? <>Noise group — không có suggestion (CV predictions phân tán)</>
            : <>Suggested action — <span className="font-mono text-[var(--text)]">{cluster.dominant_cv_label || '?'} {fraction}%</span> avg susp <span className="font-mono">{(cluster.avg_suspicion_score * 100).toFixed(0)}</span></>
          }
        </div>
        {decisionForGroup && (
          <span className="rounded bg-[var(--success)]/20 px-2 py-0.5 text-[11px] text-[var(--success)]">
            ✓ Applied: {decisionForGroup.action === 'keep' ? `keep ${current}` : `→ ${decisionForGroup.target_label}`}
          </span>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {choices.map((c) => (
          <button
            key={c.key}
            type="button"
            onClick={() => onApply(c.target)}
            className={cn(
              'flex h-8 items-center gap-1.5 rounded-[6px] border px-3 text-[12px] font-medium',
              c.primary
                ? 'border-current bg-current/15 text-white'
                : 'border-[var(--border)] bg-[var(--surface)] text-[var(--text-muted)] hover:border-current hover:text-[var(--text)]'
            )}
            style={c.primary ? { color: c.color } : undefined}
          >
            {c.primary && <IconCheck size={14} />}
            <span>{c.label}</span>
            {c.sublabel && <span className="text-[10px] opacity-70">· {c.sublabel}</span>}
          </button>
        ))}
      </div>
    </div>
  );
}

function BoxThumb({ box, currentClass, decision }) {
  const effectiveLabel = decision
    ? (decision.action === 'keep' ? currentClass : decision.target_label || 'reject')
    : currentClass;
  const ring = decision ? classBorderHex(effectiveLabel) : '';
  return (
    <div
      className={cn(
        'relative rounded-[6px]',
        decision ? 'ring-2' : 'ring-1 ring-[var(--border-muted)]'
      )}
      style={decision ? { boxShadow: `0 0 0 2px ${ring}` } : undefined}
    >
      <BoxThumbnail
        box={{ ...box, predicted_label: effectiveLabel }}
        size={110}
        showLabel={false}
      />
      <div className="absolute right-1 top-1 rounded bg-[#f472b6]/85 px-1 py-0.5 text-[9px] font-mono text-white">
        !{(box.suspicion_score * 100).toFixed(0)}
      </div>
      {box.cv_predicted_label && box.cv_predicted_label !== currentClass && !decision && (
        <div
          className="absolute left-1 top-1 rounded px-1 py-0.5 text-[9px] font-mono font-semibold text-white"
          style={{ background: classBorderHex(box.cv_predicted_label) }}
        >
          ?→{box.cv_predicted_label}
        </div>
      )}
      {decision && (
        <div
          className="absolute left-1 top-1 rounded px-1 py-0.5 text-[9px] font-mono font-semibold text-white"
          style={{ background: ring }}
        >
          {decision.action === 'keep' ? '✓ keep' : `→ ${decision.target_label || 'reject'}`}
        </div>
      )}
      <div className="absolute bottom-1 right-1 rounded bg-black/70 px-1 py-0.5 font-mono text-[9px] text-white">
        #{box.result_id}
      </div>
    </div>
  );
}

function ReviewScreen({
  classTotals, decisions, runId,
  activeClass, onSelectClass,
  clusters, clustersLoading,
  selectedClusterId, onSelectCluster,
  detail, detailLoading,
  saving, lastSavedAt,
  onApplyGroup, onSkipGroup,
  onBack,
}) {
  const detailMemberIds = useMemo(
    () => (detail?.members || []).map((m) => Number(m.result_id)),
    [detail]
  );

  const totalSuspects = useMemo(
    () => classTotals.reduce((acc, c) => acc + (c.total_suspects || 0), 0),
    [classTotals]
  );

  const decidedByClass = useMemo(() => {
    const out = {};
    for (const d of Object.values(decisions)) {
      const cls = d?.current_label_at_decision;
      if (cls) out[cls] = (out[cls] || 0) + 1;
    }
    return out;
  }, [decisions]);

  // Per-cluster decided count derived from detail (only known for selected cluster).
  const groupDecided = useMemo(() => {
    if (!detail) return 0;
    let n = 0;
    for (const id of detailMemberIds) {
      if (decisions[id]) n += 1;
    }
    return n;
  }, [detail, detailMemberIds, decisions]);

  // First decision in detail group (if uniform across all members) for the "Applied" badge.
  const decisionForGroup = useMemo(() => {
    if (!detail || detailMemberIds.length === 0) return null;
    const first = decisions[detailMemberIds[0]];
    if (!first) return null;
    for (const id of detailMemberIds) {
      const d = decisions[id];
      if (!d || d.action !== first.action || (d.target_label || null) !== (first.target_label || null)) {
        return null;
      }
    }
    return first;
  }, [detail, detailMemberIds, decisions]);

  // Hotkeys: arrow keys = prev/next cluster; Enter = accept suggested; K = keep all; 1/2/3/4 = change all
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      const idx = clusters.findIndex((c) => c.cluster_id === selectedClusterId);
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        const next = clusters[Math.min(clusters.length - 1, idx + 1)];
        if (next) onSelectCluster(next.cluster_id);
        return;
      }
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        const prev = clusters[Math.max(0, idx - 1)];
        if (prev) onSelectCluster(prev.cluster_id);
        return;
      }
      if (e.key === 'Enter' && detail?.cluster?.suggested_action === 'change' && detail.cluster.suggested_target_label) {
        e.preventDefault();
        onApplyGroup(detail.cluster.suggested_target_label, /*advance*/ true);
        return;
      }
      if (e.key === 'k' || e.key === 'K') onApplyGroup('__keep__', true);
      else if (e.key === '1') onApplyGroup('mold', true);
      else if (e.key === '2') onApplyGroup('spall', true);
      else if (e.key === '3') onApplyGroup('crack', true);
      else if (e.key === '4') onApplyGroup('reject', true);
      else if (e.key === 's' || e.key === 'S') onSkipGroup(true);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [clusters, selectedClusterId, onSelectCluster, detail, onApplyGroup, onSkipGroup]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Top bar */}
      <div className="flex items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2">
        <button
          type="button"
          onClick={onBack}
          className="flex h-7 items-center gap-1 rounded-[5px] border border-[var(--border)] px-2 text-[12px] text-[var(--text-muted)] hover:text-[var(--text)]"
        >
          <IconArrowLeft size={14} /> Setup
        </button>
        <div className="text-[12px] text-[var(--text-muted)]">
          Run <span className="font-mono text-[var(--text)]">{runId.slice(0, 12)}</span>
          <span className="ml-2">· {formatNum(totalSuspects)} suspects total (non-suspects auto-accept)</span>
        </div>
        <div className="ml-auto flex items-center gap-3 text-[11px] text-[var(--text-muted)]">
          <span>
            Decided: <span className="font-mono text-[var(--text)]">{formatNum(Object.keys(decisions).length)}</span>
          </span>
          <span className="flex items-center gap-1">
            {saving ? (
              <><IconLoader2 size={12} className="animate-spin" /> Saving...</>
            ) : lastSavedAt ? (
              <><IconChecks size={12} className="text-[var(--success)]" /> Saved {formatSince(lastSavedAt)}</>
            ) : (
              <span>Not saved yet</span>
            )}
          </span>
        </div>
      </div>

      <ClassTabs
        classTotals={classTotals}
        activeClass={activeClass}
        onSelect={onSelectClass}
        decidedByClass={decidedByClass}
      />

      <div className="flex min-h-0 flex-1">
        {/* Sidebar */}
        <aside className="flex w-[260px] shrink-0 flex-col border-r border-[var(--border-muted)] bg-[var(--surface)]">
          <div className="border-b border-[var(--border-muted)] px-3 py-2 text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
            Clusters ({clusters.length})
          </div>
          <div className="flex-1 overflow-auto p-2">
            {clustersLoading && (
              <div className="flex h-20 items-center justify-center text-[12px] text-[var(--text-muted)]">
                <IconLoader2 size={14} className="mr-1 animate-spin" /> Loading...
              </div>
            )}
            {!clustersLoading && clusters.length === 0 && (
              <div className="px-2 py-4 text-[12px] text-[var(--text-muted)]">
                Không có cluster nào cho class này.
              </div>
            )}
            {clusters.map((c) => (
              <ClusterListRow
                key={c.cluster_id}
                cluster={c}
                isActive={c.cluster_id === selectedClusterId}
                decided={c.cluster_id === selectedClusterId ? groupDecided : 0}
                onClick={() => onSelectCluster(c.cluster_id)}
              />
            ))}
          </div>
        </aside>

        {/* Main canvas */}
        <main className="flex min-w-0 flex-1 flex-col">
          {detail?.cluster ? (
            <>
              <div className="border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2.5">
                <div className="mb-2 text-[13px] font-semibold text-[var(--text)]">
                  {detail.cluster.is_noise_cluster
                    ? 'Noise group'
                    : `Cluster #${detail.cluster.cluster_id}`}
                  <span className="ml-2 text-[12px] text-[var(--text-muted)]">
                    · {formatNum(detail.cluster.size)} suspects
                    · current: <span className={classColor(activeClass)}>{activeClass}</span>
                  </span>
                </div>
                <SuggestionCard
                  cluster={{ ...detail.cluster, current_label: activeClass }}
                  decisionForGroup={decisionForGroup}
                  onApply={(target) => onApplyGroup(target, true)}
                />
              </div>

              <div className="flex-1 overflow-auto px-4 py-3">
                {detail.representatives.length > 0 && (
                  <div className="mb-3">
                    <div className="mb-1.5 text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
                      Representatives ({detail.representatives.length}) — top suspicion
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {detail.representatives.map((box) => (
                        <BoxThumb
                          key={box.result_id}
                          box={box}
                          currentClass={activeClass}
                          decision={decisions[box.result_id]}
                        />
                      ))}
                    </div>
                  </div>
                )}
                <div className="mb-1.5 text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
                  All members ({detail.members.length}{detail.total_members > detail.members.length && ` of ${detail.total_members}`})
                </div>
                <div className="grid grid-cols-[repeat(auto-fill,minmax(110px,1fr))] gap-1.5">
                  {detail.members.map((box) => (
                    <BoxThumb
                      key={box.result_id}
                      box={box}
                      currentClass={activeClass}
                      decision={decisions[box.result_id]}
                    />
                  ))}
                </div>
                {detailLoading && (
                  <div className="mt-3 text-center text-[11px] text-[var(--text-muted)]">
                    <IconLoader2 size={12} className="inline animate-spin" /> Loading thumbnails...
                  </div>
                )}
              </div>

              <div className="border-t border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2 text-[10px] text-[var(--text-muted)]">
                Enter = accept suggestion · K = keep · 1/2/3/4 = mold/spall/crack/reject · S = skip · ←/→ prev/next
              </div>
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center text-[13px] text-[var(--text-muted)]">
              {detailLoading
                ? <><IconLoader2 size={14} className="mr-1 animate-spin" /> Loading cluster...</>
                : 'Chọn một cluster để bắt đầu review.'}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

// ── Top-level container ────────────────────────────────────────────────────
export default function LabelReview() {
  const [paths, setPaths] = useState({
    suspectClusterDbPath: '',
    sourceDbPath: '',
    imageRootPath: '',
    sessionsDir: '',
  });
  const [pathsLoaded, setPathsLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [runs, setRuns] = useState([]);
  const [classTotalsAll, setClassTotalsAll] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState('');

  const [sessions, setSessions] = useState([]);
  const [selectedSessionId, setSelectedSessionId] = useState('');
  const [sessionTitle, setSessionTitle] = useState('');
  const [sessionData, setSessionData] = useState(null);

  const [screen, setScreen] = useState('setup');
  const [activeClass, setActiveClass] = useState('mold');
  const [clusters, setClusters] = useState([]);
  const [clustersLoading, setClustersLoading] = useState(false);
  const [classTotalsForRun, setClassTotalsForRun] = useState([]);

  const [selectedClusterId, setSelectedClusterId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const [saving, setSaving] = useState(false);
  const [lastSavedAt, setLastSavedAt] = useState('');
  const saveTimer = useRef(null);

  const decisions = useMemo(() => sessionData?.decisions || {}, [sessionData]);

  useEffect(() => {
    if (pathsLoaded) return;
    let cancelled = false;
    window.electronAPI.getLabelReviewDefaults()
      .then((d) => {
        if (cancelled) return;
        setPaths({
          suspectClusterDbPath: d.suspectClusterDbPath || '',
          sourceDbPath: d.sourceDbPath || '',
          imageRootPath: d.imageRootPath || '',
          sessionsDir: d.sessionsDir || '',
        });
      })
      .finally(() => { if (!cancelled) setPathsLoaded(true); });
    return () => { cancelled = true; };
  }, [pathsLoaded]);

  const handlePathChange = useCallback((field, value) => {
    setPaths((prev) => ({ ...prev, [field]: value }));
  }, []);

  const loadRuns = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const result = await window.electronAPI.listSuspectRuns({
        suspectClusterDbPath: paths.suspectClusterDbPath,
      });
      setRuns(result.runs || []);
      setClassTotalsAll(result.class_totals || []);
      if ((result.runs || []).length > 0) {
        setSelectedRunId(result.runs[0].run_id);
      }
    } catch (e) {
      setError(String(e?.message || e));
      setRuns([]);
    } finally {
      setLoading(false);
    }
  }, [paths.suspectClusterDbPath]);

  useEffect(() => {
    if (!selectedRunId) {
      setSessions([]);
      setSelectedSessionId('');
      return;
    }
    let cancelled = false;
    window.electronAPI.listLabelReviewSessions({
      sessionsDir: paths.sessionsDir,
      subclusterRunId: selectedRunId,
    })
      .then((res) => { if (!cancelled) setSessions(res.sessions || []); })
      .catch(() => { if (!cancelled) setSessions([]); });
    return () => { cancelled = true; };
  }, [selectedRunId, paths.sessionsDir]);

  const handleCreateSession = useCallback(async () => {
    if (!selectedRunId) return;
    try {
      const data = await window.electronAPI.createLabelReviewSession({
        sessionsDir: paths.sessionsDir,
        subclusterRunId: selectedRunId,
        title: sessionTitle.trim(),
      });
      setSessionTitle('');
      const list = await window.electronAPI.listLabelReviewSessions({
        sessionsDir: paths.sessionsDir,
        subclusterRunId: selectedRunId,
      });
      setSessions(list.sessions || []);
      setSelectedSessionId(data.session_id);
    } catch (e) { setError(String(e?.message || e)); }
  }, [selectedRunId, paths.sessionsDir, sessionTitle]);

  const handleDeleteSession = useCallback(async (sid) => {
    if (!confirm(`Xoá session ${sid}?`)) return;
    try {
      await window.electronAPI.deleteLabelReviewSession({
        sessionsDir: paths.sessionsDir,
        sessionId: sid,
      });
      setSessions((prev) => prev.filter((s) => s.session_id !== sid));
      if (selectedSessionId === sid) setSelectedSessionId('');
    } catch (e) { setError(String(e?.message || e)); }
  }, [paths.sessionsDir, selectedSessionId]);

  const startReview = useCallback(async () => {
    if (!selectedSessionId || !selectedRunId) return;
    setLoading(true);
    try {
      const data = await window.electronAPI.loadLabelReviewSession({
        sessionsDir: paths.sessionsDir,
        sessionId: selectedSessionId,
      });
      setSessionData(data);
      // Pick the first class that has suspects
      const firstClass = classTotalsAll.find((c) => c.total_suspects > 0)?.current_label || 'mold';
      setActiveClass(firstClass);
      setScreen('review');
      setSelectedClusterId(null);
      setDetail(null);
    } catch (e) { setError(String(e?.message || e)); }
    finally { setLoading(false); }
  }, [selectedRunId, selectedSessionId, paths.sessionsDir, classTotalsAll]);

  // Load clusters when class/run changes
  useEffect(() => {
    if (screen !== 'review' || !selectedRunId || !activeClass) return;
    let cancelled = false;
    setClustersLoading(true);
    window.electronAPI.listSuspectClusters({
      suspectClusterDbPath: paths.suspectClusterDbPath,
      runId: selectedRunId,
      currentLabel: activeClass,
    })
      .then((res) => {
        if (cancelled) return;
        setClusters(res.clusters || []);
        setClassTotalsForRun(res.classes || []);
        if ((res.clusters || []).length > 0) {
          setSelectedClusterId(res.clusters[0].cluster_id);
        } else {
          setSelectedClusterId(null);
        }
      })
      .catch((e) => { if (!cancelled) setError(String(e?.message || e)); })
      .finally(() => { if (!cancelled) setClustersLoading(false); });
    return () => { cancelled = true; };
  }, [screen, selectedRunId, activeClass, paths.suspectClusterDbPath]);

  // Load detail when cluster changes
  useEffect(() => {
    if (screen !== 'review' || selectedClusterId == null) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    window.electronAPI.getSuspectClusterMembers({
      suspectClusterDbPath: paths.suspectClusterDbPath,
      sourceDbPath: paths.sourceDbPath,
      imageRootPath: paths.imageRootPath,
      runId: selectedRunId,
      currentLabel: activeClass,
      clusterId: selectedClusterId,
      memberLimit: 120,
    })
      .then((res) => { if (!cancelled) setDetail(res); })
      .catch((e) => { if (!cancelled) setError(String(e?.message || e)); })
      .finally(() => { if (!cancelled) setDetailLoading(false); });
    return () => { cancelled = true; };
  }, [screen, selectedClusterId, activeClass, selectedRunId, paths.suspectClusterDbPath, paths.sourceDbPath, paths.imageRootPath]);

  const scheduleSave = useCallback((nextData) => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(async () => {
      setSaving(true);
      try {
        const result = await window.electronAPI.saveLabelReviewSession({
          sessionsDir: paths.sessionsDir,
          sessionId: nextData.session_id,
          payload: nextData,
        });
        setLastSavedAt(result.last_updated_utc);
      } catch (e) { setError(String(e?.message || e)); }
      finally { setSaving(false); }
    }, 400);
  }, [paths.sessionsDir]);

  const advanceToNext = useCallback(() => {
    const idx = clusters.findIndex((c) => c.cluster_id === selectedClusterId);
    const next = clusters[idx + 1];
    if (next) setSelectedClusterId(next.cluster_id);
  }, [clusters, selectedClusterId]);

  // target: '__keep__' (apply keep to all) or one of mold/spall/crack/reject
  const handleApplyGroup = useCallback((target, advance) => {
    if (!sessionData || !detail?.cluster) return;
    const ids = (detail.members || []).map((m) => Number(m.result_id));
    if (ids.length === 0) return;
    const decision = target === '__keep__'
      ? buildKeepDecision(activeClass)
      : target === activeClass
        ? buildKeepDecision(activeClass)
        : buildChangeDecision(activeClass, target);
    const nextDecisions = { ...sessionData.decisions };
    for (const id of ids) nextDecisions[id] = decision;
    const nextData = { ...sessionData, decisions: nextDecisions };
    setSessionData(nextData);
    scheduleSave(nextData);
    if (advance) setTimeout(advanceToNext, 80);
  }, [sessionData, detail, activeClass, scheduleSave, advanceToNext]);

  const handleSkipGroup = useCallback((advance) => {
    if (!detail?.cluster) return;
    if (advance) advanceToNext();
  }, [detail, advanceToNext]);

  if (screen === 'setup') {
    return (
      <SetupScreen
        paths={paths}
        onPathChange={handlePathChange}
        onLoad={loadRuns}
        loading={loading}
        error={error}
        runs={runs}
        selectedRunId={selectedRunId}
        onSelectRun={setSelectedRunId}
        classTotals={classTotalsAll}
        sessions={sessions}
        selectedSessionId={selectedSessionId}
        onSelectSession={setSelectedSessionId}
        onCreateSession={handleCreateSession}
        onDeleteSession={handleDeleteSession}
        sessionTitle={sessionTitle}
        onSessionTitleChange={setSessionTitle}
        onStart={startReview}
        canStart={Boolean(selectedRunId && selectedSessionId)}
      />
    );
  }

  return (
    <ReviewScreen
      classTotals={classTotalsForRun.length > 0 ? classTotalsForRun : classTotalsAll}
      decisions={decisions}
      runId={selectedRunId}
      activeClass={activeClass}
      onSelectClass={setActiveClass}
      clusters={clusters}
      clustersLoading={clustersLoading}
      selectedClusterId={selectedClusterId}
      onSelectCluster={setSelectedClusterId}
      detail={detail}
      detailLoading={detailLoading}
      saving={saving}
      lastSavedAt={lastSavedAt}
      onApplyGroup={handleApplyGroup}
      onSkipGroup={handleSkipGroup}
      onBack={() => setScreen('setup')}
    />
  );
}
