import { IconLoader2, IconPlus, IconTrash } from '@tabler/icons-react';
import { Button } from '../../../components/ui/index.js';
import ReviewBadge from './ReviewBadge.jsx';
import { formatNum } from '../reviewConstants.js';

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

export default function SetupScreen({
  paths, onPathChange, onLoad, loading, error,
  runs, selectedRunId, onSelectRun,
  reviewer, onReviewerChange,
  sessions, onCreateSession, onOpenSession, onDeleteSession,
}) {
  const selectedRun = runs.find((r) => r.run_id === selectedRunId);
  return (
    <div className="mx-auto flex h-full w-full max-w-[960px] flex-col gap-5 overflow-auto p-6">
      <header>
        <h1 className="text-[20px] font-semibold text-[var(--text)]">Review Console</h1>
        <p className="mt-1 text-[12px] text-[var(--text-muted)]">
          Cluster/prototype/disagreement review for the resemi semi-labeling pipeline.
          Every action is audit-trailed and committed back into the resemi SQLite artifact.
        </p>
      </header>

      <section className="grid gap-3 rounded-[10px] border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="text-[13px] font-semibold text-[var(--text)]">1 · Resemi database</div>
        <PathField
          label="resemi.sqlite3"
          value={paths.resemiDbPath}
          placeholder="/path/to/resemi.sqlite3"
          onChange={(v) => onPathChange('resemiDbPath', v)}
        />
        <PathField
          label="Image root (ROI crops)"
          value={paths.imageRootPath}
          placeholder="/path/to/data/HinhAnh"
          onChange={(v) => onPathChange('imageRootPath', v)}
        />
        <div className="flex items-center justify-between pt-1">
          <span className="text-[11px] text-[var(--text-muted)]">{runs.length > 0 ? `${runs.length} runs loaded` : 'No runs loaded'}</span>
          <Button variant="primary" onClick={onLoad} disabled={loading}>
            {loading ? <IconLoader2 size={15} className="animate-spin" /> : 'Load runs'}
          </Button>
        </div>
        {error && <div className="text-[12px] text-[var(--danger)]">{error}</div>}
      </section>

      {runs.length > 0 && (
        <section className="grid gap-3 rounded-[10px] border border-[var(--border)] bg-[var(--surface)] p-5">
          <div className="text-[13px] font-semibold text-[var(--text)]">2 · Run + reviewer</div>
          <div className="grid max-h-[240px] gap-1 overflow-auto">
            {runs.map((r) => (
              <button
                key={r.run_id}
                type="button"
                onClick={() => onSelectRun(r.run_id)}
                className={`flex items-center gap-3 rounded-[6px] border px-3 py-2 text-left text-[12px] ${selectedRunId === r.run_id ? 'border-[var(--primary)] bg-[var(--active)]' : 'border-[var(--border-muted)] hover:bg-[var(--hover)]'}`}
              >
                <span className="font-medium text-[var(--text)]">{r.run_id}</span>
                <span className="text-[var(--text-muted)]">{formatNum(r.total_detections)} det</span>
                <span className="ml-auto flex gap-1.5">
                  <ReviewBadge tone="green">{formatNum(r.cleaned_count)} clean</ReviewBadge>
                  <ReviewBadge tone="amber">{formatNum(r.suspect_count)} suspect</ReviewBadge>
                </span>
              </button>
            ))}
          </div>
          <label className="grid max-w-[280px] gap-1.5">
            <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">Reviewer name</span>
            <input
              type="text"
              value={reviewer}
              onChange={(e) => onReviewerChange(e.target.value)}
              placeholder="your name"
              className="h-9 rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] px-3 text-[12px] text-[var(--text)] focus:border-[var(--primary)] focus:outline-none"
            />
          </label>
        </section>
      )}

      {selectedRun && (
        <section className="grid gap-3 rounded-[10px] border border-[var(--border)] bg-[var(--surface)] p-5">
          <div className="flex items-center justify-between">
            <div className="text-[13px] font-semibold text-[var(--text)]">3 · Review session</div>
            <Button variant="primary" onClick={onCreateSession}>
              <IconPlus size={15} className="mr-1" />New session
            </Button>
          </div>
          {sessions.length === 0 && <div className="text-[12px] text-[var(--text-muted)]">No sessions for this run yet.</div>}
          {sessions.map((s) => (
            <div key={s.session_id} className="flex items-center gap-3 rounded-[6px] border border-[var(--border-muted)] px-3 py-2 text-[12px]">
              <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left" onClick={() => onOpenSession(s.session_id)}>
                <span className="truncate font-medium text-[var(--text)]">{s.title || s.session_id}</span>
                <ReviewBadge tone={s.status === 'committed' ? 'green' : 'gray'}>{s.status}</ReviewBadge>
                <span className="text-[var(--text-muted)]">{formatNum(s.stats?.reviewed)} reviewed</span>
              </button>
              <button type="button" className="text-[var(--text-muted)] hover:text-[var(--danger)]" onClick={() => onDeleteSession(s.session_id)} title="Delete session">
                <IconTrash size={15} />
              </button>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
