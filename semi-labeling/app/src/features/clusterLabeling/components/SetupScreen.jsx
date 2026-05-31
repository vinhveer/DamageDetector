import { useState } from 'react';
import {
  IconArrowLeft,
  IconArrowRight,
  IconCheck,
  IconCircleCheck,
  IconClock,
  IconDatabase,
  IconPlus,
  IconRefresh,
  IconTrash,
} from '@tabler/icons-react';

const STEPS = [
  { id: 1, label: 'Kết nối', subtitle: 'Database paths' },
  { id: 2, label: 'Cluster run', subtitle: 'Chọn run cần label' },
  { id: 3, label: 'Session', subtitle: 'Tạo hoặc chọn session' },
];

const formatSince = (iso) => {
  if (!iso) return '';
  const ts = Date.parse(iso);
  if (!ts) return '';
  const secs = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
};

function Stepper({ currentStep, runsLoaded, runPicked, sessionPicked }) {
  const completed = [false, runsLoaded, runPicked, sessionPicked];
  return (
    <div className="flex items-center justify-center gap-1.5">
      {STEPS.map((step, idx) => {
        const isActive = step.id === currentStep;
        const isDone = completed[step.id];
        return (
          <div key={step.id} className="flex items-center gap-1.5">
            <div
              className={`flex h-8 items-center gap-2 rounded-full border px-3 text-[12px] font-medium transition-colors ${
                isActive
                  ? 'border-[var(--primary)] bg-[var(--primary)] text-white'
                  : isDone
                    ? 'border-[var(--success)] bg-[var(--surface)] text-[var(--success)]'
                    : 'border-[var(--border)] bg-[var(--surface)] text-[var(--text-muted)]'
              }`}
            >
              <div
                className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold ${
                  isActive
                    ? 'bg-white text-[var(--primary)]'
                    : isDone
                      ? 'bg-[var(--success)] text-white'
                      : 'bg-[var(--surface-2)] text-[var(--text-muted)]'
                }`}
              >
                {isDone ? <IconCheck size={11} /> : step.id}
              </div>
              <span>{step.label}</span>
            </div>
            {idx < STEPS.length - 1 && (
              <div
                className={`h-px w-6 ${
                  completed[step.id + 1] || step.id < currentStep ? 'bg-[var(--success)]' : 'bg-[var(--border)]'
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

function PathField({ label, value, placeholder, onChange }) {
  return (
    <label className="grid gap-1.5">
      <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
        {label}
      </span>
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

function StepConnect({ paths, onPathChange, onLoad, loading, error, runs }) {
  const hasPaths = paths?.clusterDbPath && paths?.sourceDbPath;
  return (
    <div className="grid gap-5">
      <div>
        <h2 className="text-[18px] font-semibold text-[var(--text)]">Kết nối databases</h2>
        <p className="mt-1 text-[12px] text-[var(--text-muted)]">
          Step 5 cần đọc <code className="rounded bg-[var(--surface-2)] px-1 py-0.5 text-[11px]">clusters.sqlite3</code> và join với{' '}
          <code className="rounded bg-[var(--surface-2)] px-1 py-0.5 text-[11px]">damage_scan.sqlite3</code> để lấy bbox coords.
        </p>
      </div>
      <div className="grid gap-3 rounded-[10px] border border-[var(--border)] bg-[var(--surface)] p-5">
        <PathField
          label="clusters.sqlite3 (Step 5 output)"
          value={paths.clusterDbPath}
          placeholder="/path/to/clusters.sqlite3"
          onChange={(v) => onPathChange('clusterDbPath', v)}
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
          placeholder="/path/to/HinhAnh"
          onChange={(v) => onPathChange('imageRootPath', v)}
        />
        <div className="mt-1 flex items-center gap-3">
          <button
            type="button"
            onClick={onLoad}
            disabled={!hasPaths || loading}
            className="inline-flex h-9 items-center gap-2 rounded-[6px] bg-[var(--primary)] px-4 text-[12px] font-semibold text-white shadow-sm disabled:opacity-40"
          >
            <IconRefresh size={14} className={loading ? 'animate-spin' : ''} />
            {loading ? 'Loading…' : runs.length ? 'Reload runs' : 'Load runs'}
          </button>
          {runs.length > 0 && (
            <span className="inline-flex items-center gap-1.5 text-[12px] text-[var(--success)]">
              <IconCircleCheck size={14} />
              Tìm thấy {runs.length} run{runs.length === 1 ? '' : 's'}
            </span>
          )}
          {error && <span className="text-[12px] text-[var(--danger)]">{error}</span>}
        </div>
      </div>
    </div>
  );
}

function StepPickRun({ runs, selectedRunId, onSelectRun }) {
  return (
    <div className="grid gap-5">
      <div>
        <h2 className="text-[18px] font-semibold text-[var(--text)]">Chọn cluster run</h2>
        <p className="mt-1 text-[12px] text-[var(--text-muted)]">
          Mỗi run là 1 lần chạy Step 5 với threshold/PCA khác nhau.
        </p>
      </div>
      <div className="grid gap-2">
        {runs.length === 0 ? (
          <div className="rounded-[10px] border border-dashed border-[var(--border)] bg-[var(--surface)] p-8 text-center text-[12px] text-[var(--text-muted)]">
            Chưa load runs. Quay về Bước 1.
          </div>
        ) : (
          runs.map((run) => {
            const active = run.cluster_run_id === selectedRunId;
            return (
              <button
                key={run.cluster_run_id}
                type="button"
                onClick={() => onSelectRun(run.cluster_run_id)}
                className={`flex items-center gap-3 rounded-[10px] border px-4 py-3.5 text-left transition-colors ${
                  active
                    ? 'border-[var(--primary)] bg-[var(--active)] ring-1 ring-[var(--primary)]'
                    : 'border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--hover)]'
                }`}
              >
                <div
                  className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border ${
                    active ? 'border-[var(--primary)] bg-[var(--primary)]' : 'border-[var(--border)]'
                  }`}
                >
                  {active && <div className="h-2 w-2 rounded-full bg-white" />}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <IconDatabase size={13} className="text-[var(--text-muted)]" />
                    <span className="font-mono text-[12px] font-medium text-[var(--text)]">
                      {run.cluster_run_id.slice(0, 12)}…
                    </span>
                    <span className="rounded-full bg-[var(--surface-2)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-muted)]">
                      {run.algorithm}
                    </span>
                  </div>
                  <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                    {run.total_boxes.toLocaleString()} boxes ·
                    {' '}{run.total_clusters} clusters ·
                    {' '}PCA {run.pca_dim} ({Number(run.pca_explained_ratio).toFixed(2)} variance)
                  </div>
                </div>
                <span className="text-[11px] text-[var(--text-muted)]">
                  {formatSince(run.created_at_utc)}
                </span>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}

function NewSessionForm({ defaultTitle, onCancel, onCreate }) {
  const [title, setTitle] = useState(defaultTitle);
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onCreate(title.trim() || defaultTitle);
      }}
      className="grid gap-3 rounded-[10px] border border-[var(--primary)] bg-[var(--surface)] p-4"
    >
      <label className="grid gap-1.5">
        <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
          Tên session
        </span>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          autoFocus
          placeholder={defaultTitle}
          className="h-9 rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] px-3 text-[13px] text-[var(--text)] focus:border-[var(--primary)] focus:outline-none"
        />
      </label>
      <div className="flex items-center gap-2">
        <button
          type="submit"
          className="inline-flex h-8 items-center gap-1.5 rounded-[6px] bg-[var(--primary)] px-3 text-[12px] font-semibold text-white"
        >
          <IconCheck size={13} /> Tạo session
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="h-8 rounded-[6px] border border-[var(--border)] px-3 text-[12px] font-medium text-[var(--text)] hover:bg-[var(--hover)]"
        >
          Hủy
        </button>
      </div>
    </form>
  );
}

function StepSession({
  sessions,
  selectedSessionId,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  defaultSessionTitle,
}) {
  const [creating, setCreating] = useState(false);

  return (
    <div className="grid gap-5">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-[18px] font-semibold text-[var(--text)]">Chọn labeling session</h2>
          <p className="mt-1 text-[12px] text-[var(--text-muted)]">
            Session là 1 lần label. Bạn có thể tạo nhiều session cho cùng 1 run.
          </p>
        </div>
        {!creating && (
          <button
            type="button"
            onClick={() => setCreating(true)}
            className="inline-flex h-9 items-center gap-1.5 rounded-[6px] border border-[var(--primary)] bg-[var(--surface)] px-3 text-[12px] font-semibold text-[var(--primary)] hover:bg-[var(--primary)] hover:text-white"
          >
            <IconPlus size={13} /> New session
          </button>
        )}
      </div>

      {creating && (
        <NewSessionForm
          defaultTitle={defaultSessionTitle}
          onCancel={() => setCreating(false)}
          onCreate={async (title) => {
            await onCreateSession(title);
            setCreating(false);
          }}
        />
      )}

      <div className="grid gap-2">
        {sessions.length === 0 && !creating ? (
          <div className="rounded-[10px] border border-dashed border-[var(--border)] bg-[var(--surface)] p-8 text-center text-[12px] text-[var(--text-muted)]">
            Chưa có session nào. Bấm <strong>"New session"</strong> ở trên để tạo.
          </div>
        ) : (
          sessions.map((session) => {
            const active = session.session_id === selectedSessionId;
            const reviewed = Number(session.stats?.reviewed || 0);
            return (
              <div
                key={session.session_id}
                className={`flex items-center gap-3 rounded-[10px] border px-4 py-3 transition-colors ${
                  active
                    ? 'border-[var(--primary)] bg-[var(--active)] ring-1 ring-[var(--primary)]'
                    : 'border-[var(--border)] bg-[var(--surface)]'
                }`}
              >
                <button
                  type="button"
                  onClick={() => onSelectSession(session.session_id)}
                  className="flex flex-1 items-center gap-3 text-left"
                >
                  <div
                    className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border ${
                      active ? 'border-[var(--primary)] bg-[var(--primary)]' : 'border-[var(--border)]'
                    }`}
                  >
                    {active && <div className="h-2 w-2 rounded-full bg-white" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="truncate text-[13px] font-medium text-[var(--text)]">
                      {session.title || session.session_id}
                    </div>
                    <div className="mt-0.5 flex items-center gap-3 text-[11px] text-[var(--text-muted)]">
                      <span className="inline-flex items-center gap-1">
                        <IconCircleCheck size={11} /> {reviewed} reviewed
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <IconClock size={11} /> {formatSince(session.last_updated_utc)}
                      </span>
                    </div>
                  </div>
                </button>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); onDeleteSession(session.session_id); }}
                  className="text-[var(--text-muted)] hover:text-[var(--danger)]"
                  title="Delete session"
                >
                  <IconTrash size={14} />
                </button>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export default function SetupScreen({
  paths,
  onPathChange,
  onLoad,
  loading,
  error,
  runs,
  selectedRunId,
  onSelectRun,
  sessions,
  selectedSessionId,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  onStartLabeling,
}) {
  const [currentStep, setCurrentStep] = useState(1);
  const runsLoaded = runs.length > 0;
  const runPicked = Boolean(selectedRunId);
  const sessionPicked = Boolean(selectedSessionId);

  const canNext = currentStep === 1
    ? runsLoaded
    : currentStep === 2
      ? runPicked
      : sessionPicked;

  const handleNext = () => {
    if (currentStep < 3) {
      setCurrentStep(currentStep + 1);
    } else if (sessionPicked) {
      onStartLabeling();
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-[760px] flex-col gap-8 px-6 py-10">
      <div className="text-center">
        <h1 className="text-[22px] font-semibold text-[var(--text)]">Cluster Labeling</h1>
        <p className="mt-1 text-[12px] text-[var(--text-muted)]">
          Review từng cluster của Step 5, gán nhãn và lưu vào JSON session.
        </p>
      </div>

      <Stepper
        currentStep={currentStep}
        runsLoaded={runsLoaded}
        runPicked={runPicked}
        sessionPicked={sessionPicked}
      />

      <div className="min-h-[360px]">
        {currentStep === 1 && (
          <StepConnect
            paths={paths}
            onPathChange={onPathChange}
            onLoad={onLoad}
            loading={loading}
            error={error}
            runs={runs}
          />
        )}
        {currentStep === 2 && (
          <StepPickRun
            runs={runs}
            selectedRunId={selectedRunId}
            onSelectRun={onSelectRun}
          />
        )}
        {currentStep === 3 && (
          <StepSession
            sessions={sessions}
            selectedSessionId={selectedSessionId}
            onSelectSession={onSelectSession}
            onCreateSession={onCreateSession}
            onDeleteSession={onDeleteSession}
            defaultSessionTitle={`Pass ${sessions.length + 1}`}
          />
        )}
      </div>

      {/* Navigation footer */}
      <div className="flex items-center justify-between border-t border-[var(--border-muted)] pt-5">
        <button
          type="button"
          disabled={currentStep === 1}
          onClick={() => setCurrentStep(currentStep - 1)}
          className="inline-flex h-10 items-center gap-2 rounded-[8px] border border-[var(--border)] bg-[var(--surface)] px-4 text-[12px] font-medium text-[var(--text)] hover:bg-[var(--hover)] disabled:opacity-30"
        >
          <IconArrowLeft size={14} />
          Quay lại
        </button>
        <div className="text-[11px] text-[var(--text-muted)]">
          Bước {currentStep} / 3
        </div>
        <button
          type="button"
          disabled={!canNext}
          onClick={handleNext}
          className="inline-flex h-10 items-center gap-2 rounded-[8px] bg-[var(--primary)] px-5 text-[12px] font-semibold text-white shadow-sm disabled:opacity-40"
        >
          {currentStep === 3 ? 'Start labeling' : 'Tiếp tục'}
          <IconArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}
