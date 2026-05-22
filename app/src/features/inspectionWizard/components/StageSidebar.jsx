import { useDispatch, useSelector } from 'react-redux';
import { IconCheck, IconPhoto, IconScissors, IconSearch, IconLayersIntersect } from '@tabler/icons-react';
import { goToStep } from '../inspectionWizardSlice.js';
import { cn } from '../../../components/ui/cn.js';

const STAGES = [
  { id: 0, step: 0, label: 'Source', desc: 'Select images', icon: IconPhoto },
  { id: 1, step: 1, label: 'Crop', desc: 'Isolate objects', icon: IconScissors },
  { id: 2, step: 3, label: 'Detect', desc: 'Find damage', icon: IconSearch },
  { id: 3, step: 5, label: 'Segment', desc: 'Generate masks', icon: IconLayersIntersect },
];

function stageFromStep(step) {
  if (step <= 0) return 0;
  if (step <= 2) return 1;
  if (step <= 4) return 2;
  return 3;
}

export default function StageSidebar() {
  const dispatch = useDispatch();
  const { step, skipCrop, source, crop, detection, segmentation } = useSelector((s) => s.inspectionWizard);

  const activeStage = stageFromStep(step);

  const stageStatus = (stageId) => {
    if (stageId === 0) {
      return source.paths.length > 0 ? 'done' : 'pending';
    }
    if (stageId === 1) {
      if (skipCrop) return 'skipped';
      const done = Object.values(crop.perImage).filter((i) => i.status === 'done' || i.skipped).length;
      if (done > 0 && crop.croppedPaths.length > 0) return 'done';
      if (done > 0) return 'progress';
      return 'pending';
    }
    if (stageId === 2) {
      if (detection.status === 'done') return 'done';
      if (detection.status === 'running') return 'running';
      if (Object.keys(detection.boxesByImage).length > 0) return 'done';
      return 'pending';
    }
    if (stageId === 3) {
      if (segmentation.status === 'done') return 'done';
      if (segmentation.status === 'running') return 'running';
      return 'pending';
    }
    return 'pending';
  };

  return (
    <nav className="flex w-[220px] shrink-0 flex-col border-r border-[var(--border-muted)] bg-[var(--surface)]">
      <div className="border-b border-[var(--border-muted)] px-4 py-3">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Pipeline</span>
      </div>

      <div className="flex-1 overflow-y-auto py-2">
        {STAGES.map((s) => {
          const status = stageStatus(s.id);
          const isActive = s.id === activeStage;
          const isSkipped = s.id === 1 && skipCrop;
          const isCompleted = status === 'done' && !isSkipped;
          const isRunning = status === 'running';
          const Icon = s.icon;

          return (
            <button
              key={s.id}
              type="button"
              onClick={() => {
                if (isSkipped || isRunning) return;
                dispatch(goToStep(s.step));
              }}
              className={cn(
                'mx-2 flex w-[calc(100%-16px)] items-start gap-3 rounded-[6px] px-3 py-2.5 text-left transition-colors',
                isActive && 'bg-[var(--active)]',
                isSkipped && 'cursor-default opacity-40',
                !isActive && !isSkipped && 'hover:bg-[var(--hover)]',
              )}
            >
              <div
                className={cn(
                  'mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full',
                  isActive && 'bg-[var(--primary)] text-white',
                  isCompleted && !isActive && 'bg-[var(--success)] text-white',
                  isRunning && 'bg-[var(--primary-bg)] text-[var(--primary)]',
                  !isActive && !isCompleted && !isRunning && 'border border-[var(--border)] text-[var(--text-muted)]',
                  isSkipped && 'border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)]',
                )}
              >
                {isCompleted ? (
                  <IconCheck size={13} stroke={3} />
                ) : isRunning ? (
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-[var(--primary)] border-t-transparent" />
                ) : (
                  <Icon size={13} />
                )}
              </div>

              <div className="min-w-0 flex-1">
                <div
                  className={cn(
                    'text-[12px] font-medium leading-tight',
                    isActive ? 'text-[var(--text)]' : 'text-[var(--text-muted)]',
                    isSkipped && 'line-through',
                  )}
                >
                  {s.label}
                </div>
                <div className="text-[11px] leading-tight text-[var(--text-muted)]">
                  {isSkipped ? 'skipped' : isRunning ? 'running…' : s.desc}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      <div className="border-t border-[var(--border-muted)] px-4 py-3">
        <div className="text-[11px] text-[var(--text-muted)]">
          {source.paths.length > 0
            ? `${source.paths.length} image${source.paths.length !== 1 ? 's' : ''}`
            : 'No images selected'}
        </div>
      </div>
    </nav>
  );
}
