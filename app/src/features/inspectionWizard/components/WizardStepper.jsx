import { useDispatch, useSelector } from 'react-redux';
import { IconCheck } from '@tabler/icons-react';
import { goToStep } from '../inspectionWizardSlice.js';
import { cn } from '../../../components/ui/cn.js';

const STEPS = [
  { label: 'Source', group: null },
  { label: 'Crop config', group: 'crop' },
  { label: 'Crop apply', group: 'crop' },
  { label: 'DINO config', group: 'detect' },
  { label: 'Detection', group: 'detect' },
  { label: 'SAM config', group: 'detect' },
  { label: 'Results', group: 'detect' },
];

export default function WizardStepper() {
  const dispatch = useDispatch();
  const { step, skipCrop } = useSelector((s) => s.inspectionWizard);

  return (
    <div className="sticky top-0 z-10 flex shrink-0 items-center gap-1 border-b border-[var(--border-muted)] bg-[var(--surface)] px-6 py-3">
      {STEPS.map(({ label, group }, i) => {
        const isCropStep = group === 'crop';
        const isSkipped = skipCrop && isCropStep;
        const isCurrent = step === i;
        const isCompleted = step > i;
        const isDisabled = step < i && !isCompleted;
        const isClickable = isCompleted && !isCurrent && !isSkipped;

        // Group separators
        const showCropSeparator = i === 1; // before first crop step
        const showDetectSeparator = i === 3; // before first detect step

        return (
          <div key={i} className="flex items-center gap-1">
            {/* Group label separators */}
            {showCropSeparator && (
              <span className={cn(
                'mr-1 text-[10px] font-semibold uppercase tracking-wide',
                isSkipped ? 'text-[var(--text-muted)] opacity-50' : 'text-[var(--text-muted)]'
              )}>
                Crop ›
              </span>
            )}
            {showDetectSeparator && (
              <span className="ml-2 mr-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                Detect ›
              </span>
            )}

            {/* Connecting line between steps within same group */}
            {i > 0 && !showCropSeparator && !showDetectSeparator && (
              <span className="mx-0.5 h-px w-3 bg-[var(--border-muted)]" />
            )}

            {/* Step pill */}
            <button
              type="button"
              disabled={!isClickable}
              onClick={() => isClickable && dispatch(goToStep(i))}
              className={cn(
                'relative flex items-center gap-1.5 rounded-[5px] px-2.5 py-1 text-[12px] font-medium transition-colors',
                isCurrent && 'bg-[var(--primary)] text-white',
                isCompleted && !isCurrent && !isSkipped && 'border border-[var(--primary)] text-[var(--primary)] hover:bg-[var(--primary-bg)] cursor-pointer',
                isSkipped && 'opacity-40 cursor-default',
                isDisabled && !isSkipped && 'text-[var(--text-muted)] cursor-default',
              )}
            >
              {isCompleted && !isSkipped && <IconCheck size={12} stroke={2.5} />}
              {label}
            </button>

            {/* Skipped label */}
            {isSkipped && (
              <span className="text-[10px] text-[var(--text-muted)] opacity-60">skipped</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
