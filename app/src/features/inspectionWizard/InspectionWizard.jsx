import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { IconX } from '@tabler/icons-react';
import { goToStep, clearError } from './inspectionWizardSlice.js';
import StageSidebar from './components/StageSidebar.jsx';
import Step0Source from './components/Step0Source.jsx';
import Step1CropSetup from './components/Step1CropSetup.jsx';
import Step2CropApply from './components/Step2CropApply.jsx';
import Step3GdinoConfig from './components/Step3GdinoConfig.jsx';
import Step4DetectResults from './components/Step4DetectResults.jsx';
import Step5SamConfig from './components/Step5SamConfig.jsx';
import Step6SamResults from './components/Step6SamResults.jsx';

function stageFromStep(step) {
  if (step <= 0) return 0;
  if (step <= 2) return 1;
  if (step <= 4) return 2;
  return 3;
}

export default function InspectionWizard() {
  const dispatch = useDispatch();
  const { step, skipCrop, error } = useSelector((s) => s.inspectionWizard);

  useEffect(() => {
    if (skipCrop && (step === 1 || step === 2)) {
      dispatch(goToStep(3));
    }
  }, [skipCrop, step, dispatch]);

  const activeStage = stageFromStep(step);

  return (
    <div className="flex h-full flex-col bg-[var(--bg)] rv-font">
      {error !== null && (
        <div className="flex shrink-0 items-center gap-3 border-b border-[var(--danger)] bg-[var(--danger-bg)] px-6 py-2.5">
          <span className="min-w-0 flex-1 text-[13px] text-[var(--danger)]">{error}</span>
          <button
            type="button"
            onClick={() => dispatch(clearError())}
            className="shrink-0 rounded p-0.5 text-[var(--danger)] hover:bg-[var(--danger)] hover:text-white transition-colors"
            aria-label="Dismiss error"
          >
            <IconX size={14} />
          </button>
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        <StageSidebar />

        <div className="min-w-0 flex-1 overflow-auto">
          {/* Stage 0: Source */}
          {activeStage === 0 && <Step0Source />}

          {/* Stage 1: Crop */}
          {activeStage === 1 && (
            <CropStage step={step} />
          )}

          {/* Stage 2: Detect */}
          {activeStage === 2 && (
            <DetectStage step={step} />
          )}

          {/* Stage 3: Segment */}
          {activeStage === 3 && (
            <SegmentStage step={step} />
          )}
        </div>
      </div>
    </div>
  );
}

function CropStage({ step }) {
  const dispatch = useDispatch();

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center border-b border-[var(--border-muted)] bg-[var(--surface)]">
        <button
          type="button"
          onClick={() => dispatch(goToStep(1))}
          className={`px-4 py-2.5 text-[12px] font-medium border-b-2 transition-colors ${
            step === 1
              ? 'border-[var(--primary)] text-[var(--primary)]'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
          }`}
        >
          Setup
        </button>
        <button
          type="button"
          onClick={() => dispatch(goToStep(2))}
          className={`px-4 py-2.5 text-[12px] font-medium border-b-2 transition-colors ${
            step === 2
              ? 'border-[var(--primary)] text-[var(--primary)]'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
          }`}
        >
          Export
        </button>
      </div>
      <div className="min-h-0 flex-1">
        {step === 1 && <Step1CropSetup />}
        {step === 2 && <Step2CropApply />}
      </div>
    </div>
  );
}

function DetectStage({ step }) {
  const dispatch = useDispatch();
  const hasResults = useSelector((s) => {
    const d = s.inspectionWizard.detection;
    return Object.keys(d.boxesByImage).length > 0;
  });

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center border-b border-[var(--border-muted)] bg-[var(--surface)]">
        <button
          type="button"
          onClick={() => dispatch(goToStep(3))}
          className={`px-4 py-2.5 text-[12px] font-medium border-b-2 transition-colors ${
            step === 3
              ? 'border-[var(--primary)] text-[var(--primary)]'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
          }`}
        >
          Config
        </button>
        <button
          type="button"
          onClick={() => { if (hasResults) dispatch(goToStep(4)); }}
          disabled={!hasResults}
          className={`px-4 py-2.5 text-[12px] font-medium border-b-2 transition-colors ${
            step === 4
              ? 'border-[var(--primary)] text-[var(--primary)]'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
          } ${!hasResults ? 'cursor-not-allowed opacity-40' : ''}`}
        >
          Results
        </button>
      </div>
      <div className="min-h-0 flex-1">
        {step === 3 && <Step3GdinoConfig />}
        {step === 4 && <Step4DetectResults />}
      </div>
    </div>
  );
}

function SegmentStage({ step }) {
  const dispatch = useDispatch();
  const hasResults = useSelector((s) => {
    const seg = s.inspectionWizard.segmentation;
    return Object.keys(seg.masksByImage).length > 0;
  });

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center border-b border-[var(--border-muted)] bg-[var(--surface)]">
        <button
          type="button"
          onClick={() => dispatch(goToStep(5))}
          className={`px-4 py-2.5 text-[12px] font-medium border-b-2 transition-colors ${
            step === 5
              ? 'border-[var(--primary)] text-[var(--primary)]'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
          }`}
        >
          Config
        </button>
        <button
          type="button"
          onClick={() => { if (hasResults) dispatch(goToStep(6)); }}
          disabled={!hasResults}
          className={`px-4 py-2.5 text-[12px] font-medium border-b-2 transition-colors ${
            step === 6
              ? 'border-[var(--primary)] text-[var(--primary)]'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
          } ${!hasResults ? 'cursor-not-allowed opacity-40' : ''}`}
        >
          Results
        </button>
      </div>
      <div className="min-h-0 flex-1">
        {step === 5 && <Step5SamConfig />}
        {step === 6 && <Step6SamResults />}
      </div>
    </div>
  );
}
