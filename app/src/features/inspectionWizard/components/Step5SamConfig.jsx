import { useDispatch, useSelector } from 'react-redux';
import { IconFolderOpen, IconPlayerPlay, IconArrowLeft } from '@tabler/icons-react';
import { setSam, goToStep, runSegmentation } from '../inspectionWizardSlice.js';
import { Button, IconButton } from '../../../components/ui/index.js';

const Row = ({ label, children }) => (
  <div className="flex items-center gap-3">
    <span className="w-48 shrink-0 text-[12px] text-[var(--text-muted)]">{label}</span>
    {children}
  </div>
);

const SectionTitle = ({ children }) => (
  <span className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">{children}</span>
);

export default function Step5SamConfig() {
  const dispatch = useDispatch();
  const { sam, segmentation, detection } = useSelector(s => s.inspectionWizard);
  const running = segmentation.status === 'running';
  const canRun = detection.status === 'done' || Object.keys(detection.boxesByImage).length > 0;

  const browseCkpt = async (field) => {
    const p = await window.electronAPI.browsePath('file');
    if (p) dispatch(setSam({ [field]: p }));
  };

  const handleRunSam = () => {
    dispatch(runSegmentation());
    dispatch(goToStep(6));
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center justify-between border-b border-[var(--border-muted)] bg-[var(--surface)] px-6 py-3">
        <Button variant="secondary" onClick={() => dispatch(goToStep(4))}>
          <IconArrowLeft size={14} /> Back
        </Button>
        <Button variant="primary" disabled={running || !sam.checkpointPath || !canRun}
          onClick={handleRunSam}>
          {running ? (
            <><span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" /> Running…</>
          ) : (
            <><IconPlayerPlay size={14} /> Run SAM</>
          )}
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto max-w-[680px] space-y-10 p-8">
          <div>
            <h2 className="text-[17px] font-semibold text-[var(--text)]">SAM configuration</h2>
            <p className="mt-1.5 text-[13px] text-[var(--text-muted)]">
              Configure the Segment Anything Model for mask generation on detected boxes.
            </p>
          </div>

          <section className="space-y-4">
            <SectionTitle>Backend</SectionTitle>
            <div className="flex gap-6">
              {[
                ['sam', 'SAM (zero-shot)'],
                ['sam_finetune', 'SAM-LoRA finetuned'],
              ].map(([v, label]) => (
                <label key={v} className="flex items-center gap-2 text-[13px] text-[var(--text)] cursor-pointer">
                  <input
                    type="radio"
                    value={v}
                    checked={sam.backend === v}
                    onChange={() => dispatch(setSam({ backend: v }))}
                    className="text-[var(--primary)] focus:ring-[var(--primary)]"
                  />
                  {label}
                </label>
              ))}
            </div>
          </section>

          {sam.backend === 'sam' && (
              <section className="space-y-4">
                <SectionTitle>Model type</SectionTitle>
              <Row label="Type">
                <select value={sam.modelType} onChange={e => dispatch(setSam({ modelType: e.target.value }))}
                  className="flat-input w-32">
                  {['vit_b', 'vit_l', 'vit_h'].map(m => <option key={m}>{m}</option>)}
                </select>
              </Row>
            </section>
          )}

          <section className="space-y-4">
            <SectionTitle>Checkpoints</SectionTitle>
            <Row label="SAM checkpoint">
              <div className="flex gap-1">
                <input value={sam.checkpointPath} onChange={e => dispatch(setSam({ checkpointPath: e.target.value }))}
                  placeholder="Required" className="flat-input w-64" />
                <IconButton label="Browse" onClick={() => browseCkpt('checkpointPath')}>
                  <IconFolderOpen size={14} />
                </IconButton>
              </div>
            </Row>
            {sam.backend === 'sam_finetune' && (
              <Row label="LoRA checkpoint">
                <div className="flex gap-1">
                  <input value={sam.loraCheckpointPath} onChange={e => dispatch(setSam({ loraCheckpointPath: e.target.value }))}
                    placeholder="Required" className="flat-input w-64" />
                  <IconButton label="Browse" onClick={() => browseCkpt('loraCheckpointPath')}>
                    <IconFolderOpen size={14} />
                  </IconButton>
                </div>
              </Row>
            )}
          </section>

          <section className="space-y-4">
            <SectionTitle>Device</SectionTitle>
            <Row label="Device">
              <select value={sam.device} onChange={e => dispatch(setSam({ device: e.target.value }))}
                className="flat-input w-32">
                {['auto', 'cpu', 'cuda', 'mps'].map(d => <option key={d}>{d}</option>)}
              </select>
            </Row>
          </section>

          <section className="space-y-4">
            <SectionTitle>Box tuning</SectionTitle>
            <Row label="Multimask output">
              <input type="checkbox" checked={sam.multimask}
                onChange={e => dispatch(setSam({ multimask: e.target.checked }))}
                className="rounded-[4px] border-[var(--border)] bg-[var(--surface-2)] text-[var(--primary)]" />
            </Row>
            <Row label="Expand box (px)">
              <input type="number" value={sam.expandBoxPx} min={0} className="flat-input w-24"
                onChange={e => dispatch(setSam({ expandBoxPx: parseInt(e.target.value) || 0 }))} />
            </Row>
            <Row label="Min mask area (px²)">
              <input type="number" value={sam.minMaskArea} min={0} className="flat-input w-28"
                onChange={e => dispatch(setSam({ minMaskArea: parseInt(e.target.value) || 0 }))} />
            </Row>
          </section>
        </div>
      </div>
    </div>
  );
}
