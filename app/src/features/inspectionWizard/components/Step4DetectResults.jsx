import { useDispatch, useSelector } from 'react-redux';
import { useState } from 'react';
import { IconArrowLeft, IconRefresh, IconPlayerPlay, IconEye, IconEyeOff } from '@tabler/icons-react';
import { goToStep, runDetection, setSelectedImage } from '../inspectionWizardSlice.js';
import BoxOverlay from '../../../components/overlays/BoxOverlay.jsx';
import { Button } from '../../../components/ui/index.js';

export default function Step4DetectResults() {
  const dispatch = useDispatch();
  const { detection, selectedImagePath } = useSelector(s => s.inspectionWizard);
  const [showSuspect, setShowSuspect] = useState(false);
  const running = detection.status === 'running';

  const images = Object.keys(detection.boxesByImage);
  const currentPath = selectedImagePath || images[0] || null;
  const boxes = currentPath ? (detection.boxesByImage[currentPath] || []) : [];
  const suspect = currentPath ? (detection.suspectByImage?.[currentPath] || []) : [];

  const rerun = () => { dispatch(runDetection()); };

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center justify-between border-b border-[var(--border-muted)] bg-[var(--surface)] px-6 py-3">
        <Button variant="secondary" onClick={() => dispatch(goToStep(3))}>
          <IconArrowLeft size={14} /> Back to config
        </Button>
        <div className="flex items-center gap-2">
          <Button variant="secondary" disabled={running} onClick={rerun}>
            <IconRefresh size={14} /> Re-run
          </Button>
          <Button variant="primary" disabled={detection.status !== 'done' || images.length === 0}
            onClick={() => dispatch(goToStep(5))}>
            <IconPlayerPlay size={14} /> Continue to SAM
          </Button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1">
        <aside className="flex w-[260px] shrink-0 flex-col border-r border-[var(--border-muted)] bg-[var(--surface)]">
          <div className="flex items-center justify-between border-b border-[var(--border-muted)] px-3 py-2">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Log</span>
            {running && (
              <span className="flex items-center gap-1.5 text-[11px] text-[var(--primary)]">
                <span className="h-2.5 w-2.5 animate-spin rounded-full border-2 border-[var(--primary)] border-t-transparent" />
                Running
              </span>
            )}
          </div>
          <pre className="rv-mono min-h-0 flex-1 overflow-auto p-3 text-[11px] leading-relaxed whitespace-pre-wrap break-words text-[var(--text)]">
            {detection.logs.join('\n') || 'Waiting for process…'}
          </pre>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex shrink-0 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2">
            <select
              value={currentPath || ''}
              onChange={e => dispatch(setSelectedImage(e.target.value))}
              className="flat-input max-w-[360px]"
              disabled={images.length === 0}
            >
              {images.map(p => <option key={p} value={p}>{p.split('/').pop()}</option>)}
              {images.length === 0 && <option>—</option>}
            </select>

            <div className="h-5 w-px bg-[var(--border-muted)]" />

            <label className="flex items-center gap-2 text-[12px] text-[var(--text)] cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showSuspect}
                onChange={e => setShowSuspect(e.target.checked)}
                className="rounded-[4px] border-[var(--border)] bg-[var(--surface-2)] text-[var(--primary)]"
              />
              Show suspect
            </label>

            <span className="ml-auto text-[11px] text-[var(--text-muted)]">
              {boxes.length} detection{boxes.length !== 1 ? 's' : ''}
              {suspect.length > 0 ? ` · ${suspect.length} suspect` : ''}
            </span>
          </div>

          <div className="relative min-h-0 flex-1 bg-[var(--surface-2)]">
            {currentPath
              ? <BoxOverlay imageSrc={`file://${currentPath}`} boxes={boxes} suspectBoxes={suspect} showSuspect={showSuspect} />
              : <div className="flex h-full items-center justify-center text-[13px] text-[var(--text-muted)]">No image selected</div>
            }

            {running && (
              <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg)]/50">
                <div className="flex flex-col items-center gap-3 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] px-6 py-4">
                  <span className="h-7 w-7 animate-spin rounded-full border-2 border-[var(--primary)] border-t-transparent" />
                  <span className="text-[12px] text-[var(--text-muted)]">Detecting…</span>
                </div>
              </div>
            )}
          </div>

          {boxes.length > 0 && (
            <div className="max-h-[180px] shrink-0 overflow-auto border-t border-[var(--border-muted)]">
              <table className="w-full text-[12px]">
                <thead className="sticky top-0 z-10 bg-[var(--surface)] text-[var(--text-muted)]">
                  <tr>
                    {['Label','Score','Sem. label','Sem. prob','x1','y1','x2','y2'].map(h => (
                      <th key={h} className="px-3 py-1.5 text-left font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {boxes.map((b, i) => (
                    <tr key={i} className="border-t border-[var(--border-muted)] hover:bg-[var(--hover)] transition-colors">
                      <td className="px-3 py-1 font-medium text-[var(--text)]">{b.label}</td>
                      <td className="px-3 py-1">{(b.score * 100).toFixed(1)}%</td>
                      <td className="px-3 py-1">{b.semantic_label || '—'}</td>
                      <td className="px-3 py-1">{b.semantic_prob != null ? `${(b.semantic_prob * 100).toFixed(1)}%` : '—'}</td>
                      <td className="px-3 py-1 text-[var(--text-muted)]">{Math.round(b.x1)}</td>
                      <td className="px-3 py-1 text-[var(--text-muted)]">{Math.round(b.y1)}</td>
                      <td className="px-3 py-1 text-[var(--text-muted)]">{Math.round(b.x2)}</td>
                      <td className="px-3 py-1 text-[var(--text-muted)]">{Math.round(b.y2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
