import { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { IconArrowLeft, IconRefresh, IconFileExport, IconRotateClockwise } from '@tabler/icons-react';
import { goToStep, runSegmentation, setSelectedImage } from '../inspectionWizardSlice.js';
import MaskOverlay from '../../../components/overlays/MaskOverlay.jsx';
import { Button } from '../../../components/ui/index.js';

export default function Step6SamResults() {
  const dispatch = useDispatch();
  const { segmentation, selectedImagePath } = useSelector(s => s.inspectionWizard);
  const [alpha, setAlpha] = useState(0.55);
  const [highlightId, setHighlightId] = useState(null);
  const running = segmentation.status === 'running';

  const images = Object.keys(segmentation.masksByImage);
  const currentPath = selectedImagePath || images[0] || null;
  const masks = currentPath ? (segmentation.masksByImage[currentPath] || []) : [];

  const exportResults = async () => {
    if (!segmentation.outputDir) return;
    await window.electronAPI.browsePath('directory');
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center justify-between border-b border-[var(--border-muted)] bg-[var(--surface)] px-6 py-3">
        <Button variant="secondary" onClick={() => dispatch(goToStep(5))}>
          <IconArrowLeft size={14} /> Back to config
        </Button>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => dispatch(goToStep(0))}>
            <IconRotateClockwise size={14} /> Start over
          </Button>
          <Button variant="secondary" disabled={running} onClick={() => dispatch(runSegmentation())}>
            <IconRefresh size={14} /> Re-run
          </Button>
          <Button variant="primary" disabled={!segmentation.outputDir} onClick={exportResults}>
            <IconFileExport size={14} /> Export
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
            {segmentation.logs.join('\n') || 'Waiting for process…'}
          </pre>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex shrink-0 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2">
            <select
              value={currentPath || ''}
              onChange={e => { dispatch(setSelectedImage(e.target.value)); setHighlightId(null); }}
              className="flat-input max-w-[360px]"
              disabled={images.length === 0}
            >
              {images.map(p => <option key={p} value={p}>{p.split('/').pop()}</option>)}
              {images.length === 0 && <option>—</option>}
            </select>

            <div className="h-5 w-px bg-[var(--border-muted)]" />

            <label className="flex items-center gap-2 text-[12px] text-[var(--text-muted)]">
              Alpha
              <input
                type="range" min={0} max={1} step={0.05} value={alpha}
                onChange={e => setAlpha(parseFloat(e.target.value))}
                className="w-28"
              />
              <span className="w-8 text-[11px] tabular-nums text-[var(--text)]">{(alpha * 100).toFixed(0)}%</span>
            </label>

            <span className="ml-auto text-[11px] text-[var(--text-muted)]">
              {masks.length} mask{masks.length !== 1 ? 's' : ''}
            </span>
          </div>

          <div className="relative min-h-0 flex-1 bg-[var(--surface-2)]">
            {currentPath
              ? <MaskOverlay imageSrc={`file://${currentPath}`} masks={masks} alpha={alpha} highlightId={highlightId} />
              : <div className="flex h-full items-center justify-center text-[13px] text-[var(--text-muted)]">No image selected</div>
            }

            {running && (
              <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg)]/50">
                <div className="flex flex-col items-center gap-3 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] px-6 py-4">
                  <span className="h-7 w-7 animate-spin rounded-full border-2 border-[var(--primary)] border-t-transparent" />
                  <span className="text-[12px] text-[var(--text-muted)]">Segmenting…</span>
                </div>
              </div>
            )}
          </div>

          {masks.length > 0 && (
            <div className="max-h-[180px] shrink-0 overflow-auto border-t border-[var(--border-muted)]">
              <table className="w-full text-[12px]">
                <thead className="sticky top-0 z-10 bg-[var(--surface)] text-[var(--text-muted)]">
                  <tr>
                    {['ID', 'Score', 'Area (px²)', 'Box'].map(h => (
                      <th key={h} className="px-3 py-1.5 text-left font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {masks.map((m, i) => (
                    <tr
                      key={i}
                      onClick={() => setHighlightId(m.detection_id === highlightId ? null : m.detection_id)}
                      className={`cursor-pointer border-t border-[var(--border-muted)] hover:bg-[var(--hover)] transition-colors ${m.detection_id === highlightId ? 'bg-[var(--active)]' : ''}`}
                    >
                      <td className="px-3 py-1 font-medium text-[var(--text)]">{m.detection_id}</td>
                      <td className="px-3 py-1">{m.score != null ? (m.score * 100).toFixed(1) + '%' : '—'}</td>
                      <td className="px-3 py-1">{m.area?.toLocaleString() ?? '—'}</td>
                      <td className="px-3 py-1 text-[var(--text-muted)]">
                        {m.box ? `${Math.round(m.box.x1)},${Math.round(m.box.y1)},${Math.round(m.box.x2)},${Math.round(m.box.y2)}` : '—'}
                      </td>
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
