import { useDispatch, useSelector } from 'react-redux';
import { IconPlus, IconTrash, IconFolderOpen, IconPlayerPlay, IconArrowLeft } from '@tabler/icons-react';
import { setGdino, setSemantic, setSpatial, setPromptGroups, goToStep, runDetection, selectDetectionInputs } from '../inspectionWizardSlice.js';
import { Button, IconButton } from '../../../components/ui/index.js';

const SectionTitle = ({ children }) => (
  <span className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">{children}</span>
);
const Row = ({ label, children }) => (
  <div className="flex items-center gap-3">
    <span className="w-48 shrink-0 text-[12px] text-[var(--text-muted)]">{label}</span>
    {children}
  </div>
);

const TILE_SCALES = ['small', 'medium', 'large'];
const DEVICES = ['auto', 'cpu', 'cuda', 'mps'];

export default function Step3GdinoConfig() {
  const dispatch = useDispatch();
  const { gdino, semantic, spatial, detection, skipCrop } = useSelector(s => s.inspectionWizard);
  const imagePaths = useSelector(selectDetectionInputs);
  const running = detection.status === 'running';

  const browseCkpt = async () => {
    const p = await window.electronAPI.browsePath('file');
    if (p) dispatch(setGdino({ checkpointPath: p }));
  };

  const addGroup = () => dispatch(setPromptGroups([
    ...gdino.promptGroups,
    { id: Date.now(), name: 'new', prompt: '' },
  ]));
  const updateGroup = (id, field, val) =>
    dispatch(setPromptGroups(gdino.promptGroups.map(g => g.id === id ? { ...g, [field]: val } : g)));
  const removeGroup = (id) => dispatch(setPromptGroups(gdino.promptGroups.filter(g => g.id !== id)));

  const toggleTile = (s) => {
    const cur = gdino.tileScales;
    dispatch(setGdino({ tileScales: cur.includes(s) ? cur.filter(x => x !== s) : [...cur, s] }));
  };

  const handleRunDetection = () => {
    dispatch(runDetection());
    dispatch(goToStep(4));
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center justify-between border-b border-[var(--border-muted)] bg-[var(--surface)] px-6 py-3">
        <Button variant="secondary" onClick={() => dispatch(goToStep(skipCrop ? 0 : 2))}>
          <IconArrowLeft size={14} /> Back
        </Button>
        <div className="flex items-center gap-3">
          <span className="text-[12px] text-[var(--text-muted)]">
            {imagePaths.length} image{imagePaths.length !== 1 ? 's' : ''}
          </span>
          <Button
            variant="primary"
            disabled={running || imagePaths.length === 0}
            onClick={handleRunDetection}
          >
            {running ? (
              <><span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" /> Running…</>
            ) : (
              <><IconPlayerPlay size={14} /> Run detection</>
            )}
          </Button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto max-w-[760px] space-y-10 p-8">
          <div>
            <h2 className="text-[17px] font-semibold text-[var(--text)]">Detection configuration</h2>
            <p className="mt-1.5 text-[13px] text-[var(--text-muted)]">Configure GroundingDINO thresholds, prompts, and filters.</p>
          </div>

          <section className="space-y-4">
            <SectionTitle>Prompt groups</SectionTitle>
            {gdino.promptGroups.map(g => (
              <div key={g.id} className="flex items-start gap-2">
                <input value={g.name} onChange={e => updateGroup(g.id, 'name', e.target.value)}
                  placeholder="name" className="flat-input w-24 shrink-0" />
                <input value={g.prompt} onChange={e => updateGroup(g.id, 'prompt', e.target.value)}
                  placeholder="prompt text (. separated)" className="flat-input min-w-0 flex-1" />
                <IconButton label="Remove" onClick={() => removeGroup(g.id)}
                  className="text-[var(--danger)] hover:bg-[var(--danger-bg)]">
                  <IconTrash size={13} />
                </IconButton>
              </div>
            ))}
            <Button variant="ghost" onClick={addGroup} className="gap-1">
              <IconPlus size={13} /> Add group
            </Button>
          </section>

          <section className="space-y-4">
            <SectionTitle>DINO thresholds</SectionTitle>
            <Row label="Box threshold">
              <input type="number" value={gdino.boxThreshold} step={0.01} min={0} max={1}
                className="flat-input w-28"
                onChange={e => dispatch(setGdino({ boxThreshold: parseFloat(e.target.value) }))} />
            </Row>
            <Row label="Text threshold">
              <input type="number" value={gdino.textThreshold} step={0.01} min={0} max={1}
                className="flat-input w-28"
                onChange={e => dispatch(setGdino({ textThreshold: parseFloat(e.target.value) }))} />
            </Row>
            <Row label="Max detections">
              <input type="number" value={gdino.maxDets} min={1}
                className="flat-input w-28"
                onChange={e => dispatch(setGdino({ maxDets: parseInt(e.target.value) }))} />
            </Row>
            <Row label="Tiled threshold (px)">
              <input type="number" value={gdino.tiledThreshold} min={0}
                className="flat-input w-28"
                onChange={e => dispatch(setGdino({ tiledThreshold: parseInt(e.target.value) }))} />
            </Row>
            <Row label="Tile scales">
              <div className="flex gap-2">
                {TILE_SCALES.map(s => (
                  <button key={s} type="button" onClick={() => toggleTile(s)}
                    className={`rounded-[5px] border px-2.5 py-1 text-[12px] font-medium transition-colors ${gdino.tileScales.includes(s) ? 'border-[var(--primary)] bg-[var(--primary-bg)] text-[var(--primary)]' : 'border-[var(--border)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]'}`}>
                    {s}
                  </button>
                ))}
              </div>
            </Row>
          </section>

          <section className="space-y-4">
            <SectionTitle>Checkpoint &amp; device</SectionTitle>
            <Row label="GDino checkpoint">
              <div className="flex gap-1">
                <input value={gdino.checkpointPath} onChange={e => dispatch(setGdino({ checkpointPath: e.target.value }))}
                  placeholder="(auto)" className="flat-input w-64" />
                <IconButton label="Browse" onClick={browseCkpt}><IconFolderOpen size={14} /></IconButton>
              </div>
            </Row>
            <Row label="Device">
              <select value={gdino.device} onChange={e => dispatch(setGdino({ device: e.target.value }))}
                className="flat-input w-32">
                {DEVICES.map(d => <option key={d}>{d}</option>)}
              </select>
            </Row>
          </section>

          <section className="space-y-4">
            <div className="flex items-center gap-3">
              <SectionTitle>Semantic relabel (OpenCLIP)</SectionTitle>
              <input type="checkbox" checked={semantic.enabled}
                onChange={e => dispatch(setSemantic({ enabled: e.target.checked }))}
                className="rounded-[4px] border-[var(--border)] bg-[var(--surface-2)] text-[var(--primary)]" />
            </div>
            {semantic.enabled && (
              <>
                <Row label="Model">
                  <input type="text" value={semantic.modelName}
                    className="flat-input w-48"
                    onChange={e => dispatch(setSemantic({ modelName: e.target.value }))} />
                </Row>
                <Row label="Pretrained">
                  <input type="text" value={semantic.pretrained}
                    className="flat-input w-56"
                    onChange={e => dispatch(setSemantic({ pretrained: e.target.value }))} />
                </Row>
                <Row label="Batch size">
                  <input type="number" value={semantic.batchSize} min={1}
                    className="flat-input w-24"
                    onChange={e => dispatch(setSemantic({ batchSize: parseInt(e.target.value) }))} />
                </Row>
              </>
            )}
          </section>

          <section className="space-y-4">
            <div className="flex items-center gap-3">
              <SectionTitle>Spatial filter</SectionTitle>
              <input type="checkbox" checked={spatial.enabled}
                onChange={e => dispatch(setSpatial({ enabled: e.target.checked }))}
                className="rounded-[4px] border-[var(--border)] bg-[var(--surface-2)] text-[var(--primary)]" />
            </div>
            {spatial.enabled && (
              <>
                <Row label="IoU threshold">
                  <input type="number" value={spatial.iouThreshold} step={0.05} min={0} max={1}
                    className="flat-input w-28"
                    onChange={e => dispatch(setSpatial({ iouThreshold: parseFloat(e.target.value) }))} />
                </Row>
                <Row label="Containment threshold">
                  <input type="number" value={spatial.containmentThreshold} step={0.05} min={0} max={1}
                    className="flat-input w-28"
                    onChange={e => dispatch(setSpatial({ containmentThreshold: parseFloat(e.target.value) }))} />
                </Row>
              </>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
