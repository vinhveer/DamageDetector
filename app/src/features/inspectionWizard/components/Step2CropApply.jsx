import { useState, useMemo, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { IconFolderOpen, IconArrowRight, IconRefresh, IconCircleCheck, IconAlertCircle } from '@tabler/icons-react';
import {
  setCropPadding,
  setTransparentBg,
  setCropOutputDir,
  setCroppedPaths,
  goToStep,
} from '../inspectionWizardSlice.js';
import { Button, IconButton } from '../../../components/ui/index.js';
import { loadImageAsync, useResolvedImageSrc } from '../../../utils/tiffSrc.js';

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

async function createCroppedPngBase64({ srcPath, rawMaskB64, bbox, padding, transparentBg }) {
  const sourceImg = await loadImageAsync(`file://${srcPath}`);
  const sw = sourceImg.naturalWidth || sourceImg.width;
  const sh = sourceImg.naturalHeight || sourceImg.height;
  const box = Array.isArray(bbox) && bbox.length === 4 ? bbox.map(Number) : [0, 0, sw, sh];
  const x1 = Math.floor(clamp(box[0] - padding, 0, sw - 1));
  const y1 = Math.floor(clamp(box[1] - padding, 0, sh - 1));
  const x2 = Math.ceil(clamp(box[2] + padding, x1 + 1, sw));
  const y2 = Math.ceil(clamp(box[3] + padding, y1 + 1, sh));
  const w = Math.max(1, x2 - x1);
  const h = Math.max(1, y2 - y1);

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(sourceImg, x1, y1, w, h, 0, 0, w, h);

  if (transparentBg && rawMaskB64) {
    const maskImg = await loadImageAsync(`data:image/png;base64,${rawMaskB64}`);
    ctx.globalCompositeOperation = 'destination-in';
    ctx.drawImage(maskImg, x1, y1, w, h, 0, 0, w, h);
    ctx.globalCompositeOperation = 'source-over';
  }

  return canvas.toDataURL('image/png').split(',')[1] || '';
}

function SourceThumb({ path }) {
  const src = `file://${path}`;
  const resolved = useResolvedImageSrc(src);
  if (!resolved) {
    return <div className="flex h-24 w-full items-center justify-center bg-[var(--surface-3)] text-[10px] text-[var(--text-muted)]">Loading…</div>;
  }
  return <img src={resolved} alt="" className="h-24 w-full object-cover" />;
}

function CropThumb({ imgData }) {
  const b64 = imgData.cropPreviewB64 || imgData.maskB64;
  if (!b64) {
    return <div className="flex h-24 w-full items-center justify-center bg-[var(--surface-3)] text-[10px] text-[var(--text-muted)]">No preview</div>;
  }
  return <img src={`data:image/png;base64,${b64}`} alt="" className="h-24 w-full object-cover" />;
}

export default function Step2CropApply() {
  const dispatch = useDispatch();
  const crop = useSelector((s) => s.inspectionWizard.crop);
  const paths = useSelector((s) => s.inspectionWizard.source.paths);
  const { perImage, cropPadding, transparentBg, cropOutputDir } = crop;

  const [cropping, setCropping] = useState(false);
  const [progress, setProgress] = useState(0);
  const [total, setTotal] = useState(0);
  const [failedFiles, setFailedFiles] = useState([]);

  const doneImages = useMemo(
    () => paths.filter((p) => perImage[p]?.status === 'done'),
    [paths, perImage],
  );

  const defaultOutputDir = useMemo(() => {
    if (paths.length === 0) return '';
    const firstPath = paths[0];
    const dir = firstPath.substring(0, firstPath.lastIndexOf('/'));
    return `${dir}/_cropped/`;
  }, [paths]);

  const effectiveOutputDir = cropOutputDir || defaultOutputDir;

  const browseOutputDir = useCallback(async () => {
    const p = await window.electronAPI.browsePath('directory');
    if (p) dispatch(setCropOutputDir(p.endsWith('/') ? p : `${p}/`));
  }, [dispatch]);

  const handleCropAll = useCallback(async () => {
    setCropping(true);
    setProgress(0);
    setTotal(doneImages.length);
    setFailedFiles([]);

    const outPaths = [];
    const failures = [];

    for (let i = 0; i < doneImages.length; i++) {
      const path = doneImages[i];
      const imgData = perImage[path];
      try {
        const pngB64 = await createCroppedPngBase64({
          srcPath: path,
          rawMaskB64: imgData.rawMaskB64,
          bbox: imgData.bbox,
          padding: cropPadding,
          transparentBg,
        });
        const result = await window.electronAPI.saveCroppedImage({
          srcPath: path,
          pngBase64: pngB64,
          outputDir: effectiveOutputDir,
        });
        if (result.error) {
          failures.push({ path, error: result.error });
        } else {
          outPaths.push(result.outPath);
        }
      } catch (err) {
        failures.push({ path, error: err?.message || 'Unknown error' });
      }
      setProgress(i + 1);
    }

    setCropping(false);
    setFailedFiles(failures);

    if (failures.length === 0) {
      dispatch(setCroppedPaths(outPaths));
      dispatch(goToStep(3));
    } else if (outPaths.length > 0) {
      dispatch(setCroppedPaths(outPaths));
    }
  }, [doneImages, perImage, effectiveOutputDir, cropPadding, transparentBg, dispatch]);

  const handleRetry = useCallback(async (failedPath) => {
    const imgData = perImage[failedPath];
    try {
      const pngB64 = await createCroppedPngBase64({
        srcPath: failedPath,
        rawMaskB64: imgData?.rawMaskB64,
        bbox: imgData?.bbox,
        padding: cropPadding,
        transparentBg,
      });
      const result = await window.electronAPI.saveCroppedImage({
        srcPath: failedPath,
        pngBase64: pngB64,
        outputDir: effectiveOutputDir,
      });
      if (result.error) {
        setFailedFiles((prev) => prev.map((f) => (f.path === failedPath ? { ...f, error: result.error } : f)));
      } else {
        setFailedFiles((prev) => prev.filter((f) => f.path !== failedPath));
        dispatch(setCroppedPaths([...crop.croppedPaths, result.outPath]));
      }
    } catch (err) {
      setFailedFiles((prev) =>
        prev.map((f) => (f.path === failedPath ? { ...f, error: err?.message || 'Unknown error' } : f)),
      );
    }
  }, [perImage, effectiveOutputDir, crop.croppedPaths, cropPadding, transparentBg, dispatch]);

  return (
    <div className="mx-auto max-w-[900px] p-8">
      <h2 className="text-[17px] font-semibold text-[var(--text)]">Crop &amp; Export</h2>
      <p className="mt-1.5 text-[13px] text-[var(--text-muted)]">
        Review crop previews, adjust settings, then batch export all cropped images.
      </p>

      {doneImages.length === 0 ? (
        <div className="mt-8 mb-8 flex flex-col items-center justify-center rounded-[6px] border border-dashed border-[var(--border)] py-16">
          <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full border border-[var(--border)] bg-[var(--surface)]">
            <IconAlertCircle size={24} strokeWidth={1.2} className="text-[var(--text-muted)]" />
          </div>
          <p className="text-[13px] font-medium text-[var(--text-muted)]">No images with completed segmentation</p>
          <p className="mt-1 text-[12px] text-[var(--text-muted)]">Go back to Setup and run SAM on your images first.</p>
        </div>
      ) : (
        <div className="my-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {doneImages.map((p) => (
            <div key={p} className="overflow-hidden rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)]">
              <div className="flex">
                <div className="relative w-1/2">
                  <SourceThumb path={p} />
                  <span className="absolute bottom-1 left-1 rounded-[3px] bg-[var(--bg)]/80 px-1.5 py-0.5 text-[9px] font-medium text-[var(--text-muted)]">Original</span>
                </div>
                <div className="relative w-1/2">
                  <CropThumb imgData={perImage[p]} />
                  <span className="absolute bottom-1 right-1 rounded-[3px] bg-[var(--bg)]/80 px-1.5 py-0.5 text-[9px] font-medium text-[var(--primary)]">Crop</span>
                </div>
              </div>
              <div className="truncate px-2.5 pb-1.5 pt-1 text-[10px] text-[var(--text-muted)]">
                {p.split('/').pop()}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mb-6 space-y-4 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-4">
        <div className="flex items-center gap-4">
          <span className="w-[120px] shrink-0 text-[12px] text-[var(--text-muted)]">Crop padding</span>
          <input
            type="range" min={0} max={50} value={cropPadding}
            onChange={(e) => dispatch(setCropPadding(Number(e.target.value)))}
            className="flex-1"
          />
          <span className="w-10 text-right text-[12px] tabular-nums text-[var(--text)]">{cropPadding}px</span>
        </div>

        <label className="flex items-center gap-3 text-[12px] text-[var(--text)] cursor-pointer select-none">
          <input
            type="checkbox"
            checked={transparentBg}
            onChange={(e) => dispatch(setTransparentBg(e.target.checked))}
            className="rounded-[4px] border-[var(--border)] bg-[var(--surface-2)] text-[var(--primary)]"
          />
          Transparent background
        </label>

        <div className="flex items-center gap-3">
          <span className="w-[120px] shrink-0 text-[12px] text-[var(--text-muted)]">Output directory</span>
          <input
            type="text"
            value={effectiveOutputDir}
            onChange={(e) => dispatch(setCropOutputDir(e.target.value))}
            className="flex-1 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-1.5 text-[12px] text-[var(--text)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
            placeholder="Output directory for cropped images"
          />
          <IconButton label="Browse" onClick={browseOutputDir}>
            <IconFolderOpen size={16} />
          </IconButton>
        </div>
      </div>

      {cropping && (
        <div className="mb-4">
          <div className="mb-1.5 flex items-center justify-between text-[12px] text-[var(--text-muted)]">
            <span>Exporting…</span>
            <span className="tabular-nums">{progress}/{total}</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--surface-3)]">
            <div
              className="h-full rounded-full bg-[var(--primary)] transition-all duration-300"
              style={{ width: `${total > 0 ? (progress / total) * 100 : 0}%` }}
            />
          </div>
        </div>
      )}

      {failedFiles.length > 0 && (
        <div className="mb-4 rounded-[6px] border border-[var(--danger)] bg-[var(--danger-bg)] p-3">
          <p className="mb-2 text-[12px] font-medium text-[var(--danger)]">
            {failedFiles.length} file{failedFiles.length !== 1 ? 's' : ''} failed
          </p>
          {failedFiles.map((f) => (
            <div key={f.path} className="flex items-center gap-2 rounded-[4px] bg-[var(--surface)] px-2.5 py-1.5 mb-1.5 last:mb-0">
              <span className="min-w-0 flex-1 truncate text-[11px] text-[var(--text)]">
                {f.path.split('/').pop()}
                <span className="ml-2 text-[var(--text-muted)]">— {f.error}</span>
              </span>
              <button
                type="button"
                onClick={() => handleRetry(f.path)}
                className="flex shrink-0 items-center gap-1 rounded-[4px] border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1 text-[11px] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]"
              >
                <IconRefresh size={12} /> Retry
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex justify-end border-t border-[var(--border-muted)] pt-5">
        <Button
          variant="primary"
          onClick={handleCropAll}
          disabled={cropping || doneImages.length === 0}
          className="flex items-center gap-2 px-5"
        >
          {cropping ? (
            <>
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
              Exporting…
            </>
          ) : (
            <>
              <IconCircleCheck size={15} /> Export all &amp; continue <IconArrowRight size={14} />
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
