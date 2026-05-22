import { useCallback } from 'react';
import { IconFolderOpen, IconTrash, IconPlus, IconPhoto, IconUpload } from '@tabler/icons-react';
import { useDispatch, useSelector } from 'react-redux';
import {
  addSourcePaths,
  removeSourcePath,
  setRecursive,
  setSkipCrop,
  goToStep,
} from '../inspectionWizardSlice.js';
import { Button, IconButton } from '../../../components/ui/index.js';
import { useResolvedImageSrc, isTifSrc } from '../../../utils/tiffSrc.js';

function Thumb({ path }) {
  const src = `file://${path}`;
  const resolved = useResolvedImageSrc(src);
  if (!resolved) {
    return (
      <div className="flex h-28 w-full items-center justify-center bg-[var(--surface-3)] text-[10px] text-[var(--text-muted)]">
        {isTifSrc(src) ? 'Decoding TIFF…' : 'Loading…'}
      </div>
    );
  }
  return <img src={resolved} alt="" className="h-28 w-full object-cover" />;
}

export default function Step0Source() {
  const dispatch = useDispatch();
  const { paths, recursive } = useSelector((s) => s.inspectionWizard.source);
  const skipCrop = useSelector((s) => s.inspectionWizard.skipCrop);

  const addFiles = useCallback(async () => {
    const result = await window.electronAPI.browseFiles();
    if (result?.length) dispatch(addSourcePaths(result));
  }, [dispatch]);

  const addFolder = useCallback(async () => {
    const p = await window.electronAPI.browsePath('directory');
    if (!p) return;
    const files = await window.electronAPI.listImageFiles({ rootPath: p, recursive });
    if (files?.length) dispatch(addSourcePaths(files));
  }, [dispatch, recursive]);

  return (
    <div className="mx-auto max-w-[780px] p-8">
      <div className="mb-8">
        <h2 className="text-[17px] font-semibold text-[var(--text)]">Select source images</h2>
        <p className="mt-1.5 text-[13px] text-[var(--text-muted)]">
          Add image files or folders to begin the inspection pipeline.
        </p>
      </div>

      <div className="mb-6 flex flex-wrap items-center gap-3">
        <Button variant="primary" onClick={addFiles}>
          <IconUpload size={15} /> Add files
        </Button>
        <Button onClick={addFolder}>
          <IconFolderOpen size={15} /> Add folder
        </Button>

        <div className="ml-2 h-5 w-px bg-[var(--border-muted)]" />

        <label className="flex items-center gap-2 text-[13px] text-[var(--text)] cursor-pointer select-none">
          <input
            type="checkbox"
            checked={recursive}
            onChange={(e) => dispatch(setRecursive(e.target.checked))}
            className="rounded-[4px] border-[var(--border)] bg-[var(--surface-2)] text-[var(--primary)] focus:ring-[var(--primary)]"
          />
          <span className="text-[var(--text-muted)]">Recursive scan</span>
        </label>
      </div>

      <div className="mb-6 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-4">
        <label className="flex items-center gap-3 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={skipCrop}
            onChange={(e) => dispatch(setSkipCrop(e.target.checked))}
            className="rounded-[4px] border-[var(--border)] bg-[var(--surface-2)] text-[var(--primary)] focus:ring-[var(--primary)]"
          />
          <div>
            <span className="text-[13px] font-medium text-[var(--text)]">Skip crop &mdash; use source images directly</span>
            <p className="text-[11px] text-[var(--text-muted)]">Detection and segmentation will run on the original images without isolating objects first.</p>
          </div>
        </label>
      </div>

      {paths.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-[6px] border border-dashed border-[var(--border)] py-16">
          <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full border border-[var(--border)] bg-[var(--surface)]">
            <IconPhoto size={28} strokeWidth={1.2} className="text-[var(--text-muted)]" />
          </div>
          <p className="mb-1 text-[14px] font-medium text-[var(--text-muted)]">No images selected</p>
          <p className="text-[12px] text-[var(--text-muted)]">Click "Add files" or "Add folder" to get started</p>
        </div>
      ) : (
        <div className="grid grid-cols-4 gap-3">
          {paths.map((p) => (
            <div key={p} className="group relative overflow-hidden rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)]">
              <Thumb path={p} />
              <div className="truncate px-2.5 pb-1.5 pt-1 text-[11px] text-[var(--text-muted)]">
                {p.split('/').pop()}
              </div>
              <button
                type="button"
                onClick={() => dispatch(removeSourcePath(p))}
                className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-[4px] bg-[var(--danger-bg)] text-[var(--danger)] opacity-0 transition-opacity group-hover:opacity-100"
                aria-label="Remove"
              >
                <IconTrash size={12} />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="mt-6 flex items-center justify-between border-t border-[var(--border-muted)] pt-5">
        <p className="text-[12px] text-[var(--text-muted)]">
          {paths.length > 0
            ? `${paths.length} image${paths.length !== 1 ? 's' : ''} selected`
            : ''}
        </p>
        <Button
          variant="primary"
          disabled={paths.length === 0}
          onClick={() => dispatch(goToStep(skipCrop ? 3 : 1))}
          className="px-5"
        >
          {skipCrop ? 'Continue to Detection' : 'Continue to Crop'}
        </Button>
      </div>
    </div>
  );
}
