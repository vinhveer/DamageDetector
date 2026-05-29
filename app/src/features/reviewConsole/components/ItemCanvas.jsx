import { useMemo, useState } from 'react';
import ResultImage from '../../shared/ResultImage.jsx';

const VIEW_ORDER = ['tight', 'pad10', 'pad25', 'context', 'openclip_crop'];

// Large crop viewer for item review: zoom 1x/2x/fit + crop-source tabs (SPEC §7.4).
// Falls back to the row's own bbox when multi-crop views are absent.
export default function ItemCanvas({ item, evidence }) {
  const [zoom, setZoom] = useState('fit');
  const [view, setView] = useState(null);

  const views = useMemo(() => {
    const list = evidence?.crop_views || [];
    return [...list].sort((a, b) => VIEW_ORDER.indexOf(a.view_name) - VIEW_ORDER.indexOf(b.view_name));
  }, [evidence]);

  if (!item?.image_uri) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-1 text-[12px] text-[var(--text-muted)]">
        <div>Crop image missing</div>
        <div className="font-mono text-[11px]">{item?.image_rel_path || 'unknown'} · id {item?.result_id}</div>
      </div>
    );
  }

  const activeView = views.find((v) => v.view_name === view) || null;
  const row = activeView
    ? { ...activeView, image_uri: activeView.image_uri }
    : item;
  const baseSize = zoom === 'fit' ? 360 : zoom === '1x' ? 280 : 560;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-[var(--border-muted)] px-3 py-1.5 text-[11px]">
        {views.length > 1 && (
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setView(null)}
              className={`rounded-[4px] px-1.5 py-0.5 ${!view ? 'bg-[var(--active)] text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}`}
            >default</button>
            {views.map((v) => (
              <button
                key={v.view_name}
                type="button"
                onClick={() => setView(v.view_name)}
                className={`rounded-[4px] px-1.5 py-0.5 ${view === v.view_name ? 'bg-[var(--active)] text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}`}
              >{v.view_name}</button>
            ))}
          </div>
        )}
        <div className="ml-auto flex items-center gap-1">
          {['fit', '1x', '2x'].map((z) => (
            <button
              key={z}
              type="button"
              onClick={() => setZoom(z)}
              className={`rounded-[4px] px-1.5 py-0.5 ${zoom === z ? 'bg-[var(--active)] text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}`}
            >{z}</button>
          ))}
        </div>
      </div>
      <div className="flex flex-1 items-center justify-center overflow-auto bg-[var(--surface-2)] p-4">
        <ResultImage row={row} imageSize={baseSize} />
      </div>
    </div>
  );
}
