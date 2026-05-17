export const LABELS = ['crack', 'mold', 'spall'];

export const VIEW_MODES = [
  { label: 'All clusters', value: 'all' },
  { label: 'Non-outlier', value: 'non_outlier' },
  { label: 'Outliers only', value: 'outlier' },
  { label: 'Label suspect', value: 'label_suspect' }
];

export const DETAIL_VIEWS = ['grid', 'table'];

export const LABEL_BADGE_CLASSES = {
  crack: 'border-[var(--warning)] bg-[var(--warning-bg)] text-[var(--warning)]',
  mold: 'border-[var(--success)] bg-[var(--success-bg)] text-[var(--success)]',
  spall: 'border-[var(--primary)] bg-[var(--primary-bg)] text-[var(--primary)]'
};

export const initialPaths = { featureDbPath: '', sourceDbPath: '', imageRootPath: '' };

const numberFormat = new Intl.NumberFormat('en-US');

export const shortId = (value) => String(value || '').slice(0, 8);

export const formatNumber = (value) => numberFormat.format(Number(value || 0));

export const formatFloat = (value, digits = 3) => {
  const next = Number(value);
  return Number.isFinite(next) ? next.toFixed(digits) : '-';
};

export const groupRowsByImage = (rows) => {
  const groups = new Map();
  rows.forEach((row) => {
    const key = row.image_rel_path || row.image_path || String(row.result_id);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  });
  return Array.from(groups.values());
};