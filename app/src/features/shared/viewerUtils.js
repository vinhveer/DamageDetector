export const LABELS = ['crack', 'mold', 'spall'];

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
