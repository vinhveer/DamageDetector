// SPEC 11 — shared constants for the resemi human review console.

export const LABELS = ['crack', 'mold', 'spall', 'stain', 'reject'];

// Class colors mirror the established map in labelReview/LabelReview.jsx.
export const classBorderHex = (c) => (
  c === 'crack' ? '#fbbf24'
    : c === 'mold' ? '#10b981'
      : c === 'spall' ? '#60a5fa'
        : c === 'stain' ? '#a78bfa'
          : c === 'reject' ? '#f472b6'
            : '#7d8590'
);

// Decision-type → semantic tone (SPEC §7.2). Color is never the only channel — pair with text.
export const decisionTone = (decisionType) => {
  const t = String(decisionType || '');
  if (t.startsWith('auto_accept')) return t.includes('low_priority') ? 'gray' : 'green';
  if (t === 'suspect' || t.startsWith('suspect')) return 'amber';
  if (t === 'relabel_candidate') return 'blue';
  if (t === 'reject') return 'red';
  if (t.startsWith('manual_')) return 'blue';
  return 'gray';
};

// Draft action → tone for session badges.
export const actionTone = (action) => {
  switch (action) {
    case 'accept':
    case 'accept_cluster':
    case 'valid_rare_case':
      return 'green';
    case 'relabel':
    case 'relabel_cluster':
    case 'add_prototype':
      return 'blue';
    case 'reject':
    case 'reject_cluster':
    case 'add_reject_prototype':
      return 'red';
    case 'defer':
    case 'mark_ambiguous':
    case 'mixed_cluster':
      return 'amber';
    default:
      return 'gray';
  }
};

export const STAGES = [
  { value: 'prototype', label: '1 · Prototype Picks', countKey: 'prototype_candidates' },
  { value: 'core', label: '2 · Core Clusters', countKey: 'core_clusters' },
  { value: 'disagreement', label: '3 · Disagreement', countKey: 'disagreement' },
  { value: 'outlier', label: '4 · Outliers', countKey: 'outliers' },
  { value: 'relabel', label: '5 · Relabel Batches', countKey: 'relabel_batches' },
];

// SPEC §6.2 keyboard map. Surfaced in the bottom bar + tooltips.
export const SHORTCUTS = [
  { key: 'A', label: 'Accept' },
  { key: 'R', label: 'Relabel' },
  { key: 'X', label: 'Reject' },
  { key: 'M', label: 'Ambiguous' },
  { key: 'D', label: 'Defer' },
  { key: 'P', label: 'Add prototype' },
  { key: 'N', label: 'Next' },
  { key: 'B', label: 'Prev' },
  { key: 'U', label: 'Undo' },
  { key: '/', label: 'Search' },
];

export const formatNum = (n) => Number(n || 0).toLocaleString();

export const formatFloat = (value, digits = 3) => {
  const next = Number(value);
  return Number.isFinite(next) ? next.toFixed(digits) : '-';
};

export const formatPct = (value, digits = 0) => {
  const next = Number(value);
  return Number.isFinite(next) ? `${(next * 100).toFixed(digits)}%` : '-';
};
