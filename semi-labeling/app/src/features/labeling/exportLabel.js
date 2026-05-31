// Static export_label map mirrored from shared/taxonomy/label_taxonomy.py
// (default taxonomy). Used for the immediate UI update in updateCleanedLabel.
// The Python export tool re-derives the mapping from label_taxonomy at export
// time, so this map only drives instant display; parity is guarded by a test.
export const EXPORT_MAP = {
  crack: 'crack',
  mold: 'mold',
  spall: 'spall',
  stain: 'stain',
  efflorescence: 'stain',
  shadow: 'reject',
  edge: 'reject',
  background: 'reject',
  object: 'reject',
  unknown: 'reject',
  reject: 'reject',
};

// Map any working label to its export label. Unknown labels normalize to reject
// (matches label_taxonomy normalize_label -> unknown -> reject).
export const exportLabel = (label) => {
  const k = String(label || '').trim().toLowerCase();
  return Object.prototype.hasOwnProperty.call(EXPORT_MAP, k) ? EXPORT_MAP[k] : 'reject';
};
