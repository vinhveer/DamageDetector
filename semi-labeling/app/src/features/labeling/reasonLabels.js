// Vietnamese labels for known defer/decision reason codes. Shared by the
// Labeling review panel and the Versions detail view. Unknown codes fall back
// to the raw code so the UI never breaks on a new reason string.
export const REASON_LABELS_VI = {
  classifier_confidence_low: 'Độ tin cậy classifier thấp',
  classifier_margin_low: 'Khoảng cách top-1/top-2 thấp',
  missing_core_or_prototype_agreement: 'Thiếu đồng thuận core/prototype',
  geometry_conflict: 'Xung đột hình học',
};

export const reasonLabelVi = (code) => REASON_LABELS_VI[code] || String(code || '');
