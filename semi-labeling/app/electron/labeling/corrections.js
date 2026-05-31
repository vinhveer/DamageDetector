// Pure helpers for review sessions / corrections — NO node:sqlite import here,
// so this module is testable on system Node v20 via `node --test`.

import { exportLabel } from '../../src/features/labeling/exportLabel.js';

// Build a "<name>_<YYYYMMDD_HHMMSS>" id from an optional English name.
// Falls back to `fallbackPrefix` when the cleaned name is empty.
export const makeStampedId = (name, fallbackPrefix) => {
  const d = new Date();
  const p2 = (n) => String(n).padStart(2, '0');
  const stamp = `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}_${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}`;
  const clean = String(name || '').trim().replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
  const prefix = clean || fallbackPrefix;
  return `${prefix}_${stamp}`;
};

const VALID_ACTIONS = new Set(['manual_relabel', 'manual_reject', 'manual_accept']);

// Resolve the final (new_label, new_decision_type) for one correction edit.
export const resolveDecision = (action, newLabel) => {
  if (action === 'manual_reject') return { newLabel: 'reject', newDecisionType: 'reject' };
  return { newLabel: String(newLabel || '').trim(), newDecisionType: 'manual_accept' };
};

// Turn correction edits into review_decisions rows (plain objects). Each row is
// a single result target carrying the human_correction reason code. Invalid
// actions or non-finite result ids are skipped.
// edits: [{ resultId, action, previousLabel, newLabel }]
export const buildReviewDecisions = ({ edits } = {}) => {
  const list = Array.isArray(edits) ? edits : [];
  const rows = [];
  for (const edit of list) {
    const resultId = Number(edit?.resultId);
    if (!Number.isFinite(resultId)) continue;
    const action = String(edit?.action || '').trim();
    if (!VALID_ACTIONS.has(action)) continue;
    const { newLabel, newDecisionType } = resolveDecision(action, edit?.newLabel);
    rows.push({
      targetType: 'result',
      targetId: String(resultId),
      resultId,
      action,
      previousLabel: String(edit?.previousLabel || ''),
      newLabel,
      newDecisionType,
      exportLabel: exportLabel(newLabel),
      reasonCodesJson: JSON.stringify(['human_correction']),
      affectedResultIdsJson: JSON.stringify([resultId]),
      note: String(edit?.note || ''),
    });
  }
  return rows;
};
