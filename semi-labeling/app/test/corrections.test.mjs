import { test } from 'node:test';
import assert from 'node:assert/strict';
import fc from 'fast-check';
import { buildReviewDecisions, makeStampedId } from '../electron/labeling/corrections.js';
import { exportLabel } from '../src/features/labeling/exportLabel.js';

const ACTIONS = ['manual_relabel', 'manual_reject', 'manual_accept'];

// Feature: semi-labeling-review-loop, Property 7: bất biến khi dựng
// review_decisions từ chỉnh sửa — every built row carries target_type='result',
// target_id == String(resultId), affected_result_ids_json == JSON([resultId]),
// reason_codes contains human_correction, and new_label follows the action.
// Validates: Requirements 3.3, 3.4, 3.5, 3.6, 3.7
test('Property 7: buildReviewDecisions invariants hold for any edit list', () => {
  fc.assert(
    fc.property(
      fc.array(
        fc.record({
          resultId: fc.integer({ min: 1, max: 1_000_000 }),
          action: fc.constantFrom(...ACTIONS),
          previousLabel: fc.constantFrom('crack', 'mold', 'spall', 'stain', ''),
          newLabel: fc.constantFrom('crack', 'mold', 'spall', 'stain', 'reject', ''),
        }),
        { maxLength: 30 },
      ),
      (edits) => {
        const rows = buildReviewDecisions({ edits });
        // only valid actions survive -> here all are valid so counts match
        assert.equal(rows.length, edits.length);
        rows.forEach((row, i) => {
          const edit = edits[i];
          assert.equal(row.targetType, 'result');
          assert.equal(row.targetId, String(edit.resultId));
          assert.equal(row.resultId, edit.resultId);
          assert.deepEqual(JSON.parse(row.affectedResultIdsJson), [edit.resultId]);
          assert.ok(JSON.parse(row.reasonCodesJson).includes('human_correction'));
          if (edit.action === 'manual_reject') {
            assert.equal(row.newLabel, 'reject');
            assert.equal(row.newDecisionType, 'reject');
          } else {
            assert.equal(row.newLabel, String(edit.newLabel || '').trim());
            assert.equal(row.newDecisionType, 'manual_accept');
          }
          // export_label is consistent with the resolved new_label
          assert.equal(row.exportLabel, exportLabel(row.newLabel));
        });
      },
    ),
    { numRuns: 100 },
  );
});

test('buildReviewDecisions: skips invalid actions and non-finite ids', () => {
  const rows = buildReviewDecisions({
    edits: [
      { resultId: 1, action: 'manual_accept', newLabel: 'crack' },
      { resultId: 'nope', action: 'manual_accept', newLabel: 'crack' },
      { resultId: 2, action: 'bogus_action', newLabel: 'crack' },
    ],
  });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].resultId, 1);
});

// Feature: semi-labeling-review-loop, Property 8: định dạng review_session_id có
// dấu thời gian — makeStampedId(name, 'corrections') always matches
// ^[A-Za-z0-9_-]+_\d{8}_\d{6}$ and uses 'corrections' when the name is empty.
// Validates: Requirements 3.2
test('Property 8: makeStampedId always produces a stamped id', () => {
  const PATTERN = /^[A-Za-z0-9_-]+_\d{8}_\d{6}$/;
  fc.assert(
    fc.property(fc.string(), (name) => {
      const id = makeStampedId(name, 'corrections');
      assert.match(id, PATTERN);
      const cleaned = String(name || '').trim().replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
      if (cleaned === '') {
        assert.ok(id.startsWith('corrections_'));
      }
    }),
    { numRuns: 100 },
  );
});
