import { test } from 'node:test';
import assert from 'node:assert/strict';
import fc from 'fast-check';
import { exportLabel, EXPORT_MAP } from '../src/features/labeling/exportLabel.js';

// Expected default export mapping, mirrored from
// shared/taxonomy/label_taxonomy.py build_label_taxonomy() (stain_export_label='stain').
// The Python parity test (semi-labeling/tests/test_export_label_parity.py)
// guards that this expected table stays in sync with the taxonomy.
const EXPECTED = {
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
const WORKING_LABELS = Object.keys(EXPECTED);

// Feature: semi-labeling-review-loop, Property 2: bản đồ export_label trong JS
// khớp Python — for any working label, exportLabel() returns the same category
// as build_label_taxonomy().export_label().
// Validates: Requirements 2.5, 4.3
test('Property 2: JS exportLabel matches the Python taxonomy mapping', () => {
  fc.assert(
    fc.property(fc.constantFrom(...WORKING_LABELS), (label) => {
      assert.equal(exportLabel(label), EXPECTED[label]);
    }),
    { numRuns: 100 },
  );
});

test('Property 2: EXPORT_MAP equals the expected default mapping', () => {
  assert.deepEqual({ ...EXPORT_MAP }, EXPECTED);
});

test('exportLabel: case-insensitive, trims, unknown -> reject', () => {
  assert.equal(exportLabel('  CRACK '), 'crack');
  assert.equal(exportLabel('Efflorescence'), 'stain');
  assert.equal(exportLabel('totally-unknown-label'), 'reject');
  assert.equal(exportLabel(''), 'reject');
  assert.equal(exportLabel(null), 'reject');
});
