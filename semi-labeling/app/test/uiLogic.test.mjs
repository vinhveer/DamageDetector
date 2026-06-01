import { test } from 'node:test';
import assert from 'node:assert/strict';
import { reasonLabelVi } from '../src/features/labeling/reasonLabels.js';
import { shouldEmitDataChanged, needsHumanLabelConfirm } from '../src/features/labeling/runFlow.js';

test('reasonLabelVi: known codes map to Vietnamese, unknown returns raw', () => {
  assert.equal(reasonLabelVi('classifier_confidence_low'), 'Độ tin cậy classifier thấp');
  assert.equal(reasonLabelVi('classifier_margin_low'), 'Khoảng cách top-1/top-2 thấp');
  assert.equal(reasonLabelVi('missing_core_or_prototype_agreement'), 'Thiếu đồng thuận core/prototype');
  assert.equal(reasonLabelVi('geometry_conflict'), 'Xung đột hình học');
  assert.equal(reasonLabelVi('some_new_unknown_code'), 'some_new_unknown_code');
});

test('shouldEmitDataChanged: true only when both step codes are 0', () => {
  assert.equal(shouldEmitDataChanged(0, 0), true);
  assert.equal(shouldEmitDataChanged(0, 1), false);
  assert.equal(shouldEmitDataChanged(1, 0), false);
  assert.equal(shouldEmitDataChanged(2, 3), false);
});

test('needsHumanLabelConfirm: true only when zero review decisions', () => {
  assert.equal(needsHumanLabelConfirm(0), true);
  assert.equal(needsHumanLabelConfirm(1), false);
  assert.equal(needsHumanLabelConfirm(150), false);
});
