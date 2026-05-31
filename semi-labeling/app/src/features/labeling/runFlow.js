// Pure decision helpers for the Run-steps flow (R7/R8). No DOM, no IPC, no
// node:sqlite — testable on system Node v20.

// Emit a data-changed signal only when BOTH step08 and step09 succeeded.
export const shouldEmitDataChanged = (step08Code, step09Code) =>
  Number(step08Code) === 0 && Number(step09Code) === 0;

// Require confirmation when there are no human labels yet for the run.
export const needsHumanLabelConfirm = (reviewDecisionCount) =>
  Number(reviewDecisionCount) === 0;
