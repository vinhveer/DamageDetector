import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { cleanPath, connectRo, connectRw, expandHome, resolveDbPath } from './db.js';
import { decodeVec, selectDiverseSample } from './sampling.js';
import { buildReviewDecisions, makeStampedId } from './corrections.js';
import { exportLabel } from '../../src/features/labeling/exportLabel.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// electron/labeling -> electron -> app -> semi-labeling -> DamageDetector -> Lab
const labRoot = path.resolve(__dirname, '..', '..', '..', '..', '..');

// resemi fresh-run output (matches shared/runtime/paths.py default_resemi_db)
const defaultResemiDb = () => path.join(labRoot, 'model_with_inference', 'semi_labeling', 'resemi.sqlite3');
const defaultImageRoot = () => path.join(labRoot, 'data', 'HinhAnh');

export const labelingDefaults = () => ({
  resemiDbPath: defaultResemiDb(),
  imageRootPath: defaultImageRoot(),
  labels: ['crack', 'mold', 'spall', 'stain', 'reject'],
});

// ── runs that have a review_queue ───────────────────────────────────────────
export const listRuns = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const db = connectRo(dbPath);
  try {
    const rows = db.prepare(`
      SELECT r.run_id, r.created_at_utc,
             (SELECT COUNT(*) FROM review_queue q WHERE q.run_id = r.run_id) AS queue_count,
             (SELECT COUNT(*) FROM cleaned_labels c WHERE c.run_id = r.run_id) AS cleaned_count
      FROM resemi_runs r
      ORDER BY r.created_at_utc DESC, r.run_id DESC
    `).all();
    return { resemiDbPath: resolveDbPath(dbPath), runs: rows };
  } finally {
    db.close();
  }
};

const resolveImagePath = (imageRoot, relPath) => {
  const root = String(imageRoot || '').trim();
  const rel = String(relPath || '').trim();
  if (!root || !rel) return '';
  const full = path.resolve(expandHome(root), rel);
  return fs.existsSync(full) ? full : '';
};

const fileUri = (p) => (p ? pathToFileURL(p).toString() : '');

// ── one review_queue item per row, joined to box coords + image ─────────────
export const listQueue = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const imageRoot = String(payload.imageRootPath || '').trim();
  const queueType = String(payload.queueType || '').trim(); // '' = all
  // sampleRatio in (0,1) -> pick a diverse subset (~ratio per class). >=1 or 0 = all.
  const sampleRatioRaw = Number(payload.sampleRatio);
  const sampleRatio = Number.isFinite(sampleRatioRaw) ? sampleRatioRaw : 0;

  const db = connectRo(dbPath);
  try {
    // latest classifier + self-training run for this run_id (for R1 enrichment)
    const latestClf = db.prepare(`
      SELECT classifier_run_id FROM classifier_runs
      WHERE run_id = ? ORDER BY created_at_utc DESC, classifier_run_id DESC LIMIT 1
    `).get(runId);
    const latestClfId = latestClf ? latestClf.classifier_run_id : '';
    const latestSt = db.prepare(`
      SELECT self_training_run_id FROM self_training_runs
      WHERE run_id = ? ORDER BY created_at_utc DESC, self_training_run_id DESC LIMIT 1
    `).get(runId);
    const latestStId = latestSt ? latestSt.self_training_run_id : '';

    const typeClause = queueType && queueType !== 'all' ? 'AND q.queue_type = ?' : '';
    // params order: rd subquery runId, latestClfId, latestStId, runId, [queueType]
    const params = [runId, latestClfId, latestStId, runId];
    if (queueType && queueType !== 'all') params.push(queueType);
    const rows = db.prepare(`
      SELECT q.result_id, q.image_rel_path, q.crop_path, q.initial_label, q.suggested_label,
             q.queue_type, q.reliability_score, q.reason_codes_json,
             cv.x1, cv.y1, cv.x2, cv.y2,
             rd.action AS decided_action, rd.new_label AS decided_label,
             cps.predicted_label, cps.predicted_probability, cps.margin,
             cps.second_label, cps.second_probability,
             cps.disagrees_with_policy, cps.policy_label,
             stp.reason_codes_json AS defer_reason_codes_json
      FROM review_queue q
      LEFT JOIN crop_views cv ON cv.run_id = q.run_id AND cv.result_id = q.result_id AND cv.view_name = 'tight'
      LEFT JOIN review_decisions rd ON rd.result_id = q.result_id
        AND rd.review_session_id IN (SELECT review_session_id FROM review_sessions WHERE run_id = ?)
      LEFT JOIN classifier_prediction_summary cps
        ON cps.classifier_run_id = ? AND cps.result_id = q.result_id
      LEFT JOIN self_training_promotions stp
        ON stp.self_training_run_id = ? AND stp.result_id = q.result_id AND stp.action = 'defer_review'
      WHERE q.run_id = ? ${typeClause}
      ORDER BY q.reliability_score ASC, q.result_id ASC
    `).all(...params);

    // ── optional diverse sampling using cached DINOv2 tight embeddings ───────
    let selectedIds = null; // null = keep all
    if (sampleRatio > 0 && sampleRatio < 1 && rows.length > 0) {
      const embRunRow = db.prepare(`
        SELECT embedding_run_id FROM embedding_runs
        WHERE run_id = ? AND view_name = 'tight'
        ORDER BY created_at_utc DESC, rowid DESC LIMIT 1
      `).get(runId);
      const embRunId = embRunRow ? embRunRow.embedding_run_id : '';

      const vecById = new Map();
      if (embRunId) {
        const embRows = db.prepare(`
          SELECT result_id, embedding_blob FROM crop_embeddings
          WHERE embedding_run_id = ? AND view_name = 'tight'
        `).all(embRunId);
        for (const er of embRows) vecById.set(Number(er.result_id), decodeVec(er.embedding_blob));
      }

      const sampleRows = rows.map((row) => ({
        resultId: Number(row.result_id),
        label: String(row.suggested_label || row.initial_label || 'unknown'),
        reliability: Number(row.reliability_score || 0),
        vec: vecById.get(Number(row.result_id)) || null,
      }));
      selectedIds = selectDiverseSample(sampleRows, sampleRatio);
    }

    const allItems = rows.map((row) => {
      const imgPath = resolveImagePath(imageRoot, row.image_rel_path);
      let reasons;
      try { reasons = JSON.parse(String(row.reason_codes_json || '[]')); } catch { reasons = []; }
      if (!Array.isArray(reasons)) reasons = [];
      let deferReasons;
      try { deferReasons = JSON.parse(String(row.defer_reason_codes_json || '[]')); } catch { deferReasons = []; }
      if (!Array.isArray(deferReasons)) deferReasons = [];
      return {
        resultId: Number(row.result_id),
        imageRelPath: String(row.image_rel_path || ''),
        imageUri: fileUri(imgPath),
        cropUri: row.crop_path && fs.existsSync(row.crop_path) ? fileUri(row.crop_path) : '',
        initialLabel: String(row.initial_label || ''),
        suggestedLabel: String(row.suggested_label || ''),
        queueType: String(row.queue_type || ''),
        reliabilityScore: Number(row.reliability_score || 0),
        reasons,
        box: row.x1 == null ? null : {
          x1: Number(row.x1), y1: Number(row.y1), x2: Number(row.x2), y2: Number(row.y2),
        },
        decidedAction: row.decided_action ? String(row.decided_action) : '',
        decidedLabel: row.decided_label ? String(row.decided_label) : '',
        prediction: row.predicted_label == null ? null : {
          predictedLabel: String(row.predicted_label),
          predictedProbability: Number(row.predicted_probability || 0),
          margin: Number(row.margin || 0),
          secondLabel: row.second_label ? String(row.second_label) : '',
          secondProbability: row.second_probability == null ? null : Number(row.second_probability),
          disagreesWithPolicy: Number(row.disagrees_with_policy) === 1,
          policyLabel: row.policy_label ? String(row.policy_label) : '',
        },
        deferReasons,
      };
    });

    const items = selectedIds ? allItems.filter((it) => selectedIds.has(it.resultId)) : allItems;

    const counts = {};
    for (const it of items) counts[it.queueType] = (counts[it.queueType] || 0) + 1;
    return { items, counts, total: items.length, queueTotal: allItems.length, sampled: Boolean(selectedIds) };
  } finally {
    db.close();
  }
};

const utcNow = () => new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');

// ── commit a labeling session into review_sessions + review_decisions ───────
// decisions: [{ resultId, action: 'manual_accept'|'manual_relabel'|'manual_reject',
//               previousLabel, newLabel, note }]
export const commitSession = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const reviewer = String(payload.reviewer || '').trim();
  const decisions = Array.isArray(payload.decisions) ? payload.decisions : [];
  if (decisions.length === 0) return { error: 'No decisions to commit.' };

  // Optional custom English name -> "<name>_<YYYYMMDD_HHMMSS>" for easy listing.
  const sessionId = makeStampedId(payload.sessionName, 'review');
  const now = utcNow();
  const db = connectRw(dbPath);
  try {
    db.exec('BEGIN');
    db.prepare(`
      INSERT INTO review_sessions (review_session_id, run_id, reviewer, status, created_at_utc, committed_at_utc, notes)
      VALUES (?, ?, ?, 'committed', ?, ?, ?)
    `).run(sessionId, runId, reviewer, now, now, String(payload.notes || ''));

    const insert = db.prepare(`
      INSERT INTO review_decisions (
        review_session_id, target_type, target_id, result_id, action,
        previous_label, new_label, new_decision_type, reason_codes_json,
        affected_result_ids_json, note, created_at_utc
      ) VALUES (?, 'result', ?, ?, ?, ?, ?, ?, '[]', ?, ?, ?)
    `);
    let count = 0;
    for (const d of decisions) {
      const resultId = Number(d.resultId);
      if (!Number.isFinite(resultId)) continue;
      const action = String(d.action || '').trim();
      if (!['manual_accept', 'manual_relabel', 'manual_reject'].includes(action)) continue;
      const newLabel = action === 'manual_reject' ? 'reject' : String(d.newLabel || '').trim();
      const newDecisionType = action === 'manual_reject' ? 'reject' : 'manual_accept';
      insert.run(
        sessionId,
        String(resultId),
        resultId,
        action,
        String(d.previousLabel || ''),
        newLabel,
        newDecisionType,
        JSON.stringify([resultId]),
        String(d.note || ''),
        now,
      );
      count += 1;
    }
    db.exec('COMMIT');
    return { committed: true, reviewSessionId: sessionId, decisionCount: count, committedAtUtc: now };
  } catch (err) {
    try { db.exec('ROLLBACK'); } catch { /* ignore */ }
    return { error: err?.message || 'Commit failed' };
  } finally {
    db.close();
  }
};

// ── resources for a run (used to auto-fill "run next steps" flags) ──────────
export const getRunResources = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const db = connectRo(dbPath);
  const one = (sql) => {
    try { const r = db.prepare(sql).get(runId); return r || null; } catch { return null; }
  };
  try {
    const emb = one(`SELECT embedding_run_id FROM embedding_runs WHERE run_id = ? AND view_name='tight' ORDER BY created_at_utc DESC LIMIT 1`);
    const proto = one(`SELECT prototype_version_id FROM prototype_versions WHERE run_id = ? ORDER BY created_at_utc DESC LIMIT 1`);
    const core = one(`SELECT core_mining_run_id FROM core_mining_runs WHERE run_id = ? ORDER BY created_at_utc DESC LIMIT 1`);
    const clf = one(`SELECT classifier_run_id, created_at_utc FROM classifier_runs WHERE run_id = ? ORDER BY created_at_utc DESC LIMIT 1`);
    const counts = {
      reviewQueue: db.prepare(`SELECT COUNT(*) n FROM review_queue WHERE run_id = ?`).get(runId).n,
      cleaned: db.prepare(`SELECT COUNT(*) n FROM cleaned_labels WHERE run_id = ?`).get(runId).n,
      reviewDecisions: db.prepare(`
        SELECT COUNT(*) n FROM review_decisions d
        JOIN review_sessions s ON s.review_session_id = d.review_session_id
        WHERE s.run_id = ?`).get(runId).n,
    };
    return {
      embeddingRunId: emb ? emb.embedding_run_id : '',
      prototypeVersionId: proto ? proto.prototype_version_id : '',
      coreMiningRunId: core ? core.core_mining_run_id : '',
      classifierRunId: clf ? clf.classifier_run_id : '',
      counts,
    };
  } finally {
    db.close();
  }
};

// ── management: list human review sessions for a run ────────────────────────
export const listSessions = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const db = connectRo(dbPath);
  try {
    const rows = db.prepare(`
      SELECT s.review_session_id, s.reviewer, s.status, s.created_at_utc, s.committed_at_utc, s.notes,
             (SELECT COUNT(*) FROM review_decisions d WHERE d.review_session_id = s.review_session_id) AS decision_count
      FROM review_sessions s
      WHERE s.run_id = ?
      ORDER BY s.created_at_utc DESC
    `).all(runId);
    return { sessions: rows };
  } finally {
    db.close();
  }
};

// ── management: list self-training (filter) runs for a run ──────────────────
export const listSelfTrainingRuns = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const db = connectRo(dbPath);
  try {
    const rows = db.prepare(`
      SELECT self_training_run_id, classifier_run_id, created_at_utc, round_index,
             candidate_count, promoted_count, rejected_count, deferred_count
      FROM self_training_runs
      WHERE run_id = ?
      ORDER BY created_at_utc DESC
    `).all(runId);
    return { runs: rows };
  } finally {
    db.close();
  }
};

// ── R2: list cleaned_labels (machine-promoted final labels) ─────────────────
export const listCleaned = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const imageRoot = String(payload.imageRootPath || '').trim();
  const finalLabel = String(payload.finalLabel || '').trim();
  const decisionType = String(payload.decisionType || '').trim();

  const db = connectRo(dbPath);
  try {
    const clauses = ['run_id = ?'];
    const params = [runId];
    if (finalLabel && finalLabel !== 'all') { clauses.push('final_label = ?'); params.push(finalLabel); }
    if (decisionType && decisionType !== 'all') { clauses.push('decision_type = ?'); params.push(decisionType); }
    const rows = db.prepare(`
      SELECT result_id, image_rel_path, crop_path, final_label, export_label,
             decision_type, reliability_score, reason_codes_json,
             x1, y1, x2, y2, self_training_run_id, decision_policy_run_id
      FROM cleaned_labels
      WHERE ${clauses.join(' AND ')}
      ORDER BY result_id ASC
    `).all(...params);

    const items = rows.map((row) => {
      const imgPath = resolveImagePath(imageRoot, row.image_rel_path);
      let reasons;
      try { reasons = JSON.parse(String(row.reason_codes_json || '[]')); } catch { reasons = []; }
      if (!Array.isArray(reasons)) reasons = [];
      return {
        resultId: Number(row.result_id),
        imageRelPath: String(row.image_rel_path || ''),
        imageUri: fileUri(imgPath),
        cropUri: row.crop_path && fs.existsSync(row.crop_path) ? fileUri(row.crop_path) : '',
        finalLabel: String(row.final_label || ''),
        exportLabel: String(row.export_label || ''),
        decisionType: String(row.decision_type || ''),
        reliabilityScore: Number(row.reliability_score || 0),
        reasons,
        box: row.x1 == null ? null : {
          x1: Number(row.x1), y1: Number(row.y1), x2: Number(row.x2), y2: Number(row.y2),
        },
        selfTrainingRunId: row.self_training_run_id ? String(row.self_training_run_id) : '',
        decisionPolicyRunId: row.decision_policy_run_id ? String(row.decision_policy_run_id) : '',
      };
    });

    // total = all cleaned rows for the run (ignoring filters)
    const total = db.prepare('SELECT COUNT(*) n FROM cleaned_labels WHERE run_id = ?').get(runId).n;
    return { items, total, filtered: items.length };
  } finally {
    db.close();
  }
};

// ── R2: update one cleaned_labels row's final_label + export_label ──────────
export const updateCleanedLabel = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const resultId = Number(payload.resultId);
  const newLabel = String(payload.newLabel || '').trim();
  if (!Number.isFinite(resultId)) return { error: 'Invalid resultId' };
  if (!newLabel) return { error: 'Invalid newLabel' };
  const finalLabel = newLabel.toLowerCase();
  const exp = exportLabel(finalLabel);

  const db = connectRw(dbPath);
  try {
    const info = db.prepare(`
      UPDATE cleaned_labels SET final_label = ?, export_label = ?
      WHERE run_id = ? AND result_id = ?
    `).run(finalLabel, exp, runId, resultId);
    if (!info.changes) return { updated: false, error: 'Row not found' };
    return { updated: true, finalLabel, exportLabel: exp };
  } catch (err) {
    return { error: err?.message || 'Update failed' };
  } finally {
    db.close();
  }
};

// ── R3: commit corrections as a new review_session + review_decisions ───────
// payload: { resemiDbPath, runId, sessionName?, reviewer?, notes?,
//   edits: [{ resultId, action, previousLabel, newLabel }] }
export const commitCorrections = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const reviewer = String(payload.reviewer || '').trim();
  const edits = Array.isArray(payload.edits) ? payload.edits : [];
  const rows = buildReviewDecisions({ edits });
  if (rows.length === 0) return { error: 'Chưa có chỉnh sửa nào' };

  const sessionId = makeStampedId(payload.sessionName, 'corrections');
  const now = utcNow();
  const db = connectRw(dbPath);
  try {
    db.exec('BEGIN');
    db.prepare(`
      INSERT INTO review_sessions (review_session_id, run_id, reviewer, status, created_at_utc, committed_at_utc, notes)
      VALUES (?, ?, ?, 'committed', ?, ?, ?)
    `).run(sessionId, runId, reviewer, now, now, String(payload.notes || ''));

    const insert = db.prepare(`
      INSERT INTO review_decisions (
        review_session_id, target_type, target_id, result_id, action,
        previous_label, new_label, new_decision_type, reason_codes_json,
        affected_result_ids_json, note, created_at_utc
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    for (const r of rows) {
      insert.run(
        sessionId, r.targetType, r.targetId, r.resultId, r.action,
        r.previousLabel, r.newLabel, r.newDecisionType, r.reasonCodesJson,
        r.affectedResultIdsJson, r.note, now,
      );
    }
    db.exec('COMMIT');
    return { committed: true, reviewSessionId: sessionId, decisionCount: rows.length, committedAtUtc: now };
  } catch (err) {
    try { db.exec('ROLLBACK'); } catch { /* ignore */ }
    return { error: err?.message || 'Commit failed' };
  } finally {
    db.close();
  }
};

// ── R5: review_decisions of one session (read-only detail) ──────────────────
export const getSessionDecisions = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const sessionId = cleanPath(payload.reviewSessionId, 'reviewSessionId');
  const imageRoot = String(payload.imageRootPath || '').trim();
  const db = connectRo(dbPath);
  try {
    const rows = db.prepare(`
      SELECT d.result_id, d.action, d.previous_label, d.new_label,
             cv.image_rel_path, cv.crop_path, cv.x1, cv.y1, cv.x2, cv.y2
      FROM review_decisions d
      LEFT JOIN crop_views cv ON cv.run_id = ? AND cv.result_id = d.result_id AND cv.view_name = 'tight'
      WHERE d.review_session_id = ?
      ORDER BY d.result_id ASC
    `).all(runId, sessionId);
    const decisions = rows.map((row) => ({
      resultId: Number(row.result_id),
      action: String(row.action || ''),
      previousLabel: row.previous_label ? String(row.previous_label) : '',
      newLabel: row.new_label ? String(row.new_label) : '',
      imageUri: fileUri(resolveImagePath(imageRoot, row.image_rel_path)),
      cropUri: row.crop_path && fs.existsSync(row.crop_path) ? fileUri(row.crop_path) : '',
      box: row.x1 == null ? null : {
        x1: Number(row.x1), y1: Number(row.y1), x2: Number(row.x2), y2: Number(row.y2),
      },
    }));
    return { decisions };
  } finally {
    db.close();
  }
};

// ── R5: self_training_promotions of one round (read-only detail) ────────────
export const getSelfTrainingPromotions = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const stId = cleanPath(payload.selfTrainingRunId, 'selfTrainingRunId');
  const action = String(payload.action || '').trim();
  const imageRoot = String(payload.imageRootPath || '').trim();
  const db = connectRo(dbPath);
  try {
    const actionClause = action && action !== 'all' ? 'AND p.action = ?' : '';
    const params = action && action !== 'all' ? [runId, stId, action] : [runId, stId];
    const rows = db.prepare(`
      SELECT p.result_id, p.action, p.predicted_label, p.classifier_confidence, p.classifier_margin,
             cv.image_rel_path, cv.crop_path, cv.x1, cv.y1, cv.x2, cv.y2
      FROM self_training_promotions p
      LEFT JOIN crop_views cv ON cv.run_id = ? AND cv.result_id = p.result_id AND cv.view_name = 'tight'
      WHERE p.self_training_run_id = ? ${actionClause}
      ORDER BY p.result_id ASC
    `).all(...params);
    const promotions = rows.map((row) => ({
      resultId: Number(row.result_id),
      action: String(row.action || ''),
      predictedLabel: row.predicted_label ? String(row.predicted_label) : '',
      classifierConfidence: Number(row.classifier_confidence || 0),
      classifierMargin: Number(row.classifier_margin || 0),
      imageUri: fileUri(resolveImagePath(imageRoot, row.image_rel_path)),
      cropUri: row.crop_path && fs.existsSync(row.crop_path) ? fileUri(row.crop_path) : '',
      box: row.x1 == null ? null : {
        x1: Number(row.x1), y1: Number(row.y1), x2: Number(row.x2), y2: Number(row.y2),
      },
    }));
    return { promotions };
  } finally {
    db.close();
  }
};

// ── R6: aggregated progress metrics across self-training rounds ─────────────
export const getRunMetrics = (payload = {}) => {
  const dbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const db = connectRo(dbPath);
  try {
    const stRows = db.prepare(`
      SELECT self_training_run_id, classifier_run_id, round_index,
             candidate_count, promoted_count, rejected_count, deferred_count
      FROM self_training_runs
      WHERE run_id = ?
      ORDER BY round_index ASC, created_at_utc ASC
    `).all(runId);

    const accuracyOf = (clfId) => {
      if (!clfId) return null;
      let evalRow;
      try {
        evalRow = db.prepare('SELECT evaluation_json FROM classifier_runs WHERE classifier_run_id = ?').get(clfId);
      } catch { return null; }
      if (!evalRow || !evalRow.evaluation_json) return null;
      let evaluation;
      try { evaluation = JSON.parse(String(evalRow.evaluation_json)); } catch { return null; }
      if (!evaluation || evaluation.cv_status !== 'ok') return null;
      const report = evaluation.classification_report;
      const acc = report && typeof report.accuracy === 'number' ? report.accuracy : null;
      return acc && acc > 0 ? acc : null;
    };

    const oofRatioOf = (clfId) => {
      if (!clfId) return null;
      let row;
      try {
        row = db.prepare(`
          SELECT COUNT(*) n, SUM(is_disagreement) d
          FROM classifier_oof_predictions WHERE classifier_run_id = ?
        `).get(clfId);
      } catch { return null; }
      if (!row || !row.n) return null;
      return Number(row.d || 0) / Number(row.n);
    };

    const rounds = stRows.map((row) => ({
      selfTrainingRunId: String(row.self_training_run_id),
      roundIndex: Number(row.round_index),
      candidateCount: Number(row.candidate_count || 0),
      promotedCount: Number(row.promoted_count || 0),
      rejectedCount: Number(row.rejected_count || 0),
      deferredCount: Number(row.deferred_count || 0),
      classifierRunId: String(row.classifier_run_id || ''),
      classifierAccuracy: accuracyOf(row.classifier_run_id),
      oofDisagreementRatio: oofRatioOf(row.classifier_run_id),
    }));

    const cleanedCount = db.prepare('SELECT COUNT(*) n FROM cleaned_labels WHERE run_id = ?').get(runId).n;
    const reviewQueueCount = db.prepare('SELECT COUNT(*) n FROM review_queue WHERE run_id = ?').get(runId).n;

    return {
      rounds,
      deferredTrend: rounds.map((r) => r.deferredCount),
      cleanedCount,
      reviewQueueCount,
    };
  } finally {
    db.close();
  }
};
