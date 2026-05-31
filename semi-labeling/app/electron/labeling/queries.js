import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { cleanPath, connectRo, connectRw, expandHome, resolveDbPath } from './db.js';

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

  const db = connectRo(dbPath);
  try {
    const typeClause = queueType && queueType !== 'all' ? 'AND q.queue_type = ?' : '';
    const params = queueType && queueType !== 'all' ? [runId, runId, queueType] : [runId, runId];
    const rows = db.prepare(`
      SELECT q.result_id, q.image_rel_path, q.crop_path, q.initial_label, q.suggested_label,
             q.queue_type, q.reliability_score, q.reason_codes_json,
             cv.x1, cv.y1, cv.x2, cv.y2,
             rd.action AS decided_action, rd.new_label AS decided_label
      FROM review_queue q
      LEFT JOIN crop_views cv ON cv.run_id = q.run_id AND cv.result_id = q.result_id AND cv.view_name = 'tight'
      LEFT JOIN review_decisions rd ON rd.result_id = q.result_id
        AND rd.review_session_id IN (SELECT review_session_id FROM review_sessions WHERE run_id = ?)
      WHERE q.run_id = ? ${typeClause}
      ORDER BY q.reliability_score ASC, q.result_id ASC
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
      };
    });

    const counts = {};
    for (const it of items) counts[it.queueType] = (counts[it.queueType] || 0) + 1;
    return { items, counts, total: items.length };
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

  const sessionId = `rev_${Date.now().toString(16)}_${Math.random().toString(16).slice(2, 8)}`;
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
