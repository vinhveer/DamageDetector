import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { cleanPath, connectRo, connectRw, expandHome, resolveDbPath } from './db.js';
import { decodeVec, selectDiverseSample } from './sampling.js';

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

    const items = selectedIds ? allItems.filter((it) => selectedIds.has(it.resultId)) : allItems;

    const counts = {};
    for (const it of items) counts[it.queueType] = (counts[it.queueType] || 0) + 1;
    return { items, counts, total: items.length, queueTotal: allItems.length, sampled: Boolean(selectedIds) };
  } finally {
    db.close();
  }
};

const utcNow = () => new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');

// Build a "<name>_<YYYYMMDD_HHMMSS>" id from an optional English name.
const makeStampedId = (name, fallbackPrefix) => {
  const d = new Date();
  const p2 = (n) => String(n).padStart(2, '0');
  const stamp = `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}_${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}`;
  const clean = String(name || '').trim().replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
  const prefix = clean || fallbackPrefix;
  return `${prefix}_${stamp}`;
};

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
