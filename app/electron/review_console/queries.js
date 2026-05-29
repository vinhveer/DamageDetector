import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { cachedPrepare, connectRo, expandHome, resolveDbPath } from '../dedup_groups/db.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const labRoot = path.resolve(repoRoot, '..');
const PRIMARY_VIEW = 'openclip_crop';

const cleanPath = (value, fieldName) => {
  const raw = String(value || '').trim();
  if (!raw || raw.includes('\0')) throw new Error(`Invalid ${fieldName}`);
  return raw;
};

const cleanRunId = (value) => {
  const raw = String(value || '').trim();
  if (!raw || !/^[a-zA-Z0-9_.-]+$/.test(raw)) throw new Error('Invalid runId');
  return raw;
};

const defaultResemiDbPath = () => path.join(labRoot, 'infer_results', 'semi-labeling', 'resemi', 'resemi.sqlite3');
const defaultImageRootPath = () => path.join(labRoot, 'data', 'HinhAnh');
const defaultSessionsDir = () => path.join(labRoot, 'infer_results', 'semi-labeling', 'resemi', 'review_sessions');

const resolveImagePath = (row, imageRootPath) => {
  const imageRoot = String(imageRootPath || '').trim();
  const relPath = String(row.image_rel_path || '').trim();
  const cropPath = String(row.crop_path || '').trim();
  const candidates = [];
  if (cropPath) {
    const expanded = expandHome(cropPath);
    if (path.isAbsolute(expanded)) candidates.push(expanded);
  }
  if (imageRoot && relPath) candidates.push(path.resolve(expandHome(imageRoot), relPath));
  const seen = new Set();
  for (const c of candidates) {
    if (!c || seen.has(c)) continue;
    seen.add(c);
    if (fs.existsSync(c)) return c;
  }
  return candidates[candidates.length - 1] || '';
};

const imageUriForPath = (p) => (p ? pathToFileURL(p).toString() : '');

const parseJson = (value, fallback) => {
  try {
    const parsed = JSON.parse(String(value || ''));
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
};

export const reviewConsoleDefaults = () => ({
  resemiDbPath: defaultResemiDbPath(),
  imageRootPath: defaultImageRootPath(),
  sessionsDir: defaultSessionsDir(),
});

// ── Runs + top-bar selectors ────────────────────────────────────────────────
export const listRuns = (payload = {}) => {
  const resemiDbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const db = connectRo(resemiDbPath);
  try {
    const runs = cachedPrepare(db, `
      SELECT run_id, created_at_utc, total_detections, cleaned_count, suspect_count,
             reject_count, prototype_version_id
      FROM resemi_runs
      ORDER BY created_at_utc DESC, run_id DESC
    `).all();

    const latestByRun = (table, idCol, runId) => {
      try {
        return cachedPrepare(db, `
          SELECT ${idCol} AS id FROM ${table}
          WHERE run_id = ? ORDER BY created_at_utc DESC LIMIT 1
        `).get(runId)?.id ?? null;
      } catch {
        return null;
      }
    };

    return {
      resemiDbPath: resolveDbPath(resemiDbPath),
      runs: runs.map((row) => ({
        run_id: String(row.run_id),
        created_at_utc: String(row.created_at_utc || ''),
        total_detections: Number(row.total_detections || 0),
        cleaned_count: Number(row.cleaned_count || 0),
        suspect_count: Number(row.suspect_count || 0),
        reject_count: Number(row.reject_count || 0),
        prototype_version_id: row.prototype_version_id ? String(row.prototype_version_id) : null,
        reliability_run_id: latestByRun('reliability_scoring_runs', 'reliability_run_id', String(row.run_id)),
        decision_policy_run_id: latestByRun('decision_policy_runs', 'decision_policy_run_id', String(row.run_id)),
      })),
    };
  } finally {
    db.close();
  }
};

// ── Queue counts for the left rail ──────────────────────────────────────────
export const getQueueCounts = (payload = {}) => {
  const resemiDbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanRunId(payload.runId);
  const db = connectRo(resemiDbPath);
  const safeCount = (sql, ...args) => {
    try {
      return Number(cachedPrepare(db, sql).get(...args)?.n || 0);
    } catch {
      return 0;
    }
  };
  try {
    return {
      run_id: runId,
      prototype_candidates: safeCount(
        `SELECT COUNT(*) AS n FROM cleaned_labels WHERE run_id = ? AND decision_type IN ('auto_accept')`,
        runId,
      ),
      core_clusters: safeCount(`SELECT COUNT(*) AS n FROM core_clusters WHERE run_id = ?`, runId),
      disagreement: safeCount(`SELECT COUNT(*) AS n FROM review_queue WHERE run_id = ?`, runId),
      outliers: safeCount(
        `SELECT COUNT(*) AS n FROM core_outliers o
         JOIN core_mining_runs m ON m.core_mining_run_id = o.core_mining_run_id
         WHERE m.run_id = ?`,
        runId,
      ),
      relabel_batches: safeCount(
        `SELECT COUNT(*) AS n FROM semantic_decisions WHERE run_id = ? AND decision_type = 'relabel_candidate'`,
        runId,
      ),
      committed: safeCount(
        `SELECT COUNT(*) AS n FROM review_decisions d
         JOIN review_sessions s ON s.review_session_id = d.review_session_id
         WHERE s.run_id = ? AND s.status = 'committed'`,
        runId,
      ),
    };
  } finally {
    db.close();
  }
};

// ── Stage 3: Disagreement / suspect items ───────────────────────────────────
export const listDisagreementItems = (payload = {}) => {
  const resemiDbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanRunId(payload.runId);
  const imageRootPath = String(payload.imageRootPath || '').trim();
  const filters = payload.filters && typeof payload.filters === 'object' ? payload.filters : {};
  const limit = Math.min(Math.max(Number(payload.limit) || 200, 1), 2000);
  const offset = Math.max(Number(payload.offset) || 0, 0);

  const where = ['rq.run_id = ?'];
  const args = [runId];
  if (filters.label) {
    where.push('rq.initial_label = ?');
    args.push(String(filters.label));
  }
  if (filters.queueType) {
    where.push('rq.queue_type = ?');
    args.push(String(filters.queueType));
  }
  if (filters.reasonCode) {
    where.push('rq.reason_codes_json LIKE ?');
    args.push(`%${String(filters.reasonCode)}%`);
  }
  if (filters.reliabilityMin != null && filters.reliabilityMin !== '') {
    where.push('rq.reliability_score >= ?');
    args.push(Number(filters.reliabilityMin));
  }
  if (filters.reliabilityMax != null && filters.reliabilityMax !== '') {
    where.push('rq.reliability_score <= ?');
    args.push(Number(filters.reliabilityMax));
  }

  const db = connectRo(resemiDbPath);
  try {
    const total = Number(cachedPrepare(db, `
      SELECT COUNT(*) AS n FROM review_queue rq WHERE ${where.join(' AND ')}
    `).get(...args)?.n || 0);

    const rows = cachedPrepare(db, `
      SELECT rq.result_id, rq.image_rel_path, rq.crop_path, rq.initial_label,
             rq.suggested_label, rq.queue_type, rq.reliability_score AS rq_reliability,
             rq.reason_codes_json,
             sd.final_label, sd.decision_type, sd.nearest_core_class,
             sd.nearest_core_similarity, sd.prototype_class, sd.prototype_similarity,
             sd.model_agreement,
             cv.x1, cv.y1, cv.x2, cv.y2,
             sa.majority_label, sa.agreement_ratio, sa.conflict_labels_json
      FROM review_queue rq
      LEFT JOIN semantic_decisions sd ON sd.run_id = rq.run_id AND sd.result_id = rq.result_id
      LEFT JOIN crop_views cv ON cv.run_id = rq.run_id AND cv.result_id = rq.result_id AND cv.view_name = '${PRIMARY_VIEW}'
      LEFT JOIN semantic_agreements sa ON sa.run_id = rq.run_id AND sa.result_id = rq.result_id
      WHERE ${where.join(' AND ')}
      ORDER BY rq.reliability_score ASC, rq.result_id ASC
      LIMIT ? OFFSET ?
    `).all(...args, limit, offset);

    return {
      total,
      limit,
      offset,
      items: rows.map((row) => {
        const imagePath = resolveImagePath(row, imageRootPath);
        return {
          result_id: Number(row.result_id),
          image_rel_path: String(row.image_rel_path || ''),
          image_uri: imageUriForPath(imagePath),
          image_path: imagePath,
          x1: Number(row.x1 || 0),
          y1: Number(row.y1 || 0),
          x2: Number(row.x2 || 0),
          y2: Number(row.y2 || 0),
          initial_label: String(row.initial_label || ''),
          suggested_label: String(row.suggested_label || ''),
          final_label: row.final_label ? String(row.final_label) : String(row.initial_label || ''),
          decision_type: String(row.decision_type || row.queue_type || 'suspect'),
          queue_type: String(row.queue_type || 'suspect'),
          reliability_score: Number(row.rq_reliability || 0),
          model_agreement: Number(row.model_agreement || 0),
          majority_label: row.majority_label ? String(row.majority_label) : '',
          agreement_ratio: Number(row.agreement_ratio || 0),
          conflict_labels: parseJson(row.conflict_labels_json, []),
          nearest_core_class: row.nearest_core_class ? String(row.nearest_core_class) : null,
          nearest_core_similarity: row.nearest_core_similarity == null ? null : Number(row.nearest_core_similarity),
          prototype_class: row.prototype_class ? String(row.prototype_class) : null,
          prototype_similarity: row.prototype_similarity == null ? null : Number(row.prototype_similarity),
          reason_codes: parseJson(row.reason_codes_json, []),
        };
      }),
    };
  } finally {
    db.close();
  }
};

// ── Right inspector: full evidence for one result ───────────────────────────
export const getItemEvidence = (payload = {}) => {
  const resemiDbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanRunId(payload.runId);
  const resultId = Number(payload.resultId);
  if (!Number.isInteger(resultId)) throw new Error('Invalid resultId');
  const imageRootPath = String(payload.imageRootPath || '').trim();

  const db = connectRo(resemiDbPath);
  try {
    const modelOutputs = cachedPrepare(db, `
      SELECT model_name, source_type, top1_label, top1_score, top2_label, top2_score, margin, entropy
      FROM semantic_model_outputs WHERE run_id = ? AND result_id = ?
      ORDER BY model_name
    `).all(runId, resultId).map((row) => ({
      model_name: String(row.model_name),
      source_type: String(row.source_type || ''),
      top1_label: String(row.top1_label || ''),
      top1_score: Number(row.top1_score || 0),
      top2_label: row.top2_label ? String(row.top2_label) : null,
      top2_score: row.top2_score == null ? null : Number(row.top2_score),
      margin: Number(row.margin || 0),
      entropy: row.entropy == null ? null : Number(row.entropy),
    }));

    const reliability = cachedPrepare(db, `
      SELECT reliability_score, reason_codes_json, score_components_json
      FROM reliability_scores WHERE run_id = ? AND result_id = ?
    `).get(runId, resultId) || {};

    const boxQuality = (() => {
      try {
        return cachedPrepare(db, `
          SELECT box_quality_score, area_ratio_to_image, aspect_ratio, elongation,
                 child_count, composite_penalty
          FROM box_quality_scores WHERE result_id = ? LIMIT 1
        `).get(resultId) || null;
      } catch {
        return null;
      }
    })();

    const cropViews = cachedPrepare(db, `
      SELECT view_name, x1, y1, x2, y2, image_rel_path, crop_path
      FROM crop_views WHERE run_id = ? AND result_id = ?
    `).all(runId, resultId).map((row) => {
      const imagePath = resolveImagePath(row, imageRootPath);
      return {
        view_name: String(row.view_name),
        x1: Number(row.x1 || 0),
        y1: Number(row.y1 || 0),
        x2: Number(row.x2 || 0),
        y2: Number(row.y2 || 0),
        image_rel_path: String(row.image_rel_path || ''),
        image_uri: imageUriForPath(imagePath),
      };
    });

    return {
      result_id: resultId,
      model_outputs: modelOutputs,
      reliability_score: Number(reliability.reliability_score || 0),
      reason_codes: parseJson(reliability.reason_codes_json, []),
      score_components: parseJson(reliability.score_components_json, {}),
      box_quality: boxQuality
        ? {
            box_quality_score: Number(boxQuality.box_quality_score || 0),
            area_ratio_to_image: Number(boxQuality.area_ratio_to_image || 0),
            aspect_ratio: Number(boxQuality.aspect_ratio || 0),
            elongation: Number(boxQuality.elongation || 0),
            child_count: Number(boxQuality.child_count || 0),
            composite_penalty: Number(boxQuality.composite_penalty || 0),
          }
        : null,
      crop_views: cropViews,
    };
  } finally {
    db.close();
  }
};

// ── Stage 1: Prototype candidates ───────────────────────────────────────────
export const listPrototypeCandidates = (payload = {}) => {
  const resemiDbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanRunId(payload.runId);
  const imageRootPath = String(payload.imageRootPath || '').trim();
  const label = payload.label ? String(payload.label) : null;
  const limit = Math.min(Math.max(Number(payload.limit) || 120, 1), 1000);

  const where = ['run_id = ?', "decision_type = 'auto_accept'"];
  const args = [runId];
  if (label) {
    where.push('final_label = ?');
    args.push(label);
  }

  const db = connectRo(resemiDbPath);
  try {
    const rows = cachedPrepare(db, `
      SELECT result_id, image_rel_path, crop_path, final_label, reliability_score,
             x1, y1, x2, y2
      FROM cleaned_labels
      WHERE ${where.join(' AND ')}
      ORDER BY reliability_score DESC, result_id ASC
      LIMIT ?
    `).all(...args, limit);

    const labelCounts = cachedPrepare(db, `
      SELECT final_label AS label, COUNT(*) AS n
      FROM cleaned_labels WHERE run_id = ? AND decision_type = 'auto_accept'
      GROUP BY final_label ORDER BY final_label
    `).all(runId).map((row) => ({ label: String(row.label), count: Number(row.n || 0) }));

    return {
      label_counts: labelCounts,
      items: rows.map((row) => {
        const imagePath = resolveImagePath(row, imageRootPath);
        return {
          result_id: Number(row.result_id),
          image_rel_path: String(row.image_rel_path || ''),
          image_uri: imageUriForPath(imagePath),
          final_label: String(row.final_label || ''),
          reliability_score: Number(row.reliability_score || 0),
          x1: Number(row.x1 || 0),
          y1: Number(row.y1 || 0),
          x2: Number(row.x2 || 0),
          y2: Number(row.y2 || 0),
        };
      }),
    };
  } finally {
    db.close();
  }
};

// ── Stages 2/4/5: thin queries (return [] gracefully when smoke-only/empty) ──
export const getCoreClusters = (payload = {}) => {
  const resemiDbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanRunId(payload.runId);
  const db = connectRo(resemiDbPath);
  try {
    const rows = cachedPrepare(db, `
      SELECT core_cluster_id, label, member_count, density_score, agreement_score, status
      FROM core_clusters WHERE run_id = ?
      ORDER BY member_count DESC
    `).all(runId);
    return {
      clusters: rows.map((row) => ({
        core_cluster_id: String(row.core_cluster_id),
        label: String(row.label || ''),
        member_count: Number(row.member_count || 0),
        density_score: row.density_score == null ? null : Number(row.density_score),
        agreement_score: row.agreement_score == null ? null : Number(row.agreement_score),
        status: String(row.status || 'core'),
      })),
    };
  } finally {
    db.close();
  }
};

export const listOutliers = (payload = {}) => {
  const resemiDbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanRunId(payload.runId);
  const db = connectRo(resemiDbPath);
  try {
    const rows = cachedPrepare(db, `
      SELECT o.result_id, o.label, o.outlier_type, o.nearest_cluster_id, o.similarity,
             o.reason_codes_json
      FROM core_outliers o
      JOIN core_mining_runs m ON m.core_mining_run_id = o.core_mining_run_id
      WHERE m.run_id = ?
      ORDER BY o.similarity ASC
      LIMIT 1000
    `).all(runId);
    return {
      outliers: rows.map((row) => ({
        result_id: Number(row.result_id),
        label: String(row.label || ''),
        outlier_type: String(row.outlier_type || ''),
        nearest_cluster_id: row.nearest_cluster_id ? String(row.nearest_cluster_id) : null,
        similarity: row.similarity == null ? null : Number(row.similarity),
        reason_codes: parseJson(row.reason_codes_json, []),
      })),
    };
  } finally {
    db.close();
  }
};

export const listRelabelBatches = (payload = {}) => {
  const resemiDbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
  const runId = cleanRunId(payload.runId);
  const db = connectRo(resemiDbPath);
  try {
    const rows = cachedPrepare(db, `
      SELECT initial_label AS from_label, final_label AS to_label, COUNT(*) AS count,
             AVG(reliability_score) AS avg_reliability
      FROM semantic_decisions
      WHERE run_id = ? AND decision_type = 'relabel_candidate'
      GROUP BY initial_label, final_label
      ORDER BY count DESC
    `).all(runId);
    return {
      batches: rows.map((row, idx) => ({
        batch_id: `relabel_${String(row.from_label)}_${String(row.to_label)}_${idx}`,
        from_label: String(row.from_label || ''),
        to_label: String(row.to_label || ''),
        count: Number(row.count || 0),
        avg_reliability: Number(row.avg_reliability || 0),
      })),
    };
  } finally {
    db.close();
  }
};
