import fs from 'node:fs';
import path from 'node:path';
import { cachedPrepare, connectRo, connectRw, refreshRunFlagCounts, runTransaction } from '../result_viewer/db.js';
import { imageUriForPath, resolveImagePath } from '../result_viewer/images.js';

const IN_CHUNK_SIZE = 900;
const IN_CHUNK_PLACEHOLDERS = Array.from({ length: IN_CHUNK_SIZE }, () => '?').join(',');

const expandHome = (value) => {
  if (!value || !value.startsWith('~')) return value || '';
  return path.join(process.env.HOME || process.env.USERPROFILE || '', value.slice(1));
};

const repoRoot = () => path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', '..', '..');

const labRoot = () => path.dirname(repoRoot());

const connectRoIfExists = (dbPath) => {
  if (!dbPath) return null;
  const resolvedDbPath = path.resolve(expandHome(dbPath));
  return fs.existsSync(resolvedDbPath) ? connectRo(resolvedDbPath) : null;
};

const closeDb = (db) => {
  if (db) db.close();
};

const padChunk = (chunk) => chunk.length === IN_CHUNK_SIZE
  ? chunk
  : [...chunk, ...Array(IN_CHUNK_SIZE - chunk.length).fill(null)];

export const defaultPrototypeReviewDb = () => path.join(labRoot(), 'infer_results', 'semi-labeling', 'step5_prototype_review', 'prototype_review.sqlite3');

const workingDirFor = (originalReviewDbPath) => path.join(path.dirname(originalReviewDbPath), 'working');

const workingCopyOf = (originalReviewDbPath) => path.join(workingDirFor(originalReviewDbPath), path.basename(originalReviewDbPath));

// Clone the default review DB + every feature DB it references into a sibling
// `working/` folder, then rewrite feature_db_path inside the cloned review DB
// so the app only ever mutates the copies. The originals stay untouched.
const ensureWorkingReviewDb = (originalReviewDbPath) => {
  const original = path.resolve(expandHome(originalReviewDbPath));
  if (!fs.existsSync(original)) return original;

  const workingDb = workingCopyOf(original);
  const workingDir = path.dirname(workingDb);

  if (fs.existsSync(workingDb)) return workingDb;

  fs.mkdirSync(workingDir, { recursive: true });
  fs.copyFileSync(original, workingDb);

  const db = connectRw(workingDb);
  try {
    const rows = cachedPrepare(db, `
      SELECT DISTINCT feature_db_path FROM prototype_review_runs
      WHERE feature_db_path IS NOT NULL AND feature_db_path != ''
    `).all();
    const updateStmt = cachedPrepare(db,
      `UPDATE prototype_review_runs SET feature_db_path = ? WHERE feature_db_path = ?`
    );
    runTransaction(db, () => {
      for (const row of rows) {
        const featureOriginal = path.resolve(expandHome(row.feature_db_path));
        if (!fs.existsSync(featureOriginal)) continue;
        const featureCopy = path.join(workingDir, path.basename(featureOriginal));
        if (!fs.existsSync(featureCopy)) {
          fs.copyFileSync(featureOriginal, featureCopy);
        }
        updateStmt.run(featureCopy, row.feature_db_path);
      }
    });
  } finally {
    db.close();
  }

  return workingDb;
};

export const prototypeReviewDefaults = () => ({
  reviewDbPath: ensureWorkingReviewDb(defaultPrototypeReviewDb()),
  imageRootPath: path.join(labRoot(), 'HinhAnh')
});

const ALLOWED_LABELS = new Set(['crack', 'mold', 'spall']);

export const setAssignmentsLabel = (payload) => {
  const resultIds = (payload.resultIds || []).map(Number).filter(Number.isFinite);
  const { featureDbPath, groupingRunId, label } = payload;
  if (!featureDbPath || !resultIds.length || !groupingRunId) return { changed: 0 };
  if (!ALLOWED_LABELS.has(label)) throw new Error(`Invalid label: ${label}`);
  const db = connectRw(featureDbPath);
  try {
    const changed = runTransaction(db, () => {
      let total = 0;
      for (let i = 0; i < resultIds.length; i += IN_CHUNK_SIZE) {
        const chunk = padChunk(resultIds.slice(i, i + IN_CHUNK_SIZE));
        total += cachedPrepare(db,
          `UPDATE feature_group_assignments SET predicted_label = ?
           WHERE grouping_run_id = ? AND result_id IN (${IN_CHUNK_PLACEHOLDERS})`
        ).run(label, groupingRunId, ...chunk).changes;
      }
      return total;
    });
    return { changed };
  } finally {
    db.close();
  }
};

export const markAssignmentsAsOutlier = (payload) => {
  const resultIds = (payload.resultIds || []).map(Number).filter(Number.isFinite);
  const { featureDbPath, groupingRunId } = payload;
  if (!featureDbPath || !resultIds.length || !groupingRunId) return { changed: 0 };
  const db = connectRw(featureDbPath);
  try {
    const changed = runTransaction(db, () => {
      let total = 0;
      for (let i = 0; i < resultIds.length; i += IN_CHUNK_SIZE) {
        const chunk = padChunk(resultIds.slice(i, i + IN_CHUNK_SIZE));
        total += cachedPrepare(db,
          `UPDATE feature_group_assignments SET is_outlier = 1
           WHERE grouping_run_id = ? AND result_id IN (${IN_CHUNK_PLACEHOLDERS}) AND is_outlier = 0`
        ).run(groupingRunId, ...chunk).changes;
      }
      cachedPrepare(db, `
        UPDATE feature_group_clusters
        SET outlier_count = (
          SELECT COUNT(*) FROM feature_group_assignments a
          WHERE a.grouping_run_id = feature_group_clusters.grouping_run_id
            AND a.cluster_key = feature_group_clusters.cluster_key
            AND a.is_outlier = 1
        )
        WHERE grouping_run_id = ?
      `).run(groupingRunId);
      refreshRunFlagCounts(db, groupingRunId);
      return total;
    });
    return { changed };
  } finally {
    db.close();
  }
};

export const listPrototypeReviewRuns = (payload) => {
  const reviewDbPath = path.resolve(expandHome(payload.reviewDbPath || defaultPrototypeReviewDb()));
  if (!fs.existsSync(reviewDbPath)) return { runs: [] };
  const db = connectRo(reviewDbPath);
  try {
    const runs = cachedPrepare(db, `
      SELECT review_run_id, created_at_utc, grouping_run_id, source_db_path,
             feature_db_path, model_name, device, prototype_config_json,
             thresholds_json, total_clusters, auto_accept_clusters,
             need_review_clusters
      FROM prototype_review_runs
      ORDER BY created_at_utc DESC
    `).all();
    return { runs };
  } finally {
    db.close();
  }
};

export const listPrototypeReviewScores = (payload) => {
  const clauses = ['review_run_id = ?'];
  const params = [payload.reviewRunId];
  if (payload.bucket && payload.bucket !== 'all') {
    clauses.push('review_bucket = ?');
    params.push(payload.bucket);
  }
  if (payload.label && payload.label !== 'all') {
    clauses.push('recommended_label = ?');
    params.push(payload.label);
  }
  const db = connectRo(payload.reviewDbPath);
  try {
    const scores = cachedPrepare(db, `
      SELECT review_run_id, cluster_key, original_label_scope, original_major_label,
             cluster_size, purity, recommended_label, crop_vote_label,
             crop_vote_ratio, mixed_ratio, score_crack, score_spall, score_mold,
             top_score, second_score, confidence_gap, review_bucket, reason,
             is_prototype
      FROM prototype_cluster_scores
      WHERE ${clauses.join(' AND ')}
      ORDER BY
        CASE review_bucket
          WHEN 'label_conflict' THEN 0
          WHEN 'ambiguous' THEN 1
          WHEN 'mixed' THEN 2
          WHEN 'unknown' THEN 3
          WHEN 'need_review' THEN 4
          WHEN 'prototype' THEN 5
          ELSE 6
        END,
        confidence_gap ASC,
        top_score ASC,
        cluster_size DESC,
        cluster_key
    `).all(...params);
    return { scores };
  } finally {
    db.close();
  }
};

const sourceMeta = (db, resultIds) => {
  if (!db || resultIds.length === 0) return new Map();
  const rows = [];
  for (let index = 0; index < resultIds.length; index += IN_CHUNK_SIZE) {
    const chunk = padChunk(resultIds.slice(index, index + IN_CHUNK_SIZE));
    rows.push(...cachedPrepare(db, `
      SELECT res.result_id, res.image_path, src_run.input_dir AS source_input_dir,
             res.x1, res.y1, res.x2, res.y2
      FROM openclip_semantic_results res
      JOIN runs src_run ON src_run.run_id = res.source_run_id
      WHERE res.result_id IN (${IN_CHUNK_PLACEHOLDERS})
    `).all(...chunk));
  }
  return new Map(rows.map((row) => [Number(row.result_id), row]));
};

const votesById = (db, reviewRunId, resultIds) => {
  if (!db || resultIds.length === 0) return new Map();
  const rows = [];
  for (let index = 0; index < resultIds.length; index += IN_CHUNK_SIZE) {
    const chunk = padChunk(resultIds.slice(index, index + IN_CHUNK_SIZE));
    rows.push(...cachedPrepare(db, `
      SELECT result_id, vote_label, vote_score
      FROM prototype_assignment_votes
      WHERE review_run_id = ? AND result_id IN (${IN_CHUNK_PLACEHOLDERS})
    `).all(reviewRunId, ...chunk));
  }
  return new Map(rows.map((row) => [Number(row.result_id), row]));
};

const listAssignmentsFromFeatureDb = (db, groupingRunId, clusterKey) => cachedPrepare(db, `
  SELECT result_id, image_rel_path, predicted_label, predicted_probability_pct,
         detector_score, cluster_key, is_outlier, distance_to_center,
         suggested_label, label_suspect, cluster_purity, cluster_size
  FROM feature_group_assignments
  WHERE grouping_run_id = ? AND cluster_key = ?
  ORDER BY distance_to_center ASC, predicted_probability_pct ASC, result_id
`).all(groupingRunId, clusterKey);

const listBulkAssignmentsFromFeatureDb = (db, groupingRunId, clusterKeys) => {
  if (clusterKeys.length === 0) return [];
  const rows = [];
  for (let index = 0; index < clusterKeys.length; index += IN_CHUNK_SIZE) {
    const chunk = padChunk(clusterKeys.slice(index, index + IN_CHUNK_SIZE));
    rows.push(...cachedPrepare(db, `
      SELECT result_id, image_rel_path, predicted_label, predicted_probability_pct,
             detector_score, cluster_key, is_outlier, distance_to_center,
             suggested_label, label_suspect, cluster_purity, cluster_size
      FROM feature_group_assignments
      WHERE grouping_run_id = ? AND cluster_key IN (${IN_CHUNK_PLACEHOLDERS})
      ORDER BY cluster_key, distance_to_center ASC, predicted_probability_pct ASC, result_id
    `).all(groupingRunId, ...chunk));
  }
  return rows;
};

const hydrateAssignments = (assignments, metaById, voteById, imageRoot) => assignments.map((row) => {
  const meta = metaById.get(Number(row.result_id)) || {};
  const vote = voteById.get(Number(row.result_id)) || {};
  const merged = {
    ...row,
    ...meta,
    vote_label: vote.vote_label || '',
    vote_score: vote.vote_score ?? null,
    image_path: meta.image_path || row.image_path || '',
    source_input_dir: meta.source_input_dir || row.source_input_dir || '',
    x1: meta.x1 ?? row.x1 ?? 0,
    y1: meta.y1 ?? row.y1 ?? 0,
    x2: meta.x2 ?? row.x2 ?? 0,
    y2: meta.y2 ?? row.y2 ?? 0
  };
  const resolved = resolveImagePath(merged, imageRoot);
  return {
    ...merged,
    resolved_image_path: resolved,
    image_uri: resolved ? imageUriForPath(resolved) : ''
  };
});

const uniqueNumbers = (rows) => Array.from(new Set(rows.map((row) => Number(row.result_id)).filter(Number.isFinite)));

const closePrototypeAssignmentDbs = ({ featureDb, sourceDb, reviewDb }) => {
  closeDb(reviewDb);
  closeDb(sourceDb);
  closeDb(featureDb);
};

const openPrototypeAssignmentDbs = () => ({
  featureDb: null,
  sourceDb: null,
  reviewDb: null
});

const connectPrototypeAssignmentDbs = (payload) => {
  const dbs = openPrototypeAssignmentDbs();
  try {
    dbs.featureDb = connectRo(payload.featureDbPath);
    dbs.sourceDb = connectRoIfExists(payload.sourceDbPath || '');
    dbs.reviewDb = connectRoIfExists(payload.reviewDbPath || '');
    return dbs;
  } catch (event) {
    closePrototypeAssignmentDbs(dbs);
    throw event;
  }
};

export const listPrototypeReviewAssignments = (payload) => {
  const dbs = connectPrototypeAssignmentDbs(payload);
  try {
    const assignments = listAssignmentsFromFeatureDb(dbs.featureDb, payload.groupingRunId, payload.clusterKey);
    const resultIds = uniqueNumbers(assignments);
    const metaById = sourceMeta(dbs.sourceDb, resultIds);
    const voteById = votesById(dbs.reviewDb, payload.reviewRunId, resultIds);
    return { assignments: hydrateAssignments(assignments, metaById, voteById, payload.imageRootPath || '') };
  } finally {
    closePrototypeAssignmentDbs(dbs);
  }
};

export const listPrototypeReviewAssignmentsBulk = (payload) => {
  const clusterKeys = Array.from(new Set((payload.clusterKeys || []).map(String).filter(Boolean)));
  if (clusterKeys.length === 0) return { assignmentsByClusterKey: {}, errors: {} };

  const dbs = connectPrototypeAssignmentDbs(payload);
  try {
    const assignments = listBulkAssignmentsFromFeatureDb(dbs.featureDb, payload.groupingRunId, clusterKeys);
    const resultIds = uniqueNumbers(assignments);
    const metaById = sourceMeta(dbs.sourceDb, resultIds);
    const voteById = votesById(dbs.reviewDb, payload.reviewRunId, resultIds);
    const assignmentsByClusterKey = Object.fromEntries(clusterKeys.map((key) => [key, []]));
    for (const row of hydrateAssignments(assignments, metaById, voteById, payload.imageRootPath || '')) {
      const key = String(row.cluster_key || '');
      if (!assignmentsByClusterKey[key]) assignmentsByClusterKey[key] = [];
      assignmentsByClusterKey[key].push(row);
    }
    return { assignmentsByClusterKey, errors: {} };
  } finally {
    closePrototypeAssignmentDbs(dbs);
  }
};


// ── Versioning handlers ──────────────────────────────────────────────────────

const hasColumn = (db, table, col) => {
  const info = cachedPrepare(db, `PRAGMA table_info(${table})`).all();
  return info.some((row) => row.name === col);
};

export const listVersions = (payload) => {
  const reviewDbPath = path.resolve(expandHome(payload.reviewDbPath || defaultPrototypeReviewDb()));
  if (!fs.existsSync(reviewDbPath)) return { runs: [] };
  const db = connectRo(reviewDbPath);
  try {
    if (!hasColumn(db, 'prototype_review_runs', 'is_active')) return { runs: [] };
    const where = ['1=1'];
    const params = [];
    if (payload.grouping_run_id) {
      where.push('r.grouping_run_id = ?');
      params.push(payload.grouping_run_id);
    }
    if (!payload.include_archived) {
      where.push('r.is_archived = 0');
    }
    const runs = cachedPrepare(db, `
      SELECT r.review_run_id, r.display_name, r.parent_review_run_id, r.grouping_run_id,
             r.created_at_utc, r.is_active, r.is_archived, r.model_name, r.device,
             r.total_clusters, r.auto_accept_clusters, r.need_review_clusters
      FROM prototype_review_runs r
      WHERE ${where.join(' AND ')}
      ORDER BY r.created_at_utc DESC
    `).all(...params);
    // Attach prototype_counts per run
    for (const run of runs) {
      const counts = cachedPrepare(db, `
        SELECT target_label, COUNT(*) AS cnt FROM prototype_groups WHERE review_run_id = ? GROUP BY target_label
      `).all(run.review_run_id);
      run.prototype_counts = { crack: 0, spall: 0, mold: 0 };
      for (const c of counts) run.prototype_counts[c.target_label] = c.cnt;
    }
    return { runs };
  } finally {
    db.close();
  }
};

export const getVersionDetail = (payload) => {
  const reviewDbPath = path.resolve(expandHome(payload.reviewDbPath || defaultPrototypeReviewDb()));
  const db = connectRo(reviewDbPath);
  try {
    const run = cachedPrepare(db, `
      SELECT review_run_id, display_name, parent_review_run_id, grouping_run_id,
             created_at_utc, is_active, is_archived, model_name, device,
             source_db_path, feature_db_path,
             prototype_config_json, thresholds_json, total_clusters,
             auto_accept_clusters, need_review_clusters
      FROM prototype_review_runs WHERE review_run_id = ?
    `).get(payload.review_run_id);
    if (!run) return { error: 'Not found' };
    const scores = cachedPrepare(db, `
      SELECT cluster_key, original_label_scope, original_major_label, cluster_size,
             purity, recommended_label, crop_vote_label, crop_vote_ratio, mixed_ratio,
             score_crack, score_spall, score_mold, top_score, second_score,
             confidence_gap, review_bucket, reason, is_prototype
      FROM prototype_cluster_scores WHERE review_run_id = ?
      ORDER BY top_score DESC
    `).all(payload.review_run_id);
    return { run, scores };
  } finally {
    db.close();
  }
};

const ALL_BUCKETS = ['auto_accept', 'prototype', 'need_review', 'unknown', 'ambiguous', 'mixed', 'label_conflict', 'excluded'];

export const getCandidates = (payload) => {
  const reviewDbPath = path.resolve(expandHome(payload.reviewDbPath || defaultPrototypeReviewDb()));
  const db = connectRo(reviewDbPath);
  try {
    // Get parent run info
    const parent = cachedPrepare(db, `
      SELECT grouping_run_id, feature_db_path, source_db_path
      FROM prototype_review_runs WHERE review_run_id = ?
    `).get(payload.parent_review_run_id);
    if (!parent) return { candidates: [], error: 'Parent not found' };

    // Get scores from parent
    const buckets = (payload.filters?.review_buckets?.length ? payload.filters.review_buckets : ALL_BUCKETS);
    const placeholders = buckets.map(() => '?').join(',');
    let sql = `
      SELECT cluster_key, recommended_label, cluster_size, purity, top_score,
             confidence_gap, review_bucket
      FROM prototype_cluster_scores
      WHERE review_run_id = ? AND review_bucket IN (${placeholders})
    `;
    const params = [payload.parent_review_run_id, ...buckets];
    if (payload.filters?.min_purity) {
      sql += ' AND purity >= ?';
      params.push(payload.filters.min_purity);
    }
    if (payload.filters?.min_top_score) {
      sql += ' AND top_score >= ?';
      params.push(payload.filters.min_top_score);
    }
    if (payload.filters?.current_labels?.length) {
      const lp = payload.filters.current_labels.map(() => '?').join(',');
      sql += ` AND recommended_label IN (${lp})`;
      params.push(...payload.filters.current_labels);
    }
    const sortCol = payload.sort === 'purity' ? 'purity' : payload.sort === 'cluster_size' ? 'cluster_size' : 'top_score';
    sql += ` ORDER BY ${sortCol} DESC`;
    const scores = cachedPrepare(db, sql).all(...params);

    // Get thumbnails from feature DB
    // Use ?? so that explicit 0 or -1 is not replaced by default.
    // -1 means "no limit" — omit the LIMIT clause entirely.
    const thumbsPerCluster = payload.thumbnails_per_cluster ?? 4;
    const useLimit = thumbsPerCluster > 0;
    const featureDbPath = path.resolve(expandHome(parent.feature_db_path));
    if (!fs.existsSync(featureDbPath)) return { candidates: scores.map((s) => ({ ...s, current_label: s.recommended_label, thumbnails: [] })) };

    const featureDb = connectRo(featureDbPath);
    const sourceDb = connectRoIfExists(parent.source_db_path);
    try {
      const thumbSql = `
        SELECT result_id, distance_to_center FROM feature_group_assignments
        WHERE grouping_run_id = ? AND cluster_key = ?
        ORDER BY distance_to_center ASC${useLimit ? ' LIMIT ?' : ''}
      `;
      const candidates = scores.map((s) => {
        const thumbParams = useLimit
          ? [parent.grouping_run_id, s.cluster_key, thumbsPerCluster]
          : [parent.grouping_run_id, s.cluster_key];
        const rows = cachedPrepare(featureDb, thumbSql).all(...thumbParams);
        const resultIds = rows.map((r) => Number(r.result_id));
        const meta = sourceMeta(sourceDb, resultIds);
        const thumbnails = rows.map((r) => {
          const m = meta.get(Number(r.result_id)) || {};
          return {
            result_id: Number(r.result_id),
            image_path: m.image_path || '',
            bbox: [m.x1 || 0, m.y1 || 0, m.x2 || 0, m.y2 || 0],
            distance_to_center: r.distance_to_center
          };
        });
        return { ...s, current_label: s.recommended_label, thumbnails };
      });
      return { candidates };
    } finally {
      closeDb(featureDb);
      closeDb(sourceDb);
    }
  } finally {
    db.close();
  }
};

export const createVersion = (payload) => {
  // Returns args for spawning run_prototype_review.py — actual spawn done in main.js
  const reviewDbPath = path.resolve(expandHome(payload.reviewDbPath || defaultPrototypeReviewDb()));
  const db = connectRo(reviewDbPath);
  try {
    const parent = cachedPrepare(db, `
      SELECT grouping_run_id, source_db_path, feature_db_path, model_name, device, thresholds_json
      FROM prototype_review_runs WHERE review_run_id = ?
    `).get(payload.parent_review_run_id);
    if (!parent) return { error: 'Parent not found' };
    const thresholds = JSON.parse(parent.thresholds_json || '{}');
    const overrides = payload.thresholds || {};
    // v2 shape: { crack: { clusters: [...], images: [...] }, spall: {...}, mold: {...}, excluded: {...} }
    // Legacy shape (still accepted by backend): { crack: [...], spall: [...], mold: [...] }
    const prototypeJson = JSON.stringify(payload.selections);
    return {
      args: [
        '--source-db', parent.source_db_path,
        '--feature-db', parent.feature_db_path,
        '--output-db', reviewDbPath,
        '--grouping-run-id', parent.grouping_run_id,
        '--model-name', parent.model_name,
        '--device', parent.device,
        '--prototype-json', prototypeJson,
        '--parent-review-run-id', payload.parent_review_run_id,
        '--display-name', payload.display_name || '',
        ...(payload.set_active ? ['--set-active'] : []),
        '--unknown-threshold', String(overrides.unknown_threshold ?? thresholds.unknown_threshold ?? 0.55),
        '--auto-threshold', String(overrides.auto_threshold ?? thresholds.auto_threshold ?? 0.78),
        '--gap-threshold', String(overrides.gap_threshold ?? thresholds.gap_threshold ?? 0.03),
        '--vote-threshold', String(overrides.vote_threshold ?? thresholds.vote_threshold ?? 0.65),
      ],
      grouping_run_id: parent.grouping_run_id
    };
  } finally {
    db.close();
  }
};

export const setVersionActive = (payload) => {
  const reviewDbPath = path.resolve(expandHome(payload.reviewDbPath || defaultPrototypeReviewDb()));
  const db = connectRw(reviewDbPath);
  try {
    const row = cachedPrepare(db, 'SELECT grouping_run_id FROM prototype_review_runs WHERE review_run_id = ?').get(payload.review_run_id);
    if (!row) return { error: 'Not found' };
    runTransaction(db, () => {
      cachedPrepare(db, 'UPDATE prototype_review_runs SET is_active = 0 WHERE grouping_run_id = ? AND is_active = 1').run(row.grouping_run_id);
      cachedPrepare(db, 'UPDATE prototype_review_runs SET is_active = 1 WHERE review_run_id = ?').run(payload.review_run_id);
    });
    return { ok: true };
  } finally {
    db.close();
  }
};

export const archiveVersion = (payload) => {
  const reviewDbPath = path.resolve(expandHome(payload.reviewDbPath || defaultPrototypeReviewDb()));
  const db = connectRw(reviewDbPath);
  try {
    runTransaction(db, () => {
      cachedPrepare(db, 'UPDATE prototype_review_runs SET is_archived = 1, is_active = 0 WHERE review_run_id = ?').run(payload.review_run_id);
    });
    return { ok: true };
  } finally {
    db.close();
  }
};

export const unarchiveVersion = (payload) => {
  const reviewDbPath = path.resolve(expandHome(payload.reviewDbPath || defaultPrototypeReviewDb()));
  const db = connectRw(reviewDbPath);
  try {
    cachedPrepare(db, 'UPDATE prototype_review_runs SET is_archived = 0 WHERE review_run_id = ?').run(payload.review_run_id);
    return { ok: true };
  } finally {
    db.close();
  }
};

export const renameVersion = (payload) => {
  const name = String(payload.display_name || '').trim();
  if (!name || name.length > 64) return { error: 'Name must be 1-64 chars' };
  const reviewDbPath = path.resolve(expandHome(payload.reviewDbPath || defaultPrototypeReviewDb()));
  const db = connectRw(reviewDbPath);
  try {
    const row = cachedPrepare(db, 'SELECT grouping_run_id FROM prototype_review_runs WHERE review_run_id = ?').get(payload.review_run_id);
    if (!row) return { error: 'Not found' };
    const dup = cachedPrepare(db, `
      SELECT 1 FROM prototype_review_runs
      WHERE grouping_run_id = ? AND review_run_id != ? AND LOWER(display_name) = LOWER(?)
    `).get(row.grouping_run_id, payload.review_run_id, name);
    if (dup) return { error: 'Name already exists' };
    cachedPrepare(db, 'UPDATE prototype_review_runs SET display_name = ? WHERE review_run_id = ?').run(name, payload.review_run_id);
    return { ok: true };
  } finally {
    db.close();
  }
};
