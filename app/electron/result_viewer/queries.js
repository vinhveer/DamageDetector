import fs from 'node:fs';
import path from 'node:path';
import { cachedPrepare, connectRo } from './db.js';
import { imageUriForPath, resolveImagePath } from './images.js';

const IN_CHUNK_SIZE = 900;
const IN_CHUNK_PLACEHOLDERS = Array.from({ length: IN_CHUNK_SIZE }, () => '?').join(',');

const expandHome = (value) => {
  if (!value || !value.startsWith('~')) return value || '';
  return path.join(process.env.HOME || process.env.USERPROFILE || '', value.slice(1));
};

const connectRoIfExists = (dbPath) => {
  if (!dbPath) return null;
  const resolvedDbPath = path.resolve(expandHome(dbPath));
  return fs.existsSync(resolvedDbPath) ? connectRo(resolvedDbPath) : null;
};

const padChunk = (chunk) => chunk.length === IN_CHUNK_SIZE
  ? chunk
  : [...chunk, ...Array(IN_CHUNK_SIZE - chunk.length).fill(null)];

export const listRuns = (payload) => {
  const db = connectRo(payload.featureDbPath);
  try {
    const runs = cachedPrepare(db, `
      SELECT grouping_run_id, created_at_utc, source_db_path, filtered_db_path,
             source_filter_run_id, model_name, device, total_boxes, total_clusters,
             outlier_boxes, label_suspect_boxes
      FROM feature_group_runs
      ORDER BY created_at_utc DESC
    `).all();
    return { runs };
  } finally {
    db.close();
  }
};

export const listClusters = (payload) => {
  const clauses = ['grouping_run_id = ?', 'predicted_label_scope = ?'];
  const params = [payload.runId, payload.labelScope];

  if (payload.mode === 'non_outlier') {
    clauses.push('outlier_count = 0');
  } else if (payload.mode === 'outlier') {
    clauses.push('outlier_count > 0');
  } else if (payload.mode === 'label_suspect') {
    clauses.push(`
      EXISTS (
        SELECT 1
        FROM feature_group_assignments a
        WHERE a.grouping_run_id = feature_group_clusters.grouping_run_id
          AND a.cluster_key = feature_group_clusters.cluster_key
          AND a.label_suspect != 0
      )
    `);
  }

  const db = connectRo(payload.featureDbPath);
  try {
    const clusters = cachedPrepare(db, `
      SELECT cluster_key, predicted_label_scope, cluster_id, cluster_size,
             major_label, purity, crack_count, mold_count, spall_count,
             outlier_count, representative_nearest_result_id,
             representative_farthest_result_id,
             representative_low_confidence_result_id,
             representative_mismatch_result_id
      FROM feature_group_clusters
      WHERE ${clauses.join(' AND ')}
      ORDER BY cluster_size DESC, purity ASC, cluster_key
    `).all(...params);
    return { clusters };
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

export const listAssignments = (payload) => {
  const featureDb = connectRo(payload.featureDbPath);
  let sourceDb = null;
  try {
    sourceDb = connectRoIfExists(payload.sourceDbPath || '');
    const assignments = cachedPrepare(featureDb, `
      SELECT result_id, image_rel_path, predicted_label, predicted_probability_pct,
             detector_score, cluster_key, is_outlier, distance_to_center,
             suggested_label, label_suspect, cluster_purity, cluster_size
      FROM feature_group_assignments
      WHERE grouping_run_id = ? AND cluster_key = ?
      ORDER BY distance_to_center ASC, predicted_probability_pct ASC, result_id
    `).all(payload.runId, payload.clusterKey);

    const resultIds = Array.from(new Set(assignments.map((row) => Number(row.result_id)).filter(Number.isFinite)));
    const metaById = sourceMeta(sourceDb, resultIds);
    const imageRoot = payload.imageRootPath || '';
    const nextAssignments = assignments.map((row) => {
      const meta = metaById.get(Number(row.result_id)) || {};
      const merged = {
        ...row,
        ...meta,
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
    return { assignments: nextAssignments };
  } finally {
    if (sourceDb) sourceDb.close();
    featureDb.close();
  }
};
