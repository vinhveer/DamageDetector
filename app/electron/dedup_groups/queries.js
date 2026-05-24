import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { cachedPrepare, connectRo, connectRoIfExists, expandHome, resolveDbPath } from './db.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const labRoot = path.resolve(repoRoot, '..');
const IN_CHUNK_SIZE = 900;
const IN_CHUNK_PLACEHOLDERS = Array.from({ length: IN_CHUNK_SIZE }, () => '?').join(',');

const DEFAULT_KEEPER_LABELS = ['all', 'crack', 'spall', 'mold'];

const cleanPath = (value, fieldName) => {
  const raw = String(value || '').trim();
  if (!raw || raw.includes('\0')) throw new Error(`Invalid ${fieldName}`);
  return raw;
};

const defaultDedupDbPath = () => path.join(labRoot, 'infer_results', 'semi-labeling', 'step4_class_aware_dedup', 'dedup.sqlite3');
const defaultSourceDbPath = () => path.join(labRoot, 'infer_results', 'semi-labeling', 'step2_sematic', 'damage_scan.sqlite3');
const defaultImageRootPath = () => path.join(labRoot, 'HinhAnh');

const toNumber = (value) => Number(value || 0);

const padChunk = (chunk) => chunk.length === IN_CHUNK_SIZE
  ? chunk
  : [...chunk, ...Array(IN_CHUNK_SIZE - chunk.length).fill(null)];

const parseJsonArray = (value) => {
  if (!value) return null;
  try {
    const parsed = JSON.parse(String(value));
    return Array.isArray(parsed) ? parsed.map(Number) : null;
  } catch {
    return null;
  }
};

const imageUriForPath = (imagePath) => imagePath ? pathToFileURL(imagePath).toString() : '';

const resolveImagePath = (row, imageRootPath) => {
  const imageRoot = String(imageRootPath || '').trim();
  const relPath = String(row.image_rel_path || '').trim();
  const storedPath = String(row.image_path || '').trim();
  const sourceInputDir = String(row.source_input_dir || '').trim();
  const candidates = [];

  if (imageRoot && relPath) candidates.push(path.resolve(expandHome(imageRoot), relPath));
  if (storedPath) {
    const expandedStored = expandHome(storedPath);
    candidates.push(path.isAbsolute(expandedStored) ? expandedStored : path.resolve(expandHome(sourceInputDir), expandedStored));
  }
  if (sourceInputDir && relPath) candidates.push(path.resolve(expandHome(sourceInputDir), relPath));
  if (imageRoot && storedPath) candidates.push(path.resolve(expandHome(imageRoot), path.basename(storedPath)));

  const seen = new Set();
  for (const candidate of candidates) {
    if (!candidate || seen.has(candidate)) continue;
    seen.add(candidate);
    if (fs.existsSync(candidate)) return candidate;
  }
  return candidates[0] || '';
};

const sourceMeta = (db, resultIds) => {
  if (!db || resultIds.length === 0) return new Map();
  const rows = [];
  for (let index = 0; index < resultIds.length; index += IN_CHUNK_SIZE) {
    const chunk = padChunk(resultIds.slice(index, index + IN_CHUNK_SIZE));
    rows.push(...cachedPrepare(db, `
      SELECT res.result_id, res.image_path, src_run.input_dir AS source_input_dir,
             res.detector_score, res.predicted_probability_pct,
             res.x1, res.y1, res.x2, res.y2
      FROM openclip_semantic_results res
      LEFT JOIN runs src_run ON src_run.run_id = res.source_run_id
      WHERE res.result_id IN (${IN_CHUNK_PLACEHOLDERS})
    `).all(...chunk));
  }
  return new Map(rows.map((row) => [Number(row.result_id), row]));
};

export const dedupGroupsDefaults = () => ({
  dedupDbPath: defaultDedupDbPath(),
  sourceDbPath: defaultSourceDbPath(),
  imageRootPath: defaultImageRootPath(),
  labels: DEFAULT_KEEPER_LABELS,
});

export const listDedupRuns = (payload = {}) => {
  const dedupDbPath = cleanPath(payload.dedupDbPath, 'dedupDbPath');
  const db = connectRo(dedupDbPath);
  try {
    const runs = cachedPrepare(db, `
      SELECT dedup_run_id, created_at_utc, source_db_path, embedding_db_path, embedding_run_id,
             total_detections, kept_count, fused_count, dropped_count, options_json
      FROM dedup_runs
      ORDER BY created_at_utc DESC, dedup_run_id DESC
    `).all();
    return { runs, dedupDbPath: resolveDbPath(dedupDbPath) };
  } finally {
    db.close();
  }
};

export const listDedupGroups = (payload = {}) => {
  const dedupDbPath = cleanPath(payload.dedupDbPath, 'dedupDbPath');
  const dedupRunId = cleanPath(payload.dedupRunId, 'dedupRunId');
  const label = String(payload.label || 'all').trim();
  const limit = Math.max(1, Math.min(5000, Number(payload.limit || 1000)));
  const labelClause = label && label !== 'all'
    ? `AND duplicate_group_id IN (
        SELECT duplicate_group_id FROM dedup_results
        WHERE dedup_run_id = ? AND predicted_label = ?
      )`
    : '';
  const params = label && label !== 'all' ? [dedupRunId, dedupRunId, label, limit] : [dedupRunId, limit];

  const db = connectRo(dedupDbPath);
  try {
    const groups = cachedPrepare(db, `
      SELECT duplicate_group_id,
             MIN(image_rel_path) AS image_rel_path,
             GROUP_CONCAT(DISTINCT predicted_label) AS labels,
             COUNT(*) AS result_count,
             SUM(CASE WHEN keep != 0 THEN 1 ELSE 0 END) AS kept_count,
             SUM(CASE WHEN fused != 0 THEN 1 ELSE 0 END) AS fused_count,
             SUM(CASE WHEN keep = 0 THEN 1 ELSE 0 END) AS dropped_count,
             SUM(CASE WHEN drop_reason = 'duplicate' THEN 1 ELSE 0 END) AS duplicate_drop_count,
             SUM(CASE WHEN drop_reason = 'low_quality' THEN 1 ELSE 0 END) AS low_quality_drop_count,
             SUM(CASE WHEN predicted_label = 'crack' THEN 1 ELSE 0 END) AS crack_count,
             SUM(CASE WHEN predicted_label = 'spall' THEN 1 ELSE 0 END) AS spall_count,
             SUM(CASE WHEN predicted_label = 'mold' THEN 1 ELSE 0 END) AS mold_count,
             MAX(p_dup_max) AS p_dup_max,
             AVG(p_good) AS p_good_avg,
             MIN(p_good) AS p_good_min,
             MAX(p_good) AS p_good_max
      FROM dedup_results
      WHERE dedup_run_id = ? ${labelClause}
      GROUP BY duplicate_group_id
      ORDER BY result_count DESC, p_dup_max DESC, duplicate_group_id
      LIMIT ?
    `).all(...params).map((row) => ({
      ...row,
      result_count: toNumber(row.result_count),
      kept_count: toNumber(row.kept_count),
      fused_count: toNumber(row.fused_count),
      dropped_count: toNumber(row.dropped_count),
      duplicate_drop_count: toNumber(row.duplicate_drop_count),
      low_quality_drop_count: toNumber(row.low_quality_drop_count),
      crack_count: toNumber(row.crack_count),
      spall_count: toNumber(row.spall_count),
      mold_count: toNumber(row.mold_count),
    }));
    return { groups };
  } finally {
    db.close();
  }
};

export const listDedupImages = (payload = {}) => {
  const dedupDbPath = cleanPath(payload.dedupDbPath, 'dedupDbPath');
  const dedupRunId = cleanPath(payload.dedupRunId, 'dedupRunId');
  const label = String(payload.label || 'all').trim();
  const sourceDbPath = String(payload.sourceDbPath || '').trim();
  const imageRootPath = String(payload.imageRootPath || '').trim();
  const limit = Math.max(1, Math.min(20000, Number(payload.limit || 5000)));
  const labelClause = label && label !== 'all' ? 'AND predicted_label = ?' : '';
  const params = label && label !== 'all' ? [dedupRunId, label, limit] : [dedupRunId, limit];
  const dedupDb = connectRo(dedupDbPath);
  let sourceDb = null;
  try {
    sourceDb = connectRoIfExists(sourceDbPath);
    const rows = cachedPrepare(dedupDb, `
      SELECT image_rel_path,
             MIN(result_id) AS sample_result_id,
             COUNT(*) AS before_count,
             SUM(CASE WHEN keep != 0 THEN 1 ELSE 0 END) AS after_count,
             SUM(CASE WHEN keep = 0 THEN 1 ELSE 0 END) AS dropped_count,
             SUM(CASE WHEN fused != 0 THEN 1 ELSE 0 END) AS fused_count,
             SUM(CASE WHEN predicted_label = 'crack' THEN 1 ELSE 0 END) AS crack_count,
             SUM(CASE WHEN predicted_label = 'spall' THEN 1 ELSE 0 END) AS spall_count,
             SUM(CASE WHEN predicted_label = 'mold' THEN 1 ELSE 0 END) AS mold_count,
             MAX(p_dup_max) AS p_dup_max,
             AVG(p_good) AS p_good_avg
      FROM dedup_results
      WHERE dedup_run_id = ? ${labelClause}
      GROUP BY image_rel_path
      ORDER BY dropped_count DESC, before_count DESC, image_rel_path
      LIMIT ?
    `).all(...params);
    const metaById = sourceMeta(sourceDb, rows.map((row) => Number(row.sample_result_id)).filter(Number.isFinite));
    const images = rows.map((row) => {
      const meta = metaById.get(Number(row.sample_result_id)) || {};
      const merged = { ...row, ...meta };
      const resolvedImagePath = resolveImagePath(merged, imageRootPath);
      return {
        ...merged,
        before_count: toNumber(row.before_count),
        after_count: toNumber(row.after_count),
        dropped_count: toNumber(row.dropped_count),
        fused_count: toNumber(row.fused_count),
        crack_count: toNumber(row.crack_count),
        spall_count: toNumber(row.spall_count),
        mold_count: toNumber(row.mold_count),
        resolved_image_path: resolvedImagePath,
        image_uri: imageUriForPath(resolvedImagePath),
      };
    });
    return { images };
  } finally {
    if (sourceDb) sourceDb.close();
    dedupDb.close();
  }
};

export const listDedupImageBoxes = (payload = {}) => {
  const dedupDbPath = cleanPath(payload.dedupDbPath, 'dedupDbPath');
  const dedupRunId = cleanPath(payload.dedupRunId, 'dedupRunId');
  const imageRelPath = cleanPath(payload.imageRelPath, 'imageRelPath');
  const mode = String(payload.mode || 'before') === 'after' ? 'after' : 'before';
  const label = String(payload.label || 'all').trim();
  const sourceDbPath = String(payload.sourceDbPath || '').trim();
  const imageRootPath = String(payload.imageRootPath || '').trim();
  const clauses = ['dedup_run_id = ?', 'image_rel_path = ?'];
  const params = [dedupRunId, imageRelPath];
  if (mode === 'after') clauses.push('keep != 0');
  if (label && label !== 'all') {
    clauses.push('predicted_label = ?');
    params.push(label);
  }

  const dedupDb = connectRo(dedupDbPath);
  let sourceDb = null;
  try {
    sourceDb = connectRoIfExists(sourceDbPath);
    const rows = cachedPrepare(dedupDb, `
      SELECT result_id, image_rel_path, predicted_label, keep, fused,
             duplicate_group_id, representative_id, p_dup_max, p_good,
             drop_reason, fused_bbox_json
      FROM dedup_results
      WHERE ${clauses.join(' AND ')}
      ORDER BY keep DESC, fused DESC, predicted_label, p_good DESC, result_id
    `).all(...params);
    const resultIds = rows.map((row) => Number(row.result_id)).filter(Number.isFinite);
    const metaById = sourceMeta(sourceDb, resultIds);
    const boxes = rows.map((row) => {
      const meta = metaById.get(Number(row.result_id)) || {};
      const fusedBox = parseJsonArray(row.fused_bbox_json);
      const box = fusedBox || [meta.x1 ?? 0, meta.y1 ?? 0, meta.x2 ?? 0, meta.y2 ?? 0].map(Number);
      return {
        ...row,
        ...meta,
        keep: Number(row.keep),
        fused: Number(row.fused),
        x1: Number(box[0] || 0),
        y1: Number(box[1] || 0),
        x2: Number(box[2] || 0),
        y2: Number(box[3] || 0),
        fused_bbox: fusedBox,
      };
    });
    const meta = metaById.get(Number(resultIds[0])) || { image_rel_path: imageRelPath };
    const resolvedImagePath = resolveImagePath({ ...meta, image_rel_path: imageRelPath }, imageRootPath);
    return {
      boxes,
      image: {
        image_rel_path: imageRelPath,
        resolved_image_path: resolvedImagePath,
        image_uri: imageUriForPath(resolvedImagePath),
      },
    };
  } finally {
    if (sourceDb) sourceDb.close();
    dedupDb.close();
  }
};

export const listDedupGroupMembers = (payload = {}) => {
  const dedupDbPath = cleanPath(payload.dedupDbPath, 'dedupDbPath');
  const dedupRunId = cleanPath(payload.dedupRunId, 'dedupRunId');
  const groupId = cleanPath(payload.groupId, 'groupId');
  const sourceDbPath = String(payload.sourceDbPath || '').trim();
  const imageRootPath = String(payload.imageRootPath || '').trim();
  const dedupDb = connectRo(dedupDbPath);
  let sourceDb = null;
  try {
    sourceDb = connectRoIfExists(sourceDbPath);
    const rows = cachedPrepare(dedupDb, `
      SELECT result_id, image_rel_path, predicted_label, keep, fused,
             duplicate_group_id, representative_id, p_dup_max, p_good,
             drop_reason, fused_bbox_json
      FROM dedup_results
      WHERE dedup_run_id = ? AND duplicate_group_id = ?
      ORDER BY keep DESC, fused DESC, p_good DESC, result_id
    `).all(dedupRunId, groupId);

    const resultIds = rows.map((row) => Number(row.result_id)).filter(Number.isFinite);
    const metaById = sourceMeta(sourceDb, resultIds);
    const members = rows.map((row) => {
      const meta = metaById.get(Number(row.result_id)) || {};
      const merged = {
        ...row,
        ...meta,
        keep: Number(row.keep),
        fused: Number(row.fused),
        fused_bbox: parseJsonArray(row.fused_bbox_json),
      };
      const resolvedImagePath = resolveImagePath(merged, imageRootPath);
      return {
        ...merged,
        resolved_image_path: resolvedImagePath,
        image_uri: imageUriForPath(resolvedImagePath),
      };
    });
    return { members };
  } finally {
    if (sourceDb) sourceDb.close();
    dedupDb.close();
  }
};
