import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { cachedPrepare, connectRo, expandHome, resolveDbPath } from './db.js';
import { SEMI_LABELING_DEFAULTS } from '../defaults.js';

const IN_CHUNK_SIZE = 900;

const cleanPath = (value, fieldName) => {
  const raw = String(value || '').trim();
  if (!raw || raw.includes('\0')) throw new Error(`Invalid ${fieldName}`);
  return raw;
};

const defaultClusterDbPath = () => SEMI_LABELING_DEFAULTS.cluster.clusterDbPath;
const defaultSourceDbPath = () => SEMI_LABELING_DEFAULTS.cluster.sourceDbPath;
const defaultImageRootPath = () => SEMI_LABELING_DEFAULTS.cluster.imageRootPath;
const defaultSessionsDir = () => SEMI_LABELING_DEFAULTS.cluster.sessionsDir;

const resolveImagePath = (row, imageRootPath) => {
  const imageRoot = String(imageRootPath || '').trim();
  const relPath = String(row.image_rel_path || '').trim();
  const storedPath = String(row.image_path || '').trim();
  const sourceInputDir = String(row.source_input_dir || '').trim();
  const candidates = [];
  if (imageRoot && relPath) candidates.push(path.resolve(expandHome(imageRoot), relPath));
  if (storedPath) {
    const expanded = expandHome(storedPath);
    candidates.push(path.isAbsolute(expanded) ? expanded : path.resolve(expandHome(sourceInputDir), expanded));
  }
  if (sourceInputDir && relPath) candidates.push(path.resolve(expandHome(sourceInputDir), relPath));
  const seen = new Set();
  for (const c of candidates) {
    if (!c || seen.has(c)) continue;
    seen.add(c);
    if (fs.existsSync(c)) return c;
  }
  return candidates[0] || '';
};

const imageUriForPath = (p) => p ? pathToFileURL(p).toString() : '';

export const clusterLabelingDefaults = () => ({
  clusterDbPath: defaultClusterDbPath(),
  sourceDbPath: defaultSourceDbPath(),
  imageRootPath: defaultImageRootPath(),
  sessionsDir: defaultSessionsDir(),
});

export const listClusterRuns = (payload = {}) => {
  const clusterDbPath = cleanPath(payload.clusterDbPath, 'clusterDbPath');
  const db = connectRo(clusterDbPath);
  try {
    const runs = cachedPrepare(db, `
      SELECT cluster_run_id, created_at_utc, dedup_run_id, embedding_run_id,
             algorithm, total_boxes, total_clusters, pca_dim, pca_explained_ratio,
             options_json
      FROM cluster_runs
      ORDER BY created_at_utc DESC, cluster_run_id DESC
    `).all();
    return { runs, clusterDbPath: resolveDbPath(clusterDbPath) };
  } finally {
    db.close();
  }
};

export const listClusters = (payload = {}) => {
  const clusterDbPath = cleanPath(payload.clusterDbPath, 'clusterDbPath');
  const clusterRunId = cleanPath(payload.clusterRunId, 'clusterRunId');
  const sortBy = String(payload.sortBy || 'size').trim();
  const filterLabel = String(payload.filterLabel || '').trim();

  const orderClause = sortBy === 'cluster_id'
    ? 'cluster_id ASC'
    : sortBy === 'purity'
      ? 'avg_intra_distance ASC, size DESC'
      : 'size DESC, cluster_id ASC';

  const labelClause = filterLabel && filterLabel !== 'all' ? 'AND dominant_label = ?' : '';
  const params = filterLabel && filterLabel !== 'all'
    ? [clusterRunId, filterLabel]
    : [clusterRunId];

  const db = connectRo(clusterDbPath);
  try {
    const clusters = cachedPrepare(db, `
      SELECT cluster_id, size, representative_result_id,
             label_distribution_json, dominant_label, avg_intra_distance
      FROM cluster_summary
      WHERE cluster_run_id = ? ${labelClause}
      ORDER BY ${orderClause}
    `).all(...params);
    const labels = cachedPrepare(db, `
      SELECT DISTINCT dominant_label
      FROM cluster_summary
      WHERE cluster_run_id = ?
      ORDER BY dominant_label
    `).all(clusterRunId).map((row) => row.dominant_label);
    return {
      clusters: clusters.map((row) => ({
        cluster_id: Number(row.cluster_id),
        size: Number(row.size),
        representative_result_id: Number(row.representative_result_id),
        label_distribution: parseLabelDistribution(row.label_distribution_json, row.size),
        dominant_label: String(row.dominant_label || ''),
        avg_intra_distance: Number(row.avg_intra_distance || 0),
      })),
      labels: ['all', ...labels],
    };
  } finally {
    db.close();
  }
};

const parseLabelDistribution = (json, size) => {
  let parsed;
  try {
    parsed = JSON.parse(String(json || '{}'));
  } catch {
    parsed = {};
  }
  const total = Number(size || 0) || 1;
  const out = {};
  for (const [key, value] of Object.entries(parsed)) {
    const count = Number(value || 0);
    out[key] = { count, fraction: count / total };
  }
  return out;
};

const fetchBboxRows = (sourceDb, resultIds) => {
  if (resultIds.length === 0) return new Map();
  const map = new Map();
  for (let i = 0; i < resultIds.length; i += IN_CHUNK_SIZE) {
    const chunk = resultIds.slice(i, i + IN_CHUNK_SIZE);
    const placeholders = chunk.map(() => '?').join(',');
    const rows = sourceDb.prepare(`
      SELECT res.result_id, res.image_rel_path, res.image_path,
             src_run.input_dir AS source_input_dir,
             res.predicted_label, res.detector_score, res.predicted_probability_pct,
             res.x1, res.y1, res.x2, res.y2,
             img.width AS image_width, img.height AS image_height
      FROM openclip_semantic_results res
      JOIN images img ON img.image_id = res.image_id
      LEFT JOIN runs src_run ON src_run.run_id = res.source_run_id
      WHERE res.result_id IN (${placeholders})
    `).all(...chunk);
    for (const row of rows) map.set(Number(row.result_id), row);
  }
  return map;
};

export const getClusterMembers = (payload = {}) => {
  const clusterDbPath = cleanPath(payload.clusterDbPath, 'clusterDbPath');
  const sourceDbPath = cleanPath(payload.sourceDbPath, 'sourceDbPath');
  const clusterRunId = cleanPath(payload.clusterRunId, 'clusterRunId');
  const clusterId = Number(payload.clusterId);
  const imageRootPath = String(payload.imageRootPath || '').trim();
  const memberLimit = Math.max(1, Math.min(500, Number(payload.memberLimit || 48)));

  if (!Number.isFinite(clusterId)) throw new Error('Invalid clusterId');

  const clusterDb = connectRo(clusterDbPath);
  let rows;
  let summary;
  try {
    rows = cachedPrepare(clusterDb, `
      SELECT result_id, image_rel_path, predicted_label,
             distance_to_centroid, is_representative, rank_in_cluster
      FROM cluster_results
      WHERE cluster_run_id = ? AND cluster_id = ?
      ORDER BY rank_in_cluster ASC
    `).all(clusterRunId, clusterId);
    summary = cachedPrepare(clusterDb, `
      SELECT size, representative_result_id, label_distribution_json,
             dominant_label, avg_intra_distance
      FROM cluster_summary
      WHERE cluster_run_id = ? AND cluster_id = ?
    `).get(clusterRunId, clusterId);
  } finally {
    clusterDb.close();
  }

  if (!summary) {
    return { cluster: null, representatives: [], members: [] };
  }

  const sourceDb = connectRo(sourceDbPath);
  let bboxMap;
  try {
    const ids = rows.map((row) => Number(row.result_id));
    bboxMap = fetchBboxRows(sourceDb, ids);
  } finally {
    sourceDb.close();
  }

  const enriched = rows.map((row) => {
    const bbox = bboxMap.get(Number(row.result_id));
    const imagePath = bbox ? resolveImagePath(bbox, imageRootPath) : '';
    return {
      result_id: Number(row.result_id),
      image_rel_path: String(row.image_rel_path || ''),
      predicted_label: String(row.predicted_label || ''),
      distance_to_centroid: Number(row.distance_to_centroid || 0),
      is_representative: Boolean(row.is_representative),
      rank_in_cluster: Number(row.rank_in_cluster || 0),
      x1: bbox ? Number(bbox.x1) : 0,
      y1: bbox ? Number(bbox.y1) : 0,
      x2: bbox ? Number(bbox.x2) : 0,
      y2: bbox ? Number(bbox.y2) : 0,
      image_width: bbox ? Number(bbox.image_width) : 0,
      image_height: bbox ? Number(bbox.image_height) : 0,
      detector_score: bbox ? Number(bbox.detector_score || 0) : 0,
      semantic_pct: bbox ? Number(bbox.predicted_probability_pct || 0) : 0,
      image_uri: imageUriForPath(imagePath),
    };
  });

  return {
    cluster: {
      cluster_id: clusterId,
      size: Number(summary.size),
      dominant_label: String(summary.dominant_label || ''),
      avg_intra_distance: Number(summary.avg_intra_distance || 0),
      label_distribution: parseLabelDistribution(summary.label_distribution_json, summary.size),
      representative_result_id: Number(summary.representative_result_id),
    },
    representatives: enriched.filter((row) => row.is_representative),
    members: enriched.slice(0, memberLimit),
    total_members: enriched.length,
  };
};

export const getBoxImage = (payload = {}) => {
  const sourceDbPath = cleanPath(payload.sourceDbPath, 'sourceDbPath');
  const resultId = Number(payload.resultId);
  const imageRootPath = String(payload.imageRootPath || '').trim();
  if (!Number.isFinite(resultId)) throw new Error('Invalid resultId');

  const sourceDb = connectRo(sourceDbPath);
  try {
    const bbox = sourceDb.prepare(`
      SELECT res.result_id, res.image_rel_path, res.image_path,
             src_run.input_dir AS source_input_dir,
             res.predicted_label, res.detector_score, res.predicted_probability_pct,
             res.x1, res.y1, res.x2, res.y2,
             img.width AS image_width, img.height AS image_height
      FROM openclip_semantic_results res
      JOIN images img ON img.image_id = res.image_id
      LEFT JOIN runs src_run ON src_run.run_id = res.source_run_id
      WHERE res.result_id = ?
    `).get(resultId);
    if (!bbox) throw new Error(`result_id not found: ${resultId}`);
    const imagePath = resolveImagePath(bbox, imageRootPath);
    return {
      result_id: Number(bbox.result_id),
      image_rel_path: String(bbox.image_rel_path || ''),
      predicted_label: String(bbox.predicted_label || ''),
      x1: Number(bbox.x1),
      y1: Number(bbox.y1),
      x2: Number(bbox.x2),
      y2: Number(bbox.y2),
      image_width: Number(bbox.image_width),
      image_height: Number(bbox.image_height),
      image_uri: imageUriForPath(imagePath),
    };
  } finally {
    sourceDb.close();
  }
};
