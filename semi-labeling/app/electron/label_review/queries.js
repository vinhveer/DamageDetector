import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { cachedPrepare, connectRo, expandHome, resolveDbPath } from '../dedup_groups/db.js';
import { SEMI_LABELING_DEFAULTS } from '../defaults.js';

const IN_CHUNK_SIZE = 900;

const cleanPath = (value, fieldName) => {
  const raw = String(value || '').trim();
  if (!raw || raw.includes('\0')) throw new Error(`Invalid ${fieldName}`);
  return raw;
};

const defaultSubclusterDbPath = () => SEMI_LABELING_DEFAULTS.labelReview.subclusterDbPath;
const defaultSuspectClusterDbPath = () => SEMI_LABELING_DEFAULTS.labelReview.suspectClusterDbPath;
const defaultSourceDbPath = () => SEMI_LABELING_DEFAULTS.labelReview.sourceDbPath;
const defaultImageRootPath = () => SEMI_LABELING_DEFAULTS.labelReview.imageRootPath;
const defaultSessionsDir = () => SEMI_LABELING_DEFAULTS.labelReview.sessionsDir;
const cvOofDir = () => SEMI_LABELING_DEFAULTS.labelReview.cvOofDir;

// Cache: cv_oof file path → Map(result_id → {predicted_label, suspicion_score, is_suspect, predicted_conf})
const cvOofCache = new Map();

const pickLatestCvOof = () => {
  const dir = cvOofDir();
  if (!fs.existsSync(dir)) return null;
  const candidates = fs.readdirSync(dir)
    .filter((name) => name.startsWith('cv_oof_') && name.endsWith('.json'))
    .map((name) => path.join(dir, name));
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  return candidates[0];
};

const loadCvOofMap = (filePath) => {
  if (!filePath) return null;
  const cached = cvOofCache.get(filePath);
  if (cached) return cached;
  const raw = fs.readFileSync(filePath, 'utf8');
  const data = JSON.parse(raw);
  const map = new Map();
  for (const rec of data.records || []) {
    map.set(Number(rec.result_id), {
      predicted_label: String(rec.predicted_label || ''),
      predicted_conf: Number(rec.predicted_conf || 0),
      suspicion_score: Number(rec.suspicion_score || 0),
      is_suspect: Boolean(rec.is_suspect),
    });
  }
  cvOofCache.set(filePath, map);
  return map;
};

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

export const labelReviewDefaults = () => ({
  subclusterDbPath: defaultSubclusterDbPath(),
  suspectClusterDbPath: defaultSuspectClusterDbPath(),
  sourceDbPath: defaultSourceDbPath(),
  imageRootPath: defaultImageRootPath(),
  sessionsDir: defaultSessionsDir(),
});

export const listSubclusterRuns = (payload = {}) => {
  const subclusterDbPath = cleanPath(payload.subclusterDbPath, 'subclusterDbPath');
  const db = connectRo(subclusterDbPath);
  try {
    const runs = cachedPrepare(db, `
      SELECT subcluster_run_id, created_at_utc, source_csv, embedding_run_id,
             algorithm, total_boxes, n_classes,
             hdbscan_min_cluster_size, hdbscan_min_samples, options_json
      FROM subcluster_runs
      ORDER BY created_at_utc DESC, subcluster_run_id DESC
    `).all();
    const classes = cachedPrepare(db, `
      SELECT class_name, SUM(size) AS total
      FROM subcluster_summary
      GROUP BY class_name
      ORDER BY class_name
    `).all().map((row) => ({ class_name: String(row.class_name), total: Number(row.total || 0) }));
    return {
      runs: runs.map((row) => ({
        subcluster_run_id: String(row.subcluster_run_id),
        created_at_utc: String(row.created_at_utc),
        source_csv: String(row.source_csv),
        embedding_run_id: String(row.embedding_run_id),
        algorithm: String(row.algorithm),
        total_boxes: Number(row.total_boxes),
        n_classes: Number(row.n_classes),
        hdbscan_min_cluster_size: Number(row.hdbscan_min_cluster_size),
        hdbscan_min_samples: Number(row.hdbscan_min_samples),
        options_json: String(row.options_json),
      })),
      class_totals: classes,
      subclusterDbPath: resolveDbPath(subclusterDbPath),
    };
  } finally {
    db.close();
  }
};

export const listSubclustersByClass = (payload = {}) => {
  const subclusterDbPath = cleanPath(payload.subclusterDbPath, 'subclusterDbPath');
  const subclusterRunId = cleanPath(payload.subclusterRunId, 'subclusterRunId');
  const className = cleanPath(payload.className, 'className');
  const sortBy = String(payload.sortBy || 'suspect').trim();

  const cvMap = loadCvOofMap(pickLatestCvOof());

  const db = connectRo(subclusterDbPath);
  try {
    const summaries = cachedPrepare(db, `
      SELECT sub_cluster_id, size, representative_result_id,
             avg_intra_distance, is_noise_cluster
      FROM subcluster_summary
      WHERE subcluster_run_id = ? AND class_name = ?
    `).all(subclusterRunId, className);

    const classes = cachedPrepare(db, `
      SELECT class_name, SUM(size) AS total
      FROM subcluster_summary
      WHERE subcluster_run_id = ?
      GROUP BY class_name
      ORDER BY class_name
    `).all(subclusterRunId).map((row) => ({
      class_name: String(row.class_name),
      total: Number(row.total || 0),
    }));

    let suspectsByGroup = new Map();
    if (cvMap) {
      // Pull all members for this class in one query, then aggregate suspect counts.
      const memberRows = cachedPrepare(db, `
        SELECT sub_cluster_id, result_id
        FROM subcluster_results
        WHERE subcluster_run_id = ? AND class_name = ?
      `).all(subclusterRunId, className);
      for (const row of memberRows) {
        const sid = Number(row.sub_cluster_id);
        const cv = cvMap.get(Number(row.result_id));
        const isSuspect = Boolean(cv?.is_suspect);
        const entry = suspectsByGroup.get(sid) || { count: 0 };
        if (isSuspect) entry.count += 1;
        suspectsByGroup.set(sid, entry);
      }
    }

    const items = summaries.map((row) => {
      const sid = Number(row.sub_cluster_id);
      const size = Number(row.size);
      const suspect_count = suspectsByGroup.get(sid)?.count || 0;
      const suspect_pct = size > 0 ? suspect_count / size : 0;
      return {
        sub_cluster_id: sid,
        size,
        representative_result_id: Number(row.representative_result_id),
        avg_intra_distance: Number(row.avg_intra_distance || 0),
        is_noise_cluster: Boolean(row.is_noise_cluster),
        suspect_count,
        suspect_pct,
      };
    });

    const cmp = (a, b) => {
      if (a.is_noise_cluster !== b.is_noise_cluster) return a.is_noise_cluster ? 1 : -1;
      if (sortBy === 'sub_cluster_id') return a.sub_cluster_id - b.sub_cluster_id;
      if (sortBy === 'distance') return a.avg_intra_distance - b.avg_intra_distance || b.size - a.size;
      if (sortBy === 'size') return b.size - a.size || a.sub_cluster_id - b.sub_cluster_id;
      // 'suspect' (default): suspect_pct desc, then suspect_count desc, then size desc
      return b.suspect_pct - a.suspect_pct
        || b.suspect_count - a.suspect_count
        || b.size - a.size;
    };
    items.sort(cmp);

    return {
      classes,
      subclusters: items,
      has_cv_data: Boolean(cvMap),
    };
  } finally {
    db.close();
  }
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

export const getSubclusterMembers = (payload = {}) => {
  const subclusterDbPath = cleanPath(payload.subclusterDbPath, 'subclusterDbPath');
  const sourceDbPath = cleanPath(payload.sourceDbPath, 'sourceDbPath');
  const subclusterRunId = cleanPath(payload.subclusterRunId, 'subclusterRunId');
  const className = cleanPath(payload.className, 'className');
  const subClusterId = Number(payload.subClusterId);
  const imageRootPath = String(payload.imageRootPath || '').trim();
  const memberLimit = Math.max(1, Math.min(500, Number(payload.memberLimit || 60)));
  const suspectOnly = Boolean(payload.suspectOnly);

  if (!Number.isFinite(subClusterId)) throw new Error('Invalid subClusterId');

  const cvMap = loadCvOofMap(pickLatestCvOof());

  const subDb = connectRo(subclusterDbPath);
  let rows;
  let summary;
  try {
    rows = cachedPrepare(subDb, `
      SELECT result_id, is_noise, is_representative, rank_in_cluster, core_distance
      FROM subcluster_results
      WHERE subcluster_run_id = ? AND class_name = ? AND sub_cluster_id = ?
      ORDER BY is_representative DESC, rank_in_cluster ASC, result_id ASC
    `).all(subclusterRunId, className, subClusterId);
    summary = cachedPrepare(subDb, `
      SELECT size, representative_result_id, avg_intra_distance, is_noise_cluster
      FROM subcluster_summary
      WHERE subcluster_run_id = ? AND class_name = ? AND sub_cluster_id = ?
    `).get(subclusterRunId, className, subClusterId);
  } finally {
    subDb.close();
  }

  if (!summary) {
    return { subcluster: null, representatives: [], members: [], total_members: 0, suspect_count: 0 };
  }

  const total = rows.length;
  let suspectCount = 0;
  for (const row of rows) {
    const cv = cvMap?.get(Number(row.result_id));
    if (cv?.is_suspect) suspectCount += 1;
  }

  // Filter: if suspectOnly, restrict to suspect rows. Always keep reps too if they are suspects.
  let candidateRows = rows;
  if (suspectOnly && cvMap) {
    candidateRows = rows.filter((row) => Boolean(cvMap.get(Number(row.result_id))?.is_suspect));
  }

  const repRows = candidateRows.filter((row) => Number(row.is_representative) === 1);
  const nonRepRows = candidateRows.filter((row) => Number(row.is_representative) !== 1);

  // Sort non-reps by suspicion score desc when CV data is available, else preserve rank.
  if (cvMap) {
    nonRepRows.sort((a, b) => {
      const sa = cvMap.get(Number(a.result_id))?.suspicion_score || 0;
      const sb = cvMap.get(Number(b.result_id))?.suspicion_score || 0;
      return sb - sa;
    });
  }

  const remaining = Math.max(0, memberLimit - repRows.length);
  let sampledNonReps;
  if (nonRepRows.length <= remaining) {
    sampledNonReps = nonRepRows;
  } else if (cvMap || suspectOnly) {
    sampledNonReps = nonRepRows.slice(0, remaining);
  } else {
    const stride = Math.max(1, Math.ceil(nonRepRows.length / Math.max(1, remaining)));
    sampledNonReps = nonRepRows.filter((_, idx) => idx % stride === 0).slice(0, remaining);
  }
  const memberRows = [...repRows, ...sampledNonReps];

  const sourceDb = connectRo(sourceDbPath);
  let bboxMap;
  try {
    const ids = memberRows.map((row) => Number(row.result_id));
    bboxMap = fetchBboxRows(sourceDb, ids);
  } finally {
    sourceDb.close();
  }

  const enriched = memberRows.map((row) => {
    const bbox = bboxMap.get(Number(row.result_id));
    const imagePath = bbox ? resolveImagePath(bbox, imageRootPath) : '';
    const cv = cvMap?.get(Number(row.result_id));
    return {
      result_id: Number(row.result_id),
      image_rel_path: bbox ? String(bbox.image_rel_path || '') : '',
      predicted_label: bbox ? String(bbox.predicted_label || className) : className,
      current_label: className,
      is_noise: Boolean(row.is_noise),
      is_representative: Number(row.is_representative) === 1,
      rank_in_cluster: Number(row.rank_in_cluster || 0),
      core_distance: Number(row.core_distance || 0),
      is_suspect: Boolean(cv?.is_suspect),
      cv_predicted_label: cv ? String(cv.predicted_label || '') : '',
      cv_predicted_conf: cv ? Number(cv.predicted_conf || 0) : 0,
      suspicion_score: cv ? Number(cv.suspicion_score || 0) : 0,
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
    subcluster: {
      class_name: className,
      sub_cluster_id: subClusterId,
      size: Number(summary.size),
      representative_result_id: Number(summary.representative_result_id),
      avg_intra_distance: Number(summary.avg_intra_distance || 0),
      is_noise_cluster: Boolean(summary.is_noise_cluster),
      suspect_count: suspectCount,
      suspect_pct: total > 0 ? suspectCount / total : 0,
    },
    representatives: enriched.filter((row) => row.is_representative),
    members: enriched,
    total_members: total,
    suspect_count: suspectCount,
    has_cv_data: Boolean(cvMap),
  };
};

// ──────────────────────────────────────────────────────────────────────────
// Suspect-clusters (output of cluster_suspects.py)
// ──────────────────────────────────────────────────────────────────────────

export const listSuspectRuns = (payload = {}) => {
  const dbPath = cleanPath(payload.suspectClusterDbPath, 'suspectClusterDbPath');
  const db = connectRo(dbPath);
  try {
    const runs = cachedPrepare(db, `
      SELECT run_id, created_at_utc, source_oof, embedding_run_id,
             total_suspects, n_clusters, options_json
      FROM suspect_cluster_runs
      ORDER BY created_at_utc DESC, run_id DESC
    `).all();
    const classes = cachedPrepare(db, `
      SELECT current_label, COUNT(*) AS n_clusters, SUM(size) AS total_suspects
      FROM suspect_cluster_summary
      GROUP BY current_label
      ORDER BY current_label
    `).all().map((row) => ({
      current_label: String(row.current_label),
      n_clusters: Number(row.n_clusters || 0),
      total_suspects: Number(row.total_suspects || 0),
    }));
    return {
      runs: runs.map((row) => ({
        run_id: String(row.run_id),
        created_at_utc: String(row.created_at_utc),
        embedding_run_id: String(row.embedding_run_id),
        total_suspects: Number(row.total_suspects),
        n_clusters: Number(row.n_clusters),
        options_json: String(row.options_json),
      })),
      class_totals: classes,
      suspectClusterDbPath: resolveDbPath(dbPath),
    };
  } finally {
    db.close();
  }
};

export const listSuspectClusters = (payload = {}) => {
  const dbPath = cleanPath(payload.suspectClusterDbPath, 'suspectClusterDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const currentLabel = cleanPath(payload.currentLabel, 'currentLabel');

  const db = connectRo(dbPath);
  try {
    const rows = cachedPrepare(db, `
      SELECT cluster_id, size, representative_result_id,
             dominant_cv_label, dominant_cv_fraction, avg_suspicion_score,
             suggested_action, suggested_target_label, is_noise_cluster
      FROM suspect_cluster_summary
      WHERE run_id = ? AND current_label = ?
      ORDER BY is_noise_cluster ASC, size DESC, cluster_id ASC
    `).all(runId, currentLabel);

    const classes = cachedPrepare(db, `
      SELECT current_label, COUNT(*) AS n_clusters, SUM(size) AS total_suspects
      FROM suspect_cluster_summary
      WHERE run_id = ?
      GROUP BY current_label
      ORDER BY current_label
    `).all(runId).map((row) => ({
      current_label: String(row.current_label),
      n_clusters: Number(row.n_clusters || 0),
      total_suspects: Number(row.total_suspects || 0),
    }));

    return {
      classes,
      clusters: rows.map((row) => ({
        cluster_id: Number(row.cluster_id),
        size: Number(row.size),
        representative_result_id: Number(row.representative_result_id),
        dominant_cv_label: String(row.dominant_cv_label || ''),
        dominant_cv_fraction: Number(row.dominant_cv_fraction || 0),
        avg_suspicion_score: Number(row.avg_suspicion_score || 0),
        suggested_action: String(row.suggested_action || ''),
        suggested_target_label: String(row.suggested_target_label || ''),
        is_noise_cluster: Boolean(row.is_noise_cluster),
      })),
    };
  } finally {
    db.close();
  }
};

export const getSuspectClusterMembers = (payload = {}) => {
  const dbPath = cleanPath(payload.suspectClusterDbPath, 'suspectClusterDbPath');
  const sourceDbPath = cleanPath(payload.sourceDbPath, 'sourceDbPath');
  const runId = cleanPath(payload.runId, 'runId');
  const currentLabel = cleanPath(payload.currentLabel, 'currentLabel');
  const clusterId = Number(payload.clusterId);
  const imageRootPath = String(payload.imageRootPath || '').trim();
  const memberLimit = Math.max(1, Math.min(500, Number(payload.memberLimit || 80)));

  if (!Number.isFinite(clusterId)) throw new Error('Invalid clusterId');

  const db = connectRo(dbPath);
  let rows;
  let summary;
  try {
    rows = cachedPrepare(db, `
      SELECT result_id, is_representative, rank_in_cluster, suspicion_score,
             cv_predicted_label, cv_predicted_conf
      FROM suspect_cluster_results
      WHERE run_id = ? AND current_label = ? AND cluster_id = ?
      ORDER BY is_representative DESC, rank_in_cluster ASC
    `).all(runId, currentLabel, clusterId);
    summary = cachedPrepare(db, `
      SELECT size, representative_result_id, dominant_cv_label, dominant_cv_fraction,
             avg_suspicion_score, suggested_action, suggested_target_label, is_noise_cluster
      FROM suspect_cluster_summary
      WHERE run_id = ? AND current_label = ? AND cluster_id = ?
    `).get(runId, currentLabel, clusterId);
  } finally {
    db.close();
  }

  if (!summary) {
    return { cluster: null, representatives: [], members: [], total_members: 0 };
  }

  const total = rows.length;
  const limited = rows.slice(0, memberLimit);

  const sourceDb = connectRo(sourceDbPath);
  let bboxMap;
  try {
    const ids = limited.map((row) => Number(row.result_id));
    bboxMap = fetchBboxRows(sourceDb, ids);
  } finally {
    sourceDb.close();
  }

  const enriched = limited.map((row) => {
    const bbox = bboxMap.get(Number(row.result_id));
    const imagePath = bbox ? resolveImagePath(bbox, imageRootPath) : '';
    return {
      result_id: Number(row.result_id),
      image_rel_path: bbox ? String(bbox.image_rel_path || '') : '',
      current_label: currentLabel,
      predicted_label: bbox ? String(bbox.predicted_label || currentLabel) : currentLabel,
      is_representative: Number(row.is_representative) === 1,
      rank_in_cluster: Number(row.rank_in_cluster || 0),
      is_suspect: true,
      cv_predicted_label: String(row.cv_predicted_label || ''),
      cv_predicted_conf: Number(row.cv_predicted_conf || 0),
      suspicion_score: Number(row.suspicion_score || 0),
      x1: bbox ? Number(bbox.x1) : 0,
      y1: bbox ? Number(bbox.y1) : 0,
      x2: bbox ? Number(bbox.x2) : 0,
      y2: bbox ? Number(bbox.y2) : 0,
      image_width: bbox ? Number(bbox.image_width) : 0,
      image_height: bbox ? Number(bbox.image_height) : 0,
      image_uri: imageUriForPath(imagePath),
    };
  });

  return {
    cluster: {
      current_label: currentLabel,
      cluster_id: clusterId,
      size: Number(summary.size),
      representative_result_id: Number(summary.representative_result_id),
      dominant_cv_label: String(summary.dominant_cv_label || ''),
      dominant_cv_fraction: Number(summary.dominant_cv_fraction || 0),
      avg_suspicion_score: Number(summary.avg_suspicion_score || 0),
      suggested_action: String(summary.suggested_action || ''),
      suggested_target_label: String(summary.suggested_target_label || ''),
      is_noise_cluster: Boolean(summary.is_noise_cluster),
    },
    representatives: enriched.filter((row) => row.is_representative),
    members: enriched,
    total_members: total,
  };
};
