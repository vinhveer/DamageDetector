import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { connectRo, expandHome } from '../dedup_groups/db.js';
import { SEMI_LABELING_DEFAULTS } from '../defaults.js';

const IN_CHUNK = 900;

const cleanPath = (value, fieldName) => {
  const raw = String(value || '').trim();
  if (!raw || raw.includes('\0')) throw new Error(`Invalid ${fieldName}`);
  return raw;
};

const defaultResultsDir = () => SEMI_LABELING_DEFAULTS.classifier.resultsDir;
const defaultSourceDbPath = () => SEMI_LABELING_DEFAULTS.classifier.sourceDbPath;
const defaultImageRootPath = () => SEMI_LABELING_DEFAULTS.classifier.imageRootPath;

const safeReadJson = (filePath) => {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
};

const imageUriForPath = (p) => p ? pathToFileURL(p).toString() : '';

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

export const classifierResultsDefaults = () => ({
  resultsDir: defaultResultsDir(),
  sourceDbPath: defaultSourceDbPath(),
  imageRootPath: defaultImageRootPath(),
});

export const listClassifierRuns = (payload = {}) => {
  const resultsDir = String(payload.resultsDir || '').trim() || defaultResultsDir();
  if (!fs.existsSync(resultsDir)) {
    return { applies: [], trainings: [], resultsDir };
  }
  const entries = fs.readdirSync(resultsDir, { withFileTypes: true });
  const applies = [];
  const trainings = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
    const full = path.join(resultsDir, entry.name);
    const data = safeReadJson(full);
    if (!data) continue;
    if (entry.name.startsWith('predictions_needs_split_')) {
      applies.push({
        run_id: String(data.run_id || ''),
        created_at_utc: String(data.created_at_utc || ''),
        model_path: String(data.model_path || ''),
        session_path: String(data.session_path || ''),
        cluster_run_id: String(data.cluster_run_id || ''),
        n_predictions: Number(data.n_predictions || 0),
        class_distribution: data.class_distribution || {},
        confidence_buckets: data.confidence_buckets || {},
        low_conf_count: Number(data.low_conf_count || 0),
        cv: data.cv || null,
        file: full,
      });
    } else if (entry.name.startsWith('metrics_')) {
      const mlp = data.mlp || {};
      const lr = data.logreg || {};
      trainings.push({
        run_id: String(data.run_id || ''),
        created_at_utc: String(data.created_at_utc || ''),
        classes: data.classes || [],
        best_model: String(data.best_model || ''),
        test_accuracy_mlp: Number(mlp.test_accuracy || 0),
        test_accuracy_lr: Number(lr.test_accuracy || 0),
        n_total_labeled: Number(data.n_total_labeled || 0),
        model_path: String(data.model_path || ''),
        file: full,
      });
    }
  }
  applies.sort((a, b) => String(b.created_at_utc).localeCompare(String(a.created_at_utc)));
  trainings.sort((a, b) => String(b.created_at_utc).localeCompare(String(a.created_at_utc)));
  return { applies, trainings, resultsDir };
};

const findTrainingForApply = (resultsDir, modelPath) => {
  if (!modelPath) return null;
  const base = path.basename(String(modelPath));
  const match = base.match(/classifier_([0-9a-f]+)\.joblib/i);
  if (!match) return null;
  const metricsPath = path.join(resultsDir, `metrics_${match[1]}.json`);
  if (!fs.existsSync(metricsPath)) return null;
  return safeReadJson(metricsPath);
};

const fetchBboxRows = (sourceDb, resultIds) => {
  if (resultIds.length === 0) return new Map();
  const map = new Map();
  for (let i = 0; i < resultIds.length; i += IN_CHUNK) {
    const chunk = resultIds.slice(i, i + IN_CHUNK);
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

export const getApplyRun = (payload = {}) => {
  const resultsDir = String(payload.resultsDir || '').trim() || defaultResultsDir();
  const sourceDbPath = cleanPath(payload.sourceDbPath, 'sourceDbPath');
  const imageRootPath = String(payload.imageRootPath || '').trim();
  const file = cleanPath(payload.file, 'file');
  const applyData = safeReadJson(file);
  if (!applyData) throw new Error(`Apply file not readable: ${file}`);
  const metricsData = findTrainingForApply(resultsDir, applyData.model_path);

  const predictions = Array.isArray(applyData.predictions) ? applyData.predictions : [];
  const ids = predictions.map((p) => Number(p.result_id));
  const sourceDb = connectRo(sourceDbPath);
  let bboxMap;
  try {
    bboxMap = fetchBboxRows(sourceDb, ids);
  } finally {
    sourceDb.close();
  }

  const enriched = predictions.map((p) => {
    const bbox = bboxMap.get(Number(p.result_id));
    const imagePath = bbox ? resolveImagePath(bbox, imageRootPath) : '';
    return {
      result_id: Number(p.result_id),
      cluster_id: Number(p.cluster_id || 0),
      image_rel_path: String(p.image_rel_path || ''),
      predicted_label_step2: String(p.predicted_label_step2 || ''),
      predicted_class: String(p.predicted_class || ''),
      confidence: Number(p.confidence || 0),
      probabilities: p.probabilities || {},
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
    apply: {
      run_id: String(applyData.run_id || ''),
      created_at_utc: String(applyData.created_at_utc || ''),
      model_path: String(applyData.model_path || ''),
      session_path: String(applyData.session_path || ''),
      cluster_run_id: String(applyData.cluster_run_id || ''),
      embedding_run_id: String(applyData.embedding_run_id || ''),
      n_predictions: Number(applyData.n_predictions || 0),
      class_distribution: applyData.class_distribution || {},
      confidence_buckets: applyData.confidence_buckets || {},
      low_conf_threshold: Number(applyData.low_conf_threshold || 0),
      low_conf_count: Number(applyData.low_conf_count || 0),
      cv: applyData.cv || null,
    },
    metrics: metricsData,
    predictions: enriched,
  };
};

export const getTrainingRun = (payload = {}) => {
  const file = cleanPath(payload.file, 'file');
  const data = safeReadJson(file);
  if (!data) throw new Error(`Metrics file not readable: ${file}`);
  return { metrics: data };
};
