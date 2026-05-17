import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { connectRo } from '../result_viewer/db.js';
import { resolveImagePath } from '../result_viewer/images.js';

const expandHome = (value) => {
  if (!value || !value.startsWith('~')) return value || '';
  return path.join(process.env.HOME || process.env.USERPROFILE || '', value.slice(1));
};

const repoRoot = () => path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', '..', '..');

const labRoot = () => path.dirname(repoRoot());

export const defaultPrototypeReviewDb = () => path.join(labRoot(), 'infer_results', 'semi-labeling', 'step5_prototype_review', 'prototype_review.sqlite3');

export const prototypeReviewDefaults = () => ({
  reviewDbPath: defaultPrototypeReviewDb(),
  imageRootPath: path.join(labRoot(), 'HinhAnh')
});

export const listPrototypeReviewRuns = (payload) => {
  const reviewDbPath = path.resolve(expandHome(payload.reviewDbPath || defaultPrototypeReviewDb()));
  if (!fs.existsSync(reviewDbPath)) return { runs: [] };
  const db = connectRo(reviewDbPath);
  try {
    const runs = db.prepare(`
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
    const scores = db.prepare(`
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

const sourceMeta = (sourceDbPath, resultIds) => {
  const resolvedSourceDbPath = path.resolve(expandHome(sourceDbPath));
  if (!sourceDbPath || !fs.existsSync(resolvedSourceDbPath) || resultIds.length === 0) return new Map();

  const placeholders = resultIds.map(() => '?').join(',');
  const db = connectRo(resolvedSourceDbPath);
  try {
    const rows = db.prepare(`
      SELECT res.result_id, res.image_path, src_run.input_dir AS source_input_dir,
             res.x1, res.y1, res.x2, res.y2
      FROM openclip_semantic_results res
      JOIN runs src_run ON src_run.run_id = res.source_run_id
      WHERE res.result_id IN (${placeholders})
    `).all(...resultIds);
    return new Map(rows.map((row) => [Number(row.result_id), row]));
  } finally {
    db.close();
  }
};

const votesById = (reviewDbPath, reviewRunId, resultIds) => {
  if (resultIds.length === 0) return new Map();
  const placeholders = resultIds.map(() => '?').join(',');
  const db = connectRo(reviewDbPath);
  try {
    const rows = db.prepare(`
      SELECT result_id, vote_label, vote_score
      FROM prototype_assignment_votes
      WHERE review_run_id = ? AND result_id IN (${placeholders})
    `).all(reviewRunId, ...resultIds);
    return new Map(rows.map((row) => [Number(row.result_id), row]));
  } finally {
    db.close();
  }
};

export const listPrototypeReviewAssignments = (payload) => {
  const featureDbPath = payload.featureDbPath;
  const db = connectRo(featureDbPath);
  let assignments;
  try {
    assignments = db.prepare(`
      SELECT result_id, image_rel_path, predicted_label, predicted_probability_pct,
             detector_score, cluster_key, is_outlier, distance_to_center,
             suggested_label, label_suspect, cluster_purity, cluster_size
      FROM feature_group_assignments
      WHERE grouping_run_id = ? AND cluster_key = ?
      ORDER BY distance_to_center ASC, predicted_probability_pct ASC, result_id
    `).all(payload.groupingRunId, payload.clusterKey);
  } finally {
    db.close();
  }

  const resultIds = assignments.map((row) => Number(row.result_id));
  const metaById = sourceMeta(payload.sourceDbPath || '', resultIds);
  const voteById = votesById(payload.reviewDbPath, payload.reviewRunId, resultIds);
  const imageRoot = payload.imageRootPath || '';
  const nextAssignments = assignments.map((row) => {
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
      image_uri: resolved ? pathToFileURL(resolved).href : ''
    };
  });
  return { assignments: nextAssignments };
};
