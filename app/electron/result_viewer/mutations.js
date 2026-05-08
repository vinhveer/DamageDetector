import { connectRw, refreshRunFlagCounts, runTransaction } from './db.js';

export const clearFlagsForResults = (payload) => {
  const resultIds = (payload.resultIds || []).map(Number).filter(Number.isFinite);
  if (resultIds.length === 0) return { changed: 0 };

  const placeholders = resultIds.map(() => '?').join(',');
  const runId = payload.runId;
  const db = connectRw(payload.featureDbPath);
  try {
    const changed = runTransaction(db, () => {
      const before = db.prepare(`
        SELECT cluster_key, SUM(is_outlier) AS outliers
        FROM feature_group_assignments
        WHERE grouping_run_id = ? AND result_id IN (${placeholders})
        GROUP BY cluster_key
      `).all(runId, ...resultIds);

      const changed = db.prepare(`
        UPDATE feature_group_assignments
        SET is_outlier = 0, label_suspect = 0
        WHERE grouping_run_id = ?
          AND result_id IN (${placeholders})
          AND (is_outlier != 0 OR label_suspect != 0)
      `).run(runId, ...resultIds).changes;

      const updateCluster = db.prepare(`
        UPDATE feature_group_clusters
        SET outlier_count = MAX(0, outlier_count - ?)
        WHERE grouping_run_id = ? AND cluster_key = ?
      `);
      before.forEach((row) => {
        const outliers = Number(row.outliers || 0);
        if (outliers > 0) updateCluster.run(outliers, runId, row.cluster_key);
      });

      refreshRunFlagCounts(db, runId);
      return changed;
    });
    return { changed };
  } finally {
    db.close();
  }
};

export const clearFlagsForCluster = (payload) => {
  const db = connectRw(payload.featureDbPath);
  try {
    const changed = runTransaction(db, () => {
      const changed = db.prepare(`
        UPDATE feature_group_assignments
        SET is_outlier = 0, label_suspect = 0
        WHERE grouping_run_id = ? AND cluster_key = ?
          AND (is_outlier != 0 OR label_suspect != 0)
      `).run(payload.runId, payload.clusterKey).changes;

      db.prepare(`
        UPDATE feature_group_clusters
        SET outlier_count = 0
        WHERE grouping_run_id = ? AND cluster_key = ?
      `).run(payload.runId, payload.clusterKey);

      refreshRunFlagCounts(db, payload.runId);
      return changed;
    });
    return { changed };
  } finally {
    db.close();
  }
};
