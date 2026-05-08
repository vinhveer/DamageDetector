import fs from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';

const expandHome = (value) => {
  if (!value || !value.startsWith('~')) return value || '';
  return path.join(process.env.HOME || process.env.USERPROFILE || '', value.slice(1));
};

const resolveDbPath = (dbPath) => path.resolve(expandHome(dbPath));

const openDatabase = (dbPath, readOnly) => {
  const resolved = resolveDbPath(dbPath);
  if (!fs.existsSync(resolved)) {
    throw new Error(`SQLite database not found: ${resolved}`);
  }
  return new DatabaseSync(resolved, { readOnly });
};

export const connectRo = (dbPath) => openDatabase(dbPath, true);

export const connectRw = (dbPath) => openDatabase(dbPath, false);

export const runTransaction = (db, callback) => {
  db.exec('BEGIN IMMEDIATE');
  try {
    const result = callback();
    db.exec('COMMIT');
    return result;
  } catch (error) {
    db.exec('ROLLBACK');
    throw error;
  }
};

export const refreshRunFlagCounts = (db, runId) => {
  const totals = db.prepare(`
    SELECT SUM(is_outlier) AS outliers, SUM(label_suspect) AS suspects
    FROM feature_group_assignments
    WHERE grouping_run_id = ?
  `).get(runId) || {};

  db.prepare(`
    UPDATE feature_group_runs
    SET outlier_boxes = ?, label_suspect_boxes = ?
    WHERE grouping_run_id = ?
  `).run(Number(totals.outliers || 0), Number(totals.suspects || 0), runId);
};
