import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const LABELS = ['crack', 'mold', 'spall'];

export const repoRoot = () => path.resolve(__dirname, '..', '..', '..');

export const labRoot = () => path.dirname(repoRoot());

export const defaultSourceDb = () => path.join(labRoot(), 'infer_results', 'semi-labeling', '2_sematic', 'damage_scan.sqlite3');

export const defaultFeatureDb = () => {
  const sourceDir = path.dirname(defaultSourceDb());
  const giant = path.join(sourceDir, 'step4_feature_grouping_giant_agglo045_full_restart', 'feature_groups.sqlite3');
  if (fs.existsSync(giant)) return giant;
  return path.join(sourceDir, 'step4_feature_grouping', 'feature_groups.sqlite3');
};

export const defaultImageRoot = () => path.join(labRoot(), 'HinhAnh');
