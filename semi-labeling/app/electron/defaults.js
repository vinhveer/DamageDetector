import path from 'node:path';
import { fileURLToPath } from 'node:url';

// electron/ → app/ → semi-labeling/ → DamageDetector/ → Lab/
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const LAB_ROOT = path.resolve(__dirname, '..', '..', '..', '..');

const semi = (...parts) => path.join(LAB_ROOT, 'infer_results', 'semi-labeling', ...parts);

const IMAGE_ROOT = path.join(LAB_ROOT, 'data', 'HinhAnh');
const SOURCE_DB = semi('step2_sematic', 'damage_scan.sqlite3');
const DEFAULT_KEEPER_LABELS = ['all', 'crack', 'spall', 'mold'];

/**
 * Centralized default paths for the standalone semi-labeling review app.
 * Each IPC handler's `getXxxDefaults()` reads from here. Users can still
 * override any path in Settings / the per-tab path inputs.
 */
export const SEMI_LABELING_DEFAULTS = {
  labRoot: LAB_ROOT,
  imageRoot: IMAGE_ROOT,
  sourceDbPath: SOURCE_DB,

  // Step 4 · Dedup
  dedup: {
    dedupDbPath: semi('step4_class_aware_dedup', 'dedup.sqlite3'),
    sourceDbPath: SOURCE_DB,
    imageRootPath: IMAGE_ROOT,
    labels: DEFAULT_KEEPER_LABELS,
  },

  // Step 5 · Cluster
  cluster: {
    clusterDbPath: semi('step5_clustering', 'clusters.sqlite3'),
    sourceDbPath: SOURCE_DB,
    imageRootPath: IMAGE_ROOT,
    sessionsDir: semi('step5_clustering', 'sessions'),
  },

  // Step 6 · Classifier
  classifier: {
    resultsDir: semi('step6_classifier'),
    sourceDbPath: SOURCE_DB,
    imageRootPath: IMAGE_ROOT,
  },

  // Step 7 · Review
  labelReview: {
    subclusterDbPath: semi('step7_label_review', 'subclusters.sqlite3'),
    suspectClusterDbPath: semi('step7_label_review', 'suspect_clusters.sqlite3'),
    sourceDbPath: SOURCE_DB,
    imageRootPath: IMAGE_ROOT,
    sessionsDir: semi('step7_label_review', 'sessions'),
    cvOofDir: semi('step7_label_review'),
  },

  // Step 8 · Final
  finalReview: {
    step6Dir: semi('step6_classifier'),
    step7Dir: semi('step7_label_review'),
    sourceDbPath: SOURCE_DB,
    imageRootPath: IMAGE_ROOT,
  },
};

export default SEMI_LABELING_DEFAULTS;
