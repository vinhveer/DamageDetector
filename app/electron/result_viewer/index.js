import { LABELS, defaultFeatureDb, defaultImageRoot, defaultSourceDb } from './paths.js';
import { listAssignments, listClusters, listRuns } from './queries.js';
import { clearFlagsForCluster, clearFlagsForResults } from './mutations.js';

export const resultViewerDefaults = () => ({
  labels: LABELS,
  featureDbPath: defaultFeatureDb(),
  sourceDbPath: defaultSourceDb(),
  imageRootPath: defaultImageRoot()
});

export {
  clearFlagsForCluster,
  clearFlagsForResults,
  listAssignments,
  listClusters,
  listRuns
};
