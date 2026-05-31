import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { connectRo, expandHome, resolveDbPath } from '../dedup_groups/db.js';
import { SEMI_LABELING_DEFAULTS } from '../defaults.js';

const IN_CHUNK_SIZE = 900;

const cleanPath = (value, fieldName) => {
  const raw = String(value || '').trim();
  if (!raw || raw.includes('\0')) throw new Error(`Invalid ${fieldName}`);
  return raw;
};

const defaultStep6Dir = () => SEMI_LABELING_DEFAULTS.finalReview.step6Dir;
const defaultStep7Dir = () => SEMI_LABELING_DEFAULTS.finalReview.step7Dir;
const defaultSourceDbPath = () => SEMI_LABELING_DEFAULTS.finalReview.sourceDbPath;
const defaultImageRootPath = () => SEMI_LABELING_DEFAULTS.finalReview.imageRootPath;

const pickLatest = (dir, prefix, suffix = '.csv') => {
  if (!fs.existsSync(dir)) return '';
  const matches = fs.readdirSync(dir)
    .filter((name) => name.startsWith(prefix) && name.endsWith(suffix))
    .map((name) => path.join(dir, name));
  if (matches.length === 0) return '';
  matches.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  return matches[0];
};

const parseCsv = (filePath) => {
  const text = fs.readFileSync(filePath, 'utf8');
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length === 0) return { headers: [], rows: [] };
  const headers = lines[0].split(',');
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(',');
    const row = {};
    for (let h = 0; h < headers.length; h += 1) {
      row[headers[h]] = cols[h] != null ? cols[h] : '';
    }
    rows.push(row);
  }
  return { headers, rows };
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

const imageUriForPath = (p) => (p ? pathToFileURL(p).toString() : '');

const safeRelPath = (value) => {
  const relPath = String(value || '').trim().replace(/\\/g, '/');
  if (!relPath || path.posix.isAbsolute(relPath)) return '';
  const normalized = path.posix.normalize(relPath);
  if (normalized === '.' || normalized.startsWith('../') || normalized === '..') return '';
  return normalized;
};

export const finalReviewDefaults = () => ({
  step6Dir: defaultStep6Dir(),
  step7Dir: defaultStep7Dir(),
  completeLabelsCsv: pickLatest(defaultStep6Dir(), 'complete_labels_'),
  finalLabelsCsv: pickLatest(defaultStep7Dir(), 'final_labels_'),
  sourceDbPath: defaultSourceDbPath(),
  imageRootPath: defaultImageRootPath(),
});

export const listFinalCsvs = (payload = {}) => {
  const step7Dir = String(payload.step7Dir || defaultStep7Dir()).trim();
  if (!fs.existsSync(step7Dir)) return { csvs: [] };
  const matches = fs.readdirSync(step7Dir)
    .filter((name) => name.startsWith('final_labels_') && name.endsWith('.csv'))
    .map((name) => {
      const full = path.join(step7Dir, name);
      const stat = fs.statSync(full);
      return { name, path: full, mtime_ms: stat.mtimeMs, size_bytes: stat.size };
    });
  matches.sort((a, b) => b.mtime_ms - a.mtime_ms);
  return { csvs: matches };
};

// Build per-image stats from the two CSVs
const buildImageStats = (completePath, finalPath) => {
  if (!completePath || !fs.existsSync(completePath)) {
    throw new Error(`complete_labels CSV not found: ${completePath}`);
  }
  if (!finalPath || !fs.existsSync(finalPath)) {
    throw new Error(`final_labels CSV not found: ${finalPath}`);
  }
  const complete = parseCsv(completePath);
  const finalC = parseCsv(finalPath);

  // Map result_id → {original, final, image_rel_path}
  const map = new Map();
  for (const row of complete.rows) {
    const rid = String(row.result_id || '').trim();
    if (!rid) continue;
    map.set(rid, {
      result_id: Number(rid),
      image_rel_path: String(row.image_rel_path || ''),
      original_label: String(row.final_class || ''),
      final_label: String(row.final_class || ''), // overridden below
      source: String(row.source || ''),
    });
  }
  for (const row of finalC.rows) {
    const rid = String(row.result_id || '').trim();
    if (!rid) continue;
    const existing = map.get(rid) || {
      result_id: Number(rid),
      image_rel_path: String(row.image_rel_path || ''),
      original_label: '',
      final_label: '',
      source: '',
    };
    existing.final_label = String(row.final_class || '');
    existing.source = String(row.source || existing.source);
    map.set(rid, existing);
  }

  // Group by image_rel_path
  const byImage = new Map();
  for (const rec of map.values()) {
    if (!rec.image_rel_path) continue;
    const key = rec.image_rel_path;
    const entry = byImage.get(key) || {
      image_rel_path: key,
      total: 0,
      changed: 0,
      kept: 0,
      counts_before: { crack: 0, mold: 0, spall: 0, reject: 0 },
      counts_after: { crack: 0, mold: 0, spall: 0, reject: 0 },
      result_ids: [],
    };
    entry.total += 1;
    if (rec.original_label !== rec.final_label) entry.changed += 1;
    else entry.kept += 1;
    if (rec.original_label in entry.counts_before) entry.counts_before[rec.original_label] += 1;
    if (rec.final_label in entry.counts_after) entry.counts_after[rec.final_label] += 1;
    entry.result_ids.push(rec.result_id);
    byImage.set(key, entry);
  }
  return { byImage, recordMap: map };
};

export const listFinalImages = (payload = {}) => {
  const completePath = cleanPath(payload.completeLabelsCsv, 'completeLabelsCsv');
  const finalPath = cleanPath(payload.finalLabelsCsv, 'finalLabelsCsv');
  const filterChangedOnly = Boolean(payload.changedOnly);
  const sortBy = String(payload.sortBy || 'changed_desc').trim();

  const { byImage } = buildImageStats(completePath, finalPath);
  let images = Array.from(byImage.values()).map((entry) => ({
    image_rel_path: entry.image_rel_path,
    total: entry.total,
    changed: entry.changed,
    kept: entry.kept,
    counts_before: entry.counts_before,
    counts_after: entry.counts_after,
  }));
  if (filterChangedOnly) images = images.filter((i) => i.changed > 0);
  if (sortBy === 'rel_path') {
    images.sort((a, b) => a.image_rel_path.localeCompare(b.image_rel_path));
  } else if (sortBy === 'total_desc') {
    images.sort((a, b) => b.total - a.total || a.image_rel_path.localeCompare(b.image_rel_path));
  } else {
    // changed_desc (default)
    images.sort((a, b) => b.changed - a.changed
      || b.total - a.total
      || a.image_rel_path.localeCompare(b.image_rel_path));
  }

  const totals = images.reduce((acc, i) => {
    acc.total += i.total;
    acc.changed += i.changed;
    acc.kept += i.kept;
    return acc;
  }, { images: images.length, total: 0, changed: 0, kept: 0 });

  return {
    images,
    totals,
    completeLabelsCsv: resolveDbPath(completePath),
    finalLabelsCsv: resolveDbPath(finalPath),
  };
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

// Export the final dataset as COCO JSON (keeping only crack/mold/spall).
// Optionally copies images alongside.
export const exportFinalToCoco = (payload = {}) => {
  const finalPath = cleanPath(payload.finalLabelsCsv, 'finalLabelsCsv');
  const sourceDbPath = cleanPath(payload.sourceDbPath, 'sourceDbPath');
  const imageRootPath = String(payload.imageRootPath || '').trim();
  const outputDir = path.resolve(expandHome(cleanPath(payload.outputDir, 'outputDir')));
  const copyImages = Boolean(payload.copyImages);

  if (!fs.existsSync(finalPath)) throw new Error(`final_labels CSV not found: ${finalPath}`);

  const finalRows = parseCsv(finalPath).rows;
  const categoryById = new Map([
    ['crack', 1],
    ['mold', 2],
    ['spall', 3],
  ]);
  const kept = finalRows.filter((r) => categoryById.has(String(r.final_class || '').trim()));

  const ids = kept.map((r) => Number(r.result_id)).filter(Number.isFinite);
  const sourceDb = connectRo(sourceDbPath);
  let bboxMap;
  try { bboxMap = fetchBboxRows(sourceDb, ids); }
  finally { sourceDb.close(); }

  const imageIdByRelPath = new Map();
  const imageSourceByRelPath = new Map();
  const images = [];
  const annotations = [];
  let nextImageId = 1;
  let nextAnnId = 1;
  let skippedNoBbox = 0;

  for (const row of kept) {
    const rid = Number(row.result_id);
    const bbox = bboxMap.get(rid);
    if (!bbox) { skippedNoBbox += 1; continue; }
    const relPath = safeRelPath(bbox.image_rel_path);
    if (!relPath) { skippedNoBbox += 1; continue; }
    const x = Math.max(0, Number(bbox.x1 || 0));
    const y = Math.max(0, Number(bbox.y1 || 0));
    const w = Math.max(0, Number(bbox.x2 || 0) - x);
    const h = Math.max(0, Number(bbox.y2 || 0) - y);
    if (w <= 0 || h <= 0) { skippedNoBbox += 1; continue; }
    if (!imageSourceByRelPath.has(relPath)) {
      imageSourceByRelPath.set(relPath, resolveImagePath(bbox, imageRootPath));
    }

    let imageId = imageIdByRelPath.get(relPath);
    if (!imageId) {
      imageId = nextImageId;
      nextImageId += 1;
      imageIdByRelPath.set(relPath, imageId);
      images.push({
        id: imageId,
        file_name: copyImages ? path.posix.join('images', relPath) : relPath,
        width: Number(bbox.image_width || 0),
        height: Number(bbox.image_height || 0),
      });
    }

    const finalClass = String(row.final_class || '').trim();
    annotations.push({
      id: nextAnnId,
      image_id: imageId,
      category_id: categoryById.get(finalClass),
      bbox: [x, y, w, h],
      area: w * h,
      iscrowd: 0,
      segmentation: [],
    });
    nextAnnId += 1;
  }

  const coco = {
    info: {
      description: 'Semi-labeled damage detection dataset (Step 8 export)',
      version: '1.0',
      date_created: new Date().toISOString(),
      source_csv: path.basename(finalPath),
    },
    licenses: [],
    categories: [
      { id: 1, name: 'crack', supercategory: 'damage' },
      { id: 2, name: 'mold',  supercategory: 'damage' },
      { id: 3, name: 'spall', supercategory: 'damage' },
    ],
    images,
    annotations,
  };

  fs.mkdirSync(outputDir, { recursive: true });
  const annPath = path.join(outputDir, 'annotations.json');
  fs.writeFileSync(annPath, JSON.stringify(coco, null, 2), 'utf8');

  let copiedCount = 0;
  let copyErrors = 0;
  if (copyImages) {
    const imagesDir = path.join(outputDir, 'images');
    for (const img of images) {
      // file_name was prefixed with 'images/' for COCO; recover the relative path
      const relPath = img.file_name.startsWith('images/') ? img.file_name.slice('images/'.length) : img.file_name;
      const safePath = safeRelPath(relPath);
      if (!safePath) { copyErrors += 1; continue; }
      const srcPath = imageSourceByRelPath.get(safePath);
      if (!srcPath || !fs.existsSync(srcPath)) { copyErrors += 1; continue; }
      const dstPath = path.join(imagesDir, safePath);
      try {
        fs.mkdirSync(path.dirname(dstPath), { recursive: true });
        fs.copyFileSync(srcPath, dstPath);
        copiedCount += 1;
      } catch {
        copyErrors += 1;
      }
    }
  }

  return {
    annotations_path: annPath,
    output_dir: outputDir,
    n_images: images.length,
    n_annotations: annotations.length,
    n_skipped_no_bbox: skippedNoBbox,
    n_copied: copiedCount,
    n_copy_errors: copyErrors,
    categories: coco.categories,
  };
};

export const getFinalImageBoxes = (payload = {}) => {
  const completePath = cleanPath(payload.completeLabelsCsv, 'completeLabelsCsv');
  const finalPath = cleanPath(payload.finalLabelsCsv, 'finalLabelsCsv');
  const sourceDbPath = cleanPath(payload.sourceDbPath, 'sourceDbPath');
  const imageRelPath = cleanPath(payload.imageRelPath, 'imageRelPath');
  const imageRootPath = String(payload.imageRootPath || '').trim();

  const { byImage, recordMap } = buildImageStats(completePath, finalPath);
  const entry = byImage.get(imageRelPath);
  if (!entry) {
    return { image: null, boxes: [] };
  }

  const ids = entry.result_ids;
  const sourceDb = connectRo(sourceDbPath);
  let bboxMap;
  try {
    bboxMap = fetchBboxRows(sourceDb, ids);
  } finally {
    sourceDb.close();
  }

  // Find an image_uri from the first row that has bbox data
  let imageUri = '';
  let imageWidth = 0;
  let imageHeight = 0;
  let bboxSample = null;
  for (const rid of ids) {
    const bbox = bboxMap.get(rid);
    if (bbox) {
      bboxSample = bbox;
      const p = resolveImagePath(bbox, imageRootPath);
      imageUri = imageUriForPath(p);
      imageWidth = Number(bbox.image_width || 0);
      imageHeight = Number(bbox.image_height || 0);
      break;
    }
  }

  const boxes = ids.map((rid) => {
    const rec = recordMap.get(String(rid)) || recordMap.get(rid);
    const bbox = bboxMap.get(rid);
    return {
      result_id: rid,
      original_label: rec?.original_label || '',
      final_label: rec?.final_label || '',
      changed: rec ? rec.original_label !== rec.final_label : false,
      source: rec?.source || '',
      x1: bbox ? Number(bbox.x1) : 0,
      y1: bbox ? Number(bbox.y1) : 0,
      x2: bbox ? Number(bbox.x2) : 0,
      y2: bbox ? Number(bbox.y2) : 0,
    };
  });

  return {
    image: {
      image_rel_path: imageRelPath,
      image_uri: imageUri,
      image_width: imageWidth,
      image_height: imageHeight,
      total: entry.total,
      changed: entry.changed,
      counts_before: entry.counts_before,
      counts_after: entry.counts_after,
    },
    boxes,
    has_image: Boolean(bboxSample),
  };
};
