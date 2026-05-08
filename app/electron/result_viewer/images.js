import fs from 'node:fs';
import path from 'node:path';

const expandHome = (value) => {
  if (!value || !value.startsWith('~')) return value || '';
  return path.join(process.env.HOME || process.env.USERPROFILE || '', value.slice(1));
};

const resolveExisting = (candidate) => {
  try {
    return fs.existsSync(candidate) && fs.statSync(candidate).isFile() ? path.resolve(candidate) : '';
  } catch {
    return '';
  }
};

export const resolveImagePath = (row, imageRoot) => {
  const candidates = [];
  const relPath = String(row.image_rel_path || '').trim();
  const storedPath = String(row.image_path || '').trim();
  const sourceInputDir = path.resolve(expandHome(String(row.source_input_dir || '')) || '.');

  if (imageRoot) {
    const root = path.resolve(expandHome(imageRoot));
    if (relPath) candidates.push(path.join(root, relPath));
    if (storedPath) candidates.push(path.join(root, path.basename(storedPath)));
  }
  if (storedPath) {
    const expanded = expandHome(storedPath);
    candidates.push(path.isAbsolute(expanded) ? expanded : path.join(sourceInputDir, storedPath));
  }
  if (relPath) candidates.push(path.join(sourceInputDir, relPath));
  if (storedPath) candidates.push(path.join(sourceInputDir, path.basename(storedPath)));

  const seen = new Set();
  for (const candidate of candidates) {
    const key = path.normalize(candidate);
    if (seen.has(key)) continue;
    seen.add(key);
    const resolved = resolveExisting(candidate);
    if (resolved) return resolved;
  }
  if (imageRoot && relPath) return path.resolve(expandHome(imageRoot), relPath);
  return path.resolve(sourceInputDir, relPath);
};
