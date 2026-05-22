import fs from 'node:fs';
import crypto from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

let tempImageCacheDir = '';

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

const getTempImageCacheDir = () => {
  if (!tempImageCacheDir) {
    tempImageCacheDir = path.join(os.tmpdir(), 'damage-detector-app', `image-cache-${process.pid}`);
  }
  return tempImageCacheDir;
};

const cacheKeyForPath = (resolvedPath, stat) => crypto
  .createHash('sha1')
  .update(path.resolve(resolvedPath))
  .update('\0')
  .update(String(stat.size))
  .update('\0')
  .update(String(Math.trunc(stat.mtimeMs)))
  .digest('hex');

export const materializeImagePath = (resolvedPath) => {
  if (!resolvedPath) return '';
  try {
    const stat = fs.statSync(resolvedPath);
    if (!stat.isFile()) return resolvedPath;

    const cacheRoot = getTempImageCacheDir();
    const ext = path.extname(resolvedPath) || '.img';
    const target = path.join(cacheRoot, `${cacheKeyForPath(resolvedPath, stat)}${ext}`);
    if (!fs.existsSync(target)) {
      fs.mkdirSync(cacheRoot, { recursive: true });
      const tempTarget = `${target}.${process.pid}.tmp`;
      try {
        fs.copyFileSync(resolvedPath, tempTarget);
        fs.renameSync(tempTarget, target);
      } finally {
        if (fs.existsSync(tempTarget)) fs.rmSync(tempTarget, { force: true });
      }
    }
    return target;
  } catch {
    return resolvedPath;
  }
};

export const imageUriForPath = (resolvedPath) => {
  const materializedPath = materializeImagePath(resolvedPath);
  return materializedPath ? pathToFileURL(materializedPath).href : '';
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

process.once('exit', () => {
  if (tempImageCacheDir) fs.rmSync(tempImageCacheDir, { recursive: true, force: true });
});
