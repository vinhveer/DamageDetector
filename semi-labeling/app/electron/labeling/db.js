import fs from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';

export const cleanPath = (value, fieldName) => {
  const raw = String(value || '').trim();
  if (!raw || raw.includes('\0')) throw new Error(`Invalid ${fieldName}`);
  return raw;
};

export const expandHome = (value) => {
  const raw = String(value || '').trim();
  if (!raw.startsWith('~')) return raw;
  return path.join(process.env.HOME || process.env.USERPROFILE || '', raw.slice(1));
};

export const resolveDbPath = (dbPath) => path.resolve(expandHome(dbPath));

// Read-only connection (checkpoints WAL first so latest rows are visible).
export const connectRo = (dbPath) => {
  const resolved = resolveDbPath(dbPath);
  if (!fs.existsSync(resolved)) throw new Error(`SQLite database not found: ${resolved}`);
  if (fs.existsSync(`${resolved}-shm`)) {
    try {
      const w = new DatabaseSync(resolved, { readOnly: false });
      w.exec('PRAGMA wal_checkpoint(TRUNCATE)');
      w.close();
    } catch {
      /* another process owns the WAL; read-only open still works */
    }
  }
  return new DatabaseSync(resolved, { readOnly: true });
};

export const connectRw = (dbPath) => {
  const resolved = resolveDbPath(dbPath);
  if (!fs.existsSync(resolved)) throw new Error(`SQLite database not found: ${resolved}`);
  return new DatabaseSync(resolved, { readOnly: false });
};
