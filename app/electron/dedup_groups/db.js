import fs from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';

const statementCache = new WeakMap();

export const expandHome = (value) => {
  const raw = String(value || '').trim();
  if (!raw.startsWith('~')) return raw;
  return path.join(process.env.HOME || process.env.USERPROFILE || '', raw.slice(1));
};

export const resolveDbPath = (dbPath) => path.resolve(expandHome(dbPath));

export const connectRo = (dbPath) => {
  const resolved = resolveDbPath(dbPath);
  if (!fs.existsSync(resolved)) {
    throw new Error(`SQLite database not found: ${resolved}`);
  }

  const shmPath = `${resolved}-shm`;
  if (fs.existsSync(shmPath)) {
    try {
      const writable = new DatabaseSync(resolved, { readOnly: false });
      writable.exec('PRAGMA wal_checkpoint(TRUNCATE)');
      writable.close();
    } catch {
      // Read-only open still works if another process owns the WAL.
    }
  }

  return new DatabaseSync(resolved, { readOnly: true });
};

export const connectRoIfExists = (dbPath) => {
  const raw = String(dbPath || '').trim();
  if (!raw) return null;
  const resolved = resolveDbPath(raw);
  return fs.existsSync(resolved) ? connectRo(resolved) : null;
};

export const cachedPrepare = (db, sql) => {
  let cache = statementCache.get(db);
  if (!cache) {
    cache = new Map();
    statementCache.set(db, cache);
  }
  let statement = cache.get(sql);
  if (!statement) {
    statement = db.prepare(sql);
    cache.set(sql, statement);
  }
  return statement;
};
