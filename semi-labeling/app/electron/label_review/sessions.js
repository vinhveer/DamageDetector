import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { SEMI_LABELING_DEFAULTS } from '../defaults.js';

const defaultSessionsDir = () => SEMI_LABELING_DEFAULTS.labelReview.sessionsDir;

const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
};

const cleanSessionId = (value) => {
  const raw = String(value || '').trim();
  if (!raw || !/^[a-zA-Z0-9_-]+$/.test(raw)) throw new Error('Invalid sessionId');
  return raw;
};

const cleanString = (value, fieldName, { allowEmpty = false } = {}) => {
  const raw = String(value || '').trim();
  if (!allowEmpty && !raw) throw new Error(`Missing ${fieldName}`);
  return raw;
};

const resolveSessionsDir = (sessionsDir) => {
  const raw = String(sessionsDir || '').trim();
  const target = raw || defaultSessionsDir();
  ensureDir(target);
  return target;
};

const sessionFilePath = (sessionsDir, sessionId) => path.join(sessionsDir, `${sessionId}.json`);
const sessionBackupPath = (sessionsDir, sessionId) => path.join(sessionsDir, `${sessionId}.json.bak`);

const utcNow = () => new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');

const buildSkeleton = ({ sessionId, subclusterRunId, title }) => ({
  session_id: sessionId,
  subcluster_run_id: subclusterRunId,
  title: title || `Session ${utcNow().slice(0, 19)}`,
  created_at_utc: utcNow(),
  last_updated_utc: utcNow(),
  decisions: {},
  stats: {
    reviewed: 0,
    kept: 0,
    changed: 0,
    rejected: 0,
  },
});

const computeStats = (payload) => {
  // Decisions are keyed by result_id (per-box). The presence of a decision means
  // the box was touched by the user. 'keep' = confirmed current label.
  const stats = { reviewed: 0, kept: 0, changed: 0, rejected: 0 };
  const decisions = payload?.decisions || {};
  for (const value of Object.values(decisions)) {
    if (!value || typeof value !== 'object') continue;
    const action = String(value.action || '').trim();
    if (!action) continue;
    stats.reviewed += 1;
    if (action === 'keep') stats.kept += 1;
    else if (action === 'change') {
      stats.changed += 1;
      if (String(value.target_label || '') === 'reject') stats.rejected += 1;
    } else if (action === 'reject') stats.rejected += 1;
  }
  return stats;
};

export const listSessions = (payload = {}) => {
  const sessionsDir = resolveSessionsDir(payload.sessionsDir);
  const runIdFilter = String(payload.subclusterRunId || payload.runId || '').trim();
  const entries = fs.readdirSync(sessionsDir, { withFileTypes: true });
  const sessions = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
    const filePath = path.join(sessionsDir, entry.name);
    try {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      if (runIdFilter && data.subcluster_run_id !== runIdFilter) continue;
      sessions.push({
        session_id: String(data.session_id || ''),
        subcluster_run_id: String(data.subcluster_run_id || ''),
        title: String(data.title || ''),
        created_at_utc: String(data.created_at_utc || ''),
        last_updated_utc: String(data.last_updated_utc || ''),
        stats: data.stats || {},
      });
    } catch {
      // skip malformed
    }
  }
  sessions.sort((a, b) => String(b.last_updated_utc).localeCompare(String(a.last_updated_utc)));
  return { sessions, sessionsDir };
};

export const loadSession = (payload = {}) => {
  const sessionsDir = resolveSessionsDir(payload.sessionsDir);
  const sessionId = cleanSessionId(payload.sessionId);
  const filePath = sessionFilePath(sessionsDir, sessionId);
  if (!fs.existsSync(filePath)) throw new Error(`Session not found: ${sessionId}`);
  const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  return data;
};

export const saveSession = (payload = {}) => {
  const sessionsDir = resolveSessionsDir(payload.sessionsDir);
  const sessionId = cleanSessionId(payload.sessionId);
  const data = payload.payload;
  if (!data || typeof data !== 'object') throw new Error('Missing payload');
  if (String(data.session_id || '') !== sessionId) {
    throw new Error('session_id in payload does not match sessionId');
  }
  const normalized = {
    ...data,
    last_updated_utc: utcNow(),
    stats: computeStats(data),
  };
  const filePath = sessionFilePath(sessionsDir, sessionId);
  const backupPath = sessionBackupPath(sessionsDir, sessionId);
  if (fs.existsSync(filePath)) {
    fs.copyFileSync(filePath, backupPath);
  }
  const tmpPath = `${filePath}.tmp`;
  fs.writeFileSync(tmpPath, JSON.stringify(normalized, null, 2), 'utf8');
  fs.renameSync(tmpPath, filePath);
  return { session_id: sessionId, last_updated_utc: normalized.last_updated_utc, stats: normalized.stats };
};

export const createSession = (payload = {}) => {
  const sessionsDir = resolveSessionsDir(payload.sessionsDir);
  const subclusterRunId = cleanString(payload.subclusterRunId || payload.runId, 'subclusterRunId');
  const title = String(payload.title || '').trim();
  const sessionId = randomUUID().replace(/-/g, '').slice(0, 16);
  const data = buildSkeleton({ sessionId, subclusterRunId, title });
  const filePath = sessionFilePath(sessionsDir, sessionId);
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
  return data;
};

export const deleteSession = (payload = {}) => {
  const sessionsDir = resolveSessionsDir(payload.sessionsDir);
  const sessionId = cleanSessionId(payload.sessionId);
  const filePath = sessionFilePath(sessionsDir, sessionId);
  if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
  const backup = sessionBackupPath(sessionsDir, sessionId);
  if (fs.existsSync(backup)) fs.unlinkSync(backup);
  return { session_id: sessionId, deleted: true };
};
