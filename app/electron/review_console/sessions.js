import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const labRoot = path.resolve(repoRoot, '..');

const defaultSessionsDir = () => path.join(labRoot, 'infer_results', 'semi-labeling', 'resemi', 'review_sessions');

const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
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

const buildSkeleton = ({ sessionId, runId, reviewer, sources, title }) => ({
  session_id: sessionId,
  run_id: runId,
  reviewer: reviewer || '',
  title: title || `Session ${utcNow().slice(0, 19)}`,
  status: 'draft',
  source_reliability_run_id: sources.reliabilityRunId || null,
  source_decision_policy_run_id: sources.decisionPolicyRunId || null,
  source_prototype_version_id: sources.prototypeVersionId || null,
  created_at_utc: utcNow(),
  last_updated_utc: utcNow(),
  committed_at_utc: null,
  notes: '',
  decisions: {},
  stats: { reviewed: 0, accepted: 0, relabeled: 0, rejected: 0, deferred: 0, prototypes: 0 },
});

const computeStats = (payload) => {
  const stats = { reviewed: 0, accepted: 0, relabeled: 0, rejected: 0, deferred: 0, prototypes: 0 };
  const decisions = payload?.decisions || {};
  for (const value of Object.values(decisions)) {
    if (!value || typeof value !== 'object') continue;
    const action = String(value.action || '').trim();
    if (!action) continue;
    stats.reviewed += 1;
    if (action === 'accept' || action === 'accept_cluster') stats.accepted += 1;
    else if (action === 'relabel' || action === 'relabel_cluster') stats.relabeled += 1;
    else if (action === 'reject' || action === 'reject_cluster') stats.rejected += 1;
    else if (action === 'defer') stats.deferred += 1;
    else if (action === 'add_prototype' || action === 'add_reject_prototype') stats.prototypes += 1;
  }
  return stats;
};

export const listSessions = (payload = {}) => {
  const sessionsDir = resolveSessionsDir(payload.sessionsDir);
  const runIdFilter = String(payload.runId || '').trim();
  const entries = fs.readdirSync(sessionsDir, { withFileTypes: true });
  const sessions = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
    try {
      const data = JSON.parse(fs.readFileSync(path.join(sessionsDir, entry.name), 'utf8'));
      if (runIdFilter && data.run_id !== runIdFilter) continue;
      sessions.push({
        session_id: String(data.session_id || ''),
        run_id: String(data.run_id || ''),
        reviewer: String(data.reviewer || ''),
        title: String(data.title || ''),
        status: String(data.status || 'draft'),
        created_at_utc: String(data.created_at_utc || ''),
        last_updated_utc: String(data.last_updated_utc || ''),
        committed_at_utc: data.committed_at_utc || null,
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
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
};

export const saveSession = (payload = {}) => {
  const sessionsDir = resolveSessionsDir(payload.sessionsDir);
  const sessionId = cleanSessionId(payload.sessionId);
  const data = payload.payload;
  if (!data || typeof data !== 'object') throw new Error('Missing payload');
  if (String(data.session_id || '') !== sessionId) {
    throw new Error('session_id in payload does not match sessionId');
  }
  if (String(data.status || '') === 'committed') {
    throw new Error('Cannot edit a committed session; create a superseding session.');
  }
  const normalized = {
    ...data,
    last_updated_utc: utcNow(),
    stats: computeStats(data),
  };
  const filePath = sessionFilePath(sessionsDir, sessionId);
  const backupPath = sessionBackupPath(sessionsDir, sessionId);
  if (fs.existsSync(filePath)) fs.copyFileSync(filePath, backupPath);
  const tmpPath = `${filePath}.tmp`;
  fs.writeFileSync(tmpPath, JSON.stringify(normalized, null, 2), 'utf8');
  fs.renameSync(tmpPath, filePath);
  return { session_id: sessionId, last_updated_utc: normalized.last_updated_utc, stats: normalized.stats };
};

export const createSession = (payload = {}) => {
  const sessionsDir = resolveSessionsDir(payload.sessionsDir);
  const runId = cleanString(payload.runId, 'runId');
  const sessionId = randomUUID().replace(/-/g, '').slice(0, 16);
  const data = buildSkeleton({
    sessionId,
    runId,
    reviewer: String(payload.reviewer || '').trim(),
    title: String(payload.title || '').trim(),
    sources: {
      reliabilityRunId: payload.reliabilityRunId,
      decisionPolicyRunId: payload.decisionPolicyRunId,
      prototypeVersionId: payload.prototypeVersionId,
    },
  });
  fs.writeFileSync(sessionFilePath(sessionsDir, sessionId), JSON.stringify(data, null, 2), 'utf8');
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

// Mark a session committed after a successful SQLite write (called by commit.js).
export const markSessionCommitted = ({ sessionsDir, sessionId, committedAtUtc }) => {
  const dir = resolveSessionsDir(sessionsDir);
  const id = cleanSessionId(sessionId);
  const filePath = sessionFilePath(dir, id);
  if (!fs.existsSync(filePath)) return { session_id: id, updated: false };
  const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  data.status = 'committed';
  data.committed_at_utc = committedAtUtc || utcNow();
  data.last_updated_utc = utcNow();
  const tmpPath = `${filePath}.tmp`;
  fs.writeFileSync(tmpPath, JSON.stringify(data, null, 2), 'utf8');
  fs.renameSync(tmpPath, filePath);
  return { session_id: id, updated: true };
};
