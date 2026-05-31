import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { loadSession, markSessionCommitted } from './sessions.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..', '..');
const repoRoot = path.resolve(appRoot, '..');
const labRoot = path.resolve(repoRoot, '..');
const commitModule = 'tools.review_commit';
const commitScript = path.join(repoRoot, 'semi-labeling', 'tools', 'review_commit.py');

const cleanPath = (value, fieldName) => {
  const raw = String(value || '').trim();
  if (!raw || raw.includes('\0')) throw new Error(`Invalid ${fieldName}`);
  return raw;
};

const pythonEnv = () => ({
  ...process.env,
  PYTHONPATH: [appRoot, repoRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
});

const resolvePython = (payload) => {
  const venvDir = String(payload.venvDir || '').trim();
  if (payload.useGlobalPython || !venvDir) return process.env.PYTHON || 'python3';
  const binName = process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python';
  return path.join(venvDir, binName);
};

const parsePythonJsonResult = (stdout, stderr, code) => {
  const lines = stdout.trim().split('\n').filter(Boolean);
  const resultLine = [...lines].reverse().find((line) => {
    try { JSON.parse(line); return true; } catch { return false; }
  });
  if (resultLine) {
    try {
      const parsed = JSON.parse(resultLine);
      if (parsed.error) return { error: String(parsed.error) };
      if (code === 0) return parsed;
      return { error: parsed.error || `Python exited with code ${code}` };
    } catch { /* fall through */ }
  }
  return { error: stderr.trim() || `Python exited with code ${code}` };
};

export const commitSession = async (payload = {}) => {
  let tmpPath = '';
  try {
    const resemiDbPath = cleanPath(payload.resemiDbPath, 'resemiDbPath');
    const sessionsDir = String(payload.sessionsDir || '').trim();
    const session = loadSession({ sessionsDir, sessionId: payload.sessionId });

    if (String(session.status || '') === 'committed') {
      return { error: 'Session already committed. Create a superseding session.' };
    }
    const decisions = session.decisions || {};
    if (Object.keys(decisions).length === 0) {
      return { error: 'No drafted decisions to commit.' };
    }

    if (!fs.existsSync(commitScript)) {
      return { error: `Commit script not found: ${commitScript}` };
    }

    const inputData = { output_db: path.resolve(resemiDbPath), session };
    tmpPath = path.join(os.tmpdir(), `review-commit-${Date.now()}-${Math.random().toString(16).slice(2)}.json`);
    fs.writeFileSync(tmpPath, JSON.stringify(inputData), 'utf8');

    const python = resolvePython(payload);
    const result = await new Promise((resolve) => {
      const child = spawn(python, ['-m', commitModule, '--input', tmpPath], {
        cwd: path.join(repoRoot, 'semi-labeling'),
        env: pythonEnv(),
      });
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (d) => { stdout += d; });
      child.stderr.on('data', (d) => { stderr += d; });
      child.on('error', (err) => resolve({ error: err.message }));
      child.on('close', (code) => resolve(parsePythonJsonResult(stdout, stderr, code)));
    });

    if (result.error) return result;
    if (result.committed) {
      markSessionCommitted({
        sessionsDir,
        sessionId: session.session_id,
        committedAtUtc: result.committed_at_utc,
      });
    }
    return result;
  } catch (err) {
    return { error: err?.message || 'Commit failed' };
  } finally {
    if (tmpPath) fs.promises.unlink(tmpPath).catch(() => {});
  }
};

export { labRoot };
