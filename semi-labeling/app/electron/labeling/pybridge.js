// Python bridge — spawns the semi-labeling step CLIs as subprocesses and
// streams their stdout/stderr back to the renderer.
//
// The app is standalone (SQLite only), but the "Run next steps" tab needs to
// invoke the Python pipeline (step08 classifier + step09 self-training). We
// resolve the project's venv python and the semi-labeling package root, then
// run `python -m steps.<step>.main ...` with PYTHONPATH set.

import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// electron/labeling -> electron -> app -> semi-labeling -> DamageDetector -> Lab
const semiLabelingDir = path.resolve(__dirname, '..', '..', '..');
const repoRoot = path.resolve(semiLabelingDir, '..'); // DamageDetector/

// Candidate venv python locations (project venv preferred).
const pythonCandidates = () => [
  process.env.SEMI_LABELING_PYTHON,
  path.join(repoRoot, '.venv', 'bin', 'python'),
  path.join(semiLabelingDir, '.venv', 'bin', 'python'),
  path.join(repoRoot, '.venv', 'Scripts', 'python.exe'),
].filter(Boolean);

export const resolvePython = () => {
  for (const candidate of pythonCandidates()) {
    try { if (fs.existsSync(candidate)) return candidate; } catch { /* ignore */ }
  }
  return 'python3'; // last resort: rely on PATH
};

export const bridgeInfo = () => ({
  python: resolvePython(),
  semiLabelingDir,
  repoRoot,
  pythonExists: pythonCandidates().some((c) => { try { return fs.existsSync(c); } catch { return false; } }),
});

// Build argv for a known step. Only an allow-listed set of modules can run.
const STEP_MODULES = {
  step08: 'steps.step08_classifier.main',
  step09: 'steps.step09_self_train.main',
  export_dataset: 'tools.export_dataset',
};

const isSafeScalar = (v) => typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean';

// Turn a flags object {"--run-id": "x", "--apply-promotions": true} into argv.
const flagsToArgv = (flags = {}) => {
  const argv = [];
  for (const [key, value] of Object.entries(flags)) {
    if (!/^--[a-z0-9-]+$/i.test(key)) throw new Error(`Unsafe flag name: ${key}`);
    if (value === false || value === null || value === undefined) continue;
    if (value === true) { argv.push(key); continue; }
    if (!isSafeScalar(value)) throw new Error(`Unsafe flag value for ${key}`);
    argv.push(key, String(value));
  }
  return argv;
};

// Run a step, streaming output via onData(chunk). Resolves with { code, output }.
export const runStep = ({ step, flags = {}, onData } = {}) => new Promise((resolve) => {
  const moduleName = STEP_MODULES[step];
  if (!moduleName) { resolve({ code: -1, output: `Unknown step: ${step}` }); return; }

  let argv;
  try {
    argv = ['-m', moduleName, ...flagsToArgv(flags)];
  } catch (err) {
    resolve({ code: -1, output: String(err?.message || err) });
    return;
  }

  const python = resolvePython();
  const env = { ...process.env, PYTHONPATH: semiLabelingDir, PYTHONUNBUFFERED: '1' };
  const child = spawn(python, argv, { cwd: semiLabelingDir, env });

  let output = '';
  const push = (text) => {
    output += text;
    if (typeof onData === 'function') onData(text);
  };
  push(`$ ${python} ${argv.join(' ')}\n`);

  child.stdout.on('data', (d) => push(d.toString()));
  child.stderr.on('data', (d) => push(d.toString()));
  child.on('error', (err) => { push(`\n[spawn error] ${err.message}\n`); resolve({ code: -1, output }); });
  child.on('close', (code) => { push(`\n[exit ${code}]\n`); resolve({ code, output }); });
});

export { STEP_MODULES };
