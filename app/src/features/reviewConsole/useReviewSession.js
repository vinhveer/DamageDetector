import { useCallback, useEffect, useRef, useState } from 'react';

const utcNow = () => new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');

// Owns the in-memory draft session: applies/undoes decisions, tracks dirty state,
// and autosaves to the JSON session file (debounced) via window.electronAPI.
export function useReviewSession({ session, sessionsDir, onSessionChange }) {
  const [draft, setDraft] = useState(session);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const undoStackRef = useRef([]);
  const saveTimerRef = useRef(null);

  useEffect(() => {
    setDraft(session);
    setDirty(false);
    undoStackRef.current = [];
  }, [session]);

  const committed = String(draft?.status || '') === 'committed';

  const persist = useCallback(async (next) => {
    if (!next || String(next.status || '') === 'committed') return;
    setSaving(true);
    setError('');
    try {
      const res = await window.electronAPI.saveReviewSession({
        sessionsDir,
        sessionId: next.session_id,
        payload: next,
      });
      if (res?.error) throw new Error(res.error);
      setDirty(false);
      if (onSessionChange) onSessionChange();
    } catch (err) {
      setError(err.message || 'Save failed');
    } finally {
      setSaving(false);
    }
  }, [sessionsDir, onSessionChange]);

  const scheduleSave = useCallback((next) => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => persist(next), 600);
  }, [persist]);

  const applyDecision = useCallback((decision) => {
    if (committed) return;
    const key = decision.target_id || `${decision.target_type}:${decision.result_id}`;
    setDraft((prev) => {
      const decisions = { ...(prev.decisions || {}) };
      undoStackRef.current.push({ key, previous: decisions[key] });
      decisions[key] = { ...decision, target_id: key, decided_at_utc: utcNow() };
      const next = { ...prev, decisions };
      scheduleSave(next);
      return next;
    });
    setDirty(true);
  }, [committed, scheduleSave]);

  const undo = useCallback(() => {
    if (committed) return;
    const last = undoStackRef.current.pop();
    if (!last) return;
    setDraft((prev) => {
      const decisions = { ...(prev.decisions || {}) };
      if (last.previous === undefined) delete decisions[last.key];
      else decisions[last.key] = last.previous;
      const next = { ...prev, decisions };
      scheduleSave(next);
      return next;
    });
    setDirty(true);
  }, [committed, scheduleSave]);

  const clearAll = useCallback(() => {
    if (committed) return;
    setDraft((prev) => {
      const next = { ...prev, decisions: {} };
      undoStackRef.current = [];
      scheduleSave(next);
      return next;
    });
    setDirty(true);
  }, [committed, scheduleSave]);

  const flushSave = useCallback(async () => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    await persist(draft);
  }, [draft, persist]);

  useEffect(() => () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); }, []);

  const decisionFor = useCallback(
    (key) => (draft?.decisions ? draft.decisions[key] : undefined),
    [draft],
  );

  return {
    draft,
    committed,
    dirty,
    saving,
    error,
    decisionCount: Object.keys(draft?.decisions || {}).length,
    canUndo: undoStackRef.current.length > 0,
    applyDecision,
    undo,
    clearAll,
    flushSave,
    decisionFor,
  };
}
