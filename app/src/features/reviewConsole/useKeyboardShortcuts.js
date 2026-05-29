import { useEffect } from 'react';

// SPEC §6.2 — global keyboard map for the review console. Skips when a text input
// is focused (except Escape) so typing notes / filters never triggers an action.
export function useKeyboardShortcuts(handlers, enabled = true) {
  useEffect(() => {
    if (!enabled) return undefined;
    const onKey = (event) => {
      const tag = (event.target?.tagName || '').toLowerCase();
      const typing = tag === 'input' || tag === 'textarea' || event.target?.isContentEditable;
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      const key = event.key;
      if (key === 'Escape') { handlers.onEscape?.(); return; }
      if (typing) return;

      const map = {
        a: handlers.onAccept,
        r: handlers.onRelabel,
        x: handlers.onReject,
        m: handlers.onAmbiguous,
        d: handlers.onDefer,
        p: handlers.onAddPrototype,
        n: handlers.onNext,
        b: handlers.onPrev,
        u: handlers.onUndo,
        '/': handlers.onSearch,
      };
      const lower = key.toLowerCase();

      if (key === 'P' && event.shiftKey) { handlers.onAddRejectPrototype?.(); event.preventDefault(); return; }
      if (/^[1-9]$/.test(key)) { handlers.onPickCandidate?.(Number(key)); event.preventDefault(); return; }

      const fn = map[lower];
      if (fn) { fn(); event.preventDefault(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handlers, enabled]);
}
