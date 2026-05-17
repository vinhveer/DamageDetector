import { useState } from 'react';
import { Button } from '../../../components/ui/index.js';

export default function WorkflowTerminal({ session }) {
  const [copied, setCopied] = useState(false);
  const lines = session.log ? session.log.split('\n') : [];

  const copyLog = async () => {
    await window.navigator.clipboard.writeText(session.log || '');
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  const classForLine = (line) => {
    const l = line.toLowerCase();
    if (l.includes('error') || l.startsWith('traceback')) return 'text-[var(--danger)]';
    if (l.includes('warning') || l.startsWith('warn'))     return 'text-[var(--warning)]';
    if (l.startsWith('info') || l.startsWith('[info]'))    return 'text-[var(--text-muted)]';
    return 'text-[var(--text)]';
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-[8px] border border-[var(--border)] bg-[var(--surface)]">

      {/* Toolbar */}
      <div className="flex h-10 shrink-0 items-center justify-between gap-3 border-b border-[var(--border-muted)] px-4">
        <span className="rv-mono text-[11px] text-[var(--text-muted)]">process output</span>
        <Button
          variant="ghost"
          onClick={copyLog}
          disabled={!session.log}
          className="h-6 px-2 text-[12px]"
        >
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </div>

      {/* Log content */}
      <pre className="rv-mono min-h-0 flex-1 overflow-auto p-4 text-[12px] leading-[1.7] whitespace-pre-wrap break-words">
        {lines.length > 0
          ? lines.map((line, i) => (
              <span key={i} className={classForLine(line)}>
                {line}{'\n'}
              </span>
            ))
          : <span className="text-[var(--text-subtle)]">Waiting for output…</span>
        }
      </pre>
    </div>
  );
}
