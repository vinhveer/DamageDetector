import { useState } from 'react';
import { cn } from '../../../components/ui/cn.js';
import { Button } from '../../../components/ui/index.js';

const monoFont = {
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
};

export default function WorkflowTerminal({ session }) {
  const lines = session.log ? session.log.split('\n') : [];
  const [copied, setCopied] = useState(false);

  const copyLog = async () => {
    await window.navigator.clipboard.writeText(session.log || '');
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-md border border-slate-800 bg-slate-950">
      <div className="flex h-11 shrink-0 items-center justify-between gap-3 border-b border-slate-800 px-4 text-[12px] text-slate-400">
        <span style={monoFont}>process log</span>
        <Button
          variant="ghost"
          onClick={copyLog}
          disabled={!session.log}
          className="h-7 border-slate-700 px-2.5 text-[12px] text-slate-300 hover:bg-slate-900 hover:text-white"
        >
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </div>
      <pre style={monoFont} className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap break-words p-4 text-[12px] leading-6 text-slate-200">
        {lines.length > 0 ? lines.map((line, index) => (
          <span key={`${index}-${line}`}>
            <span className={cn(line.toLowerCase().startsWith('error') && 'text-red-300', line.toLowerCase().startsWith('warning') && 'text-amber-300')}>{line}</span>
            {'\n'}
          </span>
        )) : <span className="text-slate-500">Initializing process...</span>}
      </pre>
    </div>
  );
}