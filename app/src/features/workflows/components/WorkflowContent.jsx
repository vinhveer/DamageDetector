import { useDispatch } from 'react-redux';
import { IconChevronLeft } from '@tabler/icons-react';
import { Button, StatusBadge } from '../../../components/ui/index.js';
import { cn } from '../../../components/ui/cn.js';
import { setSelectedWorkflow } from '../workflowsSlice.js';
import WorkflowForm from './WorkflowForm.jsx';
import WorkflowTerminal from './WorkflowTerminal.jsx';

export default function WorkflowContent({ workflow, session }) {
  const dispatch = useDispatch();

  return (
    <div className="rv-enter flex h-full min-w-0 flex-col bg-[var(--bg)] rv-font">

      {/* Header */}
      <header className="flex h-12 shrink-0 items-center justify-between gap-4 border-b border-[var(--border-muted)] px-6">
        <div className="flex min-w-0 items-center gap-3">
          <button
            type="button"
            onClick={() => dispatch(setSelectedWorkflow(null))}
            className="inline-flex shrink-0 items-center gap-1 text-[13px] text-[var(--text-muted)] hover:text-[var(--text)]"
          >
            <IconChevronLeft size={14} />
            Back
          </button>
          <span className="text-[var(--border-muted)]">/</span>
          <span className="truncate text-[13px] font-medium text-[var(--text)]">{workflow.name}</span>
          {session && <StatusBadge status={session.status} />}
        </div>
        <div className="flex shrink-0 gap-2">
          {session?.status === 'running' && (
            <Button variant="danger" onClick={() => window.electronAPI.stopWorkflow(session.id)}>
              Stop
            </Button>
          )}
        </div>
      </header>

      {/* Body */}
      <main className="min-h-0 flex-1 overflow-auto p-6">
        <div className={cn('mx-auto w-full max-w-[860px]', session ? 'flex h-full flex-col' : '')}>
          {session ? (
            <WorkflowTerminal session={session} />
          ) : (
            <WorkflowForm workflow={workflow} />
          )}
        </div>
      </main>
    </div>
  );
}
