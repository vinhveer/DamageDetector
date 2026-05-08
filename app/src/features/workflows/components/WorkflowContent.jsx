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
    <div className="rv-enter flex h-full min-w-0 flex-col bg-[var(--docker-bg)] rv-font">
      <header className="flex h-16 shrink-0 items-center justify-between gap-4 bg-[var(--docker-bg)] px-8">
        <div className="flex min-w-0 items-center gap-4">
          <button
            type="button"
            onClick={() => dispatch(setSelectedWorkflow(null))}
            className="inline-flex shrink-0 items-center gap-1 text-[13px] font-medium text-[var(--docker-blue)] hover:underline"
          >
            <IconChevronLeft size={15} />
            Back
          </button>
          <h1 className="truncate text-[18px] font-semibold text-[var(--docker-text)]">{workflow.name}</h1>
          {session && <StatusBadge status={session.status} />}
        </div>
        <div className="flex shrink-0 gap-2">
          {session?.status === 'running' && (
            <Button variant="danger" onClick={() => window.electronAPI.stopWorkflow(session.id)}>
              Stop / Terminate
            </Button>
          )}
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-auto p-8">
        <div className={cn('mx-auto grid w-full max-w-[900px] gap-8', session ? 'h-full min-h-0' : 'min-h-full')}>
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