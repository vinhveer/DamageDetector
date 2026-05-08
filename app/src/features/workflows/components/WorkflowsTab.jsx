import { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { TextInput, StatusBadge } from '../../../components/ui/index.js';
import { setSelectedWorkflow } from '../workflowsSlice.js';
import WorkflowContent from './WorkflowContent.jsx';

export default function WorkflowsTab() {
  const dispatch = useDispatch();
  const { items, status, selectedWorkflowId, sessions } = useSelector((state) => state.workflows);
  const [query, setQuery] = useState('');
  const filteredItems = items.filter((workflow) => workflow.name.toLowerCase().includes(query.trim().toLowerCase()));
  const selectedWorkflow = selectedWorkflowId ? items.find((workflow) => workflow.id === selectedWorkflowId) : null;
  const selectedSession = selectedWorkflow ? sessions.find((session) => session.workflowId === selectedWorkflow.id) : null;

  if (selectedWorkflow) {
    return <WorkflowContent workflow={selectedWorkflow} session={selectedSession} />;
  }

  return (
    <div className="rv-enter flex h-full min-w-0 flex-col bg-[var(--docker-bg)] rv-font">
      <header className="flex min-h-[72px] shrink-0 items-center justify-between bg-[var(--docker-bg)] px-8">
        <div>
          <h1 className="text-[18px] font-semibold text-[var(--docker-text)]">Workflows</h1>
        </div>
        <TextInput className="w-[320px]" placeholder="Search workflows" value={query} onChange={(event) => setQuery(event.currentTarget.value)} />
      </header>

      <main className="min-h-0 flex-1 overflow-auto p-8">
        <div data-ui="panel" className="mx-auto max-w-[960px] overflow-hidden rounded-md border border-[var(--docker-border)] bg-white">
          {status === 'loading' && <div className="px-5 py-4 text-[13px] text-[var(--docker-muted)]">Loading...</div>}
          {filteredItems.map((workflow) => {
            const session = sessions.find((item) => item.workflowId === workflow.id);
            return (
              <button
                key={workflow.id}
                type="button"
                onClick={() => dispatch(setSelectedWorkflow(workflow.id))}
                className="flex w-full flex-nowrap items-center justify-between gap-4 px-5 py-4 text-left hover:bg-[var(--docker-hover)]"
              >
                <div className="min-w-0">
                  <div className="truncate text-[14px] font-medium text-[var(--docker-text)]">{workflow.name}</div>
                  <div className="mt-1 truncate text-[12px] text-[var(--docker-muted)]">{workflow.type || workflow.command || workflow.id}</div>
                </div>
                <div className="shrink-0">{session && <StatusBadge status={session.status} />}</div>
              </button>
            );
          })}
          {filteredItems.length === 0 && status !== 'loading' && <div className="px-5 py-10 text-center text-[13px] text-[var(--docker-muted)]">No workflows found.</div>}
        </div>
      </main>
    </div>
  );
}