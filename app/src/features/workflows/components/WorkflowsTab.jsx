import { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { TextInput, StatusBadge } from '../../../components/ui/index.js';
import { setSelectedWorkflow } from '../workflowsSlice.js';
import WorkflowContent from './WorkflowContent.jsx';

export default function WorkflowsTab() {
  const dispatch = useDispatch();
  const { items, status, selectedWorkflowId, sessions } = useSelector((state) => state.workflows);
  const [query, setQuery] = useState('');

  const filteredItems = items.filter((w) =>
    w.name.toLowerCase().includes(query.trim().toLowerCase())
  );
  const selectedWorkflow = selectedWorkflowId ? items.find((w) => w.id === selectedWorkflowId) : null;
  const selectedSession  = selectedWorkflow ? sessions.find((s) => s.workflowId === selectedWorkflow.id) : null;

  if (selectedWorkflow) {
    return <WorkflowContent workflow={selectedWorkflow} session={selectedSession} />;
  }

  return (
    <div className="rv-enter flex h-full min-w-0 flex-col bg-[var(--bg)] rv-font">

      {/* Header */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-[var(--border-muted)] px-6">
        <span className="text-[13px] font-medium text-[var(--text)]">Workflows</span>
        <TextInput
          className="w-[240px]"
          placeholder="Search…"
          value={query}
          onChange={(e) => setQuery(e.currentTarget.value)}
        />
      </header>

      {/* List */}
      <main className="min-h-0 flex-1 overflow-auto p-6">
        <div
          data-ui="panel"
          className="mx-auto max-w-[800px] overflow-hidden rounded-[8px] border border-[var(--border)] bg-[var(--surface)]"
        >
          {status === 'loading' && (
            <div className="px-5 py-4 text-[13px] text-[var(--text-muted)]">Loading…</div>
          )}

          {filteredItems.map((workflow, idx) => {
            const session = sessions.find((s) => s.workflowId === workflow.id);
            return (
              <button
                key={workflow.id}
                type="button"
                onClick={() => dispatch(setSelectedWorkflow(workflow.id))}
                className={[
                  'flex w-full items-center justify-between gap-4 px-5 py-3.5 text-left',
                  'hover:bg-[var(--hover)]',
                  idx > 0 ? 'border-t border-[var(--border-muted)]' : '',
                ].join(' ')}
              >
                <div className="min-w-0">
                  <div className="truncate text-[13px] font-medium text-[var(--text)]">
                    {workflow.name}
                  </div>
                  <div className="mt-0.5 truncate text-[12px] text-[var(--text-muted)]">
                    {workflow.type || workflow.command || workflow.id}
                  </div>
                </div>
                {session && (
                  <div className="shrink-0">
                    <StatusBadge status={session.status} />
                  </div>
                )}
              </button>
            );
          })}

          {filteredItems.length === 0 && status !== 'loading' && (
            <div className="px-5 py-10 text-center text-[13px] text-[var(--text-muted)]">
              No workflows found.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
