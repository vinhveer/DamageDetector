import { useEffect } from 'react';
import {
  IconGitBranch,
  IconLayoutSidebarLeftCollapse,
  IconLayoutSidebarLeftExpand,
  IconPhotoSearch,
  IconSettings
} from '@tabler/icons-react';
import { useDispatch, useSelector } from 'react-redux';
import ResultViewer from './features/resultViewer/ResultViewer.jsx';
import {
  fetchWorkflows,
  sessionEvent,
  setSelectedTab,
  toggleSidebar
} from './features/workflows/workflowsSlice.js';
import WorkflowsTab from './features/workflows/components/WorkflowsTab.jsx';
import { IconButton } from './components/ui/index.js';
import { cn } from './components/ui/cn.js';
import SettingsPage from './features/settings/SettingsPage.jsx';

const NAV_ITEMS = [
  { label: 'Workflows', value: 'workflows', icon: IconGitBranch },
  { label: 'Image Viewer', value: 'resultViewer', icon: IconPhotoSearch },
  { label: 'Cài đặt', value: 'settings', icon: IconSettings }
];

export default function App() {
  const dispatch = useDispatch();
  const selectedTab = useSelector((state) => state.workflows.selectedTab);
  const sidebarOpen = useSelector((state) => state.workflows.sidebarOpen);
  const currentTitle = NAV_ITEMS.find((item) => item.value === selectedTab)?.label || 'Damage Detector';

  useEffect(() => {
    dispatch(fetchWorkflows());
    return window.electronAPI.onWorkflowEvent((payload) => dispatch(sessionEvent(payload)));
  }, [dispatch]);

  useEffect(() => {
    if (window.location.pathname !== '/') window.history.replaceState(null, '', '/');
  }, []);

  return (
    <div className="flex h-screen min-w-0 flex-col overflow-hidden bg-[var(--docker-bg)] text-[var(--docker-text)] rv-font">
      <header className="app-drag flex h-11 shrink-0 items-center bg-[var(--docker-bg)] px-3">
        <div className="w-[76px] shrink-0" />
        <IconButton className="app-no-drag" label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'} onClick={() => dispatch(toggleSidebar())}>
          {sidebarOpen ? <IconLayoutSidebarLeftCollapse size={18} /> : <IconLayoutSidebarLeftExpand size={18} />}
        </IconButton>
        <div className="ml-3 truncate text-[13px] font-semibold text-[var(--docker-text)]">{currentTitle}</div>
      </header>

      <div className="flex min-h-0 min-w-0 flex-1 overflow-hidden">
        <nav className={cn('flex shrink-0 flex-col bg-[var(--docker-bg)] transition-[width] duration-200', sidebarOpen ? 'w-[240px]' : 'w-[60px]')}>
        <div className="grid gap-1 p-3">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = selectedTab === item.value;
            return (
              <button
                key={item.value}
                type="button"
                title={item.label}
                onClick={() => dispatch(setSelectedTab(item.value))}
                className={cn(
                  'flex h-9 items-center gap-2 rounded-md px-2 text-[13px] font-medium',
                  isActive ? 'bg-[var(--docker-active)] text-[var(--docker-blue)]' : 'text-[var(--docker-text)] hover:bg-[var(--docker-hover)]',
                  !sidebarOpen && 'justify-center px-0'
                )}
              >
                <Icon size={18} className="shrink-0" />
                {sidebarOpen && <span className="truncate">{item.label}</span>}
              </button>
            );
          })}
        </div>
        </nav>

        <section className="min-w-0 flex-1 overflow-hidden">
          {selectedTab === 'workflows' && <WorkflowsTab />}
          {selectedTab === 'resultViewer' && <ResultViewer />}
          {selectedTab === 'settings' && <SettingsPage />}
        </section>
      </div>
    </div>
  );
}