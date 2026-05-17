import { useEffect } from 'react';
import {
  IconGitBranch,
  IconLayoutSidebarLeftCollapse,
  IconLayoutSidebarLeftExpand,
  IconPhotoSearch,
  IconSettings,
  IconTargetArrow
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
import PrototypeReview from './features/prototypeReview/PrototypeReview.jsx';

const NAV_MAIN = [
  { label: 'Workflows',        value: 'workflows',      icon: IconGitBranch    },
  { label: 'Image Viewer',     value: 'resultViewer',   icon: IconPhotoSearch  },
  { label: 'Prototype Review', value: 'prototypeReview',icon: IconTargetArrow  },
];

const NAV_BOTTOM = [
  { label: 'Settings', value: 'settings', icon: IconSettings },
];

function NavItem({ item, isActive, sidebarOpen, onClick }) {
  const Icon = item.icon;
  return (
    <button
      key={item.value}
      type="button"
      title={!sidebarOpen ? item.label : undefined}
      onClick={onClick}
      className={cn(
        'flex h-8 w-full items-center gap-2.5 rounded-[5px] px-2.5 text-[13px] font-medium',
        isActive
          ? 'bg-[var(--active)] text-[var(--text)]'
          : 'text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
        !sidebarOpen && 'justify-center px-0'
      )}
    >
      <Icon size={16} className="shrink-0" />
      {sidebarOpen && <span className="truncate">{item.label}</span>}
    </button>
  );
}

export default function App() {
  const dispatch = useDispatch();
  const selectedTab  = useSelector((state) => state.workflows.selectedTab);
  const sidebarOpen  = useSelector((state) => state.workflows.sidebarOpen);
  const currentTitle = [...NAV_MAIN, ...NAV_BOTTOM].find((i) => i.value === selectedTab)?.label ?? 'Damage Detector';

  useEffect(() => {
    dispatch(fetchWorkflows());
    return window.electronAPI.onWorkflowEvent((payload) => dispatch(sessionEvent(payload)));
  }, [dispatch]);

  useEffect(() => {
    if (window.location.pathname !== '/') window.history.replaceState(null, '', '/');
  }, []);

  return (
    <div className="flex h-screen min-w-0 flex-col overflow-hidden bg-[var(--bg)] text-[var(--text)] rv-font">

      {/* Title bar */}
      <header className="app-drag flex h-10 shrink-0 items-center border-b border-[var(--border-muted)] bg-[var(--surface)] px-3">
        <div className="w-[76px] shrink-0" />
        <IconButton
          className="app-no-drag"
          label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          onClick={() => dispatch(toggleSidebar())}
        >
          {sidebarOpen
            ? <IconLayoutSidebarLeftCollapse size={16} />
            : <IconLayoutSidebarLeftExpand  size={16} />
          }
        </IconButton>
        <span className="ml-2.5 text-[13px] font-medium text-[var(--text-muted)]">{currentTitle}</span>
      </header>

      <div className="flex min-h-0 min-w-0 flex-1 overflow-hidden">

        {/* Sidebar */}
        <nav
          className={cn(
            'flex shrink-0 flex-col border-r border-[var(--border-muted)] bg-[var(--surface)]',
            'transition-[width] duration-200',
            sidebarOpen ? 'w-[220px]' : 'w-[48px]'
          )}
        >
          <div className="flex flex-col gap-0.5 p-2">
            {NAV_MAIN.map((item) => (
              <NavItem
                key={item.value}
                item={item}
                isActive={selectedTab === item.value}
                sidebarOpen={sidebarOpen}
                onClick={() => dispatch(setSelectedTab(item.value))}
              />
            ))}
          </div>

          <div className="mt-auto flex flex-col gap-0.5 border-t border-[var(--border-muted)] p-2">
            {NAV_BOTTOM.map((item) => (
              <NavItem
                key={item.value}
                item={item}
                isActive={selectedTab === item.value}
                sidebarOpen={sidebarOpen}
                onClick={() => dispatch(setSelectedTab(item.value))}
              />
            ))}
          </div>
        </nav>

        {/* Main content */}
        <section className="min-w-0 flex-1 overflow-hidden bg-[var(--bg)]">
          {selectedTab === 'workflows'      && <WorkflowsTab />}
          {selectedTab === 'resultViewer'   && <ResultViewer />}
          {selectedTab === 'prototypeReview'&& <PrototypeReview />}
          {selectedTab === 'settings'       && <SettingsPage />}
        </section>

      </div>
    </div>
  );
}
