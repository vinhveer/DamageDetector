import { Component, Suspense, lazy, useEffect } from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
import {
  IconGitBranch,
  IconLayoutSidebarLeftCollapse,
  IconLayoutSidebarLeftExpand,
  IconPhotoSearch,
  IconRoute,
  IconScissors,
  IconSettings,
  IconTargetArrow
} from '@tabler/icons-react';
import { useDispatch, useSelector } from 'react-redux';
import {
  fetchWorkflows,
  sessionEvent,
  setSelectedTab,
  toggleSidebar
} from './features/workflows/workflowsSlice.js';
import { IconButton } from './components/ui/index.js';
import { cn } from './components/ui/cn.js';

const WorkflowsTab = lazy(() => import('./features/workflows/components/WorkflowsTab.jsx'));
const ResultViewer = lazy(() => import('./features/resultViewer/ResultViewer.jsx'));
const PrototypeReview = lazy(() => import('./features/prototypeReview/PrototypeReview.jsx'));
const SegmentTab = lazy(() => import('./features/segment/SegmentTab.jsx'));
const InspectionWizard = lazy(() => import('./features/inspectionWizard/InspectionWizard.jsx'));
const SettingsPage = lazy(() => import('./features/settings/SettingsPage.jsx'));

const NAV_MAIN = [
  { label: 'Workflows',        value: 'workflows',      icon: IconGitBranch    },
  { label: 'Image Viewer',     value: 'resultViewer',   icon: IconPhotoSearch  },
  { label: 'Prototype Review', value: 'prototypeReview',icon: IconTargetArrow  },
  { label: 'Segment',          value: 'segment',        icon: IconScissors     },
  { label: 'Inspection',       value: 'inspection',     icon: IconRoute        },
];

const NAV_BOTTOM = [
  { label: 'Settings', value: 'settings', icon: IconSettings },
];

function NavItem({ item, isActive, sidebarOpen, onClick }) {
  const Icon = item.icon;
  const btn = (
    <button
      type="button"
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

  if (sidebarOpen) return btn;

  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{btn}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="rd-tooltip-content" side="right" sideOffset={8}>
          {item.label}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

class TabErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidUpdate(prevProps) {
    if (prevProps.tabKey !== this.props.tabKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex h-full items-center justify-center bg-[var(--bg)] p-6">
          <div className="max-w-[520px] rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-4">
            <div className="text-[14px] font-semibold text-[var(--text)]">Tab failed to render</div>
            <p className="mt-2 text-[13px] text-[var(--text-muted)]">
              {this.state.error?.message || 'Unknown UI error'}
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

function TabFallback() {
  return (
    <div className="flex h-full items-center justify-center bg-[var(--bg)] text-[13px] text-[var(--text-muted)]">
      Loading...
    </div>
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
    <Tooltip.Provider delayDuration={300}>
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
            <TabErrorBoundary tabKey={selectedTab}>
              <Suspense fallback={<TabFallback />}>
                {selectedTab === 'workflows'      && <WorkflowsTab />}
                {selectedTab === 'resultViewer'   && <ResultViewer />}
                {selectedTab === 'prototypeReview'&& <PrototypeReview />}
                {selectedTab === 'segment'        && <SegmentTab />}
                {selectedTab === 'inspection'     && <InspectionWizard />}
                {selectedTab === 'settings'       && <SettingsPage />}
              </Suspense>
            </TabErrorBoundary>
          </section>

        </div>
      </div>
    </Tooltip.Provider>
  );
}
