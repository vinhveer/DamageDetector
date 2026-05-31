import { useEffect, useState } from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
import {
  IconLayoutSidebarLeftCollapse,
  IconLayoutSidebarLeftExpand,
  IconSettings,
} from '@tabler/icons-react';
import { IconButton } from './components/ui/index.js';
import { cn } from './components/ui/cn.js';

// Layout shell only. Feature screens were removed — add new ones to NAV and
// render them in the content section below.
const NAV_MAIN = [];

const NAV_BOTTOM = [
  { label: 'Settings', value: 'settings', icon: IconSettings },
];

const ALL_NAV_ITEMS = [...NAV_MAIN, ...NAV_BOTTOM];

const DEFAULT_TAB = ALL_NAV_ITEMS[0]?.value ?? '';

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

function EmptyContent() {
  return (
    <div className="flex h-full items-center justify-center bg-[var(--bg)] text-[13px] text-[var(--text-muted)]">
      No screens yet — this is the layout shell.
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState(DEFAULT_TAB);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const currentTitle = ALL_NAV_ITEMS.find((i) => i.value === activeTab)?.label ?? 'Semi-labeling Review';

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
            onClick={() => setSidebarOpen((v) => !v)}
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
                  isActive={activeTab === item.value}
                  sidebarOpen={sidebarOpen}
                  onClick={() => setActiveTab(item.value)}
                />
              ))}
            </div>

            <div className="mt-auto flex flex-col gap-0.5 border-t border-[var(--border-muted)] p-2">
              {NAV_BOTTOM.map((item) => (
                <NavItem
                  key={item.value}
                  item={item}
                  isActive={activeTab === item.value}
                  sidebarOpen={sidebarOpen}
                  onClick={() => setActiveTab(item.value)}
                />
              ))}
            </div>
          </nav>

          {/* Main content */}
          <section className="min-w-0 flex-1 overflow-hidden bg-[var(--bg)]">
            <EmptyContent />
          </section>

        </div>
      </div>
    </Tooltip.Provider>
  );
}
