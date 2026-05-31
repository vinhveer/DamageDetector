import * as Switch from '@radix-ui/react-switch';
import * as Tooltip from '@radix-ui/react-tooltip';
import { IconMoon, IconSun } from '@tabler/icons-react';
import { useDispatch, useSelector } from 'react-redux';
import { toggleTheme } from './themeSlice.js';

export default function ThemeToggle() {
  const dispatch = useDispatch();
  const mode = useSelector((s) => s.theme.mode);
  const isDark = mode === 'dark';

  return (
    <Tooltip.Provider delayDuration={400}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <div className="app-no-drag flex items-center gap-1.5">
            <IconSun size={13} className={isDark ? 'text-[var(--text-subtle)]' : 'text-[var(--warning)]'} />
            <Switch.Root
              checked={isDark}
              onCheckedChange={() => dispatch(toggleTheme())}
              className="rd-switch-root"
              aria-label="Toggle dark mode"
            >
              <Switch.Thumb className="rd-switch-thumb" />
            </Switch.Root>
            <IconMoon size={13} className={isDark ? 'text-[var(--primary)]' : 'text-[var(--text-subtle)]'} />
          </div>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content className="rd-tooltip-content" side="bottom" sideOffset={6}>
            {isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}
