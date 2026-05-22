import * as Switch from '@radix-ui/react-switch';
import { useDispatch, useSelector } from 'react-redux';
import { setTheme } from '../theme/themeSlice.js';
import SettingsForm from './components/SettingsForm.jsx';
import { useSettings } from './hooks/useSettings.js';

export default function SettingsPage() {
  const settings = useSettings();
  const dispatch = useDispatch();
  const mode = useSelector((s) => s.theme.mode);

  return (
    <div className="rv-enter h-full min-w-0 overflow-auto bg-[var(--bg)] p-8 rv-font">
      <div className="mx-auto grid w-full max-w-[760px] gap-6">
        <div className="grid gap-1">
          <span className="text-[15px] font-semibold text-[var(--text)]">Settings</span>
          <span className="text-[13px] text-[var(--text-muted)]">Application preferences and defaults.</span>
        </div>

        {/* Appearance */}
        <section className="grid gap-4 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-5">
          <span className="text-[12px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Appearance</span>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-[13px] font-medium text-[var(--text)]">Dark mode</div>
              <div className="text-[12px] text-[var(--text-muted)]">Toggle between light and dark theme</div>
            </div>
            <Switch.Root
              checked={mode === 'dark'}
              onCheckedChange={(v) => dispatch(setTheme(v ? 'dark' : 'light'))}
              className="rd-switch-root"
              aria-label="Dark mode"
            >
              <Switch.Thumb className="rd-switch-thumb" />
            </Switch.Root>
          </div>
        </section>

        <SettingsForm {...settings} />
      </div>
    </div>
  );
}
