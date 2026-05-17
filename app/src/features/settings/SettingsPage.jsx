import SettingsForm from './components/SettingsForm.jsx';
import { useSettings } from './hooks/useSettings.js';

export default function SettingsPage() {
  const settings = useSettings();

  return (
    <div className="rv-enter h-full min-w-0 overflow-auto bg-[var(--bg)] p-8 rv-font">
      <div className="mx-auto grid w-full max-w-[760px] gap-6">
        <div className="grid gap-1">
          <span className="text-[15px] font-semibold text-[var(--text)]">Settings</span>
          <span className="text-[13px] text-[var(--text-muted)]">Default output directory for workflow runs.</span>
        </div>

        <SettingsForm {...settings} />
      </div>
    </div>
  );
}
