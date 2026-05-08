import SettingsForm from './components/SettingsForm.jsx';
import { useSettings } from './hooks/useSettings.js';

export default function SettingsPage() {
  const settings = useSettings();

  return (
    <div className="rv-enter h-full min-w-0 overflow-auto bg-[var(--docker-bg)] p-8 rv-font">
      <div className="mx-auto grid w-full max-w-[760px] gap-6">
        <header className="grid gap-1">
          <h1 className="text-[18px] font-semibold text-[var(--docker-text)]">Cài đặt</h1>
          <p className="text-[13px] text-[var(--docker-muted)]">Chọn thư mục lưu kết quả mặc định.</p>
        </header>

        <SettingsForm {...settings} />
      </div>
    </div>
  );
}