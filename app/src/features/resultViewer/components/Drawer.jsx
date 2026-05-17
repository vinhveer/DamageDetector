import { useEffect, useState } from 'react';
import { IconX } from '@tabler/icons-react';
import { IconButton } from '../../../components/ui/index.js';
import { cn } from '../../../components/ui/cn.js';

export default function Drawer({ opened, title, onClose, children }) {
  const [mounted, setMounted] = useState(opened);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (opened) {
      setMounted(true);
      const firstFrame = window.requestAnimationFrame(() => {
        const secondFrame = window.requestAnimationFrame(() => setVisible(true));
        return secondFrame;
      });
      return () => window.cancelAnimationFrame(firstFrame);
    }
    setVisible(false);
    const timeout = window.setTimeout(() => setMounted(false), 180);
    return () => window.clearTimeout(timeout);
  }, [opened]);

  useEffect(() => {
    if (mounted && opened) {
      const firstFrame = window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => setVisible(true));
      });
      return () => window.cancelAnimationFrame(firstFrame);
    }
    return undefined;
  }, [mounted, opened]);

  useEffect(() => {
    if (opened) {
      return undefined;
    }
    setVisible(false);
    return undefined;
  }, [opened]);

  useEffect(() => {
    if (!mounted) return undefined;
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose?.();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mounted, onClose]);

  if (!mounted) return null;

  return (
    <div
      className={cn(
        'fixed inset-0 z-50 flex justify-end bg-black/40 transition-opacity duration-150 ease-out',
        visible ? 'opacity-100' : 'opacity-0'
      )}
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) onClose?.();
      }}
    >
      <aside
        className={cn(
          'flex h-full w-[380px] flex-col border-l border-[var(--border)] bg-[var(--surface)] shadow-[0_2px_8px_rgba(0,0,0,0.4)]',
          'transition-transform duration-150 ease-out will-change-transform',
          visible ? 'translate-x-0' : 'translate-x-full'
        )}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex min-h-14 items-center justify-between border-b border-[var(--border-muted)] px-5">
          <h2 className="text-[14px] font-semibold text-[var(--text)]">{title}</h2>
          <IconButton label="Close" onClick={() => onClose?.()} className="app-no-drag">
            <IconX size={16} />
          </IconButton>
        </div>
        <div className="min-h-0 flex-1 overflow-auto p-5">{children}</div>
      </aside>
    </div>
  );
}
