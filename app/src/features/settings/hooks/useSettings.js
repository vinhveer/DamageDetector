import { useEffect, useState } from 'react';
import { readSettings, writeSettings } from '../utils/settingsStorage.js';

export const useSettings = () => {
  const [saveDir, setSaveDir] = useState('');
  const [defaultDir, setDefaultDir] = useState('');

  useEffect(() => {
    let active = true;
    const settings = readSettings();

    window.electronAPI.getDownloadsPath().then((downloadsPath) => {
      if (!active) return;
      const nextSaveDir = settings.saveDir || downloadsPath;
      setDefaultDir(downloadsPath);
      setSaveDir(nextSaveDir);
      if (!settings.saveDir) {
        writeSettings({ ...settings, saveDir: nextSaveDir });
      }
    });

    return () => {
      active = false;
    };
  }, []);

  const chooseFolder = async () => {
    const selected = await window.electronAPI.browsePath('directory');
    if (!selected) return;
    setSaveDir(selected);
    writeSettings({ ...readSettings(), saveDir: selected });
  };

  const updateSaveDir = (value) => {
    setSaveDir(value);
    writeSettings({ ...readSettings(), saveDir: value });
  };

  const resetDefault = () => {
    if (!defaultDir) return;
    setSaveDir(defaultDir);
    writeSettings({ ...readSettings(), saveDir: defaultDir });
  };

  return {
    saveDir,
    defaultDir,
    chooseFolder,
    resetDefault,
    updateSaveDir
  };
};
