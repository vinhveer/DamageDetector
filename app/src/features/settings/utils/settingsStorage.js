export const SETTINGS_KEY = 'damage-detector.settings';

export const readSettings = () => {
  try {
    return JSON.parse(window.localStorage.getItem(SETTINGS_KEY) || '{}');
  } catch {
    return {};
  }
};

export const writeSettings = (settings) => {
  window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
};
