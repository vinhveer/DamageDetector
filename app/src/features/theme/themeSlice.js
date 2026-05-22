import { createSlice } from '@reduxjs/toolkit';

const KEY = 'damage-detector.theme';
const saved = localStorage.getItem(KEY);
const initial = saved === 'light' ? 'light' : 'dark';

// Apply immediately so there's no flash
document.documentElement.setAttribute('data-theme', initial);

const themeSlice = createSlice({
  name: 'theme',
  initialState: { mode: initial },
  reducers: {
    toggleTheme(state) {
      state.mode = state.mode === 'dark' ? 'light' : 'dark';
      localStorage.setItem(KEY, state.mode);
      document.documentElement.setAttribute('data-theme', state.mode);
    },
    setTheme(state, action) {
      state.mode = action.payload;
      localStorage.setItem(KEY, state.mode);
      document.documentElement.setAttribute('data-theme', state.mode);
    },
  },
});

export const { toggleTheme, setTheme } = themeSlice.actions;
export default themeSlice.reducer;
