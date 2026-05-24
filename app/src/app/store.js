import { configureStore } from '@reduxjs/toolkit';
import workflowsReducer from '../features/workflows/workflowsSlice.js';
import segmentReducer from '../features/segment/segmentSlice.js';
import themeReducer from '../features/theme/themeSlice.js';
import inspectionWizardReducer from '../features/inspectionWizard/inspectionWizardSlice.js';

export const store = configureStore({
  reducer: {
    workflows: workflowsReducer,
    segment: segmentReducer,
    theme: themeReducer,
    inspectionWizard: inspectionWizardReducer,
  }
});
