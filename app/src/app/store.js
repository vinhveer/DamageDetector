import { configureStore } from '@reduxjs/toolkit';
import workflowsReducer from '../features/workflows/workflowsSlice.js';
import resultViewerReducer from '../features/resultViewer/resultViewerSlice.js';

export const store = configureStore({
  reducer: {
    workflows: workflowsReducer,
    resultViewer: resultViewerReducer
  }
});
