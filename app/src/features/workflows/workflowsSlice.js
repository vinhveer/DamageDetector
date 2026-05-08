import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';

export const fetchWorkflows = createAsyncThunk('workflows/fetchWorkflows', async () => {
  return window.electronAPI.listWorkflows();
});

const initialState = {
  items: [],
  status: 'idle',
  selectedTab: 'workflows',
  selectedWorkflowId: null,
  sidebarOpen: true,
  formValues: {},
  sessions: []
};

const workflowsSlice = createSlice({
  name: 'workflows',
  initialState,
  reducers: {
    setSelectedTab(state, action) {
      state.selectedTab = action.payload;
    },
    setSelectedWorkflow(state, action) {
      state.selectedWorkflowId = action.payload;
    },
    toggleSidebar(state) {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setInputValue(state, action) {
      const { workflowId, name, value } = action.payload;
      state.formValues[workflowId] = {
        ...(state.formValues[workflowId] || {}),
        [name]: value
      };
    },
    sessionStarted(state, action) {
      state.sessions.unshift({
        id: action.payload.sessionId,
        workflowId: action.payload.workflowId,
        workflowName: action.payload.workflowName,
        status: 'running',
        log: ''
      });
    },
    sessionEvent(state, action) {
      const session = state.sessions.find((item) => item.id === action.payload.sessionId);
      if (!session) {
        return;
      }
      if (action.payload.type === 'closed') {
        session.status = action.payload.data === '0' ? 'done' : 'error';
        return;
      }
      session.log += `${action.payload.data}`;
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchWorkflows.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchWorkflows.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.items = action.payload;
        state.selectedWorkflowId = null;
      })
      .addCase(fetchWorkflows.rejected, (state) => {
        state.status = 'failed';
      });
  }
});

export const { setSelectedTab, setSelectedWorkflow, toggleSidebar, setInputValue, sessionStarted, sessionEvent } =
  workflowsSlice.actions;
export default workflowsSlice.reducer;
