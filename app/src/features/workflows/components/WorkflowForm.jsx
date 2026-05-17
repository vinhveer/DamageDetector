import { useEffect, useMemo, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Button } from '../../../components/ui/index.js';
import { setInputValue, sessionStarted } from '../workflowsSlice.js';
import WorkflowField from './WorkflowField.jsx';
import VenvDialog from './VenvDialog.jsx';

const EMPTY_VALUES = {};
const VENV_CONFIG_KEY = 'damage-detector.workflow-python';
const APP_SETTINGS_KEY = 'damage-detector.settings';

const readAppSettings = () => {
  try {
    return JSON.parse(window.localStorage.getItem(APP_SETTINGS_KEY) || '{}');
  } catch {
    return {};
  }
};

const inputDefaultValue = (input, settingsSaveDir = '') => {
  if (input.type === 'boolean') return Boolean(input.default);
  if (input.name === 'output_dir' && settingsSaveDir) return settingsSaveDir;
  return input.default ?? '';
};

const readVenvConfig = () => {
  try {
    return JSON.parse(window.localStorage.getItem(VENV_CONFIG_KEY) || 'null');
  } catch {
    return null;
  }
};

const writeVenvConfig = (config) => {
  window.localStorage.setItem(VENV_CONFIG_KEY, JSON.stringify(config));
};

export default function WorkflowForm({ workflow, onStarted }) {
  const dispatch = useDispatch();
  const formValues = useSelector((state) => state.workflows.formValues);
  const storedValues = formValues[workflow.id] || EMPTY_VALUES;
  const [settingsSaveDir, setSettingsSaveDir] = useState(() => readAppSettings().saveDir || '');
  const initialValues = useMemo(() => {
    const entries = (workflow.inputs || []).map((input) => [
      input.name,
      storedValues[input.name] ?? inputDefaultValue(input, settingsSaveDir)
    ]);
    return Object.fromEntries(entries);
  }, [settingsSaveDir, storedValues, workflow.id, workflow.inputs]);
  const [values, setValues] = useState(initialValues);
  const [venvDialogOpen, setVenvDialogOpen] = useState(false);
  const [venvDir, setVenvDir] = useState('');

  useEffect(() => {
    if (settingsSaveDir) return;
    let active = true;

    window.electronAPI.getDownloadsPath().then((downloadsPath) => {
      if (!active) return;
      const settings = readAppSettings();
      const saveDir = settings.saveDir || downloadsPath;
      setSettingsSaveDir(saveDir);
      if (!settings.saveDir) {
        window.localStorage.setItem(APP_SETTINGS_KEY, JSON.stringify({ ...settings, saveDir }));
      }
    });

    return () => {
      active = false;
    };
  }, [settingsSaveDir]);

  useEffect(() => {
    setValues(initialValues);
  }, [initialValues]);

  const updateValue = (name, value) => {
    setValues((current) => ({ ...current, [name]: value }));
    dispatch(setInputValue({ workflowId: workflow.id, name, value }));
  };

  const runWithConfig = async (config) => {
    const result = await window.electronAPI.startWorkflow({
      workflowId: workflow.id,
      values,
      venvDir: config?.venvDir || '',
      useGlobalPython: Boolean(config?.useGlobalPython)
    });
    dispatch(sessionStarted({ sessionId: result.sessionId, workflowId: workflow.id, workflowName: workflow.name }));
    onStarted?.();
  };

  const start = async () => {
    const config = readVenvConfig();
    if (!config) {
      setVenvDialogOpen(true);
      return;
    }
    await runWithConfig(config);
  };

  const useGlobal = async () => {
    const config = { useGlobalPython: true, venvDir: '' };
    writeVenvConfig(config);
    setVenvDialogOpen(false);
    await runWithConfig(config);
  };

  const setVenv = async () => {
    const config = { useGlobalPython: false, venvDir: venvDir.trim() };
    writeVenvConfig(config);
    setVenvDialogOpen(false);
    await runWithConfig(config);
  };

  return (
    <div className="grid gap-6">
      <section className="grid gap-4">
        <div className="text-[12px] font-medium text-[var(--text-muted)]">Parameters</div>
        {(workflow.inputs || []).map((input) => (
          <WorkflowField
            key={input.name}
            workflowId={workflow.id}
            input={input}
            value={values[input.name] ?? inputDefaultValue(input, settingsSaveDir)}
            onChange={updateValue}
          />
        ))}
      </section>

      <Button variant="primary" onClick={start} className="w-fit">
        Run
      </Button>
      {venvDialogOpen && (
        <VenvDialog value={venvDir} onChange={setVenvDir} onUseGlobal={useGlobal} onSet={setVenv} />
      )}
    </div>
  );
}