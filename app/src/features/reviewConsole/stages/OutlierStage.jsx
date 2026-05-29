import { useEffect, useState } from 'react';
import { EmptyState, ErrorMessage } from '../../../components/ui/index.js';
import { formatFloat } from '../reviewConstants.js';

// Stage 4 — wired but light until core mining emits outliers (SPEC §4.4).
export default function OutlierStage({ ctx }) {
  const { resemiDbPath, runId } = ctx;
  const [outliers, setOutliers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    window.electronAPI.listReviewOutliers({ resemiDbPath, runId })
      .then((res) => { if (res?.error) throw new Error(res.error); setOutliers(res.outliers || []); })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [resemiDbPath, runId]);

  if (error) return <div className="p-6"><ErrorMessage message={error} /></div>;
  if (!loading && outliers.length === 0) {
    return <EmptyState title="No outliers">Core mining has not flagged rare/noise crops for this run yet.</EmptyState>;
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto p-4">
      <table className="w-full text-[12px]">
        <thead className="text-left text-[var(--text-muted)]">
          <tr>
            <th className="py-1 pr-3">Result</th>
            <th className="py-1 pr-3">Label</th>
            <th className="py-1 pr-3">Outlier type</th>
            <th className="py-1 pr-3">Nearest cluster</th>
            <th className="py-1">Similarity</th>
          </tr>
        </thead>
        <tbody>
          {outliers.map((o) => (
            <tr key={o.result_id} className="border-t border-[var(--border-muted)] text-[var(--text)]">
              <td className="py-1.5 pr-3 font-mono">#{o.result_id}</td>
              <td className="py-1.5 pr-3">{o.label}</td>
              <td className="py-1.5 pr-3">{o.outlier_type}</td>
              <td className="py-1.5 pr-3 font-mono">{o.nearest_cluster_id || '-'}</td>
              <td className="py-1.5 tabular-nums">{formatFloat(o.similarity, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
