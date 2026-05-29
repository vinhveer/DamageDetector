import { useEffect, useState } from 'react';
import { EmptyState, ErrorMessage } from '../../../components/ui/index.js';
import { formatFloat, formatNum } from '../reviewConstants.js';

// Stage 2 — wired but light until full core mining runs on the dataset (SPEC §4.2).
export default function CoreClusterStage({ ctx }) {
  const { resemiDbPath, runId } = ctx;
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    window.electronAPI.getReviewCoreClusters({ resemiDbPath, runId })
      .then((res) => { if (res?.error) throw new Error(res.error); setClusters(res.clusters || []); })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [resemiDbPath, runId]);

  if (error) return <div className="p-6"><ErrorMessage message={error} /></div>;
  if (!loading && clusters.length === 0) {
    return <EmptyState title="No core clusters">Run core mining on the full dataset to populate this stage.</EmptyState>;
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto p-4">
      <table className="w-full text-[12px]">
        <thead className="text-left text-[var(--text-muted)]">
          <tr>
            <th className="py-1 pr-3">Cluster</th>
            <th className="py-1 pr-3">Label</th>
            <th className="py-1 pr-3">Members</th>
            <th className="py-1 pr-3">Density</th>
            <th className="py-1 pr-3">Agreement</th>
            <th className="py-1">Status</th>
          </tr>
        </thead>
        <tbody>
          {clusters.map((c) => (
            <tr key={c.core_cluster_id} className="border-t border-[var(--border-muted)] text-[var(--text)]">
              <td className="py-1.5 pr-3 font-mono">{c.core_cluster_id}</td>
              <td className="py-1.5 pr-3">{c.label}</td>
              <td className="py-1.5 pr-3 tabular-nums">{formatNum(c.member_count)}</td>
              <td className="py-1.5 pr-3 tabular-nums">{formatFloat(c.density_score, 2)}</td>
              <td className="py-1.5 pr-3 tabular-nums">{formatFloat(c.agreement_score, 2)}</td>
              <td className="py-1.5">{c.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
