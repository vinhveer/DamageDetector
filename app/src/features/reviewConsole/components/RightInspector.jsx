import { EmptyState } from '../../../components/ui/index.js';
import ReviewBadge from './ReviewBadge.jsx';
import { decisionTone, formatFloat, formatPct } from '../reviewConstants.js';

function Section({ title, children }) {
  return (
    <section className="border-b border-[var(--border-muted)] px-4 py-3">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">{title}</div>
      {children}
    </section>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-0.5 text-[12px]">
      <span className="text-[var(--text-muted)]">{label}</span>
      <span className="truncate text-right text-[var(--text)]">{value}</span>
    </div>
  );
}

export default function RightInspector({ item, evidence, evidenceLoading, draftDecision, history }) {
  if (!item) {
    return (
      <aside className="flex w-[320px] shrink-0 flex-col border-l border-[var(--border-muted)] bg-[var(--surface)]">
        <EmptyState title="No item selected">Pick a card to see evidence</EmptyState>
      </aside>
    );
  }

  return (
    <aside className="flex w-[320px] shrink-0 flex-col overflow-auto border-l border-[var(--border-muted)] bg-[var(--surface)]">
      <Section title="Decision summary">
        <Row label="Result id" value={`#${item.result_id}`} />
        <Row label="Current label" value={item.initial_label} />
        <Row label="Suggested" value={item.suggested_label || '-'} />
        {draftDecision && <Row label="Draft" value={<ReviewBadge tone="blue">{draftDecision.action}{draftDecision.new_label ? ` → ${draftDecision.new_label}` : ''}</ReviewBadge>} />}
        <Row label="Decision" value={<ReviewBadge tone={decisionTone(item.decision_type)}>{item.decision_type}</ReviewBadge>} />
        <Row label="Reliability" value={formatFloat(item.reliability_score)} />
      </Section>

      <Section title="Evidence — model outputs">
        {evidenceLoading && <div className="text-[12px] text-[var(--text-muted)]">Loading…</div>}
        {!evidenceLoading && (evidence?.model_outputs?.length ? (
          <div className="flex flex-col gap-1">
            {evidence.model_outputs.map((m) => (
              <div key={m.model_name} className="flex items-baseline justify-between gap-2 text-[12px]">
                <span className="truncate text-[var(--text-muted)]" title={m.source_type}>{m.model_name}</span>
                <span className="text-[var(--text)]">{m.top1_label} <span className="text-[var(--text-muted)]">{formatFloat(m.top1_score, 2)}</span></span>
              </div>
            ))}
          </div>
        ) : <div className="text-[12px] text-[var(--text-muted)]">No model outputs</div>)}
      </Section>

      <Section title="Scores">
        <Row label="Model agreement" value={formatPct(item.agreement_ratio)} />
        <Row label="Majority label" value={item.majority_label || '-'} />
        {item.conflict_labels?.length > 0 && <Row label="Conflicts" value={item.conflict_labels.join(', ')} />}
      </Section>

      <Section title="Nearest prototype / core">
        <Row label="Prototype" value={item.prototype_class ? `${item.prototype_class} (${formatFloat(item.prototype_similarity, 2)})` : '-'} />
        <Row label="Core" value={item.nearest_core_class ? `${item.nearest_core_class} (${formatFloat(item.nearest_core_similarity, 2)})` : '-'} />
      </Section>

      {evidence?.box_quality && (
        <Section title="Geometry">
          <Row label="Box quality" value={formatFloat(evidence.box_quality.box_quality_score, 2)} />
          <Row label="Area ratio" value={formatFloat(evidence.box_quality.area_ratio_to_image, 3)} />
          <Row label="Aspect ratio" value={formatFloat(evidence.box_quality.aspect_ratio, 2)} />
          <Row label="Children" value={evidence.box_quality.child_count} />
        </Section>
      )}

      <Section title="Reason codes">
        <div className="flex flex-wrap gap-1">
          {item.reason_codes?.length ? item.reason_codes.map((r) => (
            <ReviewBadge key={r} tone="amber">{r}</ReviewBadge>
          )) : <span className="text-[12px] text-[var(--text-muted)]">None</span>}
        </div>
      </Section>

      <Section title="History">
        {history?.length ? history.map((h, i) => (
          <Row key={i} label={h.session_id?.slice(0, 8) || 'prev'} value={`${h.action}${h.new_label ? ` → ${h.new_label}` : ''}`} />
        )) : <span className="text-[12px] text-[var(--text-muted)]">No prior committed decisions</span>}
      </Section>
    </aside>
  );
}
