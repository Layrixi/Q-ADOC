import './SourceChip.css';

/**
 * Converts raw cross-encoder scores (unbounded logits) into a 0-100% display
 * value, scaled RELATIVE to the other sources in this answer.
 *
 * Why not a fixed sigmoid: cross-encoder scores can drift into very negative
 * ranges (especially after fine-tuning shifts the score distribution), which
 * would squash every score down near 0% even when one source is clearly
 * more relevant than another - correct ranking, misleading display. Min-max
 * scaling within the returned batch keeps the percentages meaningful: the
 * best match among what was actually returned shows high, the weakest shows
 * low, regardless of the absolute scale that particular query landed in.
 *
 * Trade-off: percentages are only comparable WITHIN one answer's source
 * list, not across different questions/answers. That's an acceptable
 * trade-off here since sources are only ever shown grouped by answer.
 */
function scoresToPercents(rawScores) {
  const min = Math.min(...rawScores);
  const max = Math.max(...rawScores);
  const range = max - min;

  if (range === 0) {
    // All sources scored identically - show them as equally strong matches
    // rather than dividing by zero.
    return rawScores.map(() => 100);
  }

  return rawScores.map((score) => Math.round(((score - min) / range) * 100));
}

export function SourceList({ sources }) {
  const percents = scoresToPercents(sources.map((s) => s.relevanceScore));

  return (
    <ul className="message-bubble__source-list">
      {sources.map((s, i) => (
        <li key={i} className="source-chip">
          <span className="source-chip__icon" aria-hidden="true">
            ◆
          </span>
          <span className="source-chip__name">{s.source}</span>
          <span className="source-chip__score">{percents[i]}% match</span>
        </li>
      ))}
    </ul>
  );
}
