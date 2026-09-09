import ReactMarkdown from 'react-markdown';
import { SourceList } from './SourceChip';
import './MessageBubble.css';

export function MessageBubble({ role, content, sources, isLoading }) {
  const isUser = role === 'user';

  return (
    <div
      className={`message-row message-row--${isUser ? 'user' : 'assistant'}`}
      role={isUser ? undefined : 'status'}
    >
      <div className="message-bubble">
        {isLoading ? (
          <ThinkingIndicator />
        ) : isUser ? (
          <p className="message-bubble__text">{content}</p>
        ) : (
          <>
            <div className="message-bubble__text message-bubble__text--markdown">
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
            {sources && sources.length > 0 && (
              <div className="message-bubble__sources">
                <span className="sr-only">Sources used for this answer:</span>
                <SourceList
                  sources={sources.map((s) => ({
                    source: s.source,
                    relevanceScore: s.relevance_score,
                  }))}
                />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div className="thinking-indicator">
      <span className="sr-only">Generating answer…</span>
      <span className="thinking-indicator__dot" aria-hidden="true" />
      <span className="thinking-indicator__dot" aria-hidden="true" />
      <span className="thinking-indicator__dot" aria-hidden="true" />
    </div>
  );
}
