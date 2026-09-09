import { useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import './ChatArea.css';

export function ChatArea({ messages, onSend, isBusy, hasDocuments, onOpenSidebar }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-area">
      <header className="chat-area__topbar">
        <button
          type="button"
          className="chat-area__menu-button"
          onClick={onOpenSidebar}
          aria-label="Open document library"
          aria-controls="sidebar"
        >
          ☰
        </button>
        <h2 className="chat-area__title">Ask your library</h2>
      </header>

      <main id="main-content" className="chat-area__messages" ref={scrollRef}>
        {messages.length === 0 ? (
          <EmptyState hasDocuments={hasDocuments} />
        ) : (
          <div className="chat-area__message-list">
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                role={m.role}
                content={m.content}
                sources={m.sources}
                isLoading={m.isLoading}
              />
            ))}
          </div>
        )}
      </main>

      <ChatInput onSend={onSend} isBusy={isBusy} disabled={!hasDocuments} />
    </div>
  );
}

function EmptyState({ hasDocuments }) {
  return (
    <div className="empty-state">
      <span className="empty-state__icon" aria-hidden="true">
        ⚔
      </span>
      <h3 className="empty-state__title">
        {hasDocuments ? 'Ask anything about your library' : 'Your library is empty'}
      </h3>
      <p className="empty-state__body">
        {hasDocuments
          ? 'Questions are answered using only what your documents actually say, with sources cited below each answer.'
          : 'Upload a PDF, DOCX, or TXT file from the sidebar to start asking questions grounded in its contents.'}
      </p>
    </div>
  );
}
