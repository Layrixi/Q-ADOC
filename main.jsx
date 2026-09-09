import { useCallback, useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { useApi } from './hooks/useApi';
import './App.css';

let nextMessageId = 1;

export default function App() {
  const { isConnected, documents, totalChunks, uploadDocument, askQuestion } = useApi();
  const [messages, setMessages] = useState([]);
  const [isBusy, setIsBusy] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const handleSend = useCallback(
    async (question) => {
      const userMessageId = nextMessageId++;
      const loadingMessageId = nextMessageId++;

      setMessages((prev) => [
        ...prev,
        { id: userMessageId, role: 'user', content: question },
        { id: loadingMessageId, role: 'assistant', isLoading: true },
      ]);
      setIsBusy(true);

      try {
        const result = await askQuestion(question);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMessageId
              ? { ...m, isLoading: false, content: result.answer, sources: result.sources }
              : m
          )
        );
      } catch (err) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMessageId
              ? {
                  ...m,
                  isLoading: false,
                  content: `Something went wrong: ${err.message}. Check that the backend is running.`,
                }
              : m
          )
        );
      } finally {
        setIsBusy(false);
      }
    },
    [askQuestion]
  );

  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">
        Skip to chat
      </a>

      <Sidebar
        isConnected={isConnected}
        documents={documents}
        totalChunks={totalChunks}
        onUpload={uploadDocument}
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
      />

      <ChatArea
        messages={messages}
        onSend={handleSend}
        isBusy={isBusy}
        hasDocuments={documents.length > 0}
        onOpenSidebar={() => setIsSidebarOpen(true)}
      />
    </div>
  );
}
