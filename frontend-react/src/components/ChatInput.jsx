import { useState } from 'react';
import './ChatInput.css';

export function ChatInput({ onSend, isBusy, disabled }) {
  const [value, setValue] = useState('');

  const canSend = value.trim().length > 0 && !isBusy && !disabled;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!canSend) return;
    onSend(value.trim());
    setValue('');
  };

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <label htmlFor="chat-input-field" className="sr-only">
        Ask a question about your documents
      </label>
      <input
        id="chat-input-field"
        type="text"
        className="chat-input__field"
        placeholder={
          disabled ? 'Upload a document to get started…' : 'Ask a question about your documents…'
        }
        value={value}
        onChange={(e) => setValue(e.target.value)}
        disabled={disabled}
        autoComplete="off"
      />
      <button
        type="submit"
        className="chat-input__send"
        disabled={!canSend}
        aria-label="Send question"
      >
        {isBusy ? 'Thinking…' : 'Send'}
      </button>
    </form>
  );
}
