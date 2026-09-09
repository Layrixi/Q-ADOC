import { useCallback, useRef, useState } from 'react';
import './UploadDropzone.css';

const ACCEPTED_TYPES = ['.pdf', '.docx', '.txt'];

export function UploadDropzone({ onUpload }) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [status, setStatus] = useState('idle'); // idle | uploading | success | error
  const [message, setMessage] = useState('');
  const inputRef = useRef(null);

  const handleFile = useCallback(
    async (file) => {
      if (!file) return;

      const ext = '.' + file.name.split('.').pop().toLowerCase();
      if (!ACCEPTED_TYPES.includes(ext)) {
        setStatus('error');
        setMessage(`Unsupported file type: ${ext}`);
        return;
      }

      setStatus('uploading');
      setMessage(`Indexing ${file.name}…`);

      try {
        const result = await onUpload(file);
        setStatus('success');
        setMessage(`${result.filename} — ${result.chunks_added} chunks added`);
      } catch (err) {
        setStatus('error');
        setMessage(err.message || 'Upload failed');
      }
    },
    [onUpload]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragActive(false);
      handleFile(e.dataTransfer.files?.[0]);
    },
    [handleFile]
  );

  return (
    <div>
      <div
        className={`dropzone dropzone--${status}${isDragActive ? ' dropzone--drag' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragActive(true);
        }}
        onDragLeave={() => setIsDragActive(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload a document. PDF, DOCX, or TXT files accepted."
        aria-disabled={status === 'uploading'}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          onChange={(e) => handleFile(e.target.files?.[0])}
          className="sr-only"
          aria-hidden="true"
          tabIndex={-1}
        />
        <span className="dropzone__label">
          {status === 'uploading' ? 'Indexing…' : 'Drop a document, or click to browse'}
        </span>
        <span className="dropzone__hint">PDF, DOCX, or TXT</span>
      </div>

      <div className="dropzone__status" role="status" aria-live="polite">
        {message && (
          <p className={`dropzone__message dropzone__message--${status}`}>{message}</p>
        )}
      </div>
    </div>
  );
}
