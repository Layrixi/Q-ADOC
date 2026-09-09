import { StatusBadge } from './StatusBadge';
import { UploadDropzone } from './UploadDropzone';
import { DocumentList } from './DocumentList';
import './Sidebar.css';

export function Sidebar({ isConnected, documents, totalChunks, onUpload, isOpen, onClose }) {
  return (
    <>
      {/* Backdrop only rendered (and clickable) on narrow screens when the drawer is open */}
      {isOpen && <div className="sidebar-backdrop" onClick={onClose} aria-hidden="true" />}

      <aside
        id="sidebar"
        className={`sidebar${isOpen ? ' sidebar--open' : ''}`}
        aria-label="Document library"
      >
        <div className="sidebar__header">
          <h1 className="sidebar__wordmark">Q&ADOC</h1>
          <button
            type="button"
            className="sidebar__close"
            onClick={onClose}
            aria-label="Close document library"
          >
            ✕
          </button>
        </div>

        <StatusBadge isConnected={isConnected} />

        <section className="sidebar__section" aria-labelledby="upload-heading">
          <h2 id="upload-heading" className="sidebar__section-title">
            Add a document
          </h2>
          <UploadDropzone onUpload={onUpload} />
        </section>

        <section className="sidebar__section sidebar__section--grow" aria-labelledby="docs-heading">
          <h2 id="docs-heading" className="sidebar__section-title">
            Library
          </h2>
          <DocumentList documents={documents} totalChunks={totalChunks} />
        </section>
      </aside>
    </>
  );
}
