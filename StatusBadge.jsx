import './DocumentList.css';

export function DocumentList({ documents, totalChunks }) {
  if (documents.length === 0) {
    return (
      <p className="document-list__empty">
        No documents indexed yet. Upload one to get started.
      </p>
    );
  }

  return (
    <div>
      <ul className="document-list" aria-label="Indexed documents">
        {documents.map((name) => (
          <li key={name} className="document-list__item">
            <span className="document-list__icon" aria-hidden="true">
              📖
            </span>
            <span className="document-list__name">{name}</span>
          </li>
        ))}
      </ul>
      <p className="document-list__total">{totalChunks} chunks total</p>
    </div>
  );
}
