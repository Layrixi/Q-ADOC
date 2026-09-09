import './StatusBadge.css';

export function StatusBadge({ isConnected }) {
  const label =
    isConnected === null ? 'Checking connection…' : isConnected ? 'Connected' : 'Disconnected';

  const state = isConnected === null ? 'checking' : isConnected ? 'ok' : 'error';

  return (
    <div className="status-badge" role="status">
      <span className={`status-badge__dot status-badge__dot--${state}`} aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}
