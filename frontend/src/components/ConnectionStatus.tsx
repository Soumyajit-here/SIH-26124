import React from 'react';

interface ConnectionStatusProps {
  backendOnline: boolean;
  cameraActive: boolean;
  gpsStatus: string;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  backendOnline,
  cameraActive,
  gpsStatus,
}) => {
  return (
    <div className="connection-status-bar">
      <div className="status-item">
        <span className={`status-dot ${backendOnline ? 'online' : 'offline'}`}></span>
        Backend: {backendOnline ? 'CONNECTED' : 'OFFLINE'}
      </div>
      <div className="status-item">
        <span className={`status-dot ${cameraActive ? 'online' : 'offline'}`}></span>
        Camera: {cameraActive ? 'CONNECTED' : 'DISCONNECTED'}
      </div>
      <div className="status-item">
        <span className={`status-dot ${gpsStatus === 'CONNECTED' ? 'online' : gpsStatus === 'SEARCHING' ? 'warning' : 'offline'}`}></span>
        GPS: {gpsStatus}
      </div>
      <div className="status-item">
        <span className="status-dot online"></span>
        Model 1: READY
      </div>
      <div className="status-item">
        <span className="status-dot online"></span>
        Model 2: READY
      </div>
    </div>
  );
};
