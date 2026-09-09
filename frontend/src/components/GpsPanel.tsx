import React from 'react';
import type { GpsState } from '../services/types';

interface GpsPanelProps {
  gps: GpsState;
  busId: string;
  onSetSource: (source: GpsState['source']) => void;
}

export const GpsPanel: React.FC<GpsPanelProps> = ({ gps, busId, onSetSource }) => {
  return (
    <div className="panel-section gps-panel">
      <h3>Live GPS Telemetry</h3>
      
      <div className="gps-source-selector">
        <label>Source:</label>
        <select 
          value={gps.source} 
          onChange={(e) => onSetSource(e.target.value as GpsState['source'])}
        >
          <option value="SIMULATED">Simulated</option>
          <option value="REPLAY">Replay (CSV)</option>
          <option value="REAL_DEVICE">Real Device</option>
        </select>
      </div>

      <div className="stat-grid gps-stats">
        <div className="stat-box small">
          <span className="stat-label">Bus ID</span>
          <span className="stat-value text-sm">{busId}</span>
        </div>
        <div className="stat-box small">
          <span className="stat-label">Status</span>
          <span className={`stat-value text-sm ${gps.status === 'CONNECTED' ? 'text-green' : 'text-orange'}`}>
            {gps.status}
          </span>
        </div>
        <div className="stat-box small">
          <span className="stat-label">Latitude</span>
          <span className="stat-value text-sm">{gps.latitude.toFixed(6)}</span>
        </div>
        <div className="stat-box small">
          <span className="stat-label">Longitude</span>
          <span className="stat-value text-sm">{gps.longitude.toFixed(6)}</span>
        </div>
        <div className="stat-box small">
          <span className="stat-label">Speed</span>
          <span className="stat-value text-sm">{gps.speed_kmh.toFixed(1)} km/h</span>
        </div>
        <div className="stat-box small">
          <span className="stat-label">Heading</span>
          <span className="stat-value text-sm">{gps.course_deg.toFixed(1)}°</span>
        </div>
      </div>
      
      {gps.source === 'SIMULATED' && (
        <div className="warning-text text-xs mt-2 text-orange">
          Warning: Using simulated GPS coordinates. Do not treat as real hardware GPS.
        </div>
      )}
    </div>
  );
};
