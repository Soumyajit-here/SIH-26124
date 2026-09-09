import React, { useState } from 'react';
import { fetchEventObservations, type Observation } from '../services/api';

interface EventPopupProps {
  feature: any;
}

export const EventPopup: React.FC<EventPopupProps> = ({ feature }) => {
  const { properties } = feature;
  const [observations, setObservations] = useState<Observation[]>([]);
  const [loading, setLoading] = useState(false);
  const [showObs, setShowObs] = useState(false);

  const handleLoadObservations = async () => {
    if (showObs) {
      setShowObs(false);
      return;
    }
    setLoading(true);
    try {
      const data = await fetchEventObservations(properties.event_id);
      setObservations(data);
      setShowObs(true);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="popup-container">
      <h3>{properties.event_type} <span className="event-id">#{properties.event_id}</span></h3>
      
      <div className="popup-stats">
        <div className="stat-row">
          <span>Confidence:</span>
          <strong>{properties.aggregated_confidence.toFixed(2)}</strong>
        </div>
        <div className="stat-row">
          <span>Reports:</span>
          <strong>{properties.observation_count}</strong>
        </div>
        <div className="stat-row">
          <span>Unique buses:</span>
          <strong>{properties.unique_bus_count}</strong>
        </div>
        <div className="stat-row">
          <span>Status:</span>
          <span className={`status-badge ${properties.status.toLowerCase()}`}>
            {properties.status}
          </span>
        </div>
        <div className="stat-row">
          <span>Severity:</span>
          <span className={`severity-badge ${properties.severity.toLowerCase()}`}>
            {properties.severity}
          </span>
        </div>
      </div>

      <button className="obs-toggle-btn" onClick={handleLoadObservations}>
        {loading ? 'Loading...' : showObs ? 'Hide Observations' : 'View Raw Evidence'}
      </button>

      {showObs && (
        <div className="observations-list">
          <h4>Raw Evidence ({observations.length})</h4>
          {observations.map((obs) => (
            <div key={obs.observation_id} className="obs-card">
              <div className="obs-header">
                <span className="bus-id">🚌 {obs.bus_id}</span>
                <span className="obs-time">{new Date(obs.timestamp).toLocaleTimeString()}</span>
              </div>
              <div className="obs-details">
                <span>Conf: {obs.confidence.toFixed(2)}</span>
                <span>Model: {obs.model}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
