import React from 'react';
import type { FleetBusStatus } from '../services/types';

interface FleetPanelProps {
  features: any[];
  fleet: FleetBusStatus[];
  filterType: string;
  setFilterType: (type: string) => void;
  onRefresh: () => void;
}

export const FleetPanel: React.FC<FleetPanelProps> = ({ 
  features, 
  fleet,
  filterType, 
  setFilterType, 
  onRefresh 
}) => {
  
  const eventTypes = Array.from(new Set(features.map(f => f.properties.event_type)));

  return (
    <div className="panel-section">
      <div className="panel-header-small">
        <h3>Event Filters</h3>
        <button className="refresh-btn small" onClick={onRefresh}>↻</button>
      </div>
      
      <div className="filter-group">
        <select value={filterType} onChange={(e) => setFilterType(e.target.value)}>
          <option value="All">All Types</option>
          {eventTypes.map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      <div className="panel-header-small mt-3">
        <h3>Fleet Status</h3>
      </div>
      
      <ul className="fleet-list">
        {fleet.length > 0 ? (
          fleet.map(bus => (
            <li key={bus.bus_id} className="fleet-item">
              <span className="bus-icon">🚌</span>
              <div className="fleet-item-info">
                <span className="bus-name">{bus.bus_id}</span>
                <span className="fleet-last-seen">
                  {new Date(bus.last_seen).toLocaleTimeString()}
                </span>
              </div>
              <span className={`bus-status ${bus.status === 'Online' ? 'online' : 'offline'}`}>
                {bus.status}
              </span>
            </li>
          ))
        ) : (
          <li className="fleet-item empty">
            <span className="empty-state">No buses connected yet</span>
          </li>
        )}
      </ul>
      
      <div className="stat-grid mt-3">
        <div className="stat-box small">
          <span className="stat-value text-sm">{features.length}</span>
          <span className="stat-label">Active Events</span>
        </div>
        <div className="stat-box small">
          <span className="stat-value text-sm">{fleet.length}</span>
          <span className="stat-label">Buses Tracked</span>
        </div>
      </div>
    </div>
  );
};
