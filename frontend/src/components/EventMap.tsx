import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { EventPopup } from './EventPopup';
import type { FleetBusStatus } from '../services/types';

const colorMap: Record<string, string> = {
  Pothole: '#ef4444',
  Crack: '#f97316',
  Manhole: '#eab308',
  SpeedBump: '#3b82f6',
  TrafficCone: '#fb923c',
  Rock: '#78716c',
  RoadDebris: '#64748b',
  FallenTree: '#16a34a',
  HMV: '#3b82f6',
  LMV: '#22c55e',
  Pedestrian: '#eab308',
};

// Custom bus icon
const busIcon = L.divIcon({
  className: 'bus-marker-icon',
  html: '<div class="bus-marker">🚌</div>',
  iconSize: [32, 32],
  iconAnchor: [16, 16],
});

// Component to dynamically adjust map bounds
const MapBounds: React.FC<{ features: any[]; fleet: FleetBusStatus[] }> = ({ features, fleet }) => {
  const map = useMap();
  
  React.useEffect(() => {
    const points: [number, number][] = [];
    
    features.forEach(f => {
      points.push([f.geometry.coordinates[1], f.geometry.coordinates[0]]);
    });
    
    fleet.forEach(b => {
      if (b.latitude && b.longitude) {
        points.push([b.latitude, b.longitude]);
      }
    });

    if (points.length > 0) {
      const bounds = L.latLngBounds(points);
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 18 });
    }
  }, [features, fleet, map]);
  
  return null;
};

interface EventMapProps {
  features: any[];
  fleet?: FleetBusStatus[];
}

export const EventMap: React.FC<EventMapProps> = ({ features, fleet = [] }) => {
  return (
    <div className="map-wrapper">
      <MapContainer 
        center={[28.43, 77.01]}
        zoom={13} 
        style={{ height: '100%', width: '100%' }}
        zoomControl={false}
        preferCanvas={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        <MapBounds features={features} fleet={fleet} />

        {/* Event markers */}
        {features.map((feature, idx) => {
          const [lon, lat] = feature.geometry.coordinates;
          const eventType = feature.properties.event_type;
          const markerColor = colorMap[eventType] || '#9ca3af';

          return (
            <CircleMarker 
              key={`event-${feature.properties.event_id || idx}`} 
              center={[lat, lon]} 
              radius={8}
              pathOptions={{ 
                fillColor: markerColor, 
                fillOpacity: 0.8, 
                color: '#ffffff', 
                weight: 2 
              }}
            >
              <Popup className="custom-popup">
                <EventPopup feature={feature} />
              </Popup>
            </CircleMarker>
          );
        })}

        {/* Bus markers — distinct from events */}
        {fleet.map((bus) => {
          if (!bus.latitude || !bus.longitude) return null;
          return (
            <Marker 
              key={`bus-${bus.bus_id}`}
              position={[bus.latitude, bus.longitude]}
              icon={busIcon}
            >
              <Popup>
                <div className="popup-container bus-popup">
                  <h3>🚌 {bus.bus_id}</h3>
                  <div className="popup-stats">
                    <div className="stat-row">
                      <span>Status:</span>
                      <span className={`status-badge ${bus.status === 'Online' ? 'active' : ''}`}>
                        {bus.status}
                      </span>
                    </div>
                    <div className="stat-row">
                      <span>Speed:</span>
                      <strong>{bus.speed_kmh?.toFixed(1) ?? '—'} km/h</strong>
                    </div>
                    <div className="stat-row">
                      <span>Last Update:</span>
                      <strong>{new Date(bus.last_seen).toLocaleTimeString()}</strong>
                    </div>
                  </div>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
};
