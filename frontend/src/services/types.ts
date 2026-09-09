// Shared TypeScript types for the BharatPotHole dashboard

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_x1: number;
  bbox_y1: number;
  bbox_x2: number;
  bbox_y2: number;
  model: string;
}

export interface Observation {
  observation_id: number;
  bus_id: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  class_name: string;
  confidence: number;
  model: string;
  bbox_x1?: number;
  bbox_y1?: number;
  bbox_x2?: number;
  bbox_y2?: number;
  event_id?: number;
}

export interface InferResponse {
  detections: Detection[];
  observations_created: number;
  fused_events: number[];
}

export interface GeoJSONFeature {
  type: 'Feature';
  geometry: {
    type: 'Point';
    coordinates: [number, number]; // [lon, lat]
  };
  properties: {
    event_id: number;
    event_type: string;
    severity: string;
    status: string;
    aggregated_confidence: number;
    observation_count: number;
    unique_bus_count: number;
    first_seen: string | null;
    last_seen: string | null;
  };
}

export interface FleetBusStatus {
  bus_id: string;
  last_seen: string;
  latitude: number;
  longitude: number;
  speed_kmh: number | null;
  status: string;
}

export interface GpsState {
  latitude: number;
  longitude: number;
  speed_kmh: number;
  course_deg: number;
  source: 'REAL_DEVICE' | 'REPLAY' | 'SIMULATED';
  status: 'CONNECTED' | 'SEARCHING' | 'UNAVAILABLE';
  accuracy_m: number;
}

export interface InferenceSession {
  sessionId: string;
  busId: string;
  sourceType: 'CAMERA' | 'VIDEO' | 'REPLAY';
  startTime: Date;
  status: 'RUNNING' | 'STOPPED' | 'ERROR';
}

export type SourceMode = 'NONE' | 'CAMERA' | 'VIDEO';
export type LayoutMode = 'normal' | 'expanded' | 'full';

export interface DemoCacheDetection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
}

export interface DemoCacheVideoMeta {
  filename: string;
  fps: number;
  width: number;
  height: number;
  total_frames: number;
  duration_s: number;
}

export interface DemoCacheModelMeta {
  model_name: string;
  checkpoint: string;
  conf_threshold: number;
  imgsz: number;
  suppressed_classes: string[];
  active_classes: string[];
}

export interface DemoCacheData {
  video_metadata: DemoCacheVideoMeta;
  model_metadata: DemoCacheModelMeta;
  frames: Record<string, DemoCacheDetection[]>;
}

export interface ClearSessionResponse {
  session_id: string;
  deleted_observations: number;
  updated_events: number;
  deleted_events: number;
  status: string;
}

export interface WebSocketMessage {
  type: 'NEW_EVENT' | 'EVENT_UPDATED' | 'BUS_UPDATED' | 'TELEMETRY_UPDATED' | 'SESSION_CLEARED';
  event_id?: number;
  event_type?: string;
  latitude?: number;
  longitude?: number;
  unique_bus_count?: number;
  observation_count?: number;
  status?: string;
  bus_id?: string;
  speed_kmh?: number;
  course_deg?: number;
  session_id?: string;
  deleted_observations?: number;
  updated_events?: number;
  deleted_events?: number;
}
