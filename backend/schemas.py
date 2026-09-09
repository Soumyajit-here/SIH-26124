from pydantic import BaseModel, Field
from typing import Optional
import datetime

class ObservationBase(BaseModel):
    bus_id: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    speed_kmh: Optional[float] = None
    course_deg: Optional[float] = None
    model: str
    class_id: int = Field(..., ge=0)
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    evidence_image: Optional[str] = None
    gps_source: Optional[str] = None
    gps_accuracy_m: Optional[float] = None
    session_id: Optional[str] = None

class ObservationCreate(ObservationBase):
    timestamp: Optional[datetime.datetime] = None

from pydantic import ConfigDict

class ObservationResponse(ObservationBase):
    observation_id: int
    timestamp: datetime.datetime
    event_id: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)

class EventResponse(BaseModel):
    event_id: int
    event_type: str
    latitude: float
    longitude: float
    first_seen: datetime.datetime
    last_seen: datetime.datetime
    severity: str
    status: str
    aggregated_confidence: float
    observation_count: int
    unique_bus_count: int
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)

class ObservationFusionResponse(BaseModel):
    observation: ObservationResponse
    event_id: int
    event_created: bool

class TelemetryBase(BaseModel):
    bus_id: str
    latitude: float
    longitude: float
    speed_kmh: Optional[float] = None
    course_deg: Optional[float] = None
    gps_source: Optional[str] = None
    gps_accuracy_m: Optional[float] = None

class TelemetryCreate(TelemetryBase):
    timestamp: Optional[datetime.datetime] = None

class TelemetryResponse(TelemetryBase):
    id: int
    timestamp: datetime.datetime
    model_config = ConfigDict(from_attributes=True)

class FleetBusStatus(BaseModel):
    bus_id: str
    last_seen: datetime.datetime
    latitude: float
    longitude: float
    speed_kmh: Optional[float] = None
    status: str

class InferRequest(BaseModel):
    bus_id: str
    frame_base64: str  # JPEG base64
    latitude: float
    longitude: float
    speed_kmh: Optional[float] = None
    course_deg: Optional[float] = None
    gps_source: Optional[str] = None
    gps_accuracy_m: Optional[float] = None
    timestamp: Optional[datetime.datetime] = None

class DetectionInfo(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    model: str
    
class InferResponse(BaseModel):
    detections: list[DetectionInfo]
    observations_created: int
    fused_events: list[int]

