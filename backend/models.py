from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
import datetime

class Telemetry(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    bus_id = Column(String(50), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)
    
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed_kmh = Column(Float, nullable=True)
    course_deg = Column(Float, nullable=True)
    
    gps_source = Column(String(50), nullable=True)
    gps_accuracy_m = Column(Float, nullable=True)

class Event(Base):
    __tablename__ = "events"

    event_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_type = Column(String(50), index=True, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    first_seen = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
    last_seen = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
    severity = Column(String(20), default="MEDIUM")
    status = Column(String(20), default="ACTIVE")
    aggregated_confidence = Column(Float, nullable=False)
    observation_count = Column(Integer, default=1)
    unique_bus_count = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), onupdate=lambda: datetime.datetime.now(datetime.timezone.utc))
    
    observations = relationship("Observation", back_populates="event")

class Observation(Base):
    __tablename__ = "observations"

    observation_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    bus_id = Column(String(50), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)
    
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed_kmh = Column(Float, nullable=True)
    course_deg = Column(Float, nullable=True)
    
    model = Column(String(50), index=True, nullable=False)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(50), index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    
    bbox_x1 = Column(Integer, nullable=True)
    bbox_y1 = Column(Integer, nullable=True)
    bbox_x2 = Column(Integer, nullable=True)
    bbox_y2 = Column(Integer, nullable=True)
    
    evidence_image = Column(String(255), nullable=True)
    
    gps_source = Column(String(50), nullable=True)
    gps_accuracy_m = Column(Float, nullable=True)
    session_id = Column(String(64), index=True, nullable=True)

    event_id = Column(Integer, ForeignKey("events.event_id"), nullable=True)
    event = relationship("Event", back_populates="observations")
