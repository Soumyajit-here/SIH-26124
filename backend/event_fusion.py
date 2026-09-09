import math
from typing import Tuple
from sqlalchemy.orm import Session
import models
import datetime
import logging

logger = logging.getLogger(__name__)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth surface.
    Returns distance in meters.
    """
    R = 6371000.0  # Radius of Earth in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def fuse_observation(db: Session, obs: models.Observation, max_radius_m: float = 10.0, max_age_hours: float = 24.0) -> Tuple[models.Event, bool]:
    """
    Given a new observation, find the best matching active event of the same type within max_radius_m.
    Returns (Event, is_new_event).
    """
    cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=max_age_hours)
    
    # Get candidate events
    # We only care about ACTIVE events of the SAME CLASS that have been seen recently
    candidates = db.query(models.Event).filter(
        models.Event.status == "ACTIVE",
        models.Event.event_type == obs.class_name,
        models.Event.last_seen >= cutoff_time
    ).all()
    
    best_event = None
    min_dist = float('inf')
    
    for event in candidates:
        dist = haversine_distance(obs.latitude, obs.longitude, event.latitude, event.longitude)
        if dist <= max_radius_m and dist < min_dist:
            min_dist = dist
            best_event = event

    event_created = False

    if best_event:
        # Lock the event row for update to prevent concurrent modification
        best_event = db.query(models.Event).with_for_update().filter(models.Event.event_id == best_event.event_id).first()
        
        # We found a matching event! Update its stats.
        n = best_event.observation_count
        best_event.latitude = ((best_event.latitude * n) + obs.latitude) / (n + 1)
        best_event.longitude = ((best_event.longitude * n) + obs.longitude) / (n + 1)
        
        best_event.aggregated_confidence = max(best_event.aggregated_confidence, obs.confidence)
        
        obs_time = obs.timestamp if obs.timestamp.tzinfo else obs.timestamp.replace(tzinfo=datetime.timezone.utc)
        event_time = best_event.last_seen if best_event.last_seen.tzinfo else best_event.last_seen.replace(tzinfo=datetime.timezone.utc)
        if obs_time > event_time:
            best_event.last_seen = obs_time
            
        existing_buses = set(r[0] for r in db.query(models.Observation.bus_id).filter(models.Observation.event_id == best_event.event_id).all())
        existing_buses.add(obs.bus_id)
        
        best_event.unique_bus_count = len(existing_buses)
        best_event.observation_count += 1
        
        logger.info(f"Fused observation into event {best_event.event_id}. Distance: {min_dist:.2f}m")
    else:
        # Create a new event
        best_event = models.Event(
            event_type=obs.class_name,
            latitude=obs.latitude,
            longitude=obs.longitude,
            first_seen=obs.timestamp,
            last_seen=obs.timestamp,
            severity="MEDIUM",
            status="ACTIVE",
            aggregated_confidence=obs.confidence,
            observation_count=1,
            unique_bus_count=1
        )
        db.add(best_event)
        db.flush() # flush to get the new event_id
        event_created = True
        min_dist = 0.0
        logger.info(f"Created new event {best_event.event_id} for class {obs.class_name}")

    # Link the observation to the event
    obs.event_id = best_event.event_id
    
    logger.info(f"Observation processed: bus_id={obs.bus_id}, class_name={obs.class_name}, lat={obs.latitude}, lon={obs.longitude}, matched_event_id={best_event.event_id}, distance_to_event={min_dist:.2f}m, new_event_created={event_created}")
    
    return best_event, event_created

def recalculate_event_after_removal(db: Session, event_id: int) -> bool:
    """
    Recalculates event stats after some observations were deleted.
    If no observations remain, deletes the event and returns False.
    If observations remain, updates centroid, counts, confidence, timestamps and returns True.
    """
    event = db.query(models.Event).filter(models.Event.event_id == event_id).first()
    if not event:
        return False

    remaining = db.query(models.Observation).filter(models.Observation.event_id == event_id).all()
    if not remaining:
        db.delete(event)
        logger.info(f"Deleted event {event_id} as all associated observations were removed.")
        return False

    event.observation_count = len(remaining)
    event.unique_bus_count = len(set(o.bus_id for o in remaining))
    event.latitude = sum(o.latitude for o in remaining) / len(remaining)
    event.longitude = sum(o.longitude for o in remaining) / len(remaining)
    event.aggregated_confidence = max(o.confidence for o in remaining)

    timestamps = [o.timestamp for o in remaining if o.timestamp]
    if timestamps:
        event.first_seen = min(timestamps)
        event.last_seen = max(timestamps)

    logger.info(f"Recalculated event {event_id}: count={event.observation_count}, unique_buses={event.unique_bus_count}, lat={event.latitude:.6f}, lon={event.longitude:.6f}")
    return True

