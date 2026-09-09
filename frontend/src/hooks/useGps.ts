import { useState, useCallback, useRef } from 'react';
import type { GpsState } from '../services/types';

// Simulated GPS: linear movement from a start point
const DEFAULT_START_LAT = 28.432738;
const DEFAULT_START_LON = 77.014969;
const DEFAULT_SPEED_KMH = 30.0;
const DEFAULT_COURSE_DEG = 90.0; // East

export function useGps() {
  const [gps, setGps] = useState<GpsState>({
    latitude: DEFAULT_START_LAT,
    longitude: DEFAULT_START_LON,
    speed_kmh: DEFAULT_SPEED_KMH,
    course_deg: DEFAULT_COURSE_DEG,
    source: 'SIMULATED',
    status: 'CONNECTED',
    accuracy_m: 10.0,
  });

  const lastUpdateRef = useRef<number>(Date.now());
  const watchIdRef = useRef<number | null>(null);

  // Advance simulated GPS by elapsed time
  const advanceSimulated = useCallback(() => {
    const now = Date.now();
    const elapsedS = (now - lastUpdateRef.current) / 1000;
    lastUpdateRef.current = now;

    setGps((prev) => {
      const speedMs = (prev.speed_kmh * 1000) / 3600;
      const distanceM = speedMs * elapsedS;
      const courseRad = (prev.course_deg * Math.PI) / 180;

      // Haversine-based offset
      const R = 6371000;
      const dLat = (distanceM * Math.cos(courseRad)) / R;
      const dLon = (distanceM * Math.sin(courseRad)) / (R * Math.cos((prev.latitude * Math.PI) / 180));

      return {
        ...prev,
        latitude: prev.latitude + (dLat * 180) / Math.PI,
        longitude: prev.longitude + (dLon * 180) / Math.PI,
      };
    });
  }, []);

  // Start real device GPS
  const startRealGps = useCallback(() => {
    if (!navigator.geolocation) {
      setGps((prev) => ({ ...prev, source: 'REAL_DEVICE', status: 'UNAVAILABLE' }));
      return;
    }

    setGps((prev) => ({ ...prev, source: 'REAL_DEVICE', status: 'SEARCHING' }));

    watchIdRef.current = navigator.geolocation.watchPosition(
      (pos) => {
        setGps({
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
          speed_kmh: (pos.coords.speed ?? 0) * 3.6,
          course_deg: pos.coords.heading ?? 0,
          source: 'REAL_DEVICE',
          status: 'CONNECTED',
          accuracy_m: pos.coords.accuracy,
        });
      },
      () => {
        setGps((prev) => ({ ...prev, status: 'UNAVAILABLE' }));
      },
      { enableHighAccuracy: true, maximumAge: 2000 }
    );
  }, []);

  const stopRealGps = useCallback(() => {
    if (watchIdRef.current !== null) {
      navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
    }
  }, []);

  const setSource = useCallback(
    (source: GpsState['source']) => {
      stopRealGps();

      if (source === 'REAL_DEVICE') {
        startRealGps();
      } else {
        setGps((prev) => ({
          ...prev,
          source,
          status: source === 'SIMULATED' ? 'CONNECTED' : prev.status,
          latitude: DEFAULT_START_LAT,
          longitude: DEFAULT_START_LON,
        }));
        lastUpdateRef.current = Date.now();
      }
    },
    [startRealGps, stopRealGps]
  );

  return {
    gps,
    advanceSimulated,
    setSource,
    stopRealGps,
  };
}
