import { useState, useCallback, useRef, useEffect } from 'react';

export function useVideoFile() {
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number>(0);
  const [active, setActive] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  // Load a video file
  const loadVideo = useCallback((f: File) => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }

    const url = URL.createObjectURL(f);
    objectUrlRef.current = url;
    setFile(f);
    setError(null);
    setCurrentTime(0);

    if (videoRef.current) {
      const vid = videoRef.current;
      vid.muted = true;
      vid.loop = true;
      vid.playsInline = true;
      vid.src = url;

      vid.onloadedmetadata = () => {
        setDuration(vid.duration || 0);
      };

      vid.oncanplay = () => {
        vid.play().then(() => {
          setActive(true);
        }).catch((err) => {
          console.warn('[useVideoFile] Autoplay warning:', err);
        });
      };

      vid.onerror = () => {
        console.error('[useVideoFile] Error decoding/loading video file');
        setError('Could not decode or play video file.');
      };

      vid.load();
    }
  }, []);

  // Start playback
  const startPlayback = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.play().then(() => {
        setActive(true);
      }).catch((err) => {
        console.warn('[useVideoFile] Playback error:', err);
      });
    }
  }, []);

  // Stop playback
  const stopPlayback = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.pause();
    }
    setActive(false);
  }, []);

  // Capture current frame as base64 JPEG
  const captureFrame = useCallback((): string | null => {
    if (!videoRef.current) return null;

    const video = videoRef.current;
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return null;

    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 640 / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    setCurrentTime(video.currentTime);
    return canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
  }, []);

  // Check if video has ended
  const isEnded = useCallback((): boolean => {
    return videoRef.current?.ended ?? false;
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, []);

  return {
    videoRef,
    file,
    duration,
    active,
    currentTime,
    error,
    loadVideo,
    startPlayback,
    stopPlayback,
    captureFrame,
    isEnded,
  };
}
