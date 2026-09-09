import { useState, useCallback, useRef, useEffect } from 'react';

export function useCamera() {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Enumerate cameras
  const enumerateDevices = useCallback(async () => {
    try {
      // Need a temporary stream to get device labels
      const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = allDevices.filter(d => d.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
      tempStream.getTracks().forEach(t => t.stop());
      setError(null);
    } catch (err: any) {
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied');
      } else {
        setError('Could not access camera');
      }
    }
  }, [selectedDeviceId]);

  // Start camera
  const startCamera = useCallback(async (deviceId?: string) => {
    try {
      const constraints: MediaStreamConstraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
      };
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);
      setActive(true);
      setError(null);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
      }
    } catch (err: any) {
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied');
      } else {
        setError(`Camera error: ${err.message}`);
      }
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
    }
    setStream(null);
    setActive(false);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [stream]);

  // Capture a single frame as base64 JPEG
  const captureFrame = useCallback((): string | null => {
    if (!videoRef.current || !active) return null;

    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return null;

    const canvas = document.createElement('canvas');
    // Downscale to 640px width for performance
    const scale = Math.min(1, 640 / video.videoWidth);
    canvas.width = video.videoWidth * scale;
    canvas.height = video.videoHeight * scale;

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
    return dataUrl.split(',')[1]; // Return just the base64 part
  }, [active]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, [stream]);

  return {
    videoRef,
    stream,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    error,
    active,
    enumerateDevices,
    startCamera,
    stopCamera,
    captureFrame,
  };
}
