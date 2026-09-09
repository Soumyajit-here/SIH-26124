import React from 'react';
import type { SourceMode } from '../services/types';

interface SourceSelectorProps {
  mode: SourceMode;
  onSelectCamera: () => void;
  onSelectVideo: (file: File) => void;
  onStop: () => void;
  cameraError?: string | null;
  videoFileName?: string | null;
}

export const SourceSelector: React.FC<SourceSelectorProps> = ({
  mode,
  onSelectCamera,
  onSelectVideo,
  onStop,
  cameraError,
  videoFileName,
}) => {
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onSelectVideo(file);
      e.target.value = '';
    }
  };

  return (
    <div className="source-selector">
      <div className="source-buttons">
        <button
          className={`source-btn ${mode === 'CAMERA' ? 'active' : ''}`}
          onClick={onSelectCamera}
          disabled={mode === 'CAMERA'}
        >
          <span className="source-icon">📷</span>
          <span>Camera</span>
        </button>

        <button
          className={`source-btn ${mode === 'VIDEO' ? 'active' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          disabled={mode === 'VIDEO'}
        >
          <span className="source-icon">🎬</span>
          <span>Upload Video</span>
        </button>

        {mode !== 'NONE' && (
          <button className="source-btn stop-btn" onClick={onStop}>
            <span className="source-icon">⏹</span>
            <span>Stop</span>
          </button>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {mode !== 'NONE' && (
        <div className="source-info">
          <span className={`mode-badge ${mode === 'CAMERA' ? 'camera' : 'video'}`}>
            {mode === 'CAMERA' ? '● CAMERA' : '● VIDEO'}
          </span>
          <span className="demo-badge">DEMO MODE</span>
          {videoFileName && <span className="file-name">{videoFileName}</span>}
        </div>
      )}

      {cameraError && <div className="source-error">{cameraError}</div>}
    </div>
  );
};
