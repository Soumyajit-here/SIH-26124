import React, { useState } from 'react';
import type { LayoutMode } from '../services/types';

interface DemoControlsProps {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  currentFrame: number;
  totalFrames: number;
  playbackRate: number;
  sessionId: string;
  layoutMode: LayoutMode;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onSeek: (time: number) => void;
  onChangeSpeed: (rate: number) => void;
  onChangeLayout: (mode: LayoutMode) => void;
  onClearSession: () => Promise<any>;
}

export const DemoControls: React.FC<DemoControlsProps> = ({
  isPlaying,
  currentTime,
  duration,
  currentFrame,
  totalFrames,
  playbackRate,
  sessionId,
  layoutMode,
  onPlay,
  onPause,
  onStop,
  onSeek,
  onChangeSpeed,
  onChangeLayout,
  onClearSession,
}) => {
  const [clearing, setClearing] = useState(false);
  const [clearMessage, setClearMessage] = useState<string | null>(null);

  const handleClear = async () => {
    if (!window.confirm('Clear all observations and events created in this demo session? Historical non-demo data will be preserved.')) {
      return;
    }
    setClearing(true);
    try {
      const res = await onClearSession();
      setClearMessage(`Session cleared! (${res?.deleted_observations || 0} observations removed)`);
      setTimeout(() => setClearMessage(null), 3500);
    } catch (err: any) {
      setClearMessage('Failed to clear session');
      setTimeout(() => setClearMessage(null), 3000);
    } finally {
      setClearing(false);
    }
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="demo-controls-panel">
      {/* Playback Seekbar */}
      <div className="demo-seekbar-container">
        <input
          type="range"
          min={0}
          max={duration || 100}
          step={0.04}
          value={currentTime}
          onChange={(e) => onSeek(parseFloat(e.target.value))}
          className="demo-seekbar"
        />
        <div className="demo-time-row">
          <span>{formatTime(currentTime)} / {formatTime(duration)}</span>
          <span>Frame: {currentFrame} {totalFrames > 0 ? `/ ${totalFrames}` : ''}</span>
        </div>
      </div>

      {/* Main Buttons Bar */}
      <div className="demo-action-bar">
        {/* Play / Pause / Stop */}
        <div className="demo-btn-group">
          {isPlaying ? (
            <button className="demo-btn pause-btn" onClick={onPause} title="Pause Video">
              ⏸ Pause
            </button>
          ) : (
            <button className="demo-btn play-btn" onClick={onPlay} title="Play Video at Native FPS">
              ▶ Play
            </button>
          )}

          <button className="demo-btn stop-btn" onClick={onStop} title="Stop & Reset">
            ⏹ Stop
          </button>
        </div>

        {/* Speed Selector */}
        <div className="demo-speed-group">
          <span className="group-label">Speed:</span>
          {[1.0, 1.5, 2.0].map(speed => (
            <button
              key={speed}
              className={`speed-btn ${playbackRate === speed ? 'active' : ''}`}
              onClick={() => onChangeSpeed(speed)}
            >
              {speed}x
            </button>
          ))}
        </div>

        {/* Layout Expand / Collapse Modes */}
        <div className="demo-layout-group">
          <span className="group-label">View:</span>
          <button
            className={`layout-btn ${layoutMode === 'normal' ? 'active' : ''}`}
            onClick={() => onChangeLayout('normal')}
            title="Normal 3-Column Layout"
          >
            Normal
          </button>
          <button
            className={`layout-btn ${layoutMode === 'expanded' ? 'active' : ''}`}
            onClick={() => onChangeLayout('expanded')}
            title="Expanded Video Layout"
          >
            ⛶ Wide Video
          </button>
          <button
            className={`layout-btn ${layoutMode === 'full' ? 'active' : ''}`}
            onClick={() => onChangeLayout('full')}
            title="Full Video Mode"
          >
            ⛶ Full
          </button>
        </div>

        {/* Clear Current Session Button */}
        <div className="demo-session-group ml-auto">
          <button
            className="demo-clear-session-btn"
            onClick={handleClear}
            disabled={clearing}
            title="Safely remove only this demo session data"
          >
            {clearing ? 'Clearing...' : '🗑️ CLEAR CURRENT SESSION'}
          </button>
        </div>
      </div>

      {/* Session Feedback Toast */}
      {clearMessage && (
        <div className="demo-toast-feedback">
          {clearMessage}
        </div>
      )}

      {/* Badges & Technical Honesty Status */}
      <div className="demo-status-row">
        <span className="demo-tag active">● DEMO REPLAY</span>
        <span className="demo-tag model">MODEL 1 ONLY</span>
        <span className="demo-tag filter">MANHOLE SUPPRESSED</span>
        <span className="demo-tag gps">GPS: SIMULATED</span>
        <span className="demo-tag live">BACKEND: LIVE</span>
        <span className="demo-tag live">EVENT FUSION: LIVE</span>
        <span className="demo-session-id">Session: {sessionId}</span>
      </div>
    </div>
  );
};
