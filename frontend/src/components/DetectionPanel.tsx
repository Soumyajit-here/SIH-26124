import React from 'react';
import type { Detection } from '../services/types';

interface DetectionPanelProps {
  detections: Detection[];
}

export const DetectionPanel: React.FC<DetectionPanelProps> = ({ detections }) => {
  // Filter out Manhole and Model 2
  const visibleDetections = detections.filter(
    (d) => d.class_name !== 'Manhole' && d.model !== 'model_2'
  );

  return (
    <div className="panel-section detection-panel">
      <div className="panel-header-small flex items-center justify-between">
        <h3>Live Detections (Current Frame)</h3>
        <span className="count-badge">{visibleDetections.length} Active</span>
      </div>

      <div className="detection-list">
        {visibleDetections.length === 0 ? (
          <div className="empty-state">No Model 1 detections in current frame</div>
        ) : (
          visibleDetections.map((det, idx) => (
            <div key={`${det.class_name}-${idx}`} className={`detection-item ${det.model}`}>
              <div className="det-name">
                <span className="model-dot model_1"></span>
                <span className="det-class">{det.class_name}</span>
              </div>
              <div className="det-conf">{(det.confidence * 100).toFixed(1)}%</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
