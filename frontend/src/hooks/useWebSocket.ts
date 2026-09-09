import { useState, useEffect, useCallback, useRef } from 'react';
import { getWebSocketUrl } from '../services/api';
import type { WebSocketMessage } from '../services/types';

export function useWebSocket(onMessage?: (msg: WebSocketMessage) => void) {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(getWebSocketUrl());

      ws.onopen = () => {
        setConnected(true);
        console.log('[WS] Connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WebSocketMessage;
          onMessageRef.current?.(data);
        } catch (e) {
          console.warn('[WS] Parse error:', e);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        console.log('[WS] Disconnected. Reconnecting in 3s...');
        reconnectTimerRef.current = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        ws.close();
      };

      wsRef.current = ws;
    } catch {
      reconnectTimerRef.current = setTimeout(connect, 3000);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { connected };
}
