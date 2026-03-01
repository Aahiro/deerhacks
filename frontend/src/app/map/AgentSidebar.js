'use client'

import { useEffect, useRef, useState, useMemo } from 'react'

const MONO = "'Barlow Condensed', 'Arial Narrow', sans-serif"
const BODY = "'Inter', -apple-system, 'Segoe UI', sans-serif"

const AGENT_COLORS = {
  commander:    '#6ee06e',
  scout:        '#c8c060',
  vibe_matcher: '#b06ee0',
  cost_analyst: '#60a8e0',
  critic:       '#60e0c8',
  synthesiser:  '#e0a060',
  graph:        '#888888',
  system:       '#888888',
}

const AGENT_LABELS = {
  commander:    'COMMANDER',
  scout:        'SCOUT',
  vibe_matcher: 'VIBE MATCHER',
  cost_analyst: 'COST ANALYST',
  critic:       'CRITIC',
  synthesiser:  'SYNTHESISER',
  graph:        'GRAPH',
  system:       'SYSTEM',
}

const PREFIX_RE = /^\[[A-Z]+\]\s*/

function stripPrefix(msg) {
  return msg.replace(PREFIX_RE, '')
}

const SIDEBAR_STYLES = `
  @keyframes sidebar-slide-in {
    from { transform: translateX(-100%); opacity: 0; }
    to   { transform: translateX(0);     opacity: 1; }
  }
  @keyframes thinking-pulse {
    0%, 100% { opacity: 0.45; }
    50%       { opacity: 1;    }
  }
  @keyframes group-appear {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0);   }
  }
  .sidebar-enter {
    animation: sidebar-slide-in 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }
  .thinking-pulse {
    animation: thinking-pulse 1.8s ease-in-out infinite;
  }
  .agent-group-appear {
    animation: group-appear 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }
  .agent-log-scroll::-webkit-scrollbar { display: none; }
`

function AgentGroup({ agent, logs, color, label, isActive, isExpanded, onToggle }) {
  const scrollRef = useRef(null)

  useEffect(() => {
    if (isExpanded && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs.length, isExpanded])

  const latestLog = logs[logs.length - 1]
  const hiddenCount = logs.length - 1

  return (
    <div
      className="agent-group-appear"
      style={{
        margin: '0 12px 6px',
        borderRadius: 8,
        background: isActive
          ? 'rgba(255,255,255,0.04)'
          : 'rgba(255,255,255,0.015)',
        border: `1px solid ${isActive ? color + '22' : 'rgba(255,255,255,0.04)'}`,
        boxShadow: isActive ? `0 0 12px ${color}11` : 'none',
        transition: 'background 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease',
        overflow: 'hidden',
      }}
    >
      {/* Header — always visible */}
      <div
        onClick={onToggle}
        style={{
          padding: '10px 12px 8px',
          cursor: 'pointer',
          userSelect: 'none',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        {/* Color dot */}
        <div style={{
          width: 7, height: 7,
          borderRadius: '50%',
          background: color,
          flexShrink: 0,
          boxShadow: isActive ? `0 0 6px ${color}88` : 'none',
          transition: 'box-shadow 0.3s ease',
        }} />

        {/* Agent name */}
        <span style={{
          fontFamily: MONO,
          fontWeight: 400,
          fontSize: 14,
          letterSpacing: '0.20em',
          color: color,
          textTransform: 'uppercase',
          flex: 1,
        }}>
          {label}
        </span>

        {/* Log count badge */}
        <span style={{
          fontFamily: BODY,
          fontWeight: 400,
          fontSize: 12,
          color: 'rgba(255,255,255,0.30)',
        }}>
          {logs.length}
        </span>
      </div>

      {!isExpanded ? (
        /* Collapsed: show latest log + toggle */
        <div style={{ padding: '0 12px 10px' }}>
          {latestLog && (
            <div style={{
              fontFamily: BODY,
              fontWeight: 400,
              fontSize: 13,
              letterSpacing: '0.01em',
              color: 'rgba(255,255,255,0.50)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>
              {stripPrefix(latestLog.message)}
            </div>
          )}
          {hiddenCount > 0 && (
            <div
              onClick={onToggle}
              style={{
                fontFamily: BODY,
                fontWeight: 400,
                fontSize: 11,
                color: 'rgba(255,255,255,0.25)',
                marginTop: 4,
                cursor: 'pointer',
              }}
            >
              ▸ {hiddenCount} more line{hiddenCount !== 1 ? 's' : ''}
            </div>
          )}
        </div>
      ) : (
        /* Expanded: show all logs */
        <div style={{ padding: '0 12px 10px' }}>
          <div
            onClick={onToggle}
            style={{
              fontFamily: BODY,
              fontWeight: 400,
              fontSize: 11,
              color: 'rgba(255,255,255,0.25)',
              cursor: 'pointer',
              marginBottom: 6,
            }}
          >
            ▾ collapse
          </div>
          <div
            ref={scrollRef}
            className="agent-log-scroll"
            style={{
              maxHeight: 200,
              overflowY: 'auto',
              scrollbarWidth: 'none',
            }}
          >
            {logs.map((entry, i) => (
              <div
                key={i}
                style={{
                  fontFamily: BODY,
                  fontWeight: 400,
                  fontSize: 13,
                  letterSpacing: '0.01em',
                  color: 'rgba(255,255,255,0.50)',
                  padding: '3px 0',
                  borderBottom: i < logs.length - 1 ? '1px solid rgba(255,255,255,0.03)' : 'none',
                  wordBreak: 'break-word',
                }}
              >
                {stripPrefix(entry.message)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function useResizable(initialWidth, minWidth = 240, maxWidth = 700) {
  const [width, setWidth] = useState(initialWidth)
  const dragging = useRef(false)

  useEffect(() => {
    const onMouseMove = (e) => {
      if (!dragging.current) return
      e.preventDefault()
      setWidth(Math.min(maxWidth, Math.max(minWidth, e.clientX)))
    }
    const onMouseUp = () => { dragging.current = false; document.body.style.cursor = '' }
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [minWidth, maxWidth])

  const onMouseDown = (e) => {
    e.preventDefault()
    dragging.current = true
    document.body.style.cursor = 'col-resize'
  }

  return { width, onMouseDown }
}

export default function AgentSidebar({ logs = [], activeAgent }) {
  const scrollRef = useRef(null)
  const [expanded, setExpanded] = useState({})
  const { width, onMouseDown } = useResizable(400)

  // Group logs by agent, preserving order of first appearance
  const agentGroups = useMemo(() => {
    const orderMap = new Map()
    const groupMap = new Map()

    logs.forEach((entry) => {
      const key = entry.agent
      if (!orderMap.has(key)) {
        orderMap.set(key, orderMap.size)
        groupMap.set(key, [])
      }
      groupMap.get(key).push(entry)
    })

    return Array.from(orderMap.keys()).map((agent) => ({
      agent,
      logs: groupMap.get(agent),
      color: AGENT_COLORS[agent] ?? 'rgba(255,255,255,0.55)',
      label: AGENT_LABELS[agent] ?? agent.toUpperCase(),
    }))
  }, [logs])

  // Auto-scroll to bottom when new groups appear
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [agentGroups.length])

  const toggleExpand = (agent) => {
    setExpanded((prev) => ({ ...prev, [agent]: !prev[agent] }))
  }

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: SIDEBAR_STYLES }} />
      <div
        className="sidebar-enter"
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          bottom: 90,
          width: width,
          background: 'rgba(14, 12, 10, 0.82)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(255,255,255,0.06)',
          zIndex: 15,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div style={{ padding: '28px 20px 16px' }}>
          <span
            className="thinking-pulse"
            style={{
              fontFamily: MONO,
              fontWeight: 400,
              fontSize: 16,
              letterSpacing: '0.30em',
              color: 'rgba(255,255,255,0.88)',
              textTransform: 'uppercase',
              display: 'block',
            }}
          >
            THINKING...
          </span>
        </div>

        {/* Divider */}
        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '0 12px' }} />

        {/* Agent groups */}
        <div
          ref={scrollRef}
          className="agent-log-scroll"
          style={{
            flex: 1,
            overflowY: 'auto',
            scrollbarWidth: 'none',
            padding: '10px 0 16px',
          }}
        >
          {agentGroups.map((group) => (
            <AgentGroup
              key={group.agent}
              agent={group.agent}
              logs={group.logs}
              color={group.color}
              label={group.label}
              isActive={activeAgent === group.agent}
              isExpanded={!!expanded[group.agent]}
              onToggle={() => toggleExpand(group.agent)}
            />
          ))}
        </div>
      </div>

      {/* Drag handle — rendered outside sidebar to avoid overflow:hidden clipping */}
      <div
        onMouseDown={onMouseDown}
        style={{
          position: 'fixed',
          top: 0,
          left: width - 5,
          bottom: 90,
          width: 10,
          cursor: 'col-resize',
          zIndex: 16,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div style={{
          width: 24,
          height: 40,
          borderRadius: 5,
          background: 'rgba(50,50,50,1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          userSelect: 'none',
          pointerEvents: 'none',
        }}>
          <svg width="12" height="12" viewBox="0 0 14 14" fill="none">
            <path d="M4.5 3L1 7L4.5 11" stroke="rgba(255,255,255,0.45)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M9.5 3L13 7L9.5 11" stroke="rgba(255,255,255,0.45)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>
    </>
  )
}
