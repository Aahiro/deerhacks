'use client'

import { useEffect, useRef } from 'react'

const MONO = "'Barlow Condensed', 'Arial Narrow', sans-serif"

const AGENT_COLORS = {
  commander:      '#6ee06e',
  scout:          '#c8c060',
  vibe_matcher:   '#b06ee0',
  cost_analyst:   '#60a8e0',
  critic:         '#60e0c8',
  synthesiser:    '#e0a060',
}

const AGENT_LABELS = {
  commander:      'COMMANDER',
  scout:          'SCOUT',
  vibe_matcher:   'VIBE MATCHER',
  cost_analyst:   'COST ANALYST',
  critic:         'CRITIC',
  synthesiser:    'SYNTHESISER',
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
  .sidebar-enter {
    animation: sidebar-slide-in 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }
  .thinking-pulse {
    animation: thinking-pulse 1.8s ease-in-out infinite;
  }
  .agent-log-scroll::-webkit-scrollbar { display: none; }
`

export default function AgentSidebar({ logs = [], activeAgent }) {
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  const latestLog = logs[logs.length - 1]
  const activeColor = activeAgent ? (AGENT_COLORS[activeAgent] ?? 'rgba(255,255,255,0.55)') : 'rgba(255,255,255,0.40)'

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
          width: 280,
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
              fontSize: 13,
              letterSpacing: '0.30em',
              color: 'rgba(255,255,255,0.88)',
              textTransform: 'uppercase',
              display: 'block',
            }}
          >
            THINKING...
          </span>
          {latestLog && (
            <span
              style={{
                fontFamily: MONO,
                fontWeight: 300,
                fontSize: 11,
                letterSpacing: '0.06em',
                color: activeColor,
                display: 'block',
                marginTop: 6,
                transition: 'color 0.3s ease',
              }}
            >
              {latestLog.message}
            </span>
          )}
        </div>

        {/* Divider */}
        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '0 20px' }} />

        {/* Log list */}
        <div
          ref={scrollRef}
          className="agent-log-scroll"
          style={{
            flex: 1,
            overflowY: 'auto',
            scrollbarWidth: 'none',
            padding: '12px 0 16px',
          }}
        >
          {logs.map((entry, i) => {
            const color = AGENT_COLORS[entry.agent] ?? 'rgba(255,255,255,0.55)'
            const label = AGENT_LABELS[entry.agent] ?? entry.agent.toUpperCase()
            return (
              <div
                key={i}
                style={{
                  padding: '7px 20px',
                }}
              >
                <span
                  style={{
                    fontFamily: MONO,
                    fontWeight: 400,
                    fontSize: 10,
                    letterSpacing: '0.18em',
                    color,
                    textTransform: 'uppercase',
                    display: 'block',
                    marginBottom: 2,
                  }}
                >
                  {label}
                </span>
                <span
                  style={{
                    fontFamily: MONO,
                    fontWeight: 300,
                    fontSize: 12,
                    letterSpacing: '0.04em',
                    color: 'rgba(255,255,255,0.60)',
                    display: 'block',
                  }}
                >
                  {entry.message}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}
