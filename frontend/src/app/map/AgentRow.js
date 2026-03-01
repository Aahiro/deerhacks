'use client'

import { useEffect, useRef, useState } from 'react'

const MONO = "'Barlow Condensed', 'Arial Narrow', sans-serif"

const AGENTS = [
  { key: 'commander',      label: 'COMMANDER',      color: '#6ee06e' },
  { key: 'scout',          label: 'SCOUT',          color: '#c8c060' },
  { key: 'vibe_matcher',   label: 'VIBE MATCHER',   color: '#b06ee0' },
  { key: 'cost_analyst',   label: 'COST ANALYST',   color: '#60a8e0' },
  { key: 'critic',         label: 'CRITIC',         color: '#60e0c8' },
  { key: 'synthesiser',    label: 'SYNTHESISER',    color: '#e0a060' },
]

const AGENT_ROW_STYLES = `
  @keyframes agent-glow-pulse {
    0%, 100% { transform: scale(1);    opacity: 1;    }
    50%       { transform: scale(1.08); opacity: 0.88; }
  }
  .agent-glow-pulse {
    animation: agent-glow-pulse 1.6s ease-in-out infinite;
  }
`

function PersonIcon({ size = 28, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" fill="none" style={{ flexShrink: 0 }}>
      <circle cx="14" cy="9" r="4.5" fill={color} />
      <path
        d="M4 26c0-5.523 4.477-10 10-10s10 4.477 10 10"
        stroke={color}
        strokeWidth="1.6"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  )
}

export default function AgentRow({ activeAgent, completedAgents = [], dismissed = false }) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    requestAnimationFrame(() => setVisible(true))
  }, [])

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: AGENT_ROW_STYLES }} />
      <div
        style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          height: 90,
          background: 'rgba(14, 12, 10, 0.88)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderTop: '1px solid rgba(255,255,255,0.06)',
          zIndex: 20,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 0,
          padding: '0 32px',
          transform: dismissed ? 'translateY(100%)' : visible ? 'translateY(0)' : 'translateY(100%)',
          transition: 'transform 0.45s cubic-bezier(0.16, 1, 0.3, 1)',
          pointerEvents: dismissed ? 'none' : 'auto',
        }}
      >
        {AGENTS.map((agent, i) => {
          const isActive = activeAgent === agent.key
          const isDone = completedAgents.includes(agent.key)

          return (
            <div
              key={agent.key}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 6,
                flex: 1,
                maxWidth: 120,
                opacity: isActive ? 1 : isDone ? 0.75 : 0.4,
                transition: 'opacity 0.3s ease',
              }}
            >
              <div
                className={isActive ? 'agent-glow-pulse' : undefined}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 6,
                  borderRadius: 12,
                  padding: '4px 8px',
                  boxShadow: isActive
                    ? `0 0 18px 6px ${agent.color}55, 0 0 40px 12px ${agent.color}22`
                    : 'none',
                  transition: 'box-shadow 0.4s ease',
                }}
              >
                <PersonIcon size={22} color={isActive ? agent.color : isDone ? agent.color : 'rgba(255,255,255,0.55)'} />
                <span
                  style={{
                    fontFamily: MONO,
                    fontWeight: 400,
                    fontSize: 12,
                    letterSpacing: '0.22em',
                    color: isActive ? agent.color : isDone ? agent.color : 'rgba(255,255,255,0.55)',
                    textTransform: 'uppercase',
                    whiteSpace: 'nowrap',
                    transition: 'color 0.3s ease',
                  }}
                >
                  {agent.label}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}