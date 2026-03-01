'use client'

import { useEffect, useRef, useState, useMemo } from 'react'

const MONO = "'Barlow Condensed', 'Arial Narrow', sans-serif"
const BODY = "'Inter', -apple-system, 'Segoe UI', sans-serif"

// ── Shared constants ──────────────────────────────────────

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

const RANK_COLORS = ['#e8c84a', '#b0b8c4', '#c8905a', 'rgba(255,255,255,0.55)']

const PREFIX_RE = /^\[[A-Z]+\]\s*/

function stripPrefix(msg) {
  return msg.replace(PREFIX_RE, '')
}

// Helper to parse **text** into JSX
function formatBoldText(text) {
  if (!text) return null;
  
  // Split by **text** and keep the delimiters in the array
  const parts = text.split(/(\*\*.*?\*\*)/g);
  
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      // Remove the stars and wrap in bold tag
      return (
        <strong key={i} style={{ fontWeight: 700 }}>
          {part.slice(2, -2)}
        </strong>
      );
    }
    return part;
  });
}

// ── Styles ────────────────────────────────────────────────

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
  .sidebar-scroll::-webkit-scrollbar { display: none; }
`

// ── Resize hook ───────────────────────────────────────────

function useResizable(initialWidth, minWidth = 280, maxWidth = 700) {
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

// ── Agent log components ──────────────────────────────────

function AgentGroup({ logs, color, label, isActive, isExpanded, onToggle }) {
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
        <div style={{
          width: 7, height: 7,
          borderRadius: '50%',
          background: color,
          flexShrink: 0,
          boxShadow: isActive ? `0 0 6px ${color}88` : 'none',
          transition: 'box-shadow 0.3s ease',
        }} />
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
            className="sidebar-scroll"
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

function LogsContent({ logs, activeAgent }) {
  const scrollRef = useRef(null)
  const [expanded, setExpanded] = useState({})

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

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [agentGroups.length])

  const toggleExpand = (agent) => {
    setExpanded((prev) => ({ ...prev, [agent]: !prev[agent] }))
  }

  return (
    <div
      ref={scrollRef}
      className="sidebar-scroll"
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
          logs={group.logs}
          color={group.color}
          label={group.label}
          isActive={activeAgent === group.agent}
          isExpanded={!!expanded[group.agent]}
          onToggle={() => toggleExpand(group.agent)}
        />
      ))}
    </div>
  )
}

// ── Results components ────────────────────────────────────

function RankBadge({ rankIdx }) {
  const color = RANK_COLORS[Math.min(rankIdx, 3)]
  return (
    <div style={{
      width: 24, height: 24,
      borderRadius: '50%',
      background: 'rgba(14,12,10,0.88)',
      border: `1.5px solid ${color}`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexShrink: 0,
      fontFamily: MONO,
      fontSize: 15,
      fontWeight: 400,
      color,
    }}>
      {rankIdx + 1}
    </div>
  )
}

function ScoreChips({ venue }) {
  const priceLabel = venue.price_range || '—'

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      marginTop: 6,
    }}>
      <span style={{
        fontFamily: BODY, fontWeight: 500, fontSize: 12,
        letterSpacing: '0.04em', color: '#b06ee0',
      }}>
        ◆{venue.vibe_score != null ? venue.vibe_score.toFixed(2) : '—'}
      </span>
      <span style={{
        fontFamily: BODY, fontWeight: 500, fontSize: 12,
        letterSpacing: '0.04em', color: '#60a8e0',
      }}>
        {priceLabel}
      </span>
      {venue.rating != null && (
        <span style={{
          fontFamily: BODY, fontWeight: 500, fontSize: 12,
          letterSpacing: '0.04em', color: '#e8c84a',
        }}>
          ⭐{venue.rating.toFixed(1)}
        </span>
      )}
    </div>
  )
}

function VenueCard({ venue, rankIdx, isSelected, onSelect }) {
  const rankColor = RANK_COLORS[Math.min(rankIdx, 3)]

  return (
    <div
      onClick={onSelect}
      style={{
        padding: '12px 20px',
        cursor: 'pointer',
        borderLeft: isSelected
          ? '3px solid rgba(255,255,255,0.22)'
          : '3px solid transparent',
        background: isSelected ? 'rgba(255,255,255,0.05)' : 'transparent',
        transition: 'background 0.2s ease, border-color 0.2s ease',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <RankBadge rankIdx={rankIdx} />
        <span style={{
          fontFamily: MONO,
          fontWeight: 400,
          fontSize: 15,
          letterSpacing: '0.08em',
          color: 'rgba(255,255,255,0.88)',
          flex: 1,
          textTransform: 'uppercase',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}>
          {venue.name}
        </span>
        {venue.vibe_score != null && (
          <span style={{
            fontFamily: MONO,
            fontWeight: 400,
            fontSize: 13,
            color: rankColor,
            flexShrink: 0,
          }}>
            {venue.vibe_score.toFixed(2)}♥
          </span>
        )}
      </div>

      {venue.address && (
        <div style={{
          fontFamily: BODY,
          fontWeight: 400,
          fontSize: 13,
          letterSpacing: '0.01em',
          color: 'rgba(255,255,255,0.38)',
          marginTop: 4,
          paddingLeft: 32,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}>
          {venue.address}
        </div>
      )}

      <div style={{ paddingLeft: 32 }}>
        <ScoreChips venue={venue} />
      </div>

      {venue.why && (
        <div style={{
          fontFamily: BODY,
          fontWeight: 400,
          fontSize: 13,
          letterSpacing: '0.01em',
          color: 'rgba(255,255,255,0.62)',
          lineHeight: 1.45,
          marginTop: 8,
          paddingLeft: 32,
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}>
          {venue.why}
        </div>
      )}

      {venue.watch_out && venue.watch_out.trim() !== '' && (
        <div style={{
          fontFamily: BODY,
          fontWeight: 400,
          fontSize: 13,
          letterSpacing: '0.01em',
          color: '#e0a060',
          marginTop: 5,
          paddingLeft: 32,
        }}>
          ⚠ {venue.watch_out}
        </div>
      )}
    </div>
  )
}

function ResultsContent({ venues, globalConsensus, selectedIdx, onSelect, onNewSearch }) {
  return (
    <>
      {/* Global consensus */}
      {globalConsensus && (
        <div style={{
          padding: '12px 20px',
          fontFamily: BODY,
          fontWeight: 400,
          fontSize: 13,
          letterSpacing: '0.01em',
          lineHeight: 1.55,
          color: 'rgba(255,255,255,0.55)',
          flexShrink: 0,
        }}>
          {formatBoldText(globalConsensus)}
        </div>
      )}

      {/* Divider */}
      <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '0 20px', flexShrink: 0 }} />

      {/* Scrollable venue list + footer */}
      <div
        className="sidebar-scroll"
        style={{
          flex: 1,
          overflowY: 'auto',
          scrollbarWidth: 'none',
        }}
      >
        <div style={{ paddingTop: 8, paddingBottom: 4 }}>
          {venues.map((venue, i) => (
            <VenueCard
              key={i}
              venue={venue}
              rankIdx={i}
              isSelected={selectedIdx === i}
              onSelect={() => onSelect(i)}
            />
          ))}
        </div>

        {/* Sticky footer: New Search button */}
        <div style={{
          position: 'sticky',
          bottom: 0,
          background: 'rgba(14,12,10,0.95)',
          borderTop: '1px solid rgba(255,255,255,0.06)',
          padding: '14px 20px',
        }}>
          <button
            onClick={onNewSearch}
            style={{
              width: '100%',
              background: 'transparent',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: 6,
              padding: '8px 0',
              cursor: 'pointer',
              fontFamily: MONO,
              fontWeight: 400,
              fontSize: 15,
              letterSpacing: '0.28em',
              color: 'rgba(255,255,255,0.55)',
              textTransform: 'uppercase',
              transition: 'border-color 0.2s ease, color 0.2s ease',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.30)'
              e.currentTarget.style.color = 'rgba(255,255,255,0.88)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'
              e.currentTarget.style.color = 'rgba(255,255,255,0.55)'
            }}
          >
            NEW SEARCH
          </button>
        </div>
      </div>
    </>
  )
}

// ── Tab button ────────────────────────────────────────────

function TabButton({ label, isActive, onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        flex: 1,
        background: isActive ? 'rgba(255,255,255,0.08)' : 'transparent',
        border: 'none',
        borderBottom: isActive ? '2px solid rgba(255,255,255,0.50)' : '2px solid transparent',
        padding: '10px 0 8px',
        cursor: disabled ? 'default' : 'pointer',
        fontFamily: MONO,
        fontWeight: 400,
        fontSize: 13,
        letterSpacing: '0.20em',
        color: disabled
          ? 'rgba(255,255,255,0.15)'
          : isActive
            ? 'rgba(255,255,255,0.88)'
            : 'rgba(255,255,255,0.35)',
        textTransform: 'uppercase',
        transition: 'color 0.2s ease, background 0.2s ease',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {label}
    </button>
  )
}

// ── Main Sidebar ──────────────────────────────────────────

export default function Sidebar({
  searchState,
  logs,
  activeAgent,
  venues,
  globalConsensus,
  selectedIdx,
  onSelect,
  onNewSearch,
}) {
  const { width, onMouseDown } = useResizable(400)
  const [activeTab, setActiveTab] = useState('logs')
  const hasResults = searchState === 'results'

  // Auto-switch to results tab when results arrive
  const prevHasResults = useRef(false)
  useEffect(() => {
    if (hasResults && !prevHasResults.current) {
      setActiveTab('results')
    }
    prevHasResults.current = hasResults
  }, [hasResults])

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: SIDEBAR_STYLES }} />

      {/* Sidebar panel */}
      <div
        className="sidebar-enter"
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          bottom: 0,
          width: width,
          background: 'rgba(14, 12, 10, 0.85)',
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
        <div style={{ padding: '24px 20px 12px', flexShrink: 0 }}>
          <span
            className={!hasResults ? 'thinking-pulse' : undefined}
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
            {hasResults ? 'RESULTS' : 'THINKING...'}
          </span>
          {hasResults && (
            <span style={{
              fontFamily: BODY,
              fontWeight: 400,
              fontSize: 13,
              letterSpacing: '0.02em',
              color: 'rgba(255,255,255,0.40)',
              display: 'block',
              marginTop: 4,
            }}>
              {venues.length} venue{venues.length !== 1 ? 's' : ''} found
            </span>
          )}
        </div>

        {/* Tab bar — only show when results exist so user can toggle */}
        {hasResults && (
          <div style={{
            display: 'flex',
            margin: '0 16px',
            flexShrink: 0,
          }}>
            <TabButton
              label="Results"
              isActive={activeTab === 'results'}
              onClick={() => setActiveTab('results')}
              disabled={false}
            />
            <TabButton
              label="Logs"
              isActive={activeTab === 'logs'}
              onClick={() => setActiveTab('logs')}
              disabled={false}
            />
          </div>
        )}

        {/* Divider */}
        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '0 16px', flexShrink: 0 }} />

        {/* Content area */}
        {activeTab === 'logs' || !hasResults ? (
          <LogsContent logs={logs} activeAgent={activeAgent} />
        ) : (
          <ResultsContent
            venues={venues}
            globalConsensus={globalConsensus}
            selectedIdx={selectedIdx}
            onSelect={onSelect}
            onNewSearch={onNewSearch}
          />
        )}
      </div>

      {/* Drag handle — rendered outside sidebar to avoid overflow:hidden clipping */}
      <div
        onMouseDown={onMouseDown}
        style={{
          position: 'fixed',
          top: 0,
          left: width - 5,
          bottom: 0,
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
