'use client'

import { useEffect, useState } from 'react'

const MONO = "'Barlow Condensed', 'Arial Narrow', sans-serif"

const RANK_COLORS = ['#e8c84a', '#b0b8c4', '#c8905a', 'rgba(255,255,255,0.55)']

const RESULTS_STYLES = `
  @keyframes results-sidebar-slide-in {
    from { transform: translateX(-100%); opacity: 0; }
    to   { transform: translateX(0);     opacity: 1; }
  }
  .results-sidebar-enter {
    animation: results-sidebar-slide-in 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }
  .results-scroll::-webkit-scrollbar { display: none; }
`

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
      fontSize: 11,
      fontWeight: 400,
      color,
    }}>
      {rankIdx + 1}
    </div>
  )
}

function ScoreChips({ venue }) {
  const perPerson = venue.cost_profile?.per_person
  const costLabel = perPerson != null ? `$${Math.round(perPerson)}/pp` : '—'

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      marginTop: 6,
    }}>
      <span style={{
        fontFamily: MONO, fontWeight: 300, fontSize: 10,
        letterSpacing: '0.12em', color: '#b06ee0',
      }}>
        ◆{venue.vibe_score != null ? venue.vibe_score.toFixed(2) : '—'}
      </span>
      <span style={{
        fontFamily: MONO, fontWeight: 300, fontSize: 10,
        letterSpacing: '0.12em', color: '#60a8e0',
      }}>
        ${costLabel}
      </span>
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
      {/* Top row: rank badge + name + vibe score */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <RankBadge rankIdx={rankIdx} />
        <span style={{
          fontFamily: MONO,
          fontWeight: 400,
          fontSize: 13,
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
            fontWeight: 300,
            fontSize: 11,
            color: rankColor,
            flexShrink: 0,
          }}>
            {venue.vibe_score.toFixed(2)}♥
          </span>
        )}
      </div>

      {/* Address */}
      {venue.address && (
        <div style={{
          fontFamily: MONO,
          fontWeight: 300,
          fontSize: 11,
          letterSpacing: '0.04em',
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

      {/* Score chips */}
      <div style={{ paddingLeft: 32 }}>
        <ScoreChips venue={venue} />
      </div>

      {/* Why text */}
      {venue.why && (
        <div style={{
          fontFamily: MONO,
          fontWeight: 300,
          fontSize: 12,
          letterSpacing: '0.02em',
          color: 'rgba(255,255,255,0.62)',
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

      {/* Watch out */}
      {venue.watch_out && venue.watch_out.trim() !== '' && (
        <div style={{
          fontFamily: MONO,
          fontWeight: 300,
          fontSize: 11,
          letterSpacing: '0.02em',
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

export default function ResultsSidebar({ venues = [], selectedIdx, onSelect, onNewSearch }) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    requestAnimationFrame(() => setVisible(true))
  }, [])

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: RESULTS_STYLES }} />
      <div
        className="results-sidebar-enter"
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          bottom: 0,
          width: 320,
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
        <div style={{ padding: '24px 20px 14px', flexShrink: 0 }}>
          <span style={{
            fontFamily: MONO,
            fontWeight: 400,
            fontSize: 13,
            letterSpacing: '0.32em',
            color: 'rgba(255,255,255,0.88)',
            textTransform: 'uppercase',
            display: 'block',
          }}>
            RESULTS
          </span>
          <span style={{
            fontFamily: MONO,
            fontWeight: 300,
            fontSize: 11,
            letterSpacing: '0.06em',
            color: 'rgba(255,255,255,0.40)',
            display: 'block',
            marginTop: 4,
          }}>
            {venues.length} venue{venues.length !== 1 ? 's' : ''} found
          </span>
        </div>

        {/* Divider */}
        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '0 20px', flexShrink: 0 }} />

        {/* Scrollable venue list + footer */}
        <div
          className="results-scroll"
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
                fontSize: 11,
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
      </div>
    </>
  )
}
