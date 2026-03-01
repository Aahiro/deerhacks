'use client'

import { useState } from 'react'

const MONO = "'Barlow Condensed', 'Arial Narrow', sans-serif"

const SEARCHBAR_STYLES = `
  .locatr-search-input::placeholder {
    font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
    font-weight: 300;
    letter-spacing: 0.04em;
    color: rgba(255,255,255,0.30);
  }
  .locatr-search-input:focus { outline: none; }
`

function SearchIcon({ size = 13, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 13 13" fill="none" style={{ flexShrink: 0 }}>
      <circle cx="5.5" cy="5.5" r="4" stroke={color} strokeWidth="1.15" />
      <line x1="8.8" y1="8.8" x2="12.2" y2="12.2" stroke={color} strokeWidth="1.15" strokeLinecap="round" />
    </svg>
  )
}

export default function SearchBar({
  onSearch = () => {},
  placeholder = "search 'basketball courts in downtown TO for 10 people at 5pm'",
  disabled = false,
}) {
  const [query, setQuery] = useState('')
  const [focused, setFocused] = useState(false)

  const handleSubmit = (e) => {
    e?.preventDefault()
    if (disabled) return
    const trimmed = query.trim()
    if (trimmed) onSearch(trimmed)
  }

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: SEARCHBAR_STYLES }} />
      <form
        onSubmit={handleSubmit}
        className="ui-reveal"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          background: focused ? 'rgba(20, 17, 13, 0.84)' : 'rgba(20, 17, 13, 0.70)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderRadius: 10,
          padding: '10px 16px 9px',
          width: 580,
          maxWidth: 'calc(100vw - 48px)',
          border: `1px solid ${focused ? 'rgba(255,255,255,0.10)' : 'rgba(255,255,255,0.05)'}`,
          boxShadow: focused
            ? '0 8px 40px rgba(0,0,0,0.32), 0 1px 0 rgba(255,255,255,0.04) inset'
            : '0 2px 16px rgba(0,0,0,0.22)',
          transition: 'background 0.22s cubic-bezier(0.16,1,0.3,1), border 0.22s cubic-bezier(0.16,1,0.3,1), box-shadow 0.22s cubic-bezier(0.16,1,0.3,1)',
          boxSizing: 'border-box',
          opacity: disabled ? 0.5 : 1,
          pointerEvents: disabled ? 'none' : 'auto',
        }}
      >
        <SearchIcon
          size={13}
          color={focused ? 'rgba(255,255,255,0.52)' : 'rgba(255,255,255,0.28)'}
        />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={placeholder}
          className="locatr-search-input"
          style={{
            flex: 1,
            background: 'none',
            border: 'none',
            outline: 'none',
            padding: 0,
            fontFamily: MONO,
            fontWeight: 300,
            fontSize: 13,
            letterSpacing: '0.04em',
            color: 'rgba(255,255,255,0.88)',
            caretColor: 'rgba(255,255,255,0.65)',
            minWidth: 0,
          }}
        />
      </form>
    </>
  )
}
