'use client'

import Link from 'next/link'

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400&display=swap');

  @keyframes locatr-reveal {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .land-1 { animation: locatr-reveal 0.55s cubic-bezier(0.16,1,0.3,1) 0.05s both; }
  .land-2 { animation: locatr-reveal 0.55s cubic-bezier(0.16,1,0.3,1) 0.15s both; }
  .land-3 { animation: locatr-reveal 0.55s cubic-bezier(0.16,1,0.3,1) 0.26s both; }
  .land-4 { animation: locatr-reveal 0.55s cubic-bezier(0.16,1,0.3,1) 0.40s both; }

  .cta {
    display: inline-flex;
    align-items: center;
    padding: 10px 28px 9px;
    border-radius: 8px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.13);
    font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
    font-weight: 400;
    font-size: 11px;
    letter-spacing: 0.40em;
    color: rgba(255,255,255,0.82);
    text-transform: uppercase;
    text-decoration: none;
    transition: background 0.2s, border-color 0.2s;
  }
  .cta:hover {
    background: rgba(255,255,255,0.13);
    border-color: rgba(255,255,255,0.24);
  }
`

function CrosshairIcon({ size = 14, color = 'currentColor' }) {
  const r = size / 2
  const gap = size * 0.30
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} fill="none" style={{ flexShrink: 0 }}>
      <circle cx={r} cy={r} r={1.4} fill={color} />
      <line x1={r} y1={0}       x2={r}      y2={r - gap} stroke={color} strokeWidth={0.9} />
      <line x1={r} y1={r + gap} x2={r}      y2={size}    stroke={color} strokeWidth={0.9} />
      <line x1={0}      y1={r}  x2={r - gap} y2={r}      stroke={color} strokeWidth={0.9} />
      <line x1={r + gap} y1={r} x2={size}    y2={r}      stroke={color} strokeWidth={0.9} />
    </svg>
  )
}

export default function Home() {
  const MONO = "'Barlow Condensed', 'Arial Narrow', sans-serif"

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: STYLES }} />

      <div style={{ position: 'fixed', inset: 0, background: '#0e0c0a' }} />

      <main style={{
        position: 'relative', zIndex: 10,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        minHeight: '100dvh', gap: 0,
      }}>

        <div className="land-1" style={{ marginBottom: 28 }}>
          <CrosshairIcon size={18} color="rgba(255,255,255,0.22)" />
        </div>

        <div className="land-2" style={{ marginBottom: 14 }}>
          <span style={{
            fontFamily: MONO, fontWeight: 400,
            fontSize: 40, letterSpacing: '0.44em',
            color: 'rgba(255,255,255,0.92)', textTransform: 'uppercase',
          }}>
            LOCATR
          </span>
        </div>

        <div className="land-3" style={{ marginBottom: 40 }}>
          <p style={{
            fontFamily: MONO, fontWeight: 300,
            fontSize: 12, letterSpacing: '0.22em',
            color: 'rgba(255,255,255,0.35)', textTransform: 'uppercase',
            margin: 0, textAlign: 'center',
          }}>
            Explore your world in 3D
          </p>
        </div>

        <div className="land-4">
          <Link href="/map" className="cta">Get Started</Link>
        </div>

      </main>
    </>
  )
}
