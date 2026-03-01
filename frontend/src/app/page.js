'use client'

import Link from 'next/link'
import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'

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

function makeTrailData() {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const starts = [
    [Math.random() * vw * 0.40, -30],
    [-30, Math.random() * vh * 0.60],
    [Math.random() * vw * 0.30, vh + 30],
    [vw + 30, Math.random() * vh * 0.35],
  ]
  const [sx, sy] = starts[Math.floor(Math.random() * starts.length)]
  const ex = vw * (0.1 + Math.random() * 0.8)
  const ey = vh * (0.1 + Math.random() * 0.8)
  return {
    d: `M ${sx} ${sy} C ${vw * Math.random()} ${vh * Math.random()}, ${vw * Math.random()} ${vh * Math.random()}, ${ex} ${ey}`,
    ex, ey, vw, vh,
  }
}

// Animated dashed trail that draws itself in toward an X mark, then loops.
function Trail() {
  const svgRef = useRef(null)
  const visPathRef = useRef(null)
  const maskPathRef = useRef(null)
  const xRef = useRef(null)
  const [trail, setTrail] = useState(null)
  const maskId = useRef(`tm-${Math.random().toString(36).slice(2, 8)}`).current

  useEffect(() => { setTrail(makeTrailData()) }, [])

  useEffect(() => {
    if (!trail || !visPathRef.current || !maskPathRef.current) return

    // Snap SVG back to visible with no transition
    const svg = svgRef.current
    if (svg) { svg.style.transition = 'none'; svg.style.opacity = '1' }

    // Reset X mark
    const x = xRef.current
    if (x) { x.style.transition = 'none'; x.style.opacity = '0' }

    // Set up mask path: fully hidden, ready to draw in
    const len = visPathRef.current.getTotalLength()
    const mp = maskPathRef.current
    mp.style.transition = 'none'
    mp.style.strokeDasharray = `${len} ${len}`
    mp.style.strokeDashoffset = String(len)

    // Draw trail (0.5s delay, 2.8s duration)
    const raf = requestAnimationFrame(() => {
      mp.style.transition = `stroke-dashoffset 2.8s cubic-bezier(0.4, 0, 0.2, 1) 0.5s`
      mp.style.strokeDashoffset = '0'
    })

    // X fades in after trail is drawn
    const xTimer = setTimeout(() => {
      if (xRef.current) {
        xRef.current.style.transition = 'opacity 0.5s ease'
        xRef.current.style.opacity = '1'
      }
    }, 3500)

    // 2s after X is fully visible, fade everything out
    const fadeTimer = setTimeout(() => {
      if (svgRef.current) {
        svgRef.current.style.transition = 'opacity 0.8s ease'
        svgRef.current.style.opacity = '0'
      }
    }, 6000)

    // After fade, generate a fresh trail
    const restartTimer = setTimeout(() => {
      setTrail(makeTrailData())
    }, 6900)

    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(xTimer)
      clearTimeout(fadeTimer)
      clearTimeout(restartTimer)
    }
  }, [trail])

  if (!trail) return null

  const { d, ex, ey, vw, vh } = trail
  const xs = 13

  return (
    <svg
      ref={svgRef}
      style={{ position: 'fixed', inset: 0, width: '100%', height: '100%', zIndex: 1, pointerEvents: 'none' }}
      viewBox={`0 0 ${vw} ${vh}`}
    >
      <defs>
        <mask id={maskId}>
          <rect width={vw} height={vh} fill="black" />
          <path
            ref={maskPathRef}
            d={d}
            stroke="white"
            strokeWidth="20"
            strokeLinecap="round"
            fill="none"
          />
        </mask>
      </defs>

      <path
        ref={visPathRef}
        d={d}
        stroke="rgba(255,255,255,0.09)"
        strokeWidth="5"
        strokeDasharray="38 18"
        strokeLinecap="round"
        fill="none"
        mask={`url(#${maskId})`}
      />

      <g ref={xRef} style={{ opacity: 0 }} transform={`translate(${ex},${ey})`}>
        <line x1={-xs} y1={-xs} x2={xs} y2={xs} stroke="rgba(110,32,32,1)" strokeWidth="7" strokeLinecap="square" />
        <line x1={xs}  y1={-xs} x2={-xs} y2={xs} stroke="rgba(110,32,32,1)" strokeWidth="7" strokeLinecap="square" />
      </g>
    </svg>
  )
}

export default function Home() {
  const MONO = "'Barlow Condensed', 'Arial Narrow', sans-serif"
  const router = useRouter()

  function handleGetStarted() {
    sessionStorage.setItem('fromLanding', '1')
    router.push('/map')
  }

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: STYLES }} />

      <div style={{ position: 'fixed', inset: 0, background: '#0e0c0a' }} />
      <div style={{ position: 'fixed', inset: 0, backgroundImage: 'url(/noise.svg)', backgroundRepeat: 'repeat', pointerEvents: 'none', zIndex: 9999 }} />

      <Trail />

      <main style={{
        position: 'relative', zIndex: 10,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        minHeight: '100dvh', gap: 0,
      }}>

        <div className="land-1" style={{ marginBottom: 28 }}>
          <CrosshairIcon size={50} color="rgba(255,255,255,0.22)" />
        </div>

        <div className="land-2" style={{ marginBottom: 14 }}>
          <span style={{
            fontFamily: MONO, fontWeight: 400,
            fontSize: 65, letterSpacing: '0.44em',
            color: 'rgba(255,255,255,0.92)', textTransform: 'uppercase',
          }}>
            LOCATR
          </span>
        </div>

        <div className="land-3" style={{ marginBottom: 40 }}>
          <p style={{
            fontFamily: "var(--font-pt-serif), ui-serif, system-ui, serif",
            fontStyle: 'italic',
            fontSize: 25, letterSpacing: '0.10em',
            color: 'rgba(255,255,255,0.35)',
            margin: 0, textAlign: 'center',
          }}>
            Find your perfect venue.
          </p>
        </div>

        <div className="land-4">
          <button className="cta" onClick={handleGetStarted}>Get Started</button>
        </div>

      </main>
    </>
  )
}
