'use client'

import { useEffect, useRef, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import SearchBar from './SearchBar'
import AgentRow from './AgentRow'
import AgentSidebar from './AgentSidebar'
import ResultsSidebar from './ResultsSidebar'
import Sidebar from './Sidebar'

const GLOBAL_STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400&family=Inter:wght@300;400;500&display=swap');

  .mapboxgl-canvas:focus { outline: none; }
  canvas { -webkit-tap-highlight-color: transparent; }

  .mapboxgl-ctrl-top-right {
    top: 24px !important;
    right: 24px !important;
  }
  .mapboxgl-ctrl-group {
    background: rgba(252, 250, 246, 0.86) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 6px 24px rgba(0,0,0,0.07) !important;
    overflow: hidden !important;
  }
  .mapboxgl-ctrl-group button {
    width: 38px !important;
    height: 38px !important;
    border-bottom: 1px solid rgba(0,0,0,0.06) !important;
  }
  .mapboxgl-ctrl-group button:last-child { border-bottom: none !important; }
  .mapboxgl-ctrl-group button:hover { background-color: rgba(0,0,0,0.04) !important; }

  @keyframes locatr-reveal {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
  }
  @keyframes locatr-pulse {
    0%, 100% { opacity: 0.35; }
    50%       { opacity: 0.80; }
  }
  .ui-reveal          { animation: locatr-reveal 0.55s cubic-bezier(0.16,1,0.3,1) forwards; }
  .ui-reveal-delayed  { opacity:0; animation: locatr-reveal 0.55s cubic-bezier(0.16,1,0.3,1) 0.14s forwards; }
`

function CrosshairIcon({ size = 14, color = 'currentColor' }) {
  const r = size / 2
  const gap = size * 0.30
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} fill="none" style={{ flexShrink: 0 }}>
      <circle cx={r} cy={r} r={1.4} fill={color} />
      <line x1={r} y1={0}        x2={r}    y2={r - gap} stroke={color} strokeWidth={0.9} />
      <line x1={r} y1={r + gap}  x2={r}    y2={size}    stroke={color} strokeWidth={0.9} />
      <line x1={0}        y1={r} x2={r - gap} y2={r}    stroke={color} strokeWidth={0.9} />
      <line x1={r + gap}  y1={r} x2={size}    y2={r}    stroke={color} strokeWidth={0.9} />
    </svg>
  )
}

function Pill({ children, style = {} }) {
  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 10,
      background: 'rgba(20, 17, 13, 0.70)',
      backdropFilter: 'blur(16px)',
      WebkitBackdropFilter: 'blur(16px)',
      borderRadius: 8,
      padding: '8px 16px 7px',
      ...style,
    }}>
      {children}
    </div>
  )
}

const MONO = "'Barlow Condensed', 'Arial Narrow', sans-serif"
const BODY = "'Inter', -apple-system, 'Segoe UI', sans-serif"

function formatCoord(val, pos, neg) {
  return `${Math.abs(val).toFixed(4)}° ${val >= 0 ? pos : neg}`
}

function createMarkerEl(rankIdx) {
  const COLORS = ['#e8c84a', '#b0b8c4', '#c8905a', 'rgba(255,255,255,0.55)']
  const color  = COLORS[Math.min(rankIdx, 3)]
  const size   = rankIdx === 0 ? 36 : 28
  const el     = document.createElement('div')
  el.dataset.rank = rankIdx
  Object.assign(el.style, {
    width: `${size}px`, height: `${size}px`,
    borderRadius: '50%',
    background: 'rgba(14,12,10,0.88)',
    border: `2px solid ${color}`,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    cursor: 'pointer',
    fontFamily: "'Barlow Condensed', sans-serif",
    fontSize: `${rankIdx === 0 ? 15 : 12}px`,
    fontWeight: '400', color,
    boxShadow: `0 0 ${rankIdx === 0 ? '14px 4px' : '8px 2px'} ${color}66`,
    transition: 'transform 0.2s ease, box-shadow 0.2s ease',
    userSelect: 'none', pointerEvents: 'auto',
  })
  el.textContent = String(rankIdx + 1)
  return el
}

export default function MapComponent() {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const wsRef = useRef(null)
  const markersRef = useRef([])

  const [loaded, setLoaded] = useState(false)
  const [center, setCenter] = useState({ lng: -79.3470, lat: 43.6515 })

  // Search state: 'idle' | 'searching' | 'results'
  const [searchState, setSearchState] = useState('idle')
  const [activeAgent, setActiveAgent] = useState(null)
  const [agentLogs, setAgentLogs] = useState([])
  const [results, setResults] = useState(null)
  const [selectedVenueIdx, setSelectedVenueIdx] = useState(0)

  // WS cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close()
    }
  }, [])

  // Highlight selected marker
  useEffect(() => {
    markersRef.current.forEach((m, i) => {
      const el = m.getElement()
      if (i === selectedVenueIdx) {
        el.style.transform = 'scale(1.2)'
        el.style.outline = '2px solid rgba(255,255,255,0.50)'
        el.style.outlineOffset = '2px'
      } else {
        el.style.transform = 'scale(1)'
        el.style.outline = 'none'
      }
    })
  }, [selectedVenueIdx])

  // Spawn markers + auto-fly when results arrive
  useEffect(() => {
    if (!results || !mapRef.current) return
    const venues = results.venues ?? []

    markersRef.current.forEach(m => m.remove())
    markersRef.current = []
    setSelectedVenueIdx(0)

    venues.forEach((venue, i) => {
      const el = createMarkerEl(i)
      const marker = new mapboxgl.Marker({ element: el, anchor: 'center' })
        .setLngLat([venue.lng, venue.lat])
        .addTo(mapRef.current)
      el.addEventListener('click', () => {
        setSelectedVenueIdx(i)
        mapRef.current.flyTo({ center: [venue.lng, venue.lat], zoom: 15, duration: 900 })
      })
      markersRef.current.push(marker)
    })

    if (venues.length > 0) {
      mapRef.current.flyTo({
        center: [venues[0].lng, venues[0].lat],
        zoom: 15, pitch: 45, bearing: -17.6, duration: 1400,
      })
    }
  }, [results])

  const handleNewSearch = () => {
    markersRef.current.forEach(m => m.remove())
    markersRef.current = []
    setResults(null)
    setSearchState('idle')
    setAgentLogs([])
    setActiveAgent(null)
    setSelectedVenueIdx(0)
  }

  const handleSearch = (query) => {
    markersRef.current.forEach(m => m.remove())
    markersRef.current = []
    setSearchState('searching')
    setActiveAgent(null)
    setAgentLogs([])
    setResults(null)

    const wsUrl = (process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000') + '/api/ws/plan'
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({ prompt: query, member_locations: [] }))
    }

    ws.onmessage = (event) => {
      let msg
      try { msg = JSON.parse(event.data) } catch { return }

      if (msg.type === 'log') {
        setActiveAgent(msg.node)
        setAgentLogs((prev) => [
          ...prev,
          { agent: msg.node, message: msg.message, time: Date.now() },
        ])
      } else if (msg.type === 'result') {
        setResults(msg.data)
        setSearchState('results')
        setActiveAgent(null)
        ws.close()
      }
    }

    ws.onerror = () => {
      setSearchState('idle')
    }

    ws.onclose = (e) => {
      if (searchState === 'searching' && !e.wasClean) {
        setSearchState('idle')
      }
    }
  }

  useEffect(() => {
    if (mapRef.current) return

    const FALLBACK = [-79.3470, 43.6515]  // downtown Toronto

    const initMap = ([lng, lat]) => {
      setCenter({ lng, lat })

      mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN

      const map = new mapboxgl.Map({
        container: containerRef.current,
        style: 'mapbox://styles/mapbox/dark-v11',
        center: [lng, lat],
        zoom: 14,
        pitch: 45,
        bearing: -17.6,
        antialias: true,
      })

      mapRef.current = map

      map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), 'top-right')
      map.addControl(
        new mapboxgl.GeolocateControl({
          positionOptions: { enableHighAccuracy: true },
          trackUserLocation: true,
          showUserHeading: true,
        }),
        'top-right'
      )

      map.on('load', () => {
        const layers = map.getStyle().layers
        let labelLayerId
        for (const layer of layers) {
          if (layer.type === 'symbol' && layer.layout?.['text-field']) {
            labelLayerId = layer.id
            break
          }
        }

        map.addLayer(
          {
            id: '3d-buildings',
            source: 'composite',
            'source-layer': 'building',
            filter: ['==', 'extrude', 'true'],
            type: 'fill-extrusion',
            minzoom: 15,
            paint: {
              'fill-extrusion-color': '#625f5a',
              'fill-extrusion-height': [
                'interpolate', ['linear'], ['zoom'],
                15, 0,
                15.05, ['get', 'height'],
              ],
              'fill-extrusion-base': [
                'interpolate', ['linear'], ['zoom'],
                15, 0,
                15.05, ['get', 'min_height'],
              ],
              'fill-extrusion-opacity': 0.68,
            },
          },
          labelLayerId
        )

        setLoaded(true)
      })

      map.on('move', () => {
        const c = map.getCenter()
        setCenter({ lng: c.lng, lat: c.lat })
      })
    }

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => initMap([pos.coords.longitude, pos.coords.latitude]),
        ()    => initMap(FALLBACK),
        { timeout: 6000, maximumAge: 60000 }
      )
    } else {
      initMap(FALLBACK)
    }

    return () => {
      mapRef.current?.remove()
      mapRef.current = null
    }
  }, [])

  const completedAgents = [...new Set(agentLogs.map((l) => l.agent))]

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: GLOBAL_STYLES }} />

      <div style={{ position: 'fixed', inset: 0, background: '#ede9e3', overflow: 'hidden' }}>

        {/* Map canvas */}
        <div
          ref={containerRef}
          style={{
            width: '100%',
            height: '100%',
            opacity: loaded ? 1 : 0,
            transition: 'opacity 1.5s cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        />

        {/* Loading screen */}
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#ede9e3',
          gap: 18,
          opacity: loaded ? 0 : 1,
          transition: 'opacity 0.9s ease',
          pointerEvents: loaded ? 'none' : 'all',
          zIndex: 30,
        }}>
          <CrosshairIcon size={50} color="rgba(28,22,16,0.22)" />
          <span style={{
            fontFamily: MONO,
            fontWeight: 300,
            fontSize: 50,
            letterSpacing: '0.48em',
            color: 'rgba(28,22,16,0.40)',
            textTransform: 'uppercase',
            animation: 'locatr-pulse 2.2s ease infinite',
          }}>
            LOCATR
          </span>
          <span style={{
            fontFamily: MONO,
            fontWeight: 300,
            marginTop: 30,
            fontSize: 25,
            color: 'rgba(28,22,16,0.25)',
            // textTransform: 'uppercase',
            // animation: 'locatr-pulse 2.2s ease infinite',
          }}>
            just give me a second...
          </span>
        </div>

        {/* Top-center: Search bar */}
        {loaded && (
          <div style={{
            position: 'absolute',
            top: 24,
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 10,
          }}>
            <SearchBar
              onSearch={handleSearch}
              disabled={searchState === 'searching'}
            />
          </div>
        )}

        {/* Top-left: Wordmark + Live coordinates */}
        {loaded && (
          <div className="ui-reveal" style={{ position: 'absolute', top: 24, left: 24, zIndex: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Pill>
              <CrosshairIcon size={10} color="rgba(255,255,255,0.60)" />
              <span style={{
                fontFamily: MONO,
                fontWeight: 400,
                fontSize: 13,
                letterSpacing: '0.40em',
                color: 'rgba(255,255,255,0.88)',
                textTransform: 'uppercase',
              }}>
                LOCATR
              </span>
            </Pill>
            <Pill style={{ gap: 8 }}>
              <span style={{
                fontFamily: MONO,
                fontWeight: 300,
                fontSize: 13,
                letterSpacing: '0.10em',
                color: 'rgba(255,255,255,0.58)',
                fontVariantNumeric: 'tabular-nums',
                whiteSpace: 'nowrap',
              }}>
                {formatCoord(center.lat, 'N', 'S')}
                <span style={{ margin: '0 10px', opacity: 0.35 }}>·</span>
                {formatCoord(center.lng, 'E', 'W')}
              </span>
            </Pill>
          </div>
        )}

        {/* Sidebar — searching + results */}
        {searchState !== 'idle' && (
          <Sidebar
          searchState={searchState}
          logs={agentLogs}
          activeAgent={activeAgent}
          venues={results?.venues}
          globalConsensus={results?.global_consensus}
          selectedIdx={selectedVenueIdx}
          onSelect={(i) => {
            setSelectedVenueIdx(i);
            const v = results?.venues[i];
            if (v) {
              mapRef.current?.flyTo({ center: [v.lng, v.lat], zoom: 15, duration: 900 });
            }
          }}
          onNewSearch={handleNewSearch}
          />
        )}

        {/* AgentRow — searching + results (dismissed when results arrive) */}
        {searchState !== 'idle' && (
          <AgentRow
            activeAgent={activeAgent}
            completedAgents={completedAgents}
            dismissed={searchState === 'results'}
          />
        )}

      </div>
    </>
  )
}