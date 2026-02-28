'use client'

import { useEffect, useRef, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'

const GLOBAL_STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400&display=swap');

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

function formatCoord(val, pos, neg) {
  return `${Math.abs(val).toFixed(4)}° ${val >= 0 ? pos : neg}`
}

export default function MapComponent() {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const [loaded, setLoaded] = useState(false)
  const [center, setCenter] = useState({ lng: -79.3470, lat: 43.6515 })

  useEffect(() => {
    if (mapRef.current) return

    const FALLBACK = [-79.3470, 43.6515]

    const initMap = ([lng, lat]) => {
      setCenter({ lng, lat })

      mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN

      const map = new mapboxgl.Map({
        container: containerRef.current,
        style: 'mapbox://styles/mapbox/dark-v11',
        // style: 'mapbox://styles/mapbox/standard-satellite',
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

    initMap(FALLBACK)

    return () => {
      mapRef.current?.remove()
      mapRef.current = null
    }
  }, [])

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
          <CrosshairIcon size={18} color="rgba(28,22,16,0.22)" />
          <span style={{
            fontFamily: MONO,
            fontWeight: 300,
            fontSize: 10,
            letterSpacing: '0.48em',
            color: 'rgba(28,22,16,0.40)',
            textTransform: 'uppercase',
            animation: 'locatr-pulse 2.2s ease infinite',
          }}>
            LOCATR
          </span>
        </div>

        {/* Bottom-left: Wordmark */}
        {loaded && (
          <div className="ui-reveal" style={{ position: 'absolute', bottom: 36, left: 24, zIndex: 10 }}>
            <Pill>
              <CrosshairIcon size={10} color="rgba(255,255,255,0.60)" />
              <span style={{
                fontFamily: MONO,
                fontWeight: 400,
                fontSize: 11,
                letterSpacing: '0.40em',
                color: 'rgba(255,255,255,0.88)',
                textTransform: 'uppercase',
              }}>
                LOCATR
              </span>
            </Pill>
          </div>
        )}

        {/* Top-left: Live coordinates */}
        {loaded && (
          <div className="ui-reveal-delayed" style={{ position: 'absolute', top: 24, left: 24, zIndex: 10 }}>
            <Pill style={{ gap: 8 }}>
              <span style={{
                fontFamily: MONO,
                fontWeight: 300,
                fontSize: 11,
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

      </div>
    </>
  )
}
