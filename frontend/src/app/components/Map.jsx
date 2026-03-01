"use client";

import { useRef, useEffect } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || "";

/**
 * Interactive Mapbox canvas with custom ranked markers,
 * popups, and isochrone overlays.
 */
export default function Map({ venues = [], onSelectVenue, selectedVenue, onMapReady }) {
    const containerRef = useRef(null);
    const mapRef = useRef(null);
    const markersRef = useRef([]);
    const popupRef = useRef(null);

    // Initialise map once
    useEffect(() => {
        if (mapRef.current || !token) return;
        mapboxgl.accessToken = token;
        const map = new mapboxgl.Map({
            container: containerRef.current,
            style: "mapbox://styles/mapbox/dark-v11",
            center: [-79.3832, 43.6532], // Toronto default
            zoom: 12,
            pitch: 45,
            bearing: -17.6,
            antialias: true,
            attributionControl: false,
        });

        map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), "bottom-right");
        map.addControl(new mapboxgl.AttributionControl({ compact: true }), "bottom-left");
        map.addControl(
            new mapboxgl.GeolocateControl({
                positionOptions: { enableHighAccuracy: true },
                trackUserLocation: true,
                showUserHeading: true,
            }),
            "bottom-right"
        );

        // Add 3D buildings on load
        map.on("load", () => {
            const layers = map.getStyle().layers;
            let labelLayerId;
            for (const layer of layers) {
                if (layer.type === "symbol" && layer.layout?.["text-field"]) {
                    labelLayerId = layer.id;
                    break;
                }
            }

            map.addLayer(
                {
                    id: "3d-buildings",
                    source: "composite",
                    "source-layer": "building",
                    filter: ["==", "extrude", "true"],
                    type: "fill-extrusion",
                    minzoom: 15,
                    paint: {
                        "fill-extrusion-color": "#625f5a",
                        "fill-extrusion-height": [
                            "interpolate", ["linear"], ["zoom"],
                            15, 0,
                            15.05, ["get", "height"],
                        ],
                        "fill-extrusion-base": [
                            "interpolate", ["linear"], ["zoom"],
                            15, 0,
                            15.05, ["get", "min_height"],
                        ],
                        "fill-extrusion-opacity": 0.68,
                    },
                },
                labelLayerId
            );
        });

        mapRef.current = map;
        onMapReady?.(map);
    }, [onMapReady]);

    // Update markers + isochrones when venues change
    useEffect(() => {
        const map = mapRef.current;
        if (!map) return;

        // Clear old markers
        markersRef.current.forEach((m) => m.remove());
        markersRef.current = [];

        // Close any open popup
        if (popupRef.current) {
            popupRef.current.remove();
            popupRef.current = null;
        }

        // Clean up old isochrone layers
        for (let i = 0; i < 20; i++) {
            const layerId = `isochrone-fill-${i}`;
            const outlineId = `isochrone-outline-${i}`;
            const sourceId = `isochrone-${i}`;
            if (map.getLayer(layerId)) map.removeLayer(layerId);
            if (map.getLayer(outlineId)) map.removeLayer(outlineId);
            if (map.getSource(sourceId)) map.removeSource(sourceId);
        }

        if (!venues.length) return;

        // Add isochrone layers first (below markers)
        venues.forEach((v, idx) => {
            if (!v.isochrone_geojson) return;
            const sourceId = `isochrone-${idx}`;
            const layerId = `isochrone-fill-${idx}`;
            const outlineId = `isochrone-outline-${idx}`;

            try {
                map.addSource(sourceId, { type: "geojson", data: v.isochrone_geojson });
                map.addLayer({
                    id: layerId,
                    type: "fill",
                    source: sourceId,
                    paint: {
                        "fill-color": "#7c3aed",
                        "fill-opacity": 0.1,
                    },
                });
                map.addLayer({
                    id: outlineId,
                    type: "line",
                    source: sourceId,
                    paint: {
                        "line-color": "#7c3aed",
                        "line-width": 1.5,
                        "line-opacity": 0.3,
                    },
                });
            } catch (e) {
                console.warn("Failed to add isochrone layer:", e);
            }
        });

        // Add markers
        venues.forEach((v) => {
            const el = document.createElement("div");
            el.className = "pf-marker";
            el.textContent = v.rank;

            const marker = new mapboxgl.Marker(el)
                .setLngLat([v.lng, v.lat])
                .addTo(map);

            el.addEventListener("click", (e) => {
                e.stopPropagation();
                onSelectVenue?.(v);

                // Show popup
                if (popupRef.current) popupRef.current.remove();

                const vibeText =
                    v.vibe_score != null
                        ? `<span style="color:#a78bfa">✦ Vibe ${Math.round(v.vibe_score * 100)}%</span>`
                        : "";
                const costText =
                    v.cost_profile?.per_person > 0
                        ? `<span style="color:#fbbf24">$ $${v.cost_profile.per_person.toFixed(0)}/person</span>`
                        : "";

                const popup = new mapboxgl.Popup({
                    offset: 20,
                    closeButton: true,
                    maxWidth: "260px",
                })
                    .setLngLat([v.lng, v.lat])
                    .setHTML(
                        `<div style="font-family:Inter,sans-serif">
                            <div style="font-weight:600;font-size:13px;margin-bottom:4px">#${v.rank} ${v.name}</div>
                            <div style="font-size:11px;color:#9ca3af;margin-bottom:6px">${v.address}</div>
                            <div style="display:flex;gap:8px;font-size:11px">${vibeText}${costText}</div>
                            ${v.why ? `<div style="font-size:11px;color:#34d399;margin-top:6px">✓ ${v.why}</div>` : ""}
                        </div>`
                    )
                    .addTo(map);

                popupRef.current = popup;
            });

            markersRef.current.push(marker);
        });

        // Fit bounds
        const bounds = new mapboxgl.LngLatBounds();
        venues.forEach((v) => bounds.extend([v.lng, v.lat]));
        map.fitBounds(bounds, { padding: { top: 120, bottom: 60, left: 60, right: 360 }, duration: 800 });
    }, [venues, onSelectVenue]);

    if (!token) {
        return (
            <div className="absolute inset-0 bg-gradient-to-br from-[#0a0a1a] via-[#0f0f2a] to-[#0a0a1a] flex items-center justify-center">
                <p className="text-zinc-600 text-sm">Configure NEXT_PUBLIC_MAPBOX_TOKEN to enable the map</p>
            </div>
        );
    }

    return <div ref={containerRef} className="absolute inset-0" id="map-container" />;
}
