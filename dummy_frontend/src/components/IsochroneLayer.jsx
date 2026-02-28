"use client";

import { useEffect } from "react";

/**
 * Renders GeoJSON isochrone overlays on the Mapbox map.
 * Expects the map instance and a venues array with isochrone_geojson.
 */
export default function IsochroneLayer({ map, venues = [] }) {
    useEffect(() => {
        if (!map) return;

        venues.forEach((venue, idx) => {
            const geojson = venue.isochrone_geojson;
            if (!geojson) return;

            const sourceId = `isochrone-${idx}`;
            const layerId = `isochrone-fill-${idx}`;

            // Add or update source
            if (map.getSource(sourceId)) {
                map.getSource(sourceId).setData(geojson);
            } else {
                map.addSource(sourceId, { type: "geojson", data: geojson });
                map.addLayer({
                    id: layerId,
                    type: "fill",
                    source: sourceId,
                    paint: {
                        "fill-color": "#7c3aed",
                        "fill-opacity": 0.15,
                    },
                });
            }
        });

        // Cleanup on unmount
        return () => {
            venues.forEach((_, idx) => {
                const layerId = `isochrone-fill-${idx}`;
                const sourceId = `isochrone-${idx}`;
                if (map.getLayer(layerId)) map.removeLayer(layerId);
                if (map.getSource(sourceId)) map.removeSource(sourceId);
            });
        };
    }, [map, venues]);

    return null; // Renders into the map, no DOM output
}
