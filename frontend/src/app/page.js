"use client";

import { useState, useRef } from "react";
import SearchBar from "@/app/components/SearchBar";
import Map from "@/app/components/Map";
import VenueCard from "@/app/components/VenueCard";
import VoiceInput from "@/app/components/VoiceInput";
import { createPlan } from "@/app/lib/api";

export default function Home() {
    const [venues, setVenues] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedVenue, setSelectedVenue] = useState(null);
    const [hasSearched, setHasSearched] = useState(false);
    const [chatHistory, setChatHistory] = useState([]);
    const mapRef = useRef(null);

    const handleSearch = async (prompt) => {
        setLoading(true);
        setError(null);
        setHasSearched(true);
        try {
            const data = await createPlan({
                prompt,
                chat_history: chatHistory.length > 0 ? chatHistory : null,
            });
            setVenues(data.venues || []);
            // Append to chat history for reprompting
            setChatHistory((prev) => [
                ...prev,
                { role: "user", content: prompt },
                { role: "assistant", content: `Found ${data.venues?.length || 0} venues.` },
            ]);
            if (!data.venues?.length) {
                setError("No venues found. Try a different search.");
            }
        } catch (err) {
            console.error("Plan request failed:", err);
            setError("Something went wrong. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const handleVenueSelect = (v) => {
        setSelectedVenue(v);
        // Pan map to venue
        if (mapRef.current && v) {
            mapRef.current.flyTo({
                center: [v.lng, v.lat],
                zoom: 15,
                duration: 1200,
            });
        }
    };

    return (
        <main className="relative h-screen w-screen overflow-hidden">
            {/* Map canvas — full bleed */}
            <Map
                venues={venues}
                onSelectVenue={handleVenueSelect}
                selectedVenue={selectedVenue}
                onMapReady={(map) => (mapRef.current = map)}
            />

            {/* Top gradient fade */}
            <div className="absolute top-0 left-0 right-0 h-32 bg-gradient-to-b from-[#0a0a0f]/80 to-transparent pointer-events-none z-[5]" />

            {/* Logo + Search overlay */}
            <div className="absolute top-5 left-1/2 -translate-x-1/2 z-10 w-full max-w-2xl px-4">
                {/* Branding */}
                <div className="flex items-center justify-center gap-2 mb-3">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-sm"
                        style={{ background: "var(--gradient-brand)" }}>
                        P
                    </div>
                    <h1 className="text-lg font-bold tracking-tight text-white">
                        PATHFINDER
                    </h1>
                    <span className="text-[10px] font-medium text-violet-400 bg-violet-500/10 px-2 py-0.5 rounded-full border border-violet-500/20">
                        BETA
                    </span>
                </div>

                <div className="flex items-center gap-2">
                    <div className="flex-1">
                        <SearchBar onSearch={handleSearch} loading={loading} />
                    </div>
                    <VoiceInput
                        onTranscript={(text) => handleSearch(text)}
                        disabled={loading}
                    />
                </div>
            </div>

            {/* Loading skeleton */}
            {loading && (
                <aside className="absolute top-36 right-4 z-10 w-80 space-y-3">
                    {[1, 2, 3].map((i) => (
                        <div
                            key={i}
                            className="glass-panel p-4 space-y-3 animate-fade-in-up"
                            style={{ animationDelay: `${i * 100}ms` }}
                        >
                            <div className="h-4 w-3/4 rounded animate-shimmer" />
                            <div className="h-3 w-1/2 rounded animate-shimmer" />
                            <div className="h-3 w-full rounded animate-shimmer" />
                        </div>
                    ))}
                    <div className="text-center py-2">
                        <div className="inline-flex items-center gap-2 text-xs text-violet-400">
                            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" opacity="0.3" />
                                <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                            </svg>
                            Agents analyzing...
                        </div>
                    </div>
                </aside>
            )}

            {/* Error state */}
            {error && !loading && (
                <div className="absolute top-36 right-4 z-10 w-80 glass-panel p-4 animate-fade-in-up">
                    <p className="text-sm text-amber-400 flex items-center gap-2">
                        <span>⚠</span> {error}
                    </p>
                </div>
            )}

            {/* Venue cards sidebar */}
            {!loading && venues.length > 0 && (
                <aside className="absolute top-36 right-4 z-10 w-80 space-y-3 max-h-[calc(100vh-10rem)] overflow-y-auto pr-1">
                    {venues.map((v, i) => (
                        <div
                            key={v.rank}
                            className="animate-fade-in-up"
                            style={{ animationDelay: `${i * 80}ms` }}
                        >
                            <VenueCard
                                venue={v}
                                active={selectedVenue?.rank === v.rank}
                                onClick={() => handleVenueSelect(v)}
                            />
                        </div>
                    ))}
                </aside>
            )}

            {/* Empty state — before first search */}
            {!loading && !hasSearched && (
                <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 text-center">
                    <p className="text-sm text-zinc-500">
                        Tell PATHFINDER what you&apos;re looking for
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2 justify-center max-w-lg">
                        {[
                            "Basketball for 10 people, budget-friendly",
                            "Birthday venue for 20 kids in Waterloo",
                            "Cozy coffee shop with wifi near me",
                        ].map((q) => (
                            <button
                                key={q}
                                onClick={() => handleSearch(q)}
                                className="text-xs px-3 py-1.5 rounded-full border border-zinc-700 text-zinc-400 hover:border-violet-500/50 hover:text-violet-400 transition-all duration-200 bg-zinc-900/50 backdrop-blur-sm"
                            >
                                {q}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* Bottom gradient fade */}
            <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-[#0a0a0f]/60 to-transparent pointer-events-none z-[5]" />
        </main>
    );
}
