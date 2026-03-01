"use client";

import { useState } from "react";

/**
 * Premium floating search bar with gradient border and loading state.
 */
export default function SearchBar({ onSearch, loading }) {
    const [value, setValue] = useState("");
    const [focused, setFocused] = useState(false);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (value.trim()) onSearch(value.trim());
    };

    return (
        <form
            onSubmit={handleSubmit}
            className={`relative flex items-center gap-3 glass-panel-strong px-5 py-3.5 transition-all duration-300 ${focused ? "glow" : ""
                }`}
            style={{
                borderColor: focused
                    ? "rgba(124, 58, 237, 0.4)"
                    : "rgba(124, 58, 237, 0.15)",
            }}
        >
            {/* Search icon */}
            <svg
                className="w-5 h-5 text-zinc-500 flex-shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
            >
                <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M21 21l-4.35-4.35m0 0A7.5 7.5 0 1 0 5.65 5.65a7.5 7.5 0 0 0 10.99 10.99z"
                />
            </svg>

            <input
                type="text"
                value={value}
                onChange={(e) => setValue(e.target.value)}
                onFocus={() => setFocused(true)}
                onBlur={() => setFocused(false)}
                placeholder="e.g. Basketball for 10 people, budget-friendly, west end"
                className="flex-1 bg-transparent outline-none text-sm text-white placeholder:text-zinc-500 min-w-0"
                id="search-input"
            />

            <button
                type="submit"
                disabled={loading || !value.trim()}
                className="flex items-center gap-2 rounded-xl px-5 py-2 text-sm font-semibold text-white transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                style={{
                    background: loading
                        ? "rgba(124, 58, 237, 0.3)"
                        : "var(--gradient-brand)",
                }}
                id="search-button"
            >
                {loading ? (
                    <>
                        <svg
                            className="w-4 h-4 animate-spin"
                            viewBox="0 0 24 24"
                            fill="none"
                        >
                            <circle
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="2"
                                opacity="0.3"
                            />
                            <path
                                d="M12 2a10 10 0 0 1 10 10"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                            />
                        </svg>
                        Thinking…
                    </>
                ) : (
                    "Search"
                )}
            </button>
        </form>
    );
}
