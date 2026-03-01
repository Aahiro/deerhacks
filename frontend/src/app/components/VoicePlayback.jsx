"use client";

import { useState, useRef } from "react";
import { synthesizeSpeech } from "@/app/lib/api";

/**
 * Speaker button that calls the ElevenLabs TTS endpoint
 * and plays back the audio. Attach to venue cards.
 */
export default function VoicePlayback({ text, voiceId = null }) {
    const [playing, setPlaying] = useState(false);
    const [loading, setLoading] = useState(false);
    const audioRef = useRef(null);

    const handlePlay = async (e) => {
        e.stopPropagation(); // Don't trigger card click

        if (playing && audioRef.current) {
            audioRef.current.pause();
            audioRef.current = null;
            setPlaying(false);
            return;
        }

        if (!text) return;

        setLoading(true);
        try {
            const blob = await synthesizeSpeech(text, voiceId);
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);

            audio.onended = () => {
                setPlaying(false);
                URL.revokeObjectURL(url);
            };

            audio.onerror = () => {
                setPlaying(false);
                URL.revokeObjectURL(url);
            };

            audioRef.current = audio;
            await audio.play();
            setPlaying(true);
        } catch (err) {
            console.error("Voice playback failed:", err);
        } finally {
            setLoading(false);
        }
    };

    if (!text) return null;

    return (
        <button
            type="button"
            onClick={handlePlay}
            disabled={loading}
            className={`flex items-center justify-center w-7 h-7 rounded-full transition-all duration-200 ${playing
                    ? "bg-violet-500/30 text-violet-300"
                    : "bg-white/5 text-zinc-500 hover:bg-violet-500/20 hover:text-violet-400"
                } disabled:opacity-40`}
            title={playing ? "Stop" : "Listen to summary"}
        >
            {loading ? (
                <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" opacity="0.3" />
                    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
            ) : playing ? (
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="4" width="4" height="16" rx="1" />
                    <rect x="14" y="4" width="4" height="16" rx="1" />
                </svg>
            ) : (
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M11.383 3.07A1 1 0 0 1 12 4v16a1 1 0 0 1-1.707.707L5.586 16H2a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1h3.586l4.707-4.707a1 1 0 0 1 1.09-.217zM14.657 2.929a1 1 0 0 1 1.414 0A9.972 9.972 0 0 1 19 10a9.972 9.972 0 0 1-2.929 7.071 1 1 0 0 1-1.414-1.414A7.971 7.971 0 0 0 17 10a7.971 7.971 0 0 0-2.343-5.657 1 1 0 0 1 0-1.414zm-2.829 2.828a1 1 0 0 1 1.415 0A5.983 5.983 0 0 1 15 10a5.984 5.984 0 0 1-1.757 4.243 1 1 0 0 1-1.415-1.415A3.984 3.984 0 0 0 13 10a3.983 3.983 0 0 0-1.172-2.828 1 1 0 0 1 0-1.415z" />
                </svg>
            )}
        </button>
    );
}
