"use client";

import { useState, useRef, useCallback } from "react";

/**
 * Microphone button that uses the browser's Web Speech API
 * for voice-to-text input. Feeds the transcript into onTranscript.
 */
export default function VoiceInput({ onTranscript, disabled = false }) {
    const [listening, setListening] = useState(false);
    const recognitionRef = useRef(null);

    const toggleListening = useCallback(() => {
        // Check browser support
        const SpeechRecognition =
            typeof window !== "undefined" &&
            (window.SpeechRecognition || window.webkitSpeechRecognition);

        if (!SpeechRecognition) {
            alert("Voice input is not supported in this browser. Try Chrome.");
            return;
        }

        if (listening) {
            // Stop listening
            recognitionRef.current?.stop();
            setListening(false);
            return;
        }

        // Start listening
        const recognition = new SpeechRecognition();
        recognition.lang = "en-US";
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        recognition.continuous = false;

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            onTranscript?.(transcript);
            setListening(false);
        };

        recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
            setListening(false);
        };

        recognition.onend = () => {
            setListening(false);
        };

        recognitionRef.current = recognition;
        recognition.start();
        setListening(true);
    }, [listening, onTranscript]);

    return (
        <button
            type="button"
            onClick={toggleListening}
            disabled={disabled}
            className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200 ${listening
                    ? "bg-red-500/20 text-red-400 animate-pulse-glow"
                    : "bg-white/5 text-zinc-400 hover:bg-violet-500/20 hover:text-violet-400"
                } disabled:opacity-40 disabled:cursor-not-allowed`}
            title={listening ? "Stop listening" : "Voice input"}
            id="voice-input-button"
        >
            {listening ? (
                /* Pulsing mic — active */
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3z" />
                    <path d="M19 11a1 1 0 1 0-2 0 5 5 0 0 1-10 0 1 1 0 1 0-2 0 7 7 0 0 0 6 6.92V20H8a1 1 0 1 0 0 2h8a1 1 0 1 0 0-2h-3v-2.08A7 7 0 0 0 19 11z" />
                </svg>
            ) : (
                /* Mic icon — idle */
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z" />
                </svg>
            )}
        </button>
    );
}
