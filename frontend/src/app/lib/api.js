const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

/**
 * POST /api/plan — send a natural-language prompt and receive ranked venues.
 * Includes timeout, structured error handling, and optional auth token.
 */
export async function createPlan(payload, accessToken = null) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120_000); // 2-minute timeout

    const headers = { "Content-Type": "application/json" };
    if (accessToken) {
        headers["Authorization"] = `Bearer ${accessToken}`;
    }

    try {
        const res = await fetch(`${API_URL}/plan`, {
            method: "POST",
            headers,
            body: JSON.stringify({
                prompt: payload.prompt,
                group_size: payload.group_size || 1,
                budget: payload.budget || null,
                location: payload.location || null,
                vibe: payload.vibe || null,
                member_locations: payload.member_locations || null,
                chat_history: payload.chat_history || null,
            }),
            signal: controller.signal,
        });

        if (!res.ok) {
            const body = await res.text().catch(() => "");
            throw new Error(`API error ${res.status}: ${body}`);
        }

        return res.json();
    } catch (err) {
        if (err.name === "AbortError") {
            throw new Error("Request timed out. The agents took too long.");
        }
        throw err;
    } finally {
        clearTimeout(timeout);
    }
}

/**
 * POST /api/voice/synthesize — send text to ElevenLabs TTS and get audio.
 * Returns an audio Blob.
 */
export async function synthesizeSpeech(text, voiceId = null) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);

    try {
        const res = await fetch(`${API_URL}/voice/synthesize`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, voice_id: voiceId }),
            signal: controller.signal,
        });

        if (!res.ok) {
            const body = await res.text().catch(() => "");
            throw new Error(`Voice API error ${res.status}: ${body}`);
        }

        return res.blob();
    } catch (err) {
        if (err.name === "AbortError") {
            throw new Error("Voice synthesis timed out.");
        }
        throw err;
    } finally {
        clearTimeout(timeout);
    }
}
