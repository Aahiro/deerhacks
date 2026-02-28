const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

/**
 * POST /api/plan — send a natural-language prompt and receive ranked venues.
 * Includes timeout and structured error handling.
 */
export async function createPlan(payload) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120_000); // 2-minute timeout

    try {
        const res = await fetch(`${API_URL}/plan`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: payload.prompt,
                group_size: payload.group_size || 1,
                budget: payload.budget || null,
                location: payload.location || null,
                vibe: payload.vibe || null,
                member_locations: payload.member_locations || null,
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
