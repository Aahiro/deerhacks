"use client";

import VoicePlayback from "@/app/components/VoicePlayback";

/**
 * Rich venue result card — glassmorphism panel with score badges,
 * cost breakdown, risk warnings, confidence badges, and gradient hover effects.
 */
export default function VenueCard({ venue, active, onClick }) {
    const vibePercent =
        venue.vibe_score != null ? Math.round(venue.vibe_score * 100) : null;
    const accessPercent =
        venue.accessibility_score != null
            ? Math.round(venue.accessibility_score * 100)
            : null;

    const getScoreClass = (pct) => {
        if (pct >= 70) return "high";
        if (pct >= 40) return "medium";
        return "low";
    };

    const costProfile = venue.cost_profile;

    // Pricing confidence badge
    const confidenceBadge = (confidence) => {
        switch (confidence) {
            case "confirmed":
                return (
                    <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/20">
                        ✓ Confirmed
                    </span>
                );
            case "estimated":
                return (
                    <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/20">
                        ⚠ Estimated
                    </span>
                );
            default:
                return (
                    <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-zinc-500/15 text-zinc-500 border border-zinc-500/20">
                        ? Unknown
                    </span>
                );
        }
    };

    return (
        <div
            onClick={onClick}
            className={`cursor-pointer glass-panel p-4 transition-all duration-200 hover:scale-[1.01] ${active
                ? "!border-violet-500/60 glow"
                : "hover:border-violet-500/30"
                }`}
            id={`venue-card-${venue.rank}`}
        >
            {/* Header: Rank + Name + Voice */}
            <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2.5 min-w-0">
                    <div
                        className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white"
                        style={{ background: "var(--gradient-brand)" }}
                    >
                        {venue.rank}
                    </div>
                    <h3 className="font-semibold text-white text-sm truncate">
                        {venue.name}
                    </h3>
                </div>
                <VoicePlayback
                    text={[venue.why, venue.watch_out].filter(Boolean).join(". ")}
                />
            </div>

            {/* Address */}
            <p className="mt-1.5 text-xs text-zinc-500 ml-[38px]">
                {venue.address}
            </p>

            {/* Score Badges */}
            <div className="mt-3 flex flex-wrap gap-1.5 ml-[38px]">
                {vibePercent != null && (
                    <span className={`score-badge ${getScoreClass(vibePercent)}`}>
                        ✦ Vibe {vibePercent}%
                    </span>
                )}
                {accessPercent != null && (
                    <span
                        className={`score-badge ${getScoreClass(accessPercent)}`}
                    >
                        ◎ Access {accessPercent}%
                    </span>
                )}
                {costProfile?.per_person > 0 && (
                    <span className="score-badge medium">
                        $ ${costProfile.per_person.toFixed(0)}/person
                    </span>
                )}
            </div>

            {/* Why this venue */}
            {venue.why && (
                <p className="mt-2.5 text-xs text-emerald-400/90 ml-[38px] leading-relaxed">
                    <span className="text-emerald-400 font-medium">✓</span>{" "}
                    {venue.why}
                </p>
            )}

            {/* Watch out */}
            {venue.watch_out && (
                <p className="mt-1.5 text-xs text-amber-400/90 ml-[38px] leading-relaxed">
                    <span className="text-amber-400 font-medium">⚠</span>{" "}
                    {venue.watch_out}
                </p>
            )}

            {/* Cost breakdown with confidence badge */}
            {costProfile && costProfile.total_cost_of_attendance > 0 && (
                <div className="mt-2.5 ml-[38px] text-xs text-zinc-500 flex items-center gap-2 flex-wrap">
                    <span className="text-zinc-400">
                        Total: ${costProfile.total_cost_of_attendance.toFixed(0)} CAD
                    </span>
                    {costProfile.pricing_confidence && confidenceBadge(costProfile.pricing_confidence)}
                    {costProfile.hidden_costs?.length > 0 && (
                        <span className="text-amber-500/70">
                            +{costProfile.hidden_costs.length} hidden fee
                            {costProfile.hidden_costs.length > 1 ? "s" : ""}
                        </span>
                    )}
                </div>
            )}
        </div>
    );
}
