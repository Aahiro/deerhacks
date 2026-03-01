# TODO: Defensive Prompt Engineering Updates

## 1. Update Node 5: Cost Analyst Prompt (`app/agents/cost_analyst.py`)
**Goal:** Prevent NoneType errors and handle slow scrapers.
- [ ] Modify System Prompt to instruct Gemini to return `0.0` for all numerical fields (never `null` or `None`) and set `pricing_confidence` to "unknown" when the provided text is empty, truncated, or mentions "Access Denied/Timeout".
- [ ] Instruct prompt to extract `base_cost` (per person/hour) and identify `hidden_costs` (service fees, minimum spends).
- [ ] Calculate `total_cost_of_attendance` for the specific group size.
- [ ] Enforce the STRICT JSON schema:
  ```json
  {
    "base_cost": float,
    "hidden_costs": [{"label": string, "amount": float}],
    "total_cost_of_attendance": float,
    "pricing_confidence": "confirmed" | "estimated" | "unknown",
    "notes": string
  }
  ```

## 2. Update Node 3: Vibe Matcher Prompt (`app/agents/vibe_matcher.py` or equivalent)
**Goal:** Fix the "302 Redirect" issue and handle missing images.
- [ ] Update INPUT HANDLING to receive 1-3 image descriptions or metadata.
- [ ] Add instructions to rely on Review Sentiment provided in text metadata if an image fails to load (e.g., Redirect Error or 404), ensuring the venue is not penalized.
- [ ] Enhance AESTHETIC SCORING to score the venue from 0.0 to 1.0 based on how well it fits the specific vibe request (e.g., Cyberpunk criteria: Neon lighting, high-contrast colors, industrial materials, tech-heavy decor).
- [ ] Ensure the output schema explicitly returns `vibe_score`, `primary_style`, and `visual_descriptors`.

## 3. Update Node 1: Commander Prompt (`app/agents/commander.py` or `graph.py`)
**Goal:** Stop Auth0 404/400 spam and optimize the "COST" node activation.
- [ ] Update OAUTH & IDENTITY LOGIC to examine `user_id`.
- [ ] If `user_id` is `auth0|local_test` or looks simulated, set `identity_context` to "standard_profile".
- [ ] Explicitly instruct the Commander NOT to request a Management API lookup if the user_id does not follow the `auth0|{id}` format.
- [ ] Annotate the plan with `requires_auth: false` if the user is just looking for public cafes.
- [ ] Update COMPLEXITY TIERING rules to skip the COST node if the intent is purely aesthetic and no booking is requested to save processing time.
