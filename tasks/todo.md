# Pricing Model Implementation Plan

This plan outlines the steps to simplify the pricing architecture, transitioning from web scraping and Gemini-based cost computation to a fast, heuristic-based price signal normalizer.

## 1. Discovery APIs (Node 2: Scout)
- [x] **Update `google_places.py`**: Add `places.priceLevel` to the `X-Goog-FieldMask`. Map the returned enum values (e.g., `PRICE_LEVEL_INEXPENSIVE`) or integers to `$`-`$$$$` and include `"price_range"` in the returned dict.
- [x] **Update `yelp.py`**: Extract the `"price"` attribute (which is already `$`-`$$$$`) and include `"price_range"` in the returned dict.

## 2. Shared State (`app/models/state.py`)
- [x] Update `PathfinderState` documentation and comments to reflect that `cost_profiles` now stores `price_range` and `confidence` rather than granular numerical costs. (We will keep the key name `cost_profiles` to minimize breakage across the graph).

## 3. Cost Analyst (Node 4)
- [x] **Remove Scraping**: Delete `_firecrawl_map`, `_firecrawl_scrape`, `_guess_pricing_pages`, and `_is_website_alive`.
- [x] **Remove Gemini Extractor**: Delete `_COST_PROMPT` and `_extract_pricing`.
- [x] **Implement Heuristics**: Rewrite `_analyze_venue_cost` (or create a new sync function) to:
  - Read `price_range` added by the Scout from Google and Yelp.
  - Implement conflict resolution (Google vs. Yelp).
  - Assign `pricing_confidence` (`high`, `medium`, `low`, `none`).
  - Calculate a simple `value_score` mapping for the Synthesiser to use. 
  - Remove CIBA/Auth0 triggers (as pricing is now informational only).

## 4. Synthesiser (Node 6)
- [x] **Update Schema**: Modify the generated `VenueResult` to include `price_range` and `price_confidence`.
- [x] **Update Prompt**: Ensure `_SYNTHESIS_PROMPT` only generates `why` and `watch_out`.
- [x] **Global Consensus**: Add a new step after ranking the top venues to pass the top 3 back to Gemini to generate the `"global_consensus"` field. Emplace this at the root of the output JSON.
- [x] **Email Automation Transition**: Change all CIBA flows through Auth0 to use pure emailing once a global consensus is reached. Allow the user to automate sending a Gemini-generated email if they provide access via the token vault.

## 5. Verification & Security Review
- [x] Run `pytest backend/tests/test_commander.py`
- [x] Run `python backend/tests/test_all_nodes_connected.py` (Integration workflow test)
- [x] Review all modified code for exposed secrets, sensitive info, and ensure it aligns with "What would Zuck do" (fast, simple, robust, production-ready).

## Review Summary
The pricing model architecture was successfully migrated to a simplified, fast, heuristic-based approach:
1. **Node 2 (Scout)** and discovery APIs were updated to capture string-based `$–$$$$` price ranges directly from Google Places and Yelp, carrying these seamlessly downstream. Oh, and the deduplication function was improved to merge source prices together if both Google and Yelp return a venue.
2. **Node 4 (Cost Analyst)** was completely shredded down to its basics. All web crawling via Firecrawl, slow HTTP requests, and generative price extraction via Gemini were removed. Now, it functions as a fast, pure heuristic node: matching price signals across sources to determine `confidence` and assigning a simple `value_score` to feed into the synthesizer's formula.
3. **Node 6 (Synthesiser)** was refactored significantly to reflect the new global consensus mechanism. Instead of pushing granular cost structures per-venue, it outputs the `price_range` array and uses Gemini to synthesize these into a strong `global_consensus` message spanning all choices.
4. **Security & Speed Focus (Zuck strategy)**: By stripping out the Firecrawl mechanism and the heavy, expensive Gemini schema extraction per-venue for prices, we drastically reduced response times, API costs, and the risk of the system crashing on protected web pages. CIBA Auth has been smoothly transitioned back to the generic "action requests" pipeline within Synthesiser for emails ("send_email"), ensuring that the Token Vault is used specifically for automated outreach rather than complex payment authorization flows. Code is clean and production-ready.
