"""
PATHFINDER API routes.
"""

import asyncio
import io
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from app.schemas import PlanRequest, PlanResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Maximum time (seconds) the full pipeline may run before we give up
_PIPELINE_TIMEOUT = 120.0

NODE_LABELS = {
    "commander": "Parsing your request...",
    "scout": "Discovering venues...",
    "parallel_analysts": "Analysing vibes, cost & risks...",
    "vibe_matcher": "Analysing vibes...",
    "cost_analyst": "Calculating costs...",
    "critic": "Running risk assessment...",
    "synthesiser": "Ranking results...",
}


@router.post("/plan", response_model=PlanResponse)
async def create_plan(request: PlanRequest):
    """
    Accept a natural-language activity request and return ranked venues.

    Flow: prompt → Commander → Scout → [Vibe, Cost, Critic] → Synthesiser → results
    """
    from app.graph import pathfinder_graph

    initial_state = {
        "raw_prompt": request.prompt,
        "parsed_intent": {},
        "complexity_tier": "tier_2",
        "active_agents": [],
        "agent_weights": {},
        "candidate_venues": [],
        "vibe_scores": {},
        "cost_profiles": {},
        "risk_flags": {},
        "veto": False,
        "veto_reason": None,
        "fast_fail": False,
        "fast_fail_reason": None,
        "retry_count": 0,
        "ranked_results": [],
        "snowflake_context": None,
        "member_locations": request.member_locations or [],
        "chat_history": request.chat_history or [],
    }

    if request.group_size > 1 or request.budget or request.location or request.vibe:
        initial_state["parsed_intent"] = {
            "group_size": request.group_size,
            "budget": request.budget,
            "location": request.location,
            "vibe": request.vibe,
        }

    try:
        result = await asyncio.wait_for(
            pathfinder_graph.ainvoke(initial_state),
            timeout=_PIPELINE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("Pipeline timed out after %ss for prompt: %s", _PIPELINE_TIMEOUT, request.prompt)
        raise HTTPException(status_code=504, detail="Pipeline timed out — please try again.")
    except Exception as exc:
        logger.error("Pipeline error for prompt '%s': %s", request.prompt, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Pipeline failed — please try again.")

    return PlanResponse(
        venues=result.get("ranked_results", []),
        execution_summary="Pipeline complete.",
    )


@router.websocket("/ws/plan")
async def websocket_plan(websocket: WebSocket):
    from app.graph import pathfinder_graph

    await websocket.accept()
    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        logger.info("WS pipeline starting for prompt: %s", prompt)

        initial_state = {
            "raw_prompt": prompt,
            "parsed_intent": {},
            "complexity_tier": "tier_2",
            "active_agents": [],
            "agent_weights": {},
            "candidate_venues": [],
            "vibe_scores": {},
            "cost_profiles": {},
            "risk_flags": {},
            "veto": False,
            "veto_reason": None,
            "fast_fail": False,
            "fast_fail_reason": None,
            "retry_count": 0,
            "ranked_results": [],
            "member_locations": data.get("member_locations", []),
        }

        accumulated = {**initial_state}

        async def _run_pipeline():
            async for event in pathfinder_graph.astream(initial_state):
                node_name = list(event.keys())[0]
                node_data = event[node_name] or {}
                accumulated.update(node_data)
                logger.info("WS node complete: %s", node_name)
                await websocket.send_json({
                    "type": "progress",
                    "node": node_name,
                    "label": NODE_LABELS.get(node_name, node_name),
                })

        try:
            await asyncio.wait_for(_run_pipeline(), timeout=_PIPELINE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("WS pipeline timed out after %ss for: %s", _PIPELINE_TIMEOUT, prompt)
            await websocket.send_json({
                "type": "error",
                "message": "Pipeline timed out — please try a simpler query.",
            })
            return
        except Exception as exc:
            logger.error("WS pipeline error for '%s': %s", prompt, exc, exc_info=True)
            await websocket.send_json({
                "type": "error",
                "message": "An internal error occurred. Please try again.",
            })
            return

        # Always send the final result — even if ranked_results is empty
        await websocket.send_json({
            "type": "result",
            "data": PlanResponse(
                venues=accumulated.get("ranked_results", []),
                execution_summary="Pipeline complete.",
            ).model_dump(),
        })
        logger.info("WS pipeline complete — %d venues returned", len(accumulated.get("ranked_results", [])))

    except WebSocketDisconnect:
        logger.info("WS client disconnected")
    except Exception as exc:
        logger.error("Unexpected WS error: %s", exc, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": "Unexpected server error."})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/health")
async def api_health():
    return {"status": "ok"}


# ── Voice TTS ────────────────────────────────────────────


class VoiceSynthRequest(BaseModel):
    """Request body for text-to-speech synthesis."""
    text: str = Field(..., description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="ElevenLabs voice ID")


@router.post("/voice/synthesize")
async def synthesize_voice(request: VoiceSynthRequest):
    """
    Convert text to speech using ElevenLabs.
    Returns an audio/mpeg stream.
    """
    from app.services.elevenlabs import synthesize_speech

    audio_bytes = await synthesize_speech(
        text=request.text,
        voice_id=request.voice_id,
    )

    if audio_bytes is None:
        return {"error": "Voice synthesis unavailable. Check ELEVENLABS_API_KEY."}

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )
