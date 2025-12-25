"""SSE streaming endpoint."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..sse import sse_event_generator, sse_manager

router = APIRouter()


@router.get("/events")
async def sse_endpoint():
    """
    Server-Sent Events endpoint for live replay streaming.

    Events:
    - replay_start: New replay beginning
    - replay_chunk: Frame data chunk
    - replay_end: Replay complete
    """
    client = await sse_manager.register()

    return StreamingResponse(
        sse_event_generator(client, sse_manager),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
