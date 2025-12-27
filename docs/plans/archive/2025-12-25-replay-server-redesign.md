# Replay Server Redesign

**Date:** 2025-12-25
**Status:** Approved
**Problem:** Current server.py barely runs - 10-50 MB replays cause memory pressure, SSE disconnects, browser crashes, server hangs.

## Overview

Replace monolithic 828-line `server.py` with chunked streaming architecture. Replays sent in ~1-2 MB chunks instead of single huge JSON blobs. Memory stays bounded on both server and client.

## Architecture

### Server Structure

```
echelon/server/
├── __init__.py          # FastAPI app, lifespan, middleware
├── sse.py               # SSE client management (simplified)
├── replay.py            # Chunked replay handling
├── nav_cache.py         # NavGraph caching by world hash
└── routes/
    ├── stream.py        # GET /events (SSE endpoint)
    ├── push.py          # POST /push (chunked replays)
    └── nav.py           # POST /nav/build, /nav/path
```

### Data Flow

```
Training script                    Server                         Viewer
     │                               │                               │
     ├─── PUT /worlds/{hash} ───────►│ (cache world)                 │
     │                               │                               │
     ├─── POST /push ───────────────►│                               │
     │    {world_ref, frames[0:100]} │                               │
     │                               ├─── SSE: chunk 0 ─────────────►│
     │                               │                               │ (start playback)
     ├─── POST /push ───────────────►│                               │
     │    {world_ref, frames[100:200]}                               │
     │                               ├─── SSE: chunk 1 ─────────────►│
     │                               │                               │ (append frames)
```

## Chunked Replay Protocol

### Push Payload

```python
{
    "replay_id": "abc123",          # Unique ID for this replay session
    "world_ref": "sha256hash",      # Reference to cached world
    "chunk_index": 0,               # Which chunk (0, 1, 2, ...)
    "chunk_count": 5,               # Total chunks expected
    "frames": [...],                # ~100 frames per chunk
    "meta": {                       # Only in chunk 0
        "episode": 42,
        "update": 100,
    }
}
```

**Chunk size target:** ~1-2 MB per chunk (100 frames).

### SSE Events

| Event Type | Payload | Purpose |
|------------|---------|---------|
| `replay_start` | `{replay_id, world_ref, chunk_count, meta}` | New replay beginning |
| `replay_chunk` | `{replay_id, chunk_index, frames}` | Frame data |
| `replay_end` | `{replay_id}` | Replay complete |
| `world` | `{hash, data}` | World data (if viewer cache miss) |

### Viewer Behavior

1. On `replay_start` → check if world cached locally, request if not, prepare scene
2. On `replay_chunk` → append frames, start/continue playback
3. On `replay_end` → mark complete, enable full scrubbing

## NavGraph Caching

Cache NavGraph by world hash. LRU eviction, max 8 entries.

```python
class NavGraphCache:
    async def get_or_build(self, world_hash: str, world_data: dict) -> NavGraph:
        # Check cache first
        if world_hash in self._cache:
            return self._cache[world_hash]

        # Build outside lock (expensive)
        graph = await asyncio.to_thread(NavGraph.build, world, clearance_z)

        # Cache with LRU eviction
        self._cache[world_hash] = graph
        return graph
```

## Simplified Viewer UI

**Remove:**
- Local file upload
- Server file list/dropdown
- "Load from JSON" button
- Channel selector

**Keep:**
- SSE connection + status indicator
- 3D viewport
- Playback controls (play/pause, step, speed, scrub)
- Nav graph + paths toggles
- Mech inspection on click

**Add:**
- Loading progress bar (chunk N/M)
- Connection state indicator

**Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│  ECHELON LIVE                              ● Connected (3 fps)  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      [ 3D VIEWPORT ]                            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ◀◀  ▶/❚❚  ▶▶  │ ═══════════●════════ │ 01:23 / 02:45  │ 1x  │
├─────────────────────────────────────────────────────────────────┤
│  [Nav Graph]  [Paths]  │  Loading: ████████░░ 80% (chunk 4/5)   │
└─────────────────────────────────────────────────────────────────┘
```

## Reliability

### Server-side

- Hard client limit: 16 (LAN use case)
- Broadcast timeout: 1s per client, disconnect on failure
- Shutdown: Cancel all tasks + 2s grace period
- Memory: Only chunks in buffer, not full replays

### Client-side

- Auto-reconnect with exponential backoff (1s → 10s max)
- Reset backoff on successful connection
- Graceful degradation: missing chunks = skip in playback, not crash
- Visible error states (connection lost, chunk gaps)

## Training Script Changes

`scripts/train_ppo.py` - replace `push_replay()`:

```python
def push_replay(url: str, replay: dict, chunk_size: int = 100) -> None:
    """Push replay in chunks to server."""
    base_url = url.rsplit("/push", 1)[0]

    # Push world if not cached
    world = replay["world"]
    world_hash = compute_world_hash(world)
    if not check_world_cached(base_url, world_hash):
        put_world(base_url, world_hash, world)

    # Generate replay ID and push chunks
    replay_id = uuid.uuid4().hex[:12]
    frames = replay["frames"]
    chunk_count = (len(frames) + chunk_size - 1) // chunk_size

    for i in range(chunk_count):
        chunk = {
            "replay_id": replay_id,
            "world_ref": world_hash,
            "chunk_index": i,
            "chunk_count": chunk_count,
            "frames": frames[i * chunk_size : (i + 1) * chunk_size],
        }
        if i == 0:
            chunk["meta"] = replay.get("meta", {})

        requests.post(url, json=chunk, timeout=10)
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | / | Serve viewer.html |
| GET | /events | SSE stream |
| GET | /health | Health check |
| POST | /push | Receive replay chunk, broadcast |
| HEAD | /worlds/{hash} | Check world cached |
| GET | /worlds/{hash} | Get cached world |
| PUT | /worlds/{hash} | Store world |
| POST | /nav/graph | Get/build NavGraph (cached) |
| POST | /nav/path | A* pathfinding |

## Implementation Order

1. Server core (`echelon/server/`) - chunked protocol, NavGraph cache
2. Training script - chunked `push_replay()`
3. Viewer - stripped UI, chunked assembly
4. Delete old `server.py`

## Success Criteria

- No browser tab crashes during 1-hour training session
- No server memory growth beyond chunk buffer size
- NavGraph toggle responds in <100ms (cached)
- Visible connection/loading state at all times
