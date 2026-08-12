# LightMem-Ego Text-Only Benchmark Ablation

This document describes the single-text-modality benchmark implementation added on top of the official LightMem-Ego backend. The goal is to expose a simple `/add` and `/search` memory API while preserving the LightMem-Ego memory method as much as possible without images, video, audio, ASR, or VLM processing.

## What This Is

`lightmem_text_memory_api_server.py` is a method-level text-only ablation of LightMem-Ego:

- It accepts benchmark text records through `/add`.
- It converts each text record into a timestamped short-term micro-event (`M_st`).
- It consolidates those micro-events into long-term EM2Mem-style memory (`M_lt`).
- It searches through the LightMem-Ego router, retrieval planner, short-term retriever, long-term text retriever, and fusion modules.
- It returns benchmark-friendly retrieval hits with `id`, `content`, `score`, and `created_at`.

It is not the full wearable multimodal LightMem-Ego pipeline. The full pipeline still lives in `api_server.py` and the online worker stack.

## What Was Changed From Official LightMem-Ego

### Added Files

- `lightmem_text_memory_api_server.py`: standalone HTTP server for text-only `/add`, `/search`, and `/health`.
- `text_embedding_server.py`: OpenAI-compatible local SentenceTransformers embedding service used by the text API.
- `scripts/start_lightmem_text_memory_api.sh`: starts the text-only memory API; default port is `8767`.
- `scripts/stop_lightmem_text_memory_api.sh`: stops the text-only memory API through its pidfile.
- `scripts/start_text_embedding_server.sh`: starts the local embedding service; default port is `8010`.
- `scripts/stop_text_embedding_server.sh`: stops the local embedding service through its pidfile.
- `scripts/smoke_lightmem_text_api.sh`: minimal end-to-end add/search smoke test.
- `docs/lightmem_text_ablation.md`: this implementation note.

### Reused Official Modules

The text-only server imports and reuses these backend modules instead of replacing the whole method with plain vector search:

- `online_short_term.MSTStore` and `online_short_term.MSTRetriever` for short-term memory storage and retrieval.
- `online_memory.evidence_to_em2mem` helpers for EM2Mem-style caption, sidecar, and semantic-memory files.
- `online_memory.em2mem_layout.Em2MemOnlineLayout` for the long-term memory directory layout.
- `online_query.MemoryRouter`, `online_query.RetrievalPlanner`, and `online_query.MemoryFusion` for query routing, planning, and evidence fusion.

### Disabled Official Multimodal Functions

The benchmark ablation disables these parts because the input is text only:

- `M_cur` current visual memory.
- Browser/glasses frame streaming.
- Video chunk preprocessing.
- Audio capture and ASR.
- VLM captioning/refinement from keyframes.
- Visual embeddings and image evidence.
- Full multimodal QA generation.

### Adapted Text Construction

The official multimodal pipeline normally treats text as timestamped transcript/ASR evidence aligned to visual micro-events. In this ablation, each `/add` message is directly converted into a pseudo micro-event:

```text
/add message -> raw text record -> text-only M_st micro-event -> EM2Mem-style M_lt artifacts
```

The original text is stored unchanged as `raw_content`. The API also creates a display string and a searchable string containing `user_id`, `session_id`, `role`, `time`, and the raw content.

## API Contract

### Health

```bash
curl http://127.0.0.1:8767/health
```

Important fields:

- `ok`: server is responding.
- `backend`: `lightmem-ego-text`.
- `method_level`: `lightmem_ego_text_ablation`.
- `text_only`: `true`.
- `cuda_visible_devices`: GPU binding inherited from `CUDA_VISIBLE_DEVICES`.
- `embedding`: configured embedding backend.
- `lightmem_method`: status of the LightMem-Ego text ablation runtime.

### Add

```http
POST /add
Content-Type: application/json
```

Request:

```json
{
  "request_id": "req-1",
  "user_id": "user-1",
  "session_id": "session-1",
  "messages": [
    {
      "role": "user",
      "timestamp": 1704067200000,
      "content": "raw memory text"
    }
  ]
}
```

Response:

```json
{
  "success": true,
  "request_id": "req-1",
  "user_id": "user-1",
  "session_id": "session-1"
}
```

Notes:

- `role` must be one of `user`, `assistant`, `system`, or `tool`.
- `timestamp` accepts Unix milliseconds; Unix seconds are also tolerated.
- The content is written as raw text, not rewritten by an LLM.
- By default `LIGHTMEM_TEXT_BUILD_ON_ADD=0`, so expensive LightMem artifacts are built lazily during `/search`.

### Search

```http
POST /search
Content-Type: application/json
```

Request:

```json
{
  "query": "question",
  "options": ["A. first option", "B. second option"],
  "user_id": "user-1",
  "session_id": "session-1",
  "top_k": 10
}
```

Response:

```json
{
  "data": [
    {
      "id": "memory-or-evidence-id",
      "content": "retrieved memory text",
      "score": 0.87,
      "created_at": "2026-08-12T00:00:00.000Z"
    }
  ]
}
```

`session_id` is optional for backward compatibility. If it is provided, only memories previously added with the same `user_id + session_id` are searched and the LightMem artifacts are built under a session-level scope. If it is omitted, all memories under `user_id` are searched.

By default, `options` are appended to the retrieval query to match the benchmark interface used during local evaluation. Set `LIGHTMEM_TEXT_INCLUDE_OPTIONS_IN_QUERY=0` for an ablation that retrieves using only `query`.

## Memory Construction

### Short-Term Memory (`M_st`)

Each message becomes one pseudo micro-event. The implementation preserves:

- role, timestamp, and raw text;
- `caption = "{role}: {raw_content}"`;
- `transcript = raw_content`;
- transcript segments with role/time/text;
- rule-extracted entities and conversation focus;
- active and archive stores through `MSTStore.save_events()` and `save_archive_events()`.

For benchmark use, the recent window is set very large (`315360000` seconds, about ten years) and max events are set to `1000000`, so old benchmark records do not fall out of short-term memory during one run.

### Long-Term Memory (`M_lt`)

Long-term memory is built from the short-term micro-events, not directly from the SQLite rows:

```text
M_st events -> 30s episodes -> evidence docs -> 30s captions -> 3min/10min/1h captions -> sidecars -> semantic facts
```

The output layout mirrors EM2Mem online memory:

```text
runtime/text_memory_api/lightmem_sessions/<scope_id>/
  short_term/
  evidence/
  captions/
  em2mem/
    caption_root/
    sidecar_root/
    semantic_root/
    memory_config.json
```

Semantic extraction and triplet construction use a deterministic rule backend in this ablation, not an LLM/OpenIE backend.

## Retrieval Logic

`/search` executes the following path:

1. Validate `query`, `user_id`, optional `session_id`, and `top_k`.
2. Build `query_text`; optionally append `options`.
3. Fetch records by `user_id + session_id` when `session_id` is provided, otherwise by `user_id`.
4. Ensure text-only `M_st` and `M_lt` artifacts exist for that scope.
5. Route the query through `MemoryRouter`, with current/visual memory forced off.
6. Plan retrieval through `RetrievalPlanner` with `retrieval_mode=text_only` and `use_image_evidence=false`.
7. Retrieve short-term results with `MSTRetriever`.
8. Retrieve long-term text candidates from captions and semantic memory, using lexical/graph/source/recency prefiltering plus dense embedding scoring.
9. Fuse evidence with `MemoryFusion`.
10. Return top-k benchmark records.

If method retrieval fails unexpectedly, the service falls back to raw text vector search over SQLite rows so the API remains usable. The last fallback reason is exposed in `/health` under `lightmem_method.last_error`.

## Deployment On GPU 4

The default scripts bind both the text embedding service and the text API to GPU 4:

```bash
cd src/backend
scripts/start_text_embedding_server.sh
scripts/start_lightmem_text_memory_api.sh
```

Equivalent explicit environment:

```bash
LIGHTMEM_TEXT_EMBED_CUDA_VISIBLE_DEVICES=4 \
LIGHTMEM_TEXT_CUDA_VISIBLE_DEVICES=4 \
scripts/start_text_embedding_server.sh

LIGHTMEM_TEXT_CUDA_VISIBLE_DEVICES=4 \
LIGHTMEM_TEXT_EMBEDDING_BASE_URL=http://127.0.0.1:8010/v1 \
scripts/start_lightmem_text_memory_api.sh
```

Default ports:

- Text memory API: `8767`.
- Local embedding service: `8010`.

## Smoke Test

```bash
cd src/backend
scripts/smoke_lightmem_text_api.sh
```

Or manually:

```bash
curl -sS http://127.0.0.1:8767/health | python -m json.tool
curl -sS -X POST http://127.0.0.1:8767/add \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-1","user_id":"demo-user","session_id":"demo-session","messages":[{"role":"user","timestamp":1704067200000,"content":"Alice left the badge on the kitchen counter."}]}'
curl -sS -X POST http://127.0.0.1:8767/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Where is the badge?","user_id":"demo-user","session_id":"demo-session","top_k":5}' | python -m json.tool
```

## Fair Benchmark Recommendations

For fair single-text-modality experiments, report these settings explicitly:

- whether `/search` includes `session_id`;
- whether `LIGHTMEM_TEXT_INCLUDE_OPTIONS_IN_QUERY` is enabled;
- whether tool/thinking/final-summary messages are included in `/add`;
- the embedding model and device;
- whether artifacts are built lazily on search or eagerly on add.

Recommended ablations:

- user-only retrieval versus user+session retrieval;
- query-only retrieval versus query+options retrieval;
- raw vector fallback disabled versus full LightMem text ablation;
- short-term only, long-term only, and fused `M_st + M_lt` retrieval.
