# LightMem-Ego Text-Only Add/Search Service

This repository is a benchmark-serving adaptation of the official **LightMem-Ego: Your AI Memory for Everyday Life** codebase.

It keeps the LightMem-Ego hierarchical memory method, then adds a standalone HTTP service for benchmark systems that call `/add` and `/search`. The deployed evaluation path is text-only: text messages are adapted into short-term micro-events (`M_st`), consolidated into multi-scale long-term memory (`M_lt`), retrieved through the LightMem-Ego routing/planning/fusion stack, and returned as ranked evidence.

This is a **LightMem-Ego text-only method ablation**, not the full wearable multimodal system and not a plain vector-database wrapper.

## Version Disclosure

- Upstream project: [`zjunlp/LightMem-Ego`](https://github.com/zjunlp/LightMem-Ego)
- Paper: [`arXiv:2607.11487`](https://arxiv.org/abs/2607.11487)
- Upstream comparison commit: [`b6aa0f719d95acfc3d92ebe338bf31fcf97bfd10`](https://github.com/zjunlp/LightMem-Ego/commit/b6aa0f719d95acfc3d92ebe338bf31fcf97bfd10)
- Upstream comparison commit date: `2026-07-28T02:17:14Z`
- Backend project version: `0.1.0` from `src/backend/pyproject.toml`
- HTTP memory API version string: `LightMemEgoTextAblationAPI/2.0`
- Text embedding API version string: `LightMemTextEmbedding/1.0`
- This repository: local adaptation for single-text-modality memory benchmark serving

The server snapshot did not include the upstream `.git` history, so the exact historical fork commit cannot be recovered. The commit above is the official revision used to recheck this implementation and is a comparison reference, not a claimed fork base.

## What Changed From Official LightMem-Ego

The changes are focused on exposing the LightMem-Ego memory method through a reproducible text-only benchmark API.

### 1. Text-only input adapter

Official LightMem-Ego receives first-person video and audio, runs media preprocessing, ASR, keyframe sampling, and VLM captioning, then aligns those signals into memory events.

`src/backend/lightmem_text_memory_api_server.py` adds a direct text path:

```text
/add message
  -> raw text record
  -> text-only M_st micro-event
  -> EM2Mem-style M_lt artifacts
```

Each submitted message becomes one timestamped pseudo micro-event:

- `content` is preserved unchanged as `raw_content` and transcript text.
- `"{role}: {content}"` becomes the event caption.
- `role`, timestamp, request, user, and session metadata are preserved.
- keyframes, visual objects, image paths, video paths, and audio fields remain empty.
- no LLM rewrites the submitted text during `/add`.

### 2. Short-term memory (`M_st`)

The adapter reuses the existing LightMem-Ego short-term modules:

- `online_short_term.MSTStore` for active and archived micro-event storage.
- `online_short_term.MSTRetriever` for short-term retrieval.
- `online_short_term.schemas.build_retrieval_text` for searchable event text.

For benchmark runs, the active window is enlarged to about ten years and the active/archive limits are set to `1,000,000` events so old benchmark messages are not evicted during one evaluation.

### 3. Long-term memory (`M_lt`)

Long-term memory is built from text-only `M_st`, not directly from raw SQLite rows:

```text
M_st micro-events
  -> base episodes and evidence documents
  -> 30s captions
  -> 3min / 10min / 1h captions
  -> episodic sidecars
  -> semantic facts
```

The adapter reuses:

- `online_memory.em2mem_layout.Em2MemOnlineLayout` for the EM2Mem directory layout.
- `online_memory.evidence_to_em2mem` writers for multi-scale captions, episodic sidecars, and semantic-memory files.

One text message produces one base episode in the 30-second artifact format. Multi-scale aggregation and semantic triplets use the deterministic `rule` backend rather than VLM, LLM, or OpenIE generation. The visual-evidence file is intentionally empty.

### 4. Text-only routing, retrieval, and fusion

Search still calls the LightMem-Ego method components:

- `online_query.MemoryRouter`
- `online_query.RetrievalPlanner`
- `online_short_term.MSTRetriever`
- `online_query.MemoryFusion`

The adapter then constrains the route and plan to text:

- `M_cur` is disabled.
- `retrieval_mode` is forced to `text_only`.
- image evidence and visual embeddings are disabled.
- available `M_st` and `M_lt` memories remain enabled.

Long-term text retrieval is adapted for benchmark serving. It loads multi-scale captions and semantic facts, applies a bounded lexical/graph/source/recency prefilter, embeds the reduced candidate set, and scores candidates as:

```text
0.46 * dense_embedding
+ 0.28 * lexical_overlap
+ 0.12 * source_scale_bonus
+ 0.08 * semantic_triplet_overlap
+ 0.06 * relative_recency
```

`MemoryFusion` combines the native `M_st` results with the adapted text-only `M_lt` results. `/search` returns the fused evidence instead of running the official multimodal final-answer generator.

### 5. Benchmark `/add` and `/search` HTTP API

`src/backend/lightmem_text_memory_api_server.py` is a new standalone service exposing:

- `GET /health` and `GET /v1/health`
- `POST /add` and `POST /v1/add`
- `POST /search` and `POST /v1/search`

Raw records and their embeddings are stored in SQLite. A stable memory ID makes repeated submission of the same `request_id + message_index` idempotent. `request_id` should therefore be globally unique within one deployment, not only unique within one user.

### 6. User and session isolation

Every `/add` record stores both `user_id` and `session_id`.

- If `/search` includes `session_id`, it searches only records with the same `user_id + session_id` and builds memory artifacts under that session-level scope.
- If `/search` omits `session_id`, it searches all records belonging to `user_id` for backward compatibility.

Scope directories use stable hashes, so raw user/session identifiers are not used as filesystem paths.

### 7. Lazy memory construction and retrieval fallback

By default, `/add` stores text and embeddings immediately while LightMem artifacts are built lazily during `/search`. Set `LIGHTMEM_TEXT_BUILD_ON_ADD=1` to move that construction cost to `/add`.

Artifact construction is a full rebuild of the selected scope rather than the official streaming worker's incremental media-update path. User-scoped artifacts are reused while their source signature is unchanged. The current session-scoped cache limitation is documented under deployment caveats below.

If method-level retrieval raises an exception or returns no evidence, the service falls back to raw SQLite text retrieval using dense and lexical similarity. Exceptions are exposed as `lightmem_method.last_error` in `/health`; an empty method result can fall back without setting that field.

### 8. Local OpenAI-compatible embedding service

`src/backend/text_embedding_server.py` is new. It serves a local SentenceTransformers model through:

- `GET /health`
- `GET /v1/models`
- `POST /v1/embeddings`

The tested setup uses `all-MiniLM-L6-v2`, 384-dimensional normalized embeddings, batch size `128`, and physical GPU 4.

### 9. Concurrency and service helpers

The HTTP server uses daemon request threads and a backlog of `256`. Search concurrency is bounded by a global semaphore; the default allows two concurrent searches and waits up to ten seconds for a slot.

The following helper scripts were added:

- `scripts/start_lightmem_text_memory_api.sh`
- `scripts/stop_lightmem_text_memory_api.sh`
- `scripts/start_text_embedding_server.sh`
- `scripts/stop_text_embedding_server.sh`
- `scripts/smoke_lightmem_text_api.sh`

Start/stop scripts use pid files, validate process command lines, and run services in separate process groups.

## Method Boundary

| Stage | Official multimodal LightMem-Ego | This text-only service |
| :--- | :--- | :--- |
| Input | Frames, video, audio, ASR, and VLM captions | Timestamped text messages |
| `M_cur` | Live current visual memory | Disabled |
| `M_st` construction | Stream boundaries, aligned transcripts, keyframes, actions, and visual changes | One pseudo micro-event per message |
| `M_st` storage/retrieval | `MSTStore` and `MSTRetriever` | Same modules, text events only |
| `M_lt` structure | EM2Mem episodic, semantic, and visual memory | EM2Mem multi-scale text captions, sidecars, and rule semantic facts |
| Router/planner | Chooses current, short-term, long-term, text, and visual evidence | Same router/planner constrained to text-only `M_st + M_lt` |
| Long-term retrieval | Full text/semantic/visual retrieval path | Adapter-specific text candidate ranking |
| Fusion | `MemoryFusion` | Same fusion module over text evidence |
| Output | Memory-grounded answer plus multimodal evidence | Ranked evidence records only |

The retained method core is hierarchical `M_st + M_lt` memory, multi-scale long-term artifacts, query-dependent routing/planning, separate short-/long-term retrieval, and evidence fusion. Results should be reported as **LightMem-Ego text-only ablation**, not as the full LightMem-Ego multimodal system.

## Repository Contents

- `src/backend/lightmem_text_memory_api_server.py`: text-only LightMem adapter and benchmark HTTP API.
- `src/backend/text_embedding_server.py`: local OpenAI-compatible text embedding service.
- `src/backend/scripts/`: start, stop, smoke-test, and original worker scripts.
- `src/backend/docs/lightmem_text_ablation.md`: detailed text-ablation implementation notes.
- `src/backend/online_short_term/`: LightMem-Ego short-term memory modules reused by the adapter.
- `src/backend/online_memory/`: EM2Mem layout and artifact writers reused by the adapter.
- `src/backend/online_query/`: router, planner, and fusion modules reused by the adapter.
- `src/backend/api_server.py`: full multimodal LightMem-Ego API, not required by the text service.
- `src/frontend/` and `src/ai_glass_app/`: original web and glasses clients, not required by the text service.

Private runtime files are intentionally excluded: `.env`, API keys, logs, pid files, SQLite databases, generated memory artifacts, model weights, caches, uploads, and benchmark data.

## Installation

The backend requires Python 3.10 or newer.

```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install a CUDA-compatible PyTorch build for your server if it is not already present. Place the embedding model at `src/backend/models/all-MiniLM-L6-v2`, or set `LIGHTMEM_TEXT_EMBEDDING_MODEL_PATH` to another local path or SentenceTransformers model identifier.

Model weights are not included in this repository.

## Start The Local Text Embedding Server

The helper script uses physical GPU 4, local port `8010`, batch size `128`, and `models/all-MiniLM-L6-v2` by default:

```bash
cd src/backend
LIGHTMEM_TEXT_EMBED_CUDA_VISIBLE_DEVICES=4 \
scripts/start_text_embedding_server.sh
```

Check it:

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

Inside the embedding process, physical GPU 4 appears as `cuda:0` because the script sets `CUDA_VISIBLE_DEVICES=4`.

## Start The LightMem-Ego Add/Search Service

Recommended benchmark configuration:

```bash
cd src/backend
LIGHTMEM_TEXT_CUDA_VISIBLE_DEVICES=4 \
LIGHTMEM_TEXT_MEMORY_API_EMBEDDING_BACKEND=openai \
LIGHTMEM_TEXT_EMBEDDING_BASE_URL=http://127.0.0.1:8010/v1 \
scripts/start_lightmem_text_memory_api.sh
```

The API listens on `0.0.0.0:8767` by default. Check it locally:

```bash
curl http://127.0.0.1:8767/health
```

Stop both services:

```bash
scripts/stop_lightmem_text_memory_api.sh
scripts/stop_text_embedding_server.sh
```

The start scripts default to `LIGHTMEM_TEXT_MEMORY_API_EMBEDDING_BACKEND=auto`, which falls back to hash embeddings if the local embedding service is unavailable. Use `openai` as shown above for benchmark runs where silent embedding fallback is undesirable.

## API Contract

### Health

```bash
curl http://127.0.0.1:8767/health
```

Important fields:

- `ok`: HTTP service is responding.
- `backend`: `lightmem-ego-text`.
- `method_level`: `lightmem_ego_text_ablation`.
- `configured`: LightMem-Ego modules imported successfully.
- `embedding`: active embedding backend, endpoint, and model.
- `build_on_add`: whether LightMem artifacts are built during `/add`.
- `include_options_in_query`: whether multiple-choice options are appended during retrieval.
- `lightmem_method.last_error`: latest method/fallback error; expected value is `null`.

### Add

```http
POST /add
Content-Type: application/json
```

Example:

```bash
curl -X POST http://127.0.0.1:8767/add \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "demo-1",
    "user_id": "user-a",
    "session_id": "session-a",
    "messages": [
      {
        "role": "user",
        "content": "Alice left the badge on the kitchen counter.",
        "timestamp": 1780000000000
      }
    ]
  }'
```

Response:

```json
{
  "success": true,
  "request_id": "demo-1",
  "user_id": "user-a",
  "session_id": "session-a"
}
```

`role` must be `user`, `assistant`, `system`, or `tool`. `timestamp` accepts Unix milliseconds; Unix seconds are also tolerated.

### Search

```http
POST /search
Content-Type: application/json
```

Example:

```bash
curl -X POST http://127.0.0.1:8767/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "user-a",
    "session_id": "session-a",
    "query": "Where did Alice leave the badge?",
    "options": ["A. Kitchen counter", "B. Office desk"],
    "top_k": 5
  }'
```

Response shape:

```json
{
  "data": [
    {
      "id": "memory-or-evidence-id",
      "content": "retrieved memory evidence",
      "score": 0.87,
      "created_at": "2026-08-12T00:00:00.000Z"
    }
  ]
}
```

`session_id` is optional for backward compatibility. Include it when each benchmark stage or conversation must search only the memories added for that same stage/session.

By default, `options` are appended to the retrieval query. Set `LIGHTMEM_TEXT_INCLUDE_OPTIONS_IN_QUERY=0` for query-only retrieval.

## Smoke Test

With both services running:

```bash
cd src/backend
scripts/smoke_lightmem_text_api.sh
```

The script checks `/health`, adds a unique memory, searches the same `user_id + session_id`, and prints the retrieved evidence.

## Runtime Layout

Raw records and generated memory artifacts are stored under:

```text
src/backend/runtime/text_memory_api/
  lightmem_text_memory_api.db
  lightmem_sessions/<hashed_scope>/
    text_source_records.jsonl
    short_term/
    evidence/
    captions/
    em2mem/
      caption_root/
      sidecar_root/
      semantic_root/
      memory_config.json
```

This directory is ignored by Git and is not included in the repository.

## Deployment Notes From The Tested Server

The tested benchmark deployment used:

- memory API on port `8767`;
- embedding API on `127.0.0.1:8010/v1`;
- `all-MiniLM-L6-v2` embeddings on physical GPU 4;
- embedding batch size `128`;
- lazy memory construction with `LIGHTMEM_TEXT_BUILD_ON_ADD=0`;
- global search concurrency `2` and queue timeout `10` seconds;
- long-term candidate cache for `8` scopes;
- prefilter base/cap of `96` / `256` candidates.

Known operational caveats:

- the first user-scoped `/search` after records change builds `M_st` and `M_lt` artifacts and can be substantially slower than later searches;
- in the current revision, session-scoped state is written to the user-level state path, so searches that include `session_id` rebuild that session's artifacts on every request instead of reusing the signature cache;
- omitting `session_id` searches the entire user scope, which can make the initial or changed-scope build much larger;
- method exceptions can be hidden by the raw-vector fallback unless `/health` is monitored, and an empty-evidence fallback is not currently identified in the response;
- runtime storage grows with users, sessions, raw embeddings, and generated multi-scale artifacts.

For production serving, use systemd, supervisor, Docker, or another process manager instead of relying only on the helper scripts.

## Fair Benchmark Reporting

Report these settings with experimental results:

- whether `/search` includes `session_id`;
- whether retrieval uses `query` only or `query + options`;
- which message roles are included in `/add`;
- embedding model, backend, and device;
- lazy build on search versus eager build on add;
- whether `lightmem_method.last_error` remained `null` and whether empty-evidence fallback was ruled out;
- whether results came from full `M_st + M_lt` fusion or a fallback path.

Useful ablations include session-scoped versus user-scoped retrieval, query-only versus query-plus-options retrieval, `M_st` only versus `M_lt` only versus fused retrieval, and rule-based text memory versus the full multimodal system.

## Full Multimodal LightMem-Ego

The original wearable pipeline remains in this repository, including `src/backend/api_server.py`, the online workers, the browser frontend, and the Rokid glasses application. Those components are not started by the text-service scripts.

For the original architecture, multimodal setup, and citations, see the [official LightMem-Ego repository](https://github.com/zjunlp/LightMem-Ego) and [`src/backend/README.md`](src/backend/README.md). Detailed notes for this adaptation are in [`src/backend/docs/lightmem_text_ablation.md`](src/backend/docs/lightmem_text_ablation.md).

## License

This adaptation keeps the upstream MIT license. See [`LICENSE`](LICENSE).
