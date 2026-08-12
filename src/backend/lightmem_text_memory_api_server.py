#!/usr/bin/env python3
"""LightMem-Ego text-only benchmark memory API.

This adapter exposes the benchmark /add and /search contract while running a
paper-method text ablation of LightMem-Ego:

- M_cur/current visual memory is disabled.
- M_st is represented by text-only micro-events and queried with the native
  MSTStore/MSTRetriever path.
- M_lt is rebuilt as EM2Mem-style multi-scale captions, episodic sidecars, and
  semantic memory from text evidence only.
- Search uses the native MemoryRouter, RetrievalPlanner, and MemoryFusion with
  retrieval_mode=text_only and use_image_evidence=false.

No ASR, VLM, image, video, or audio evidence is used.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
DEFAULT_DATA_DIR = "runtime/text_memory_api"
DEFAULT_HASH_DIM = 4096
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8010/v1"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_METHOD_MODEL = "lightmem_text_rule"
TEXT_ONLY_RECENT_WINDOW_SECONDS = 315_360_000.0  # 10 years; keep benchmark data active.
TEXT_ONLY_MAX_EVENTS = 1_000_000


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def timestamp_to_datetime(timestamp: Optional[Any]) -> dt.datetime:
    if timestamp is None or timestamp == "":
        return dt.datetime.now(dt.timezone.utc)
    try:
        value = float(timestamp)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "messages[].timestamp must be a Unix timestamp in milliseconds") from exc
    seconds = value / 1000.0 if abs(value) > 100_000_000_000 else value
    return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)


def timestamp_to_iso(timestamp: Optional[Any]) -> str:
    return timestamp_to_datetime(timestamp).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso_datetime(value: Any) -> dt.datetime:
    text = str(value or "").strip()
    if not text:
        return dt.datetime.now(dt.timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return dt.datetime.now(dt.timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def stable_memory_id(user_id: str, session_id: str, request_id: str, message_index: int) -> str:
    raw = "\0".join(["lightmem-ego", user_id, session_id, request_id, str(message_index)])
    return "mem_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def stable_scope_id(user_id: str, session_id: Optional[str] = None) -> str:
    raw = user_id if not session_id else "\0".join([user_id, session_id])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"session_{digest}" if session_id else f"user_{digest}"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def make_sparse_vector(text: str, dim: int) -> dict[int, float]:
    tokens = tokenize(text)
    features: dict[int, float] = {}

    def add_feature(feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % dim
        features[bucket] = features.get(bucket, 0.0) + weight

    for token in tokens:
        add_feature("u:" + token, 1.0)
    for left, right in zip(tokens, tokens[1:]):
        add_feature("b:" + left + " " + right, 1.25)
    return features


def vector_norm(vector: Any) -> float:
    if isinstance(vector, dict):
        return math.sqrt(sum(float(v) * float(v) for v in vector.values()))
    return math.sqrt(sum(float(v) * float(v) for v in vector))


def cosine_similarity(query_vector: Any, doc_vector: Any, query_norm: float, doc_norm: float) -> float:
    if query_norm == 0.0 or doc_norm == 0.0:
        return 0.0
    if isinstance(query_vector, dict) and isinstance(doc_vector, dict):
        if len(query_vector) > len(doc_vector):
            query_vector, doc_vector = doc_vector, query_vector
        dot = sum(float(value) * float(doc_vector.get(key, 0.0)) for key, value in query_vector.items())
    else:
        if len(query_vector) != len(doc_vector):
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(query_vector, doc_vector))
    return dot / (query_norm * doc_norm)


def lexical_score(query: str, content: str) -> float:
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    c_tokens = tokenize(content)
    if not c_tokens:
        return 0.0
    q_counts: dict[str, int] = {}
    c_counts: dict[str, int] = {}
    for token in q_tokens:
        q_counts[token] = q_counts.get(token, 0) + 1
    for token in c_tokens:
        c_counts[token] = c_counts.get(token, 0) + 1
    overlap = sum(min(count, c_counts.get(token, 0)) for token, count in q_counts.items())
    return overlap / max(1, len(q_tokens))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clamp01(value: Any) -> float:
    return max(0.0, min(1.0, safe_float(value)))


def compact_text(value: Any, max_chars: int = 900) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (list, tuple, set)):
        text = "; ".join(compact_text(item, max_chars=0) for item in value if compact_text(item, max_chars=0))
    elif isinstance(value, dict):
        parts = []
        for key, item in value.items():
            child = compact_text(item, max_chars=0)
            if child:
                parts.append(f"{key}: {child}")
        text = "; ".join(parts)
    else:
        text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


class EmbeddingBackend:
    name = "base"

    def embed_many(self, texts: list[str]) -> list[Any]:
        raise NotImplementedError

    def serialize(self, vector: Any) -> str:
        if isinstance(vector, dict):
            return json.dumps({str(k): v for k, v in vector.items()}, separators=(",", ":"))
        return json.dumps(vector, separators=(",", ":"))

    def deserialize(self, raw: str) -> Any:
        value = json.loads(raw)
        if isinstance(value, dict):
            return {int(k): float(v) for k, v in value.items()}
        return value

    def health(self) -> dict[str, Any]:
        return {"backend": self.name}


class HashEmbeddingBackend(EmbeddingBackend):
    name = "hash"

    def __init__(self, dim: int = DEFAULT_HASH_DIM):
        self.dim = dim

    def embed_many(self, texts: list[str]) -> list[dict[int, float]]:
        return [make_sparse_vector(text, self.dim) for text in texts]

    def health(self) -> dict[str, Any]:
        return {"backend": self.name, "dim": self.dim}


class OpenAIEmbeddingBackend(EmbeddingBackend):
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"input": texts, "model": self.model, "encoding_format": "float"}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + "/embeddings",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"embedding request failed: {exc}") from exc
        data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        if len(data) != len(texts):
            raise RuntimeError("embedding response count does not match request count")
        return [item["embedding"] for item in data]

    def health(self) -> dict[str, Any]:
        return {"backend": self.name, "base_url": self.base_url, "model": self.model}


@dataclass
class MemoryRow:
    memory_id: str
    display_content: str
    searchable_content: str
    created_at: str
    vector_backend: str
    vector_json: str
    vector_norm: float


@dataclass
class MemoryRecord:
    memory_id: str
    request_id: str
    user_id: str
    session_id: str
    message_index: int
    role: str
    raw_content: str
    display_content: str
    searchable_content: str
    created_at: str


class MemoryDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    display_content TEXT NOT NULL,
                    searchable_content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    vector_backend TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    vector_norm REAL NOT NULL,
                    ingested_at TEXT NOT NULL,
                    UNIQUE(request_id, message_index)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")

    def upsert_rows(self, rows: list[dict[str, Any]]) -> None:
        with self.lock:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO memories (
                        id, request_id, user_id, session_id, message_index, role,
                        raw_content, display_content, searchable_content, created_at,
                        vector_backend, vector_json, vector_norm, ingested_at
                    ) VALUES (
                        :id, :request_id, :user_id, :session_id, :message_index, :role,
                        :raw_content, :display_content, :searchable_content, :created_at,
                        :vector_backend, :vector_json, :vector_norm, :ingested_at
                    )
                    ON CONFLICT(request_id, message_index) DO UPDATE SET
                        id = excluded.id,
                        user_id = excluded.user_id,
                        session_id = excluded.session_id,
                        role = excluded.role,
                        raw_content = excluded.raw_content,
                        display_content = excluded.display_content,
                        searchable_content = excluded.searchable_content,
                        created_at = excluded.created_at,
                        vector_backend = excluded.vector_backend,
                        vector_json = excluded.vector_json,
                        vector_norm = excluded.vector_norm,
                        ingested_at = excluded.ingested_at
                    """,
                    rows,
                )

    def fetch_by_user(self, user_id: str) -> list[MemoryRow]:
        return self._fetch_rows(user_id=user_id, session_id=None)

    def fetch_by_user_session(self, user_id: str, session_id: str) -> list[MemoryRow]:
        return self._fetch_rows(user_id=user_id, session_id=session_id)

    def _fetch_rows(self, user_id: str, session_id: str | None) -> list[MemoryRow]:
        where = "user_id = ?" if session_id is None else "user_id = ? AND session_id = ?"
        params: tuple[Any, ...] = (user_id,) if session_id is None else (user_id, session_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, display_content, searchable_content, created_at,
                       vector_backend, vector_json, vector_norm
                FROM memories
                WHERE {where}
                ORDER BY created_at ASC, request_id ASC, message_index ASC
                """,
                params,
            ).fetchall()
        return [
            MemoryRow(
                memory_id=str(row["id"]),
                display_content=str(row["display_content"]),
                searchable_content=str(row["searchable_content"]),
                created_at=str(row["created_at"]),
                vector_backend=str(row["vector_backend"]),
                vector_json=str(row["vector_json"]),
                vector_norm=float(row["vector_norm"]),
            )
            for row in rows
        ]

    def fetch_records_by_user(self, user_id: str) -> list[MemoryRecord]:
        return self._fetch_records(user_id=user_id, session_id=None)

    def fetch_records_by_user_session(self, user_id: str, session_id: str) -> list[MemoryRecord]:
        return self._fetch_records(user_id=user_id, session_id=session_id)

    def _fetch_records(self, user_id: str, session_id: str | None) -> list[MemoryRecord]:
        where = "user_id = ?" if session_id is None else "user_id = ? AND session_id = ?"
        params: tuple[Any, ...] = (user_id,) if session_id is None else (user_id, session_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, request_id, user_id, session_id, message_index, role,
                       raw_content, display_content, searchable_content, created_at
                FROM memories
                WHERE {where}
                ORDER BY created_at ASC, request_id ASC, message_index ASC
                """,
                params,
            ).fetchall()
        return [
            MemoryRecord(
                memory_id=str(row["id"]),
                request_id=str(row["request_id"]),
                user_id=str(row["user_id"]),
                session_id=str(row["session_id"]),
                message_index=int(row["message_index"]),
                role=str(row["role"]),
                raw_content=str(row["raw_content"]),
                display_content=str(row["display_content"]),
                searchable_content=str(row["searchable_content"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def user_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(DISTINCT user_id) FROM memories").fetchone()[0])


class LightMemTextAblationRuntime:
    """Build and query LightMem-Ego's text-only M_st/M_lt ablation artifacts."""

    def __init__(self, data_dir: Path, model_name: str = DEFAULT_METHOD_MODEL, generation_backend: str = "rule"):
        self.data_dir = data_dir
        self.sessions_root = data_dir / "lightmem_sessions"
        self.model_name = model_name
        self.generation_backend = generation_backend
        self.lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.lt_candidate_cache: OrderedDict[Any, list[dict[str, Any]]] = OrderedDict()
        self.lt_candidate_cache_users = max(0, env_int("LIGHTMEM_TEXT_LT_CACHE_USERS", 8))
        self.lt_prefilter_base = max(8, env_int("LIGHTMEM_TEXT_LT_PREFILTER_BASE", 96))
        self.lt_prefilter_cap = max(self.lt_prefilter_base, env_int("LIGHTMEM_TEXT_LT_PREFILTER_CAP", 256))
        self.last_error: str | None = None
        self.available = False
        self._import_lightmem_modules()

    def _import_lightmem_modules(self) -> None:
        try:
            from online_memory.em2mem_layout import Em2MemOnlineLayout, ensure_em2mem_layout
            from online_memory.evidence_to_em2mem import build_caption_items, write_caption_files, write_semantic_files, write_sidecar_files
            from online_preprocess.io_utils import read_json as lm_read_json
            from online_preprocess.io_utils import write_json as lm_write_json
            from online_preprocess.io_utils import write_json_atomic as lm_write_json_atomic
            from online_query.memory_fusion import MemoryFusion
            from online_query.memory_plan import RetrievalPlanner
            from online_query.memory_router import MemoryRouter
            from online_short_term.mst_retriever import MSTRetriever
            from online_short_term.mst_store import MSTStore
            from online_short_term.schemas import build_retrieval_text

            self.Em2MemOnlineLayout = Em2MemOnlineLayout
            self.ensure_em2mem_layout = ensure_em2mem_layout
            self.build_caption_items = build_caption_items
            self.write_caption_files = write_caption_files
            self.write_semantic_files = write_semantic_files
            self.write_sidecar_files = write_sidecar_files
            self.lm_read_json = lm_read_json
            self.lm_write_json = lm_write_json
            self.lm_write_json_atomic = lm_write_json_atomic
            self.MemoryFusion = MemoryFusion
            self.RetrievalPlanner = RetrievalPlanner
            self.MemoryRouter = MemoryRouter
            self.MSTRetriever = MSTRetriever
            self.MSTStore = MSTStore
            self.build_retrieval_text = build_retrieval_text
            self.available = True
        except Exception as exc:
            self.last_error = f"LightMem module import failed: {exc}"
            self.available = False

    def health(self) -> dict[str, Any]:
        scan_limit = max(0, env_int("LIGHTMEM_TEXT_HEALTH_SCAN_LIMIT", 64))
        scopes = 0
        ready_scopes = 0
        sampled = 0
        if self.sessions_root.exists():
            for path in self.sessions_root.iterdir():
                if not path.is_dir():
                    continue
                scopes += 1
                if sampled >= scan_limit:
                    continue
                sampled += 1
                config = read_json(path / "em2mem" / "memory_config.json", default={})
                if isinstance(config, dict) and config.get("memory_build_state") == "ready":
                    ready_scopes += 1
        return {
            "method_level": "lightmem_ego_text_ablation",
            "text_only": True,
            "module_available": self.available,
            "last_error": self.last_error,
            "sessions_root": str(self.sessions_root),
            "model_name": self.model_name,
            "generation_backend": self.generation_backend,
            "scopes": scopes,
            "ready_scopes_sampled": ready_scopes,
            "health_scan_limit": scan_limit,
            "lt_candidate_cache_users": len(self.lt_candidate_cache),
            "lt_prefilter_base": self.lt_prefilter_base,
            "lt_prefilter_cap": self.lt_prefilter_cap,
            "components": {
                "M_cur": "disabled",
                "M_st": "online_short_term.MSTStore + MSTRetriever over text micro-events",
                "M_lt": "EM2Mem multi-scale text captions + episodic sidecars + rule semantic memory",
                "router": "online_query.MemoryRouter",
                "planner": "online_query.RetrievalPlanner forced to retrieval_mode=text_only",
                "fusion": "online_query.MemoryFusion",
                "visual_audio": "disabled",
            },
        }

    def scope_id(self, user_id: str, session_id: str | None = None) -> str:
        return stable_scope_id(user_id, session_id=session_id)

    def scope_dir(self, user_id: str, session_id: str | None = None) -> Path:
        return self.sessions_root / self.scope_id(user_id, session_id=session_id)

    def state_path(self, user_id: str, session_id: str | None = None) -> Path:
        return self.scope_dir(user_id, session_id=session_id) / "text_ablation_state.json"

    def source_signature(self, records: list[MemoryRecord]) -> str:
        h = hashlib.sha1()
        for row in records:
            h.update(row.memory_id.encode("utf-8"))
            h.update(b"\0")
            h.update(row.created_at.encode("utf-8"))
            h.update(b"\0")
            h.update(row.session_id.encode("utf-8"))
            h.update(b"\0")
            h.update(row.role.encode("utf-8"))
            h.update(b"\0")
            h.update(row.raw_content.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def ensure_user_memory(self, user_id: str, records: list[MemoryRecord], session_id: str | None = None) -> dict[str, Any]:
        if not records:
            return {"ready": False, "reason": "empty_user_memory", "scope_id": self.scope_id(user_id, session_id=session_id)}
        if not self.available:
            raise RuntimeError(self.last_error or "LightMem text ablation modules are unavailable")

        session_dir = self.scope_dir(user_id, session_id=session_id)
        signature = self.source_signature(records)
        state_path = self.state_path(user_id, session_id=session_id)
        old_state = read_json(state_path, default={})
        config_path = session_dir / "em2mem" / "memory_config.json"
        if isinstance(old_state, dict) and old_state.get("source_signature") == signature and config_path.exists():
            ready = dict(old_state)
            ready["rebuilt"] = False
            return ready

        with self.lock:
            old_state = read_json(state_path, default={})
            if isinstance(old_state, dict) and old_state.get("source_signature") == signature and config_path.exists():
                ready = dict(old_state)
                ready["rebuilt"] = False
                return ready
            return self._rebuild_user_memory(user_id=user_id, records=records, signature=signature, session_id=session_id)

    def runtime_state(self, user_id: str, records: list[MemoryRecord], session_id: str | None = None) -> dict[str, Any]:
        session_dir = self.scope_dir(user_id, session_id=session_id)
        mst_state = read_json(session_dir / "short_term" / "mst_state.json", default={})
        config = read_json(session_dir / "em2mem" / "memory_config.json", default={})
        active_span = mst_state.get("active_time_span") if isinstance(mst_state, dict) else None
        archive_span = mst_state.get("archive_time_span") if isinstance(mst_state, dict) else None
        return {
            "current_ready": False,
            "current_stale": True,
            "short_term_ready": bool(records) and bool(mst_state.get("short_term_ready") if isinstance(mst_state, dict) else True),
            "long_term_ready": bool(config.get("memory_build_state") == "ready") if isinstance(config, dict) else False,
            "visual_embedding_ready": False,
            "mcur_time_span": None,
            "mst_time_span": active_span or archive_span or [0.0, 0.0],
            "mlt_time_span": config.get("time_span", [0.0, 0.0]) if isinstance(config, dict) else [0.0, 0.0],
            "memory_count": len(records),
        }

    def _store(self, session_dir: Path) -> Any:
        return self.MSTStore(
            session_dir,
            recent_window_seconds=TEXT_ONLY_RECENT_WINDOW_SECONDS,
            max_events=TEXT_ONLY_MAX_EVENTS,
            archive_max_events=TEXT_ONLY_MAX_EVENTS,
        )

    def _rebuild_user_memory(self, user_id: str, records: list[MemoryRecord], signature: str, session_id: str | None = None) -> dict[str, Any]:
        scope_id = self.scope_id(user_id, session_id=session_id)
        session_dir = self.scope_dir(user_id, session_id=session_id)
        layout = self.Em2MemOnlineLayout(session_dir=session_dir, session_id=scope_id)
        self.ensure_em2mem_layout(layout)
        (session_dir / "evidence").mkdir(parents=True, exist_ok=True)
        (session_dir / "captions").mkdir(parents=True, exist_ok=True)
        (session_dir / "em2mem" / "mst_episodic").mkdir(parents=True, exist_ok=True)

        events = self._records_to_mst_events(scope_id, records)
        store = self._store(session_dir)
        store.save_events(events)
        store.save_archive_events(events, bump_version=True)

        episodes = self._events_to_episodes(scope_id, events)
        evidence_docs = self._episodes_to_evidence(scope_id, episodes)
        caption_30s = self.build_caption_items(scope_id, evidence_docs)

        self._write_source_records(session_dir, records)
        self.lm_write_json(session_dir / "evidence" / "mst_session_evidence.json", evidence_docs)
        self.lm_write_json(session_dir / "captions" / "mst_session_30sec_captioned.json", caption_30s)
        self._write_mst_episode_outputs(session_dir, scope_id, episodes, caption_30s, evidence_docs)

        caption_paths = self.write_caption_files(
            layout=layout,
            caption_30s=caption_30s,
            model_name=self.model_name,
            generation_backend=self.generation_backend,
        )
        # Keep the legacy visual-evidence slot empty in this ablation; all
        # retrieval comes from text captions, episodic sidecars, and semantics.
        self.lm_write_json(layout.visual_evidence_path, [])
        caption_by_scale = {
            "30sec": caption_30s,
            "3min": self.lm_read_json(layout.caption_3min_path, default=[]) or [],
            "10min": self.lm_read_json(layout.caption_10min_path, default=[]) or [],
            "1h": self.lm_read_json(layout.caption_1h_path, default=[]) or [],
        }
        sidecar_paths = self.write_sidecar_files(
            layout=layout,
            model_name=self.model_name,
            caption_by_scale=caption_by_scale,
            generation_backend=self.generation_backend,
        )
        semantic_candidate_path, semantic_memory_path, semantic_fact_count = self.write_semantic_files(
            layout=layout,
            model_name=self.model_name,
            caption_30s=caption_30s,
            generation_backend=self.generation_backend,
        )

        time_span = self._time_span(events)
        now = utc_now_iso()
        config = {
            "session_id": scope_id,
            "user_id_hash": scope_id,
            "method_level": "lightmem_ego_text_ablation",
            "text_only": True,
            "modalities": {"text": True, "image": False, "video": False, "audio": False, "asr": False, "vlm": False},
            "memory_build_state": "ready",
            "pipeline_mode": "mst_text_only",
            "requested_30s_source": "mst_episodic",
            "active_30s_source": "mst_session_30sec_captioned",
            "episodic_source": "mst_micro_events_text_only",
            "em2mem_update_mode": "full_rebuild_text_ablation",
            "memory_generation_backend": self.generation_backend,
            "multiscale_generation_backend": self.generation_backend,
            "episodic_triplet_generation_backend": "rule",
            "semantic_generation_backend": "rule",
            "visual_embedding_ready": False,
            "visual_embedding_skipped": True,
            "use_image_evidence": False,
            "retrieval_mode": "text_only",
            "created_at": now,
            "updated_at": now,
            "time_span": time_span,
            "counts": {
                "raw_messages": len(records),
                "mst_micro_events": len(events),
                "mst_episodes": len(episodes),
                "caption_30sec": len(caption_30s),
                "caption_multiscale_total": sum(len(v) for v in caption_by_scale.values()),
                "semantic_facts": semantic_fact_count,
            },
            "source": {
                "adapter": "lightmem_text_memory_api_server",
                "api": "benchmark_add_search",
                "scope_type": "session" if session_id else "user",
                "user_id_hash": stable_scope_id(user_id),
                "session_id_hash": stable_scope_id(user_id, session_id=session_id) if session_id else None,
            },
            "mst_episodic_ready": True,
            "mst_episodic_path": "em2mem/mst_episodic/mst_30sec_episodes.json",
            "mst_captioned_30sec_path": "captions/mst_session_30sec_captioned.json",
            "mst_evidence_path": "evidence/mst_session_evidence.json",
            "query_rag_args": {
                "subject": scope_id,
                "retriever_model": self.model_name,
                "respond_model": self.model_name,
                "until_date": "DAY1",
                "until_time": self._max_end_hhmmssff(caption_30s),
                "episodic_caption_root": "em2mem/caption_root",
                "episodic_sidecar_root": "em2mem/sidecar_root",
                "semantic_root": "em2mem/semantic_root",
                "visual_root": None,
                "visual_evidence_file": None,
            },
            "em2mem_files": {
                "caption_30sec": self._rel(caption_paths["30sec"], session_dir),
                "caption_3min": self._rel(caption_paths["3min"], session_dir),
                "caption_10min": self._rel(caption_paths["10min"], session_dir),
                "caption_1h": self._rel(caption_paths["1h"], session_dir),
                "semantic_candidates": self._rel(semantic_candidate_path, session_dir),
                "semantic_memory": self._rel(semantic_memory_path, session_dir),
                "sidecars": {
                    scale: {name: self._rel(path, session_dir) for name, path in paths.items()}
                    for scale, paths in sidecar_paths.items()
                },
            },
        }
        self.lm_write_json_atomic(layout.memory_config_path, config)
        state = {
            "ready": True,
            "rebuilt": True,
            "scope_id": scope_id,
            "session_dir": str(session_dir),
            "source_signature": signature,
            "record_count": len(records),
            "event_count": len(events),
            "semantic_fact_count": semantic_fact_count,
            "memory_config": str(layout.memory_config_path),
            "updated_at": now,
        }
        write_json_atomic(self.state_path(user_id), state)
        self.last_error = None
        return state

    def _write_source_records(self, session_dir: Path, records: list[MemoryRecord]) -> None:
        path = session_dir / "text_source_records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")

    def _records_to_mst_events(self, scope_id: str, records: list[MemoryRecord]) -> list[dict[str, Any]]:
        ordered = sorted(records, key=lambda row: (parse_iso_datetime(row.created_at), row.request_id, row.message_index))
        first_epoch = parse_iso_datetime(ordered[0].created_at).timestamp() if ordered else 0.0
        events: list[dict[str, Any]] = []
        last_end = -1.0
        for idx, row in enumerate(ordered):
            epoch = parse_iso_datetime(row.created_at).timestamp()
            proposed_start = max(0.0, epoch - first_epoch)
            start = proposed_start if proposed_start > last_end else last_end + 0.01
            end = start + 1.0
            last_end = end
            caption = f"{row.role}: {row.raw_content}".strip()
            event = {
                "event_id": "mst_" + row.memory_id,
                "session_id": scope_id,
                "source_user_id": row.user_id,
                "source_session_id": row.session_id,
                "source_request_id": row.request_id,
                "source_message_index": row.message_index,
                "chunk_id": f"text_chunk_{idx:06d}",
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "duration": 1.0,
                "available_at": round(end, 3),
                "status": "final",
                "version": 1,
                "boundary_reason": "text_message",
                "diff_score": 0.0,
                "diff_stats": {"modality": "text", "message_index": row.message_index},
                "event_caption_placeholder": caption,
                "event_caption_fast": caption,
                "event_caption_refined": caption,
                "caption_source": "refined",
                "transcript": row.raw_content,
                "transcript_segments": [
                    {
                        "segment_id": f"seg_{row.memory_id}",
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": row.raw_content,
                        "role": row.role,
                        "timestamp": row.created_at,
                    }
                ],
                "transcript_version": 1,
                "transcript_source": "benchmark_text_add",
                "transcript_updated_at": row.created_at,
                "keyframes": [],
                "entities": self._extract_entities(row.raw_content),
                "visual_objects": [],
                "main_actions": [{"actor": row.role, "action": "said", "objects": [row.raw_content]}],
                "state_changes": [],
                "conversation_focus": self._conversation_focus(row.raw_content),
                "needs_refine": False,
                "refined_stale": False,
                "stale_reason": None,
                "refine_completed_at": row.created_at,
                "refine_speed": "text_only_direct",
                "needs_reconsolidation": False,
                "dirty_reason": None,
                "dirty_window_id": None,
                "dirty_time_range": None,
                "source": {
                    "type": "benchmark_text_add",
                    "memory_id": row.memory_id,
                    "request_id": row.request_id,
                    "user_id": row.user_id,
                    "session_id": row.session_id,
                    "created_at": row.created_at,
                },
                "refine": {"backend": "text_only_direct", "refine_timeline": []},
                "merged_to_long_term": True,
                "merged_episode_id": "mst_ep_" + row.memory_id,
                "merged_at": utc_now_iso(),
                "confidence": 0.9,
                "created_at": row.created_at,
                "updated_at": utc_now_iso(),
            }
            event["retrieval_text"] = self.build_retrieval_text(event)
            events.append(event)
        return events

    def _events_to_episodes(self, scope_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        episodes = []
        for idx, event in enumerate(events):
            event_id = str(event.get("event_id") or f"event_{idx:06d}")
            episode_id = str(event.get("merged_episode_id") or f"mst_ep_{event_id}")
            caption = str(event.get("event_caption_refined") or event.get("transcript") or "")
            episode = {
                "episode_id": episode_id,
                "doc_id": episode_id,
                "segment_id": f"seg_{event_id}",
                "session_id": scope_id,
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "caption": caption,
                "fine_caption": caption,
                "transcript": event.get("transcript", ""),
                "transcript_segments": event.get("transcript_segments", []),
                "transcript_summary": compact_text(event.get("transcript", "")),
                "scene": "text_only_memory",
                "main_actions": event.get("main_actions", []),
                "state_changes": [],
                "visual_objects": [],
                "keyframe_paths": [],
                "keyframe_captions": [],
                "entities": event.get("entities", []),
                "conversation_focus": event.get("conversation_focus", []),
                "source_micro_event_ids": [event_id],
                "source_micro_events": [event],
                "refined_event_count": 1,
                "completeness_score": 1.0,
                "confidence": event.get("confidence", 0.9),
                "status": "complete",
                "episodic_source": "mst_micro_events_text_only",
                "modality": "text",
                "source": event.get("source", {}),
            }
            episodes.append(episode)
        return episodes

    def _episodes_to_evidence(self, scope_id: str, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence = []
        for episode in episodes:
            source = episode.get("source", {}) if isinstance(episode.get("source"), dict) else {}
            doc_id = str(episode.get("episode_id"))
            doc = {
                "doc_id": doc_id,
                "evidence_doc_id": doc_id,
                "episode_id": episode.get("episode_id"),
                "session_id": scope_id,
                "parent_session_id": scope_id,
                "child_session_id": source.get("session_id"),
                "source_child_session_id": source.get("session_id"),
                "segment_id": episode.get("segment_id"),
                "start_time": episode.get("start_time"),
                "end_time": episode.get("end_time"),
                "local_start_time": episode.get("start_time"),
                "local_end_time": episode.get("end_time"),
                "caption": episode.get("caption"),
                "fine_caption": episode.get("fine_caption"),
                "scene": "text_only_memory",
                "transcript": episode.get("transcript", ""),
                "transcript_segments": episode.get("transcript_segments", []),
                "transcript_summary": episode.get("transcript_summary", ""),
                "keyframe_captions": [],
                "visual_objects": [],
                "main_actions": episode.get("main_actions", []),
                "state_changes": [],
                "entities": episode.get("entities", []),
                "conversation_focus": episode.get("conversation_focus", []),
                "source_micro_event_ids": episode.get("source_micro_event_ids", []),
                "source_micro_events": episode.get("source_micro_events", []),
                "keyframe_paths": [],
                "source_video_path": "",
                "clip_path": "",
                "confidence": episode.get("confidence", 0.9),
                "completeness_score": 1.0,
                "refined_event_count": 1,
                "status": "complete",
                "episodic_source": "mst_micro_events_text_only",
                "modality": "text",
                "request_id": source.get("request_id"),
                "user_id": source.get("user_id"),
                "original_session_id": source.get("session_id"),
                "role": self._source_role(episode),
                "memory_id": source.get("memory_id"),
                "created_at": source.get("created_at"),
                "display_iso_start": source.get("created_at"),
                "display_iso_end": source.get("created_at"),
                "timezone": "UTC",
                "time_source": "benchmark_timestamp",
                "date": "DAY1",
            }
            evidence.append(doc)
        return evidence

    def _write_mst_episode_outputs(
        self,
        session_dir: Path,
        scope_id: str,
        episodes: list[dict[str, Any]],
        caption_30s: list[dict[str, Any]],
        evidence_docs: list[dict[str, Any]],
    ) -> None:
        root = session_dir / "em2mem" / "mst_episodic"
        root.mkdir(parents=True, exist_ok=True)
        episodes_path = root / "mst_30sec_episodes.json"
        episodes_jsonl_path = root / "mst_30sec_episodes.jsonl"
        mapping_path = root / "mst_to_episode_mapping.json"
        state_path = root / "mst_episodic_state.json"
        self.lm_write_json(episodes_path, episodes)
        with episodes_jsonl_path.open("w", encoding="utf-8") as f:
            for episode in episodes:
                f.write(json.dumps(episode, ensure_ascii=False) + "\n")
        mapping = {
            "event_to_episode": {
                event_id: episode.get("episode_id")
                for episode in episodes
                for event_id in episode.get("source_micro_event_ids", []) or []
            },
            "episode_to_caption_doc": {episode.get("episode_id"): episode.get("episode_id") for episode in episodes},
        }
        self.lm_write_json(mapping_path, mapping)
        self.lm_write_json(
            state_path,
            {
                "session_id": scope_id,
                "status": "ready" if episodes else "empty",
                "backend": "text_only_direct",
                "episode_count": len(episodes),
                "captioned_30sec_count": len(caption_30s),
                "evidence_count": len(evidence_docs),
                "episodes_path": "em2mem/mst_episodic/mst_30sec_episodes.json",
                "captioned_30sec_path": "captions/mst_session_30sec_captioned.json",
                "evidence_path": "evidence/mst_session_evidence.json",
                "updated_at": utc_now_iso(),
            },
        )

    def _max_end_hhmmssff(self, caption_30s: list[dict[str, Any]]) -> str:
        if not caption_30s:
            return "00000000"
        return str(max((item.get("end_time") or "00000000") for item in caption_30s))

    def _time_span(self, events: list[dict[str, Any]]) -> list[float]:
        if not events:
            return [0.0, 0.0]
        return [
            round(min(safe_float(event.get("start_time")) for event in events), 3),
            round(max(safe_float(event.get("end_time")) for event in events), 3),
        ]

    def _rel(self, path: Path, root: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except Exception:
            return str(path)

    def _source_role(self, episode: dict[str, Any]) -> str:
        events = episode.get("source_micro_events") or []
        if events and isinstance(events[0], dict):
            segs = events[0].get("transcript_segments") or []
            if segs and isinstance(segs[0], dict):
                return str(segs[0].get("role") or "")
        return ""

    def _extract_entities(self, text: str) -> list[str]:
        tokens = tokenize(text)
        entities = []
        for token in tokens:
            if len(token) >= 2 or (token and "\u4e00" <= token[0] <= "\u9fff"):
                entities.append(token)
        return list(dict.fromkeys(entities))[:24]

    def _conversation_focus(self, text: str) -> list[str]:
        clean = compact_text(text, max_chars=180)
        topics = [clean] if clean else []
        keywords = [tok for tok in tokenize(text) if len(tok) >= 3]
        topics.extend(keywords[:8])
        return list(dict.fromkeys(topics))[:10]

    def route_and_plan(self, query_text: str, top_k: int, user_id: str, records: list[MemoryRecord], session_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        runtime_state = self.runtime_state(user_id, records, session_id=session_id)
        request_options = {
            "top_k": top_k,
            "text_top_k": max(top_k, 8),
            "final_evidence_k": top_k,
            "retrieval_mode": "text_only",
            "use_image_evidence": False,
            "max_image_evidence": 0,
            "use_current": False,
            "use_interaction_cache": False,
        }
        cache_context: dict[str, Any] = {}
        decision = self.MemoryRouter().route(query_text, request_options, runtime_state, cache_context=cache_context)
        # The ablation has no current/visual memory. If a visual-looking query was routed
        # to current, force it back to text M_st/M_lt so the benchmark always gets text evidence.
        decision.setdefault("memory_route", {})["use_current"] = False
        if runtime_state.get("short_term_ready") and not decision["memory_route"].get("use_long_term"):
            decision["memory_route"]["use_short_term"] = True
        if runtime_state.get("long_term_ready"):
            decision["memory_route"]["use_long_term"] = True
        decision.setdefault("memory_priority", {})["M_cur"] = 0.0
        decision["memory_priority"]["M_st"] = max(float(decision["memory_priority"].get("M_st", 0.0)), 0.65 if runtime_state.get("short_term_ready") else 0.0)
        decision["memory_priority"]["M_lt"] = max(float(decision["memory_priority"].get("M_lt", 0.0)), 0.85 if runtime_state.get("long_term_ready") else 0.0)
        plan = self.RetrievalPlanner().plan(decision, request_options, runtime_state, cache_context=cache_context)
        plan["use_image_evidence"] = False
        plan["max_image_evidence"] = 0
        plan["retrieval_mode"] = "text_only"
        if "retrieval_plan" in plan:
            plan["retrieval_plan"].setdefault("M_cur", {})["enabled"] = False
            plan["retrieval_plan"].setdefault("M_lt", {})["mode"] = "text_only"
        return decision, plan, cache_context

    def search_method(
        self,
        user_id: str,
        records: list[MemoryRecord],
        query_text: str,
        top_k: int,
        embedding: EmbeddingBackend,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if not records:
            return {"data": [], "debug": {"reason": "empty_user_memory"}}
        state = self.ensure_user_memory(user_id, records, session_id=session_id)
        session_dir = self.scope_dir(user_id, session_id=session_id)
        decision, plan, cache_context = self.route_and_plan(query_text, top_k, user_id, records, session_id=session_id)
        memory_results: dict[str, Any] = {"current_results": [], "short_term_results": [], "text_results": [], "visual_results": [], "fused_results": []}

        mst_plan = (plan.get("retrieval_plan") or {}).get("M_st") or {}
        if mst_plan.get("enabled"):
            store = self._store(session_dir)
            retriever = self.MSTRetriever(store)
            memory_results["short_term_results"] = retriever.search(
                query_text,
                top_k=max(1, int(mst_plan.get("top_k") or top_k)),
                cache_context=cache_context,
                include_archive=True,
            )

        mlt_plan = (plan.get("retrieval_plan") or {}).get("M_lt") or {}
        if mlt_plan.get("enabled"):
            memory_results["text_results"] = self.search_long_term_text(
                session_dir=session_dir,
                query_text=query_text,
                query_type=str(decision.get("query_type") or "general_qa"),
                top_k=max(1, int(mlt_plan.get("text_top_k") or mlt_plan.get("top_k") or top_k)),
                embedding=embedding,
            )

        fused = self.MemoryFusion().fuse(
            memory_results=memory_results,
            memory_decision=decision,
            retrieval_plan=plan,
            query_type=str(decision.get("query_type") or "general_qa"),
            cache_context=cache_context,
        )
        return {
            "data": self.format_api_results(fused.get("final_evidence", []), top_k=top_k),
            "debug": {
                "scope_state": state,
                "query_type": decision.get("query_type"),
                "route": decision.get("memory_route"),
                "retrieval_mode": "text_only",
                "use_image_evidence": False,
                "fusion_summary": fused.get("fusion_summary", {}),
            },
        }

    def search_long_term_text(
        self,
        session_dir: Path,
        query_text: str,
        query_type: str,
        top_k: int,
        embedding: EmbeddingBackend,
    ) -> list[dict[str, Any]]:
        layout = self.Em2MemOnlineLayout(session_dir=session_dir, session_id=session_dir.name)
        candidates = self._cached_long_term_candidates(layout)
        if not candidates:
            return []

        ranked_candidates = self._prefilter_candidates(candidates, query_text, query_type, top_k)
        candidate_texts = [item["retrieval_text"] for item, _lex, _graph, _source_bonus, _recency in ranked_candidates]
        dense_scores = self._dense_scores(query_text, candidate_texts, embedding)

        scored = []
        for (item, lex, graph_bonus, source_bonus, recency), dense in zip(ranked_candidates, dense_scores):
            score = 0.46 * clamp01(dense) + 0.28 * clamp01(lex) + 0.12 * source_bonus + 0.08 * graph_bonus + 0.06 * recency
            out = dict(item["payload"])
            out["score"] = round(clamp01(score), 6)
            out["dense_score"] = round(clamp01(dense), 6)
            out["lexical_score"] = round(clamp01(lex), 6)
            out["semantic_score"] = round(max(clamp01(dense), clamp01(lex), graph_bonus), 6)
            out["retrieval_text"] = item["retrieval_text"]
            out["retrieval_scale"] = item.get("scale")
            out["text_only"] = True
            out["prefiltered_from"] = len(candidates)
            out["prefiltered_to"] = len(ranked_candidates)
            scored.append(out)
        scored.sort(key=lambda x: (-safe_float(x.get("score")), -safe_float(x.get("end_time")), str(x.get("memory_id") or x.get("doc_id") or "")))
        return scored[: max(1, int(top_k))]

    def _cached_long_term_candidates(self, layout: Any) -> list[dict[str, Any]]:
        paths = [
            layout.caption_30sec_path,
            layout.caption_3min_path,
            layout.caption_10min_path,
            layout.caption_1h_path,
            layout.sidecar_root / "30s" / f"episodic_triplets_30s_{self.model_name}.json",
            layout.semantic_root / f"semantic_memory_{self.model_name}.json",
        ]
        signature = []
        for path in paths:
            try:
                stat = path.stat()
                signature.append((str(path), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                signature.append((str(path), 0, 0))
        key = (str(layout.session_dir), tuple(signature))
        if self.lt_candidate_cache_users > 0:
            with self.cache_lock:
                cached = self.lt_candidate_cache.get(key)
                if cached is not None:
                    self.lt_candidate_cache.move_to_end(key)
                    return cached

        caption_items = self._load_caption_candidates(layout)
        semantic_items = self._load_semantic_candidates(layout, caption_items)
        candidates = caption_items + semantic_items

        if self.lt_candidate_cache_users > 0:
            with self.cache_lock:
                self.lt_candidate_cache[key] = candidates
                self.lt_candidate_cache.move_to_end(key)
                while len(self.lt_candidate_cache) > self.lt_candidate_cache_users:
                    self.lt_candidate_cache.popitem(last=False)
        return candidates

    def _prefilter_candidates(
        self,
        candidates: list[dict[str, Any]],
        query_text: str,
        query_type: str,
        top_k: int,
    ) -> list[tuple[dict[str, Any], float, float, float, float]]:
        if not candidates:
            return []
        limit = min(len(candidates), self.lt_prefilter_cap, max(self.lt_prefilter_base, int(top_k) * 4))
        end_times = [safe_float((item.get("payload") or {}).get("end_time")) for item in candidates]
        min_end = min(end_times) if end_times else 0.0
        max_end = max(end_times) if end_times else 0.0

        def recency_for(end_time: float) -> float:
            if max_end <= min_end:
                return 1.0
            return clamp01((end_time - min_end) / (max_end - min_end))

        ranked: list[tuple[float, float, dict[str, Any], float, float, float, float]] = []
        for item, end_time in zip(candidates, end_times):
            lex = lexical_score(query_text, item["retrieval_text"])
            graph_bonus = self._graph_bonus(query_text, item)
            source_bonus = self._source_bonus(item, query_type)
            recency = recency_for(end_time)
            rank = 0.62 * clamp01(lex) + 0.18 * graph_bonus + 0.12 * source_bonus + 0.08 * recency
            ranked.append((rank, end_time, item, lex, graph_bonus, source_bonus, recency))
        ranked.sort(key=lambda x: (-x[0], -x[1]))

        selected = ranked[:limit]
        # Preserve a few latest memories even when lexical overlap is weak.
        if len(candidates) > limit:
            selected_ids = {id(entry[2]) for entry in selected}
            latest = sorted(ranked, key=lambda x: -x[1])[: min(16, limit)]
            for entry in latest:
                if id(entry[2]) not in selected_ids:
                    selected.append(entry)
                    selected_ids.add(id(entry[2]))
                    if len(selected) >= limit:
                        break

        selected.sort(key=lambda x: (-x[0], -x[1]))
        return [(item, lex, graph_bonus, source_bonus, recency) for _rank, _end, item, lex, graph_bonus, source_bonus, recency in selected[:limit]]

    def _load_caption_candidates(self, layout: Any) -> list[dict[str, Any]]:
        paths = {
            "30sec": layout.caption_30sec_path,
            "3min": layout.caption_3min_path,
            "10min": layout.caption_10min_path,
            "1h": layout.caption_1h_path,
        }
        triplet_map = self._load_triplet_map(layout.sidecar_root / "30s" / f"episodic_triplets_30s_{self.model_name}.json")
        candidates: list[dict[str, Any]] = []
        for scale, path in paths.items():
            data = self.lm_read_json(path, default=[])
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                doc_id = str(item.get("doc_id") or item.get("segment_id") or "")
                triples = triplet_map.get(doc_id, []) if scale == "30sec" else []
                retrieval_text = compact_text(
                    [
                        f"Scale: {scale}",
                        item.get("fine_caption") or item.get("caption") or item.get("text"),
                        f"Transcript: {item.get('transcript', '')}" if item.get("transcript") else "",
                        f"Topics: {item.get('topic_threads', [])}" if item.get("topic_threads") else "",
                        f"Triplets: {triples}" if triples else "",
                    ],
                    max_chars=3000,
                )
                payload = {
                    "memory_id": doc_id,
                    "evidence_doc_id": doc_id,
                    "segment_id": item.get("segment_id") or doc_id,
                    "caption": item.get("fine_caption") or item.get("caption") or item.get("text") or "",
                    "transcript": item.get("transcript", ""),
                    "start_time": item.get("start"),
                    "end_time": item.get("end"),
                    "keyframe_paths": [],
                    "created_at": item.get("display_iso_start") or self._created_at_from_source(item),
                    "source_type": "episodic_caption",
                    "scale": scale,
                    "doc_id": doc_id,
                    "source_doc_ids": item.get("source_doc_ids", []) or [doc_id],
                    "child_ids": item.get("child_ids", []),
                    "semantic_triplets": triples,
                    "text_only": True,
                    "metadata": item,
                }
                candidates.append({"scale": scale, "retrieval_text": retrieval_text, "payload": payload, "triples": triples})
        return candidates

    def _load_semantic_candidates(self, layout: Any, caption_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_doc = {}
        for candidate in caption_candidates:
            payload = candidate.get("payload") or {}
            doc_id = str(payload.get("doc_id") or "")
            if doc_id and doc_id not in by_doc:
                by_doc[doc_id] = payload
        memory_path = layout.semantic_root / f"semantic_memory_{self.model_name}.json"
        data = self.lm_read_json(memory_path, default={})
        if not isinstance(data, dict):
            return []
        facts = data.get("facts") or []
        candidates = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            support_docs = [str(x) for x in fact.get("support_docs", []) or fact.get("source_doc_ids", []) or [] if x]
            support_payloads = [by_doc[x] for x in support_docs if x in by_doc]
            first = support_payloads[0] if support_payloads else {}
            text = compact_text(
                [
                    "Semantic fact",
                    fact.get("semantic_summary") or fact.get("triple"),
                    f"Support docs: {support_docs}" if support_docs else "",
                    first.get("caption", ""),
                ],
                max_chars=3000,
            )
            payload = {
                "memory_id": fact.get("fact_id"),
                "evidence_doc_id": support_docs[0] if support_docs else fact.get("fact_id"),
                "segment_id": support_docs[0] if support_docs else fact.get("fact_id"),
                "caption": fact.get("semantic_summary") or compact_text(fact.get("triple")),
                "transcript": first.get("transcript", ""),
                "start_time": first.get("start_time", 0.0),
                "end_time": first.get("end_time", first.get("start_time", 0.0)),
                "keyframe_paths": [],
                "created_at": first.get("created_at"),
                "source_type": "semantic_memory",
                "scale": "semantic",
                "doc_id": fact.get("fact_id"),
                "source_doc_ids": support_docs,
                "semantic_fact": fact,
                "text_only": True,
            }
            candidates.append({"scale": "semantic", "retrieval_text": text, "payload": payload, "triples": [fact.get("triple", [])]})
        return candidates

    def _load_triplet_map(self, path: Path) -> dict[str, list[list[str]]]:
        data = self.lm_read_json(path, default={})
        if not isinstance(data, dict):
            return {}
        raw = data.get("triplet_map") or {}
        if not isinstance(raw, dict):
            return {}
        return {str(key): value if isinstance(value, list) else [] for key, value in raw.items()}

    def _dense_scores(self, query_text: str, candidate_texts: list[str], embedding: EmbeddingBackend) -> list[float]:
        if not candidate_texts:
            return []
        try:
            query_vector = embedding.embed_many([query_text])[0]
            query_norm = vector_norm(query_vector)
            doc_vectors: list[Any] = []
            batch_size = max(1, int(os.getenv("LIGHTMEM_TEXT_LT_EMBED_BATCH", "64")))
            for start in range(0, len(candidate_texts), batch_size):
                doc_vectors.extend(embedding.embed_many(candidate_texts[start : start + batch_size]))
            return [
                max(0.0, cosine_similarity(query_vector, vec, query_norm, vector_norm(vec)))
                for vec in doc_vectors
            ]
        except Exception as exc:
            self.last_error = f"Long-term dense scoring fell back to lexical only: {exc}"
            return [0.0 for _ in candidate_texts]

    def _source_bonus(self, item: dict[str, Any], query_type: str) -> float:
        scale = str(item.get("scale") or "")
        if scale == "semantic":
            return 0.85
        if query_type == "long_term_summary":
            return {"1h": 0.95, "10min": 0.82, "3min": 0.65, "30sec": 0.35}.get(scale, 0.2)
        return {"30sec": 0.9, "3min": 0.65, "10min": 0.45, "1h": 0.3}.get(scale, 0.2)

    def _graph_bonus(self, query_text: str, item: dict[str, Any]) -> float:
        triples = item.get("triples") or []
        if not triples:
            return 0.0
        text = " ".join(compact_text(tri, max_chars=0) for tri in triples)
        return lexical_score(query_text, text)

    def _relative_recency(self, item: dict[str, Any], candidates: list[dict[str, Any]]) -> float:
        ends = [safe_float((cand.get("payload") or {}).get("end_time")) for cand in candidates]
        if not ends:
            return 0.0
        lo, hi = min(ends), max(ends)
        end = safe_float((item.get("payload") or {}).get("end_time"))
        if hi <= lo:
            return 1.0
        return clamp01((end - lo) / (hi - lo))

    def _created_at_from_source(self, item: dict[str, Any]) -> str | None:
        source_docs = item.get("source_micro_events") or []
        if source_docs and isinstance(source_docs[0], dict):
            source = source_docs[0].get("source") or {}
            if isinstance(source, dict) and source.get("created_at"):
                return str(source["created_at"])
        return None

    def format_api_results(self, evidence: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in sorted(evidence, key=lambda x: -safe_float(x.get("final_score", x.get("retrieval_score", 0.0)))):
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            inner = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else metadata
            created_at = item.get("created_at") or metadata.get("created_at") or inner.get("created_at") or metadata.get("display_iso_start")
            content = self._evidence_content(item)
            key = normalize_for_dedup(content)
            if key in seen:
                continue
            seen.add(key)
            data.append(
                {
                    "id": str(item.get("evidence_id") or metadata.get("memory_id") or metadata.get("doc_id") or hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]),
                    "content": content,
                    "score": round(safe_float(item.get("final_score", item.get("retrieval_score", 0.0))), 6),
                    "created_at": str(created_at or utc_now_iso()),
                }
            )
            if len(data) >= top_k:
                break
        return data

    def _evidence_content(self, item: dict[str, Any]) -> str:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source_memory = item.get("source_memory") or metadata.get("source_memory") or "M_lt"
        source_type = item.get("source_type") or metadata.get("source_type") or metadata.get("scale") or "text_memory"
        caption = compact_text(item.get("caption") or metadata.get("caption") or metadata.get("fine_caption") or metadata.get("retrieval_text"), max_chars=1200)
        transcript = compact_text(item.get("transcript") or metadata.get("transcript"), max_chars=1200)
        time_part = ""
        if item.get("start_time") is not None or item.get("end_time") is not None:
            time_part = f"t={safe_float(item.get('start_time')):.3f}-{safe_float(item.get('end_time', item.get('start_time'))):.3f}s"
        parts = [f"[{source_memory}/{source_type}]", time_part, caption]
        if transcript and transcript not in caption:
            parts.append(f"Transcript: {transcript}")
        return " ".join(part for part in parts if part).strip()


class LightMemTextMemoryService:
    def __init__(self, db: MemoryDatabase, embedding: EmbeddingBackend, data_dir: Path, min_score: float = 0.0, method_model_name: str = DEFAULT_METHOD_MODEL):
        self.db = db
        self.embedding = embedding
        self.min_score = min_score
        self.method_runtime = LightMemTextAblationRuntime(data_dir=data_dir, model_name=method_model_name, generation_backend="rule")
        self.search_concurrency = max(1, env_int("LIGHTMEM_TEXT_SEARCH_CONCURRENCY", 2))
        self.search_queue_timeout = max(0.0, env_float("LIGHTMEM_TEXT_SEARCH_QUEUE_TIMEOUT", 10.0))
        self.search_semaphore = threading.BoundedSemaphore(self.search_concurrency)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "backend": "lightmem-ego-text",
            "method_level": "lightmem_ego_text_ablation",
            "text_only": True,
            "configured": self.method_runtime.available,
            "db_path": self.db.db_path,
            "memory_count": self.db.count(),
            "user_count": self.db.user_count(),
            "embedding": self.embedding.health(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "search_concurrency": self.search_concurrency,
            "search_queue_timeout": self.search_queue_timeout,
            "build_on_add": env_bool("LIGHTMEM_TEXT_BUILD_ON_ADD", False),
            "include_options_in_query": env_bool("LIGHTMEM_TEXT_INCLUDE_OPTIONS_IN_QUERY", True),
            "search_session_filter_supported": True,
            "lightmem_method": self.method_runtime.health(),
        }

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = require_str(payload, "request_id")
        user_id = require_str(payload, "user_id")
        session_id = require_str(payload, "session_id")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ApiError(400, "messages must be a non-empty array")

        prepared: list[dict[str, Any]] = []
        embedding_inputs: list[str] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ApiError(400, "messages[] entries must be objects")
            role = require_str(message, "role", path=f"messages[{index}].role")
            if role not in {"user", "assistant", "system", "tool"}:
                raise ApiError(400, f"messages[{index}].role must be user, assistant, system, or tool")
            content = require_str(message, "content", path=f"messages[{index}].content")
            created_at = timestamp_to_iso(message.get("timestamp"))
            display_content = f"[{created_at}] {role}: {content}"
            searchable_content = "\n".join(
                [
                    f"user_id: {user_id}",
                    f"session_id: {session_id}",
                    f"role: {role}",
                    f"time: {created_at}",
                    content,
                ]
            )
            prepared.append(
                {
                    "id": stable_memory_id(user_id, session_id, request_id, index),
                    "request_id": request_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "message_index": index,
                    "role": role,
                    "raw_content": content,
                    "display_content": display_content,
                    "searchable_content": searchable_content,
                    "created_at": created_at,
                    "ingested_at": utc_now_iso(),
                }
            )
            embedding_inputs.append(searchable_content)

        vectors = self.embedding.embed_many(embedding_inputs)
        rows = []
        for row, vector in zip(prepared, vectors):
            row = dict(row)
            row["vector_backend"] = self.embedding.name
            row["vector_json"] = self.embedding.serialize(vector)
            row["vector_norm"] = vector_norm(vector)
            rows.append(row)
        self.db.upsert_rows(rows)
        if env_bool("LIGHTMEM_TEXT_BUILD_ON_ADD", False):
            records = self.db.fetch_records_by_user_session(user_id, session_id)
            self.method_runtime.ensure_user_memory(user_id, records, session_id=session_id)
        return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        acquired = self.search_semaphore.acquire(timeout=self.search_queue_timeout)
        if not acquired:
            raise ApiError(503, f"search is busy; concurrency limit={self.search_concurrency}")
        try:
            query = require_str(payload, "query")
            user_id = require_str(payload, "user_id")
            top_k = require_positive_int(payload, "top_k")
            session_id = optional_str(payload, "session_id")
            query_text = build_query_text(
                query,
                payload.get("options"),
                include_options=env_bool("LIGHTMEM_TEXT_INCLUDE_OPTIONS_IN_QUERY", True),
            )
            records = (
                self.db.fetch_records_by_user_session(user_id, session_id)
                if session_id
                else self.db.fetch_records_by_user(user_id)
            )
            if records and self.method_runtime.available:
                try:
                    method_result = self.method_runtime.search_method(user_id, records, query_text, top_k, self.embedding, session_id=session_id)
                    if method_result.get("data"):
                        return {"data": method_result["data"][:top_k]}
                except Exception as exc:
                    self.method_runtime.last_error = f"method search failed; raw fallback used: {exc}"
            return self._raw_vector_search(user_id=user_id, query_text=query_text, top_k=top_k, session_id=session_id)
        finally:
            self.search_semaphore.release()

    def _raw_vector_search(self, user_id: str, query_text: str, top_k: int, session_id: str | None = None) -> dict[str, Any]:
        query_vector = self.embedding.embed_many([query_text])[0]
        query_norm = vector_norm(query_vector)
        rows = self.db.fetch_by_user_session(user_id, session_id) if session_id else self.db.fetch_by_user(user_id)

        scored: list[tuple[float, MemoryRow]] = []
        for row in rows:
            if row.vector_backend == self.embedding.name:
                doc_vector = self.embedding.deserialize(row.vector_json)
                doc_norm = row.vector_norm
                dense_score = cosine_similarity(query_vector, doc_vector, query_norm, doc_norm)
            elif self.embedding.name == "hash":
                doc_vector = self.embedding.embed_many([row.searchable_content])[0]
                doc_norm = vector_norm(doc_vector)
                dense_score = cosine_similarity(query_vector, doc_vector, query_norm, doc_norm)
            else:
                dense_score = 0.0
            lex = lexical_score(query_text, row.searchable_content)
            score = max(float(dense_score), float(lex) * 0.65)
            if score > self.min_score:
                scored.append((score, row))

        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        data = [
            {
                "id": row.memory_id,
                "content": row.display_content,
                "score": round(float(score), 6),
                "created_at": row.created_at,
            }
            for score, row in scored[:top_k]
        ]
        return {"data": data}


def require_str(payload: dict[str, Any], key: str, path: Optional[str] = None) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, f"{path or key} must be a non-empty string")
    return value


def optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, f"{key} must be a string when provided")
    return value.strip()


def require_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ApiError(400, f"{key} must be a positive integer")
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, f"{key} must be a positive integer") from exc
    if value_int <= 0:
        raise ApiError(400, f"{key} must be a positive integer")
    return value_int


def build_query_text(query: str, options: Any, include_options: bool = True) -> str:
    if options is None or not include_options:
        return query
    if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
        raise ApiError(400, "options must be an array of strings when provided")
    if not options:
        return query
    return query + "\nOptions:\n" + "\n".join(options)


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type,Authorization")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(service: LightMemTextMemoryService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LightMemEgoTextAblationAPI/2.0"

        def do_OPTIONS(self) -> None:
            json_response(self, 200, {})

        def do_GET(self) -> None:
            if self.path.rstrip("/") in {"/health", "/v1/health"}:
                json_response(self, 200, service.health())
                return
            json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                path = self.path.rstrip("/")
                if path in {"/add", "/v1/add"}:
                    json_response(self, 200, service.add(payload))
                elif path in {"/search", "/v1/search"}:
                    json_response(self, 200, service.search(payload))
                else:
                    json_response(self, 404, {"error": "not found"})
            except ApiError as exc:
                json_response(self, exc.status, {"error": exc.message})
            except Exception as exc:
                json_response(self, 500, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def _read_json(self) -> dict[str, Any]:
            length_raw = self.headers.get("Content-Length")
            if not length_raw:
                raise ApiError(400, "missing JSON body")
            try:
                length = int(length_raw)
            except ValueError as exc:
                raise ApiError(400, "invalid Content-Length") from exc
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ApiError(400, "request body must be valid JSON") from exc
            if not isinstance(payload, dict):
                raise ApiError(400, "request body must be a JSON object")
            return payload

    return Handler


def build_embedding_backend(args: argparse.Namespace) -> EmbeddingBackend:
    if args.embedding_backend == "hash":
        return HashEmbeddingBackend(dim=args.hash_dim)
    openai_backend = OpenAIEmbeddingBackend(
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        model=args.embedding_model,
        timeout=args.embedding_timeout,
    )
    if args.embedding_backend == "openai":
        openai_backend.embed_many(["healthcheck"])
        return openai_backend
    try:
        openai_backend.embed_many(["healthcheck"])
        return openai_backend
    except Exception as exc:
        print(f"Embedding service unavailable ({exc}); falling back to hash backend.", file=sys.stderr)
        return HashEmbeddingBackend(dim=args.hash_dim)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve LightMem-Ego text-only benchmark add/search API.")
    parser.add_argument("--host", default=os.environ.get("LIGHTMEM_TEXT_MEMORY_API_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LIGHTMEM_TEXT_MEMORY_API_PORT", DEFAULT_PORT)))
    parser.add_argument("--data-dir", default=os.environ.get("LIGHTMEM_TEXT_MEMORY_API_DATA_DIR", DEFAULT_DATA_DIR))
    parser.add_argument("--db-path", default=os.environ.get("LIGHTMEM_TEXT_MEMORY_API_DB", ""))
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "hash", "openai"],
        default=os.environ.get("LIGHTMEM_TEXT_MEMORY_API_EMBEDDING_BACKEND", "auto"),
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("LIGHTMEM_TEXT_EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL),
    )
    parser.add_argument("--embedding-api-key", default=os.environ.get("LIGHTMEM_TEXT_EMBEDDING_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-model", default=os.environ.get("LIGHTMEM_TEXT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--embedding-timeout", type=float, default=float(os.environ.get("LIGHTMEM_TEXT_EMBEDDING_TIMEOUT", "30")))
    parser.add_argument("--hash-dim", type=int, default=DEFAULT_HASH_DIM)
    parser.add_argument("--min-score", type=float, default=float(os.environ.get("LIGHTMEM_TEXT_MEMORY_API_MIN_SCORE", "0.0")))
    parser.add_argument("--method-model-name", default=os.environ.get("LIGHTMEM_TEXT_METHOD_MODEL", DEFAULT_METHOD_MODEL))
    return parser.parse_args(argv)


class BenchmarkHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    db_path = args.db_path or str(data_dir / "lightmem_text_memory_api.db")
    data_dir.mkdir(parents=True, exist_ok=True)
    embedding = build_embedding_backend(args)
    service = LightMemTextMemoryService(
        MemoryDatabase(db_path),
        embedding,
        data_dir=data_dir,
        min_score=args.min_score,
        method_model_name=args.method_model_name,
    )
    handler = make_handler(service)
    server = BenchmarkHTTPServer((args.host, args.port), handler)

    def stop_server(signum: int, _frame: Any) -> None:
        print(f"Received signal {signum}; shutting down.", file=sys.stderr)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    print(
        f"Serving LightMem-Ego text ablation API on http://{args.host}:{args.port} "
        f"(embedding={embedding.name}, cuda_visible_devices={os.getenv('CUDA_VISIBLE_DEVICES', '')}, "
        f"method_available={service.method_runtime.available})",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
