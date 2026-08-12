#!/usr/bin/env python3
"""OpenAI-compatible local text embedding server for text-only evaluation."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class EmbeddingService:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, inputs: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(float).tolist()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model": self.model_name,
            "device": self.device,
            "dim": int(self.model.get_sentence_embedding_dimension()),
        }


def normalize_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ApiError(400, "input must be a string or an array of strings")


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(service: EmbeddingService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LightMemTextEmbedding/1.0"

        def do_GET(self) -> None:
            path = self.path.rstrip("/")
            if path in {"", "/health", "/v1/health"}:
                json_response(self, 200, service.health())
                return
            if path in {"/models", "/v1/models"}:
                json_response(
                    self,
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": service.model_name,
                                "object": "model",
                                "owned_by": "local",
                            }
                        ],
                    },
                )
                return
            json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                path = self.path.rstrip("/")
                if path not in {"/embeddings", "/v1/embeddings"}:
                    json_response(self, 404, {"error": "not found"})
                    return
                payload = self._read_json()
                inputs = normalize_inputs(payload.get("input"))
                vectors = service.embed(inputs)
                data = [
                    {
                        "object": "embedding",
                        "embedding": vector,
                        "index": index,
                    }
                    for index, vector in enumerate(vectors)
                ]
                json_response(
                    self,
                    200,
                    {
                        "object": "list",
                        "data": data,
                        "model": payload.get("model") or service.model_name,
                        "usage": {
                            "prompt_tokens": 0,
                            "total_tokens": 0,
                        },
                    },
                )
            except ApiError as exc:
                json_response(self, exc.status, {"error": exc.message})
            except Exception as exc:
                json_response(self, 500, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def _read_json(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            if not raw_length:
                raise ApiError(400, "missing JSON body")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ApiError(400, "invalid Content-Length") from exc
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ApiError(400, "request body must be valid JSON") from exc
            if not isinstance(payload, dict):
                raise ApiError(400, "request body must be a JSON object")
            return payload

    return Handler


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local text embeddings.")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    service = EmbeddingService(model_name=args.model, device=args.device, batch_size=args.batch_size)
    handler = make_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    def stop_server(signum: int, _frame: Any) -> None:
        print(f"Received signal {signum}; shutting down.", file=sys.stderr)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)

    print(
        f"Serving text embeddings on http://{args.host}:{args.port}/v1 "
        f"(model={args.model}, device={args.device})",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
