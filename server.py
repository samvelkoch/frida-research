"""OpenAI-compatible HTTP wrapper around FRIDA-Decisions (torch backend).

Endpoints:
    GET  /health                -> 200 when the model is loaded and warmed up, 503 before
    GET  /v1/models             -> one model
    POST /v1/chat/completions   -> last user message = JSON request {state, questions};
                                   choices[0].message.content = JSON {"answers": ...}
    POST /judge                 -> native request in, native Judge response out
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from frida_decisions import Judge, RequestError

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/FRIDA-Decisions")
MODEL_ID = os.environ.get("MODEL_ID", "frida-decisions")
DEVICE = os.environ.get("DEVICE") or None          # cuda / cpu, default: cuda when available
STATE_MAX = int(os.environ.get("STATE_MAX", "384"))
STATE_CACHE_MB = int(os.environ.get("STATE_CACHE_MB", "512"))
# What goes into message.content: "answers" (default) or "full" (answers + margins + usage).
CONTENT_MODE = os.environ.get("CONTENT_MODE", "answers")

log = logging.getLogger("frida")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI(title="FRIDA-Decisions", version="0.2.0")
judge: Judge | None = None
ready = False
# Judge keeps a mutable state K/V cache and one GPU: one forward at a time.
lock = threading.Lock()

WARMUP = {
    "state": "Здравствуйте, у меня не приходит код подтверждения уже час.",
    "questions": {"topic": {"type": "choice", "instructions": "К какой теме относится обращение?",
                            "criteria": {"login": "вход в аккаунт", "payment": "оплата"}}},
}


def load() -> None:
    global judge, ready
    started = time.perf_counter()
    try:
        judge = Judge.from_pretrained(MODEL_DIR, device=DEVICE, state_max=STATE_MAX,
                                      state_cache_mb=STATE_CACHE_MB)
        judge.judge(WARMUP)
    except Exception:
        # Crash the container instead of answering 503 forever: let the orchestrator restart it.
        log.exception("model failed to load")
        os._exit(1)
    ready = True
    log.info("model loaded on %s (%s) in %.1fs", judge.device, judge.dtype,
             time.perf_counter() - started)


@app.on_event("startup")
async def startup() -> None:
    # Load in the background so /health answers 503 instead of hanging while loading.
    threading.Thread(target=load, daemon=True).start()


def run_judge(request: dict) -> dict:
    with lock:
        return judge.judge(request)


def openai_error(message: str, status: int = 400, kind: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": kind, "param": param, "code": code}})


def not_ready() -> JSONResponse:
    return openai_error("Model is loading", 503, "service_unavailable", code="model_not_ready")


# ------------------------------------------------------------------ service
@app.get("/health")
async def health():
    if not ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "model": MODEL_ID, "device": str(judge.device)}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "ai-forever"}]}


@app.get("/v1/models/{model_id}")
async def model(model_id: str):
    if model_id != MODEL_ID:
        return openai_error(f"Model {model_id} not found", 404, code="model_not_found")
    return {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "ai-forever"}


# ------------------------------------------------------------------ native
@app.post("/judge")
async def native(request: Request):
    if not ready:
        return not_ready()
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse(status_code=400, content={
            "error": {"type": "invalid_request", "message": "Body is not valid JSON"}})
    if isinstance(body, dict):
        body.pop("model", None)
    try:
        return await asyncio.to_thread(run_judge, body)
    except RequestError as error:
        return JSONResponse(status_code=error.status, content=error.payload())


# ------------------------------------------------------------------ OpenAI
FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def message_text(content) -> str:
    """content is a string or a list of parts [{"type": "text", "text": ...}]."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return ""


def extract_request(body: dict) -> dict:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list", "messages")
    user = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    if not user:
        raise RequestError("No user message", "messages")
    text = message_text(user[-1].get("content"))
    match = FENCE.match(text)
    if match:
        text = match.group(1)
    try:
        request = json.loads(text)
    except ValueError:
        raise RequestError('Last user message must be a JSON object {"state": ..., "questions": {...}}',
                           "messages") from None
    if isinstance(request, dict):
        request.pop("model", None)
    return request


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not ready:
        return not_ready()
    try:
        body = await request.json()
    except ValueError:
        return openai_error("Body is not valid JSON")
    if not isinstance(body, dict):
        return openai_error("Body must be a JSON object")
    # temperature, max_tokens, response_format, ... are accepted and ignored.
    try:
        result = await asyncio.to_thread(run_judge, extract_request(body))
    except RequestError as error:
        return openai_error(error.message, error.status, param=error.field, code=error.kind)

    payload = result if CONTENT_MODE == "full" else {"answers": result["answers"]}
    content = json.dumps(payload, ensure_ascii=False)
    usage = result["usage"]
    prompt_tokens = int(usage.get("encoder_tokens", usage.get("state_tokens", 0)))
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_name = body.get("model") or MODEL_ID

    if body.get("stream"):
        def events():
            chunk = {"id": response_id, "object": "chat.completion.chunk", "created": created,
                     "model": model_name, "choices": [{"index": 0, "delta": {
                         "role": "assistant", "content": content}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            chunk["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": 0,
                              "total_tokens": prompt_tokens}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0,
                  "total_tokens": prompt_tokens},
    }
