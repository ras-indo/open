from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock, Thread
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field as PF, ValidationError

# =============================================================================
# CONFIG
# =============================================================================

UPSTREAM_AUTH_TOKEN: str = os.getenv(
    "UPSTREAM_AUTH_TOKEN",
    "JTrrYZ76Jm9KGnci1oyPU1o3wV4sZbzqj4yxVIMfWgvXyef6L1w8lAh6rqst4YoB",
)
UPSTREAM_BASE_URL: str = os.getenv("UPSTREAM_BASE_URL", "http://100.83.239.114:5001/v1").rstrip("/")
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-3")
DEBUG: bool = os.getenv("DEBUG", "true").lower() in {"1", "true", "yes"}

MUTEX_TTL_S: float = float(os.getenv("MUTEX_TTL_S", "60"))
CTX_MAX_CHARS: int = int(os.getenv("CTX_MAX_CHARS", "48000"))
CTX_KEEP_RECENT: int = int(os.getenv("CTX_KEEP_RECENT", "12"))

SESSION_BACKEND: str = os.getenv("SESSION_BACKEND", "sqlite").lower()  # sqlite|memory
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 24 * 7)))
SQLITE_PATH: str = os.getenv("SQLITE_PATH", "./ocsn_sessions.db")

THINK_ENABLED: bool = os.getenv("THINK_ENABLED", "true").lower() in {"1", "true", "yes"}
THINK_MAX_TOKENS: int = int(os.getenv("THINK_MAX_TOKENS", "900"))
THINK_TEMP_HIGH: float = float(os.getenv("THINK_TEMP_HIGH", "0.8"))
THINK_TEMP_LOW: float = float(os.getenv("THINK_TEMP_LOW", "0.25"))
GOT_MAX_STEPS: int = int(os.getenv("GOT_MAX_STEPS", "3"))
GOT_BRANCHES: int = int(os.getenv("GOT_BRANCHES", "3"))
GOT_ENABLED: bool = os.getenv("GOT_ENABLED", "true").lower() in {"1", "true", "yes"}

MAX_UPSTREAM_PARALLEL: int = int(os.getenv("MAX_UPSTREAM_PARALLEL", "1"))

# =============================================================================
# LOGGING
# =============================================================================

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

USE_COLORS = sys.stdout.isatty()
RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if USE_COLORS:
            level_map = {
                "DEBUG": CYAN,
                "INFO": GREEN,
                "WARNING": YELLOW,
                "ERROR": RED,
                "CRITICAL": RED,
            }
            color = level_map.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname:8}{RESET}"
            record.name = f"{MAGENTA}{record.name}{RESET}"
        return super().format(record)


root = logging.getLogger()
root.setLevel(logging.DEBUG if DEBUG else logging.INFO)
root.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter("%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s", "%H:%M:%S"))
root.addHandler(handler)

L_GW = logging.getLogger("ocsn.gateway")
L_HTTP = logging.getLogger("ocsn.http")
L_CTX = logging.getLogger("ocsn.ctx")
L_GOT = logging.getLogger("ocsn.got")

# =============================================================================
# MODELS
# =============================================================================


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    model_config = ConfigDict(extra="allow")

    @property
    def text_content(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        bits: List[str] = []
        for p in self.content:
            if isinstance(p, dict) and p.get("type") == "text":
                bits.append(str(p.get("text", "")))
            elif isinstance(p, dict) and p.get("type") == "image_url":
                iu = p.get("image_url", {})
                if isinstance(iu, dict) and iu.get("url"):
                    bits.append(f"[image:{iu['url']}]")
        return "\n".join(bits)


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="allow")


class Tool(BaseModel):
    type: str = "function"
    function: FunctionDefinition
    model_config = ConfigDict(extra="allow")


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False
    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"
    stream_options: Optional[StreamOptions] = None
    user: Optional[str] = None

    # compatibility fields
    store: Optional[bool] = None
    reasoning_effort: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class TaskType(str, Enum):
    SIMPLE = "simple"
    CHAT = "chat"
    TOOL = "tool"
    COMPLEX = "complex"


@dataclass
class HoloCtx:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    spr_memory: str = ""
    summary: str = ""
    task_type: TaskType = TaskType.CHAT


# =============================================================================
# SESSION STORE
# =============================================================================


class InMemorySessionStore:
    def __init__(self, ttl: int):
        self.ttl = ttl
        self._lock = RLock()
        self._data: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        Thread(target=self._cleanup_loop, daemon=True).start()

    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(300)
            now = time.time()
            with self._lock:
                keys = [k for k, (exp, _) in self._data.items() if exp < now]
                for k in keys:
                    self._data.pop(k, None)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            exp, payload = item
            if exp < time.time():
                self._data.pop(key, None)
                return None
            self._data[key] = (time.time() + self.ttl, payload)
            return dict(payload)

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = (time.time() + self.ttl, dict(payload))


class SQLiteSessionStore:
    def __init__(self, path: str, ttl: int):
        self.path = path
        self.ttl = ttl
        self._lock = RLock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    k TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    expire_at REAL NOT NULL
                )
                """
            )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as c:
            now = time.time()
            c.execute("DELETE FROM sessions WHERE expire_at < ?", (now,))
            row = c.execute("SELECT payload FROM sessions WHERE k = ?", (key,)).fetchone()
            if not row:
                return None
            c.execute("UPDATE sessions SET expire_at = ? WHERE k = ?", (now + self.ttl, key))
            return json.loads(row[0])

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO sessions(k, payload, expire_at) VALUES(?, ?, ?) "
                "ON CONFLICT(k) DO UPDATE SET payload = excluded.payload, expire_at = excluded.expire_at",
                (key, json.dumps(payload, ensure_ascii=False), time.time() + self.ttl),
            )


SESSION = SQLiteSessionStore(SQLITE_PATH, SESSION_TTL_SECONDS) if SESSION_BACKEND == "sqlite" else InMemorySessionStore(SESSION_TTL_SECONDS)

# =============================================================================
# GLOBAL REQUEST GATE (upstream only 1 active request)
# =============================================================================

UPSTREAM_GATE = asyncio.Semaphore(MAX_UPSTREAM_PARALLEL)


# =============================================================================
# HELPERS
# =============================================================================

def _extract_cid(req: ChatCompletionRequest) -> str:
    if req.user:
        return req.user
    seed = req.messages[0].text_content if req.messages else "anon"
    return hashlib.md5(seed.encode()).hexdigest()[:16]


def _msg_to_dict(msg: ChatMessage) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        out["content"] = msg.content
    if msg.tool_calls:
        out["tool_calls"] = msg.tool_calls
    if msg.tool_call_id:
        out["tool_call_id"] = msg.tool_call_id
    if msg.name:
        out["name"] = msg.name
    return out


def _char_count(msgs: List[Dict[str, Any]]) -> int:
    n = 0
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            n += len(json.dumps(c, ensure_ascii=False))
    return n


def _spr(msgs: List[Dict[str, Any]]) -> str:
    buf: List[str] = []
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, str):
            s = content[:120]
        else:
            s = json.dumps(content, ensure_ascii=False)[:120]
        buf.append(f"[{role}:{s}]")
    return " | ".join(buf)


def _flatten_content_to_markdown(content: Union[str, List[Dict[str, Any]], None]) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        t = part.get("type")
        if t == "text":
            parts.append(str(part.get("text", "")))
        elif t == "image_url":
            iu = part.get("image_url", {})
            if isinstance(iu, dict) and iu.get("url"):
                parts.append(f"![image]({iu['url']})")
    return "\n\n".join(p for p in parts if p)


def _normalize_payload_to_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    fixed = json.loads(json.dumps(payload, ensure_ascii=False))
    msgs = fixed.get("messages", [])
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and "content" in m:
                m["content"] = _flatten_content_to_markdown(m.get("content"))
    return fixed


def _is_content_error(status: int, body: str) -> bool:
    if status != 400:
        return False
    low = body.lower()
    needles = [
        "content must be a string",
        "invalid content format",
        "expected string",
        "content array not supported",
        "content should be a string",
    ]
    return any(n in low for n in needles)


class UpstreamError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


_HTTP: Optional[httpx.AsyncClient] = None


async def _http() -> httpx.AsyncClient:
    global _HTTP
    if _HTTP is None or _HTTP.is_closed:
        _HTTP = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=20.0, read=300.0, write=60.0, pool=30.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            headers={
                "Authorization": f"Bearer {UPSTREAM_AUTH_TOKEN}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
    return _HTTP


def _build_passthrough_payload(req: ChatCompletionRequest, messages: Optional[List[Dict[str, Any]]] = None, stream: Optional[bool] = None) -> Dict[str, Any]:
    p = req.model_dump(exclude_none=True)
    p["model"] = req.model or DEFAULT_MODEL
    p["messages"] = messages if messages is not None else [_msg_to_dict(m) for m in req.messages]
    if stream is not None:
        p["stream"] = stream
    return p


async def _call_sync(payload: Dict[str, Any], rid: str, phase: str, allow_text_fallback: bool = True) -> Dict[str, Any]:
    client = await _http()
    body = json.dumps(payload, ensure_ascii=False).encode()
    L_HTTP.info("SYNC -> rid=%s phase=%s model=%s %.1fkB", rid, phase, payload.get("model"), len(body) / 1024)

    async with UPSTREAM_GATE:
        try:
            resp = await client.post(f"{UPSTREAM_BASE_URL}/chat/completions", content=body)
        except httpx.TimeoutException:
            raise HTTPException(504, detail=f"Upstream timeout ({phase})")
        except Exception as exc:
            raise HTTPException(502, detail=f"Upstream transport error: {exc}")

    if resp.status_code != 200:
        if allow_text_fallback and _is_content_error(resp.status_code, resp.text):
            L_HTTP.warning("Fallback text-only rid=%s phase=%s", rid, phase)
            return await _call_sync(_normalize_payload_to_text(payload), rid, f"{phase}:text", False)
        raise HTTPException(resp.status_code, detail=resp.text[:400])

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(502, detail="Upstream non-JSON response")

    if not isinstance(data, dict) or "choices" not in data:
        if allow_text_fallback:
            L_HTTP.warning("Fallback malformed-response rid=%s phase=%s", rid, phase)
            return await _call_sync(_normalize_payload_to_text(payload), rid, f"{phase}:shape", False)
        raise HTTPException(502, detail="Upstream malformed completion response")

    return data


async def _call_stream(payload: Dict[str, Any], rid: str, phase: str, allow_text_fallback: bool = True) -> AsyncGenerator[Dict[str, Any], None]:
    client = await _http()
    body = json.dumps(payload, ensure_ascii=False).encode()
    L_HTTP.info("STREAM -> rid=%s phase=%s model=%s %.1fkB", rid, phase, payload.get("model"), len(body) / 1024)

    async with UPSTREAM_GATE:
        try:
            async with client.stream(
                "POST",
                f"{UPSTREAM_BASE_URL}/chat/completions",
                content=body,
                headers={"Accept": "text/event-stream"},
                timeout=300.0,
            ) as resp:
                if resp.status_code != 200:
                    err = (await resp.aread()).decode(errors="replace")
                    if allow_text_fallback and _is_content_error(resp.status_code, err):
                        L_HTTP.warning("Fallback text-only(stream) rid=%s phase=%s", rid, phase)
                        async for ch in _call_stream(_normalize_payload_to_text(payload), rid, f"{phase}:text", False):
                            yield ch
                        return
                    raise UpstreamError(resp.status_code, err[:400])

                async for raw in resp.aiter_lines():
                    line = raw.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
        except UpstreamError:
            raise
        except httpx.TimeoutException:
            raise HTTPException(504, detail=f"Upstream stream timeout ({phase})")
        except Exception as exc:
            raise HTTPException(502, detail=f"Upstream stream error: {exc}")


# =============================================================================
# GoT (single-model, sequential)
# =============================================================================

GOT_GENERATE_PROMPT = """You are GoT-Generator. Generate concise candidate reasoning paths.
Return JSON:
{"candidates":[{"id":"c1","thought":"..."}]}
Question: {question}
"""

GOT_SCORE_PROMPT = """You are GoT-Scorer. Score candidate thoughts from 0..100.
Return JSON:
{"scores":[{"id":"c1","score":87,"why":"..."}]}
Question: {question}
Candidates: {candidates}
"""

GOT_IMPROVE_PROMPT = """You are GoT-Improver. Merge best candidates into refined reasoning plan.
Return plain text reasoning plan in user's language.
Question: {question}
Top candidates: {top}
"""


def _extract_last_user(req: ChatCompletionRequest) -> str:
    for m in reversed(req.messages):
        if m.role == "user":
            return m.text_content[:5000]
    return ""


async def got_reasoning(req: ChatCompletionRequest, rid: str) -> str:
    if not GOT_ENABLED:
        return ""
    question = _extract_last_user(req)
    if not question:
        return ""

    # step 1: generate branches
    gen_req = ChatCompletionRequest(
        model=req.model,
        messages=[
            ChatMessage(role="system", content="You are a strict JSON generator."),
            ChatMessage(role="user", content=GOT_GENERATE_PROMPT.format(question=question)),
        ],
        stream=False,
        temperature=THINK_TEMP_HIGH,
        max_completion_tokens=min(THINK_MAX_TOKENS, 500),
    )
    gen_data = await _call_sync(_build_passthrough_payload(gen_req), rid, "got_generate")
    gen_text = ((gen_data.get("choices") or [{}])[0].get("message") or {}).get("content", "")

    try:
        parsed = json.loads(re.sub(r"```(?:json)?|```", "", gen_text).strip())
        cands = parsed.get("candidates", [])[: max(1, GOT_BRANCHES)]
    except Exception:
        cands = [{"id": "c1", "thought": gen_text[:600]}]

    # step 2: score
    score_req = ChatCompletionRequest(
        model=req.model,
        messages=[
            ChatMessage(role="system", content="You are a strict JSON scorer."),
            ChatMessage(role="user", content=GOT_SCORE_PROMPT.format(question=question, candidates=json.dumps(cands, ensure_ascii=False))),
        ],
        stream=False,
        temperature=0.2,
        max_completion_tokens=min(THINK_MAX_TOKENS, 350),
    )
    score_data = await _call_sync(_build_passthrough_payload(score_req), rid, "got_score")
    score_text = ((score_data.get("choices") or [{}])[0].get("message") or {}).get("content", "")

    try:
        score_json = json.loads(re.sub(r"```(?:json)?|```", "", score_text).strip())
        scores = {x.get("id"): x.get("score", 0) for x in score_json.get("scores", []) if isinstance(x, dict)}
    except Exception:
        scores = {}

    ranked = sorted(cands, key=lambda x: float(scores.get(x.get("id"), 0)), reverse=True)
    top = ranked[:2] if ranked else cands[:1]

    # step 3: improve/aggregate
    improve_req = ChatCompletionRequest(
        model=req.model,
        messages=[
            ChatMessage(role="system", content="You produce concise high-quality reasoning."),
            ChatMessage(role="user", content=GOT_IMPROVE_PROMPT.format(question=question, top=json.dumps(top, ensure_ascii=False))),
        ],
        stream=False,
        temperature=THINK_TEMP_LOW,
        max_completion_tokens=min(THINK_MAX_TOKENS, 500),
    )
    improve_data = await _call_sync(_build_passthrough_payload(improve_req), rid, "got_improve")
    return ((improve_data.get("choices") or [{}])[0].get("message") or {}).get("content", "")[:4000]


# =============================================================================
# STREAM MAPPER
# =============================================================================


def _sse_chunk(cid: str, model: str, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


# =============================================================================
# CONTEXT BUILD
# =============================================================================


async def build_ctx(req: ChatCompletionRequest, cid: str) -> HoloCtx:
    data = SESSION.get(f"session:{cid}")
    if data:
        ctx = HoloCtx(**data)
    else:
        ctx = HoloCtx()

    ctx.messages.extend([_msg_to_dict(m) for m in req.messages])

    if _char_count(ctx.messages) > CTX_MAX_CHARS:
        old = ctx.messages[:-CTX_KEEP_RECENT]
        ctx.messages = ctx.messages[-CTX_KEEP_RECENT:]
        if old:
            ctx.spr_memory = _spr(old)
    return ctx


def _classify(req: ChatCompletionRequest) -> TaskType:
    if req.tools:
        return TaskType.TOOL
    last = _extract_last_user(req).lower()
    if re.match(r"^(hi|halo|hello|thanks|ok)[\W_]*$", last):
        return TaskType.SIMPLE
    if len(last) > 200:
        return TaskType.COMPLEX
    return TaskType.CHAT


# =============================================================================
# PIPELINES
# =============================================================================


async def handle_sync(req: ChatCompletionRequest, rid: str, cid: str, client: str) -> Dict[str, Any]:
    L_GW.info("REQ sync rid=%s cid=%s client=%s", rid, cid, client)
    ctx = await build_ctx(req, cid)
    ctx.task_type = _classify(req)

    reasoning = ""
    if THINK_ENABLED and ctx.task_type != TaskType.SIMPLE:
        reasoning = await got_reasoning(req, rid)

    messages = list(ctx.messages)
    if reasoning:
        messages = [{"role": "system", "content": f"Internal reasoning context (do not expose raw unless needed):\n{reasoning}"}] + messages

    payload = _build_passthrough_payload(req, messages=messages, stream=False)
    data = await _call_sync(payload, rid, "act")

    SESSION.set(
        f"session:{cid}",
        {
            "messages": ctx.messages,
            "spr_memory": ctx.spr_memory,
            "summary": ctx.summary,
            "task_type": ctx.task_type.value,
        },
    )
    return data


async def handle_stream(req: ChatCompletionRequest, rid: str, cid: str, client: str) -> AsyncGenerator[str, None]:
    L_GW.info("REQ stream rid=%s cid=%s client=%s", rid, cid, client)
    ctx = await build_ctx(req, cid)
    ctx.task_type = _classify(req)

    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    yield _sse_chunk(chunk_id, req.model, {"role": "assistant"})

    reasoning = ""
    if THINK_ENABLED and ctx.task_type != TaskType.SIMPLE:
        try:
            reasoning = await got_reasoning(req, rid)
            if reasoning:
                for piece in re.findall(r".{1,220}(?:\s+|$)", reasoning):
                    piece = piece.strip()
                    if piece:
                        yield _sse_chunk(chunk_id, req.model, {"reasoning_content": piece})
        except Exception as exc:
            L_GOT.warning("GoT skipped rid=%s err=%s", rid, exc)

    messages = list(ctx.messages)
    if reasoning:
        messages = [{"role": "system", "content": f"Internal reasoning context:\n{reasoning}"}] + messages

    payload = _build_passthrough_payload(req, messages=messages, stream=True)

    finish_reason = "stop"
    try:
        async for chunk in _call_stream(payload, rid, "act_stream"):
            choices = chunk.get("choices", [])
            if not choices:
                if req.stream_options and req.stream_options.include_usage and "usage" in chunk:
                    yield "data: " + json.dumps(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [],
                            "usage": chunk["usage"],
                        },
                        ensure_ascii=False,
                    ) + "\n\n"
                continue

            c0 = choices[0]
            delta = c0.get("delta", {})
            fr = c0.get("finish_reason")
            if fr:
                finish_reason = fr

            out_delta: Dict[str, Any] = {}
            if "content" in delta and delta["content"] is not None:
                out_delta["content"] = delta["content"]
            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                out_delta["reasoning_content"] = delta["reasoning_content"]
            if "tool_calls" in delta and delta["tool_calls"] is not None:
                out_delta["tool_calls"] = delta["tool_calls"]

            if out_delta:
                yield _sse_chunk(chunk_id, req.model, out_delta)

    except UpstreamError as exc:
        msg = f"⚠️ Upstream error {exc.status_code}: {exc.detail}"
        yield _sse_chunk(chunk_id, req.model, {"content": msg}, finish_reason="error")
        yield "data: [DONE]\n\n"
        return

    SESSION.set(
        f"session:{cid}",
        {
            "messages": ctx.messages,
            "spr_memory": ctx.spr_memory,
            "summary": ctx.summary,
            "task_type": ctx.task_type.value,
        },
    )

    yield _sse_chunk(chunk_id, req.model, {}, finish_reason=finish_reason)
    if req.stream_options and req.stream_options.include_usage:
        yield "data: " + json.dumps(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
            ensure_ascii=False,
        ) + "\n\n"
    yield "data: [DONE]\n\n"


# =============================================================================
# FASTAPI
# =============================================================================


@asynccontextmanager
async def lifespan(_: FastAPI):
    L_GW.info("OCSN v2 starting")
    yield
    global _HTTP
    if _HTTP and not _HTTP.is_closed:
        await _HTTP.aclose()
    L_GW.info("OCSN v2 stopped")


app = FastAPI(title="OCSN v2 Cognitive Proxy", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    body = None
    if DEBUG:
        try:
            body = (await request.body()).decode(errors="replace")[:1200]
        except Exception:
            body = "<unreadable>"
    L_GW.error("422 %s body=%s", exc.errors(), body)
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": body if DEBUG else None})


@app.exception_handler(ValidationError)
async def pydantic_handler(_: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
async def http_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def general_handler(_: Request, exc: Exception):
    L_GW.error("Unhandled: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "OCSN v2 Cognitive Proxy",
        "version": "2.0.0",
        "upstream": UPSTREAM_BASE_URL,
        "features": {
            "payload_passthrough": True,
            "text_fallback": True,
            "global_upstream_gate": MAX_UPSTREAM_PARALLEL,
            "got_reasoning": GOT_ENABLED,
            "session_backend": SESSION_BACKEND,
        },
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "2.0.0",
        "time": datetime.now(timezone.utc).isoformat(),
        "upstream": UPSTREAM_BASE_URL,
        "session_backend": SESSION_BACKEND,
        "max_upstream_parallel": MAX_UPSTREAM_PARALLEL,
    }


@app.get("/v1/models")
async def models() -> Dict[str, Any]:
    c = await _http()
    async with UPSTREAM_GATE:
        r = await c.get(f"{UPSTREAM_BASE_URL}/models", timeout=20)
    if r.status_code != 200:
        raise HTTPException(r.status_code, detail=r.text[:300])
    data = r.json()
    if isinstance(data, list):
        return {"object": "list", "data": data}
    if isinstance(data, dict):
        return data
    return {"object": "list", "data": []}


@app.post("/v1/chat/completions")
async def chat(req: ChatCompletionRequest, raw: Request):
    if req.n not in (None, 1):
        raise HTTPException(400, detail="Only n=1 supported")

    rid = f"req-{uuid.uuid4().hex[:10]}"
    cid = _extract_cid(req)
    client = raw.client.host if raw.client else "unknown"

    if DEBUG:
        body = await raw.body()
        L_GW.debug("rid=%s raw=%s", rid, body.decode(errors="replace")[:500])

    if req.stream:
        return StreamingResponse(
            handle_stream(req, rid, cid, client),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": rid,
                "X-Conv-ID": cid,
            },
        )

    return await handle_sync(req, rid, cid, client)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
