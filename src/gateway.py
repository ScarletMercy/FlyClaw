from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger("myclaw.gateway")


class TokenBucketRateLimiter:
    """Simple in-memory token bucket rate limiter."""

    def __init__(self, rate: float = 10.0, capacity: int = 20):
        """Initialize rate limiter.

        Args:
            rate: Tokens added per second
            capacity: Maximum token capacity
        """
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def _get_tokens(self, key: str) -> float:
        async with self._lock:
            now = time.time()
            if key not in self._buckets:
                self._buckets[key] = {"tokens": self.capacity - 1.0, "last_update": now}
                return self.capacity - 1.0

            bucket = self._buckets[key]
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
            bucket["last_update"] = now

            if bucket["tokens"] < 1.0:
                return bucket["tokens"]

            bucket["tokens"] -= 1.0
            return bucket["tokens"]

    async def acquire(self, key: str) -> bool:
        """Try to acquire a token. Returns True if successful."""
        tokens = await self._get_tokens(key)
        return tokens >= 0.0

    async def wait_time(self, key: str) -> float:
        """Return seconds until next token is available."""
        async with self._lock:
            if key not in self._buckets:
                return 0.0
            bucket = self._buckets[key]
            if bucket["tokens"] >= 1.0:
                return 0.0
            return (1.0 - bucket["tokens"]) / self.rate


_rate_limiter: Optional[TokenBucketRateLimiter] = None


def create_gateway(app_config, compiled_graph, feishu_channel=None, cron_service=None):
    global _rate_limiter

    # Initialize rate limiter from config if available
    rate = getattr(app_config.gateway, "rate_limit", 10.0) if hasattr(app_config, "gateway") else 10.0
    capacity = getattr(app_config.gateway, "rate_limit_burst", 20) if hasattr(app_config, "gateway") else 20
    _rate_limiter = TokenBucketRateLimiter(rate=rate, capacity=capacity)

    app = FastAPI(title="MyClaw", version="0.1.0")

    # Add CORS middleware
    cors_origins = getattr(app_config.gateway, "cors_origins", []) or []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=bool(cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": "0.1.0", "ts": int(time.time())}

    @app.get("/readyz")
    async def readyz():
        return {"status": "ready"}

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request):
        # Validate Content-Type
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            return JSONResponse(
                {"error": "unsupported_media_type", "message": "Content-Type must be application/json"},
                status_code=415,
            )

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                {"error": "invalid_json", "message": "Request body is not valid JSON"},
                status_code=400,
            )

        # Auth check before rate limiting
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if app_config.gateway.auth_token and not hmac.compare_digest(token, app_config.gateway.auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        # Apply rate limiting only after auth
        client_ip = request.client.host if request.client else "unknown"
        if not await _rate_limiter.acquire(client_ip):
            wait = await _rate_limiter.wait_time(client_ip)
            return JSONResponse(
                {"error": "rate_limit_exceeded", "retry_after": wait},
                status_code=429,
                headers={"Retry-After": str(int(wait))},
            )

        messages = body.get("messages", [])
        model_name = body.get("model", "myclaw")
        stream = body.get("stream", False)
        thread_id = body.get("user", str(uuid.uuid4()))

        config = {"configurable": {"thread_id": thread_id}}

        # Use the factory function for state creation
        from src.graph import create_agent_state

        # For the OpenAI API, we need to preserve the message history
        # The create_agent_state factory creates a new HumanMessage, but we have the full history
        # So we build the state manually here to preserve the message list
        lc_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        input_state = create_agent_state(
            sender_id="openai-api",
            chat_id="openai-api",
            message_text=lc_messages[-1].content if lc_messages else "",
            chat_type="p2p",
            message_id=str(uuid.uuid4()),
            system_prompt=app_config.agents.system_prompt,
        )
        # Override messages with the full history for this endpoint
        input_state["messages"] = lc_messages

        if stream:
            return StreamingResponse(
                _stream_response(compiled_graph, input_state, config),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result = await compiled_graph.ainvoke(input_state, config)
        except Exception as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "server_error"}},
                status_code=500,
            )
        assistant_text = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                assistant_text = msg.content
                break

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def _stream_response(graph, input_state, config):
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'myclaw', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

        collected_text = []
        try:
            async for event in graph.astream_events(input_state, config, version="v2"):
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "text") and chunk.text:
                        collected_text.append(chunk.text)
                        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'myclaw', 'choices': [{'index': 0, 'delta': {'content': chunk.text}, 'finish_reason': None}]})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'myclaw', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop', 'error': str(e)}]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'myclaw', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept(max_size=1024 * 1024)

        auth_token = app_config.gateway.auth_token

        if auth_token:
            nonce = secrets.token_hex(16)
            try:
                await ws.send_json({"type": "auth_challenge", "nonce": nonce})
                raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
                resp = json.loads(raw)
                if resp.get("type") != "auth_response":
                    await ws.close(code=4001, reason="expected auth_response")
                    return
                client_mac = resp.get("mac", "")
                expected_mac = hmac.new(
                    auth_token.encode(), nonce.encode(), hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(client_mac, expected_mac):
                    await ws.send_json({"type": "auth_error", "message": "invalid mac"})
                    await ws.close(code=4003, reason="auth failed")
                    return
                await ws.send_json({"type": "auth_ok"})
            except asyncio.TimeoutError:
                await ws.close(code=4002, reason="auth timeout")
                return
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("WS auth error: %s", e)
                await ws.close(code=4001, reason="auth error")
                return

        logger.info("WS client connected (auth=%s)", "yes" if auth_token else "skip")

        try:
            reconnect_delay = 1.0
            max_reconnect_delay = 30.0

            while True:
                try:
                    raw = await ws.receive_text()
                    # Reset reconnect delay on successful message
                    reconnect_delay = 1.0

                    try:
                        frame = json.loads(raw)
                    except json.JSONDecodeError:
                        await ws.send_json(
                            {"type": "res", "ok": False, "error": {"code": "INVALID_JSON"}}
                        )
                        continue

                    frame_type = frame.get("type")
                    if frame_type == "req":
                        await _handle_ws_request(ws, frame, app_config, compiled_graph)
                    else:
                        await ws.send_json(
                            {"type": "res", "ok": False, "error": {"code": "UNKNOWN_FRAME"}}
                        )
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    logger.warning("WS error: %s", e)
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
        except WebSocketDisconnect:
            pass
        finally:
            logger.info("WS client disconnected")

    async def _handle_ws_request(ws, frame, app_config, graph):
        method = frame.get("method", "")
        params = frame.get("params", {})
        frame_id = frame.get("id", "")

        if method == "ping":
            await ws.send_json({"type": "res", "id": frame_id, "ok": True, "payload": "pong"})
        elif method == "chat.send":
            from src.graph import create_agent_state

            text = params.get("text", "")
            thread_id = params.get("thread_id", "ws-default")
            config = {"configurable": {"thread_id": thread_id}}
            input_state = create_agent_state(
                sender_id="ws",
                chat_id="ws",
                message_text=text,
                chat_type="p2p",
                message_id=str(uuid.uuid4()),
                system_prompt=app_config.agents.system_prompt,
            )
            result = await graph.ainvoke(input_state, config)
            assistant_text = ""
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    assistant_text = msg.content
                    break
            await ws.send_json(
                {"type": "res", "id": frame_id, "ok": True, "payload": {"text": assistant_text}}
            )
        elif method == "sessions.list":
            await ws.send_json({"type": "res", "id": frame_id, "ok": True, "payload": []})
        elif method == "health":
            await ws.send_json(
                {"type": "res", "id": frame_id, "ok": True, "payload": {"status": "ok"}}
            )
        elif method == "cron.list" and cron_service:
            jobs = cron_service.list_jobs()
            await ws.send_json(
                {
                    "type": "res",
                    "id": frame_id,
                    "ok": True,
                    "payload": [j.model_dump() for j in jobs],
                }
            )
        elif method == "cron.add" and cron_service:
            try:
                from src.cron.types import CronJobCreate

                job = await cron_service.add_job(CronJobCreate(**params))
                await ws.send_json(
                    {"type": "res", "id": frame_id, "ok": True, "payload": job.model_dump()}
                )
            except Exception as e:
                await ws.send_json(
                    {"type": "res", "id": frame_id, "ok": False, "error": {"code": "INVALID_PARAMS", "message": str(e)}}
                )
        elif method == "cron.run" and cron_service:
            result = await cron_service.run_job_now(params.get("job_id", ""))
            await ws.send_json(
                {
                    "type": "res",
                    "id": frame_id,
                    "ok": True,
                    "payload": result.model_dump() if result else None,
                }
            )
        else:
            await ws.send_json(
                {
                    "type": "res",
                    "id": frame_id,
                    "ok": False,
                    "error": {"code": "UNKNOWN_METHOD", "message": method},
                }
            )

    async def _verify_api_auth(request: Request):
        if not app_config.gateway.auth_token:
            return
        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token, app_config.gateway.auth_token):
            raise HTTPException(status_code=401, detail="unauthorized")

    if cron_service:
        from src.cron.types import CronJobCreate, CronJobPatch

        @app.get("/api/cron/status")
        async def cron_status(request: Request):
            await _verify_api_auth(request)
            return cron_service.status()

        @app.get("/api/cron/jobs")
        async def cron_list_jobs(request: Request):
            await _verify_api_auth(request)
            jobs = cron_service.list_jobs()
            return [j.model_dump() for j in jobs]

        @app.post("/api/cron/jobs")
        async def cron_add_job(request: Request):
            await _verify_api_auth(request)
            try:
                body = await request.json()
                job = await cron_service.add_job(CronJobCreate(**body))
                return job.model_dump()
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=422)

        @app.get("/api/cron/jobs/{job_id}")
        async def cron_get_job(job_id: str, request: Request):
            await _verify_api_auth(request)
            job = cron_service.get_job(job_id)
            if not job:
                return JSONResponse({"error": "not found"}, status_code=404)
            return job.model_dump()

        @app.patch("/api/cron/jobs/{job_id}")
        async def cron_update_job(job_id: str, request: Request):
            await _verify_api_auth(request)
            try:
                body = await request.json()
                job = await cron_service.update_job(job_id, CronJobPatch(**body))
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=422)
            if not job:
                return JSONResponse({"error": "not found"}, status_code=404)
            return job.model_dump()

        @app.delete("/api/cron/jobs/{job_id}")
        async def cron_delete_job(job_id: str, request: Request):
            await _verify_api_auth(request)
            removed = await cron_service.remove_job(job_id)
            return {"removed": removed}

        @app.post("/api/cron/jobs/{job_id}/run")
        async def cron_run_job(job_id: str, request: Request):
            await _verify_api_auth(request)
            result = await cron_service.run_job_now(job_id)
            if not result:
                return JSONResponse({"error": "not found"}, status_code=404)
            return result.model_dump()

    @app.post("/api/approval/{request_id}/resolve")
    async def resolve_approval(request_id: str, req: Request):
        await _verify_api_auth(req)
        from src.tools.approval import get_approval_manager

        body = await req.json()
        decision = body.get("decision", "deny")
        mgr = get_approval_manager()
        ok = mgr.resolve(request_id, decision)
        if not ok:
            return JSONResponse({"error": "not found or already resolved"}, status_code=404)
        return {"resolved": True, "request_id": request_id, "decision": decision}

    @app.get("/api/approval/pending")
    async def list_pending_approvals(request: Request):
        await _verify_api_auth(request)
        from src.tools.approval import get_approval_manager

        mgr = get_approval_manager()
        return [r.model_dump() for r in mgr.list_pending()]

    @app.get("/api/plugins")
    async def list_plugins(request: Request):
        await _verify_api_auth(request)
        try:
            from src.plugins.registry import get_plugin_registry

            return get_plugin_registry().list_plugins()
        except Exception:
            return []

    @app.get("/api/commands")
    async def list_commands(request: Request):
        await _verify_api_auth(request)
        try:
            from src.commands.dispatcher import get_dispatcher

            d = get_dispatcher()
            if d:
                return {"commands": d.list_commands()}
        except Exception:
            pass
        return {"commands": []}

    @app.post("/api/feishu/card-action")
    async def feishu_card_action(req: Request):
        await _verify_api_auth(req)
        body = await req.json()
        action = body.get("action", {})
        action_value = action.get("value", "")

        # Parse value: may be JSON {"request_id": ..., "decision": ...} or plain string
        request_id = ""
        decision = "deny"
        if isinstance(action_value, dict):
            request_id = action_value.get("request_id", "")
            decision = action_value.get("decision", "deny")
        elif isinstance(action_value, str):
            try:
                value_data = json.loads(action_value)
                if isinstance(value_data, dict):
                    request_id = value_data.get("request_id", "")
                    decision = value_data.get("decision", "deny")
                else:
                    request_id = action_value
            except (json.JSONDecodeError, TypeError):
                request_id = action_value

        from src.channels.cards import get_card_callback_registry

        callback_registry = get_card_callback_registry()

        callback = callback_registry.resolve(request_id) if request_id else None
        if callback:
            try:
                await callback(body)
                return {"success": True}
            except Exception as e:
                logger.error("Card callback error: %s", e)
                return {"success": False, "error": "internal error"}

        if request_id:
            from src.tools.approval import get_approval_manager

            mgr = get_approval_manager()
            pending = mgr.get_pending(request_id)
            if pending:
                if decision not in ("allow_once", "allow_always", "deny"):
                    decision = "deny"
                mgr.resolve(request_id, decision)
                return {"success": True}

        return {"success": False, "error": "unknown action"}

    return app
