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

from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.canvas.server import router as canvas_router

logger = logging.getLogger("flyclaw.gateway")

router = APIRouter()


_acp_sessions = None
_acp_runtime = None


def _get_acp_runtime():
    global _acp_sessions, _acp_runtime
    if _acp_runtime is None:
        from src.acp.session import AcpSessionManager
        from src.acp.runtime import AgentLoopRuntime

        _acp_sessions = AcpSessionManager()
        _acp_runtime = AgentLoopRuntime(_acp_sessions)
    return _acp_sessions, _acp_runtime


@router.websocket("/ws/acp")
async def acp_websocket(ws: WebSocket):
    await ws.accept()
    sessions, runtime = _get_acp_runtime()

    try:
        while True:
            raw = await ws.receive_json()
            method = raw.get("method", "")
            msg_id = raw.get("id")

            if method == "initialize":
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "0.2",
                        "agentCapabilities": {"streaming": True, "tools": True},
                        "configOptions": [],
                        "modes": ["default"],
                    },
                })
            elif method == "newSession":
                params = raw.get("params", {})
                sid = sessions.create("default", cwd=params.get("cwd", ""))
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"sessionId": sid, "configOptions": [], "modes": ["default"]},
                })
            elif method == "prompt":
                params = raw.get("params", {})
                sid = params.get("sessionId", "")
                content = params.get("content", [])
                prompt = " ".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
                stop_reason = "end_turn"
                async for event in runtime.run_turn(session_id=sid, prompt=prompt):
                    if event.type == "text_delta" and event.text:
                        await ws.send_json({
                            "jsonrpc": "2.0",
                            "method": "sessionUpdate",
                            "params": {
                                "sessionId": sid,
                                "update": {"type": "agent_message_chunk", "content": {"type": "text", "text": event.text}},
                            },
                        })
                    elif event.type == "done":
                        stop_reason = event.stop_reason or "end_turn"
                await ws.send_json({"jsonrpc": "2.0", "id": msg_id, "result": {"stopReason": stop_reason, "usage": {}}})
            elif method == "cancel":
                await runtime.cancel(raw.get("params", {}).get("sessionId", ""))
                await ws.send_json({"jsonrpc": "2.0", "id": msg_id, "result": {"cancelled": True}})
            elif method == "listSessions":
                s_list = sessions.list_sessions()
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"sessions": [{"sessionId": s.session_id, "agentId": s.agent_id} for s in s_list]},
                })
            else:
                await ws.send_json({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}})
    except WebSocketDisconnect:
        pass


class TokenBucketRateLimiter:
    def __init__(self, rate: float = 10.0, capacity: int = 20):
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str) -> bool:
        async with self._lock:
            now = time.time()
            if key not in self._buckets:
                self._buckets[key] = {"tokens": self.capacity - 1.0, "last_update": now}
                return True
            bucket = self._buckets[key]
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
            bucket["last_update"] = now
            if bucket["tokens"] < 1.0:
                return False
            bucket["tokens"] -= 1.0
            return True

    async def wait_time(self, key: str) -> float:
        async with self._lock:
            if key not in self._buckets:
                return 0.0
            bucket = self._buckets[key]
            now = time.time()
            elapsed = now - bucket["last_update"]
            available = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
            if available >= 1.0:
                return 0.0
            return (1.0 - available) / self.rate


_rate_limiter: Optional[TokenBucketRateLimiter] = None
_app_ref = None


def _get_app(request: Request):
    return _app_ref


def create_gateway(app_config, agent_loop, cron_service=None):
    global _rate_limiter
    rate = getattr(app_config.gateway, "rate_limit", 10.0) if hasattr(app_config, "gateway") else 10.0
    capacity = getattr(app_config.gateway, "rate_limit_burst", 20) if hasattr(app_config, "gateway") else 20
    _rate_limiter = TokenBucketRateLimiter(rate=rate, capacity=capacity)
    app = FastAPI(title="flyclaw", version="0.1.0")
    if getattr(app_config, "canvas", None) and app_config.canvas.enabled:
        app.include_router(canvas_router)
    app.include_router(router)
    cors_origins = getattr(app_config.gateway, "cors_origins", []) or []
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_credentials=bool(cors_origins), allow_methods=["*"], allow_headers=["*"])

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": "0.1.0", "ts": int(time.time())}

    @app.get("/readyz")
    async def readyz():
        return {"status": "ready"}

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request):
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            return JSONResponse({"error": "unsupported_media_type", "message": "Content-Type must be application/json"}, status_code=415)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid_json", "message": "Request body is not valid JSON"}, status_code=400)
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if app_config.gateway.auth_token and not hmac.compare_digest(token, app_config.gateway.auth_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        client_ip = request.client.host if request.client else "unknown"
        if not await _rate_limiter.acquire(client_ip):
            wait = await _rate_limiter.wait_time(client_ip)
            return JSONResponse({"error": "rate_limit_exceeded", "retry_after": wait}, status_code=429, headers={"Retry-After": str(int(wait))})
        messages = body.get("messages", [])
        model_name = body.get("model", "flyclaw")
        stream = body.get("stream", False)
        thread_id = body.get("user", str(uuid.uuid4()))
        from src.agent.state import AgentState
        chat_messages = []
        for m in messages:
            chat_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        input_state = AgentState(messages=chat_messages, system_prompt=app_config.agents.system_prompt, sender_id="openai-api", chat_id="openai-api", chat_type="p2p", message_id=str(uuid.uuid4()), channel="api")
        if stream:
            return StreamingResponse(_stream_response(agent_loop, input_state, thread_id), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        try:
            result_state = await agent_loop.run(input_state, thread_id)
        except Exception as e:
            return JSONResponse({"error": {"message": str(e), "type": "server_error"}}, status_code=500)
        assistant_text = ""
        for msg in reversed(result_state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                assistant_text = msg["content"]
                break
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "message": {"role": "assistant", "content": assistant_text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    async def _stream_response(loop, input_state, thread_id):
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'flyclaw', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        try:
            result_state = await loop.run(input_state, thread_id)
            for msg in reversed(result_state.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'flyclaw', 'choices': [{'index': 0, 'delta': {'content': msg['content']}, 'finish_reason': None}]})}\n\n"
                    break
        except Exception as e:
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'flyclaw', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop', 'error': str(e)}]})}\n\n"
            yield "data: [DONE]\n\n"
            return
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'flyclaw', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
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
                expected_mac = hmac.new(auth_token.encode(), nonce.encode(), hashlib.sha256).hexdigest()
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
                    reconnect_delay = 1.0
                    try:
                        frame = json.loads(raw)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "res", "ok": False, "error": {"code": "INVALID_JSON"}})
                        continue
                    if frame.get("type") == "req":
                        await _handle_ws_request(ws, frame, app_config, agent_loop)
                    else:
                        await ws.send_json({"type": "res", "ok": False, "error": {"code": "UNKNOWN_FRAME"}})
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

    async def _handle_ws_request(ws, frame, app_config, loop):
        method = frame.get("method", "")
        params = frame.get("params", {})
        frame_id = frame.get("id", "")
        if method == "ping":
            await ws.send_json({"type": "res", "id": frame_id, "ok": True, "payload": "pong"})
        elif method == "chat.send":
            from src.agent.state import AgentState
            text = params.get("text", "")
            thread_id = params.get("thread_id", "ws-default")
            input_state = AgentState(messages=[{"role": "user", "content": text}], system_prompt=app_config.agents.system_prompt, sender_id="ws", chat_id="ws", chat_type="p2p", message_id=str(uuid.uuid4()), channel="ws")
            store = loop.get_store()
            existing = await store.aload(thread_id)
            if existing:
                input_state.messages = existing.messages + input_state.messages
            result_state = await loop.run(input_state, thread_id)
            assistant_text = ""
            for msg in reversed(result_state.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    assistant_text = msg["content"]
                    break
            await ws.send_json({"type": "res", "id": frame_id, "ok": True, "payload": {"text": assistant_text}})
        elif method == "sessions.list":
            await ws.send_json({"type": "res", "id": frame_id, "ok": True, "payload": []})
        elif method == "health":
            await ws.send_json({"type": "res", "id": frame_id, "ok": True, "payload": {"status": "ok"}})
        elif method == "cron.list" and cron_service:
            await ws.send_json({"type": "res", "id": frame_id, "ok": True, "payload": [j.model_dump() for j in cron_service.list_jobs()]})
        elif method == "cron.add" and cron_service:
            try:
                from src.cron.types import CronJobCreate
                job = await cron_service.add_job(CronJobCreate(**params))
                await ws.send_json({"type": "res", "id": frame_id, "ok": True, "payload": job.model_dump()})
            except Exception as e:
                await ws.send_json({"type": "res", "id": frame_id, "ok": False, "error": {"code": "INVALID_PARAMS", "message": str(e)}})
        elif method == "cron.run" and cron_service:
            result = await cron_service.run_job_now(params.get("job_id", ""))
            await ws.send_json({"type": "res", "id": frame_id, "ok": True, "payload": result.model_dump() if result else None})
        else:
            await ws.send_json({"type": "res", "id": frame_id, "ok": False, "error": {"code": "UNKNOWN_METHOD", "message": method}})

    async def require_auth(request: Request) -> None:
        if not app_config.gateway.auth_token:
            return
        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token, app_config.gateway.auth_token):
            raise HTTPException(status_code=401, detail="unauthorized")

    async def require_rbac(request: Request):
        await require_auth(request)
        if not app_config.auth.enabled:
            raise HTTPException(status_code=400, detail="auth not enabled")
        from src.auth.rbac import get_rbac
        rbac = get_rbac()
        if rbac is None:
            raise HTTPException(status_code=500, detail="RBAC not initialized")
        return rbac

    @app.exception_handler(HTTPException)
    async def _http_error_handler(request, exc):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    if cron_service:
        from src.cron.types import CronJobCreate, CronJobPatch
        @app.get("/api/cron/status")
        async def cron_status(request: Request, _auth=Depends(require_auth)):
            return cron_service.status()
        @app.get("/api/cron/jobs")
        async def cron_list_jobs(request: Request, _auth=Depends(require_auth)):
            return [j.model_dump() for j in cron_service.list_jobs()]
        @app.post("/api/cron/jobs")
        async def cron_add_job(request: Request, _auth=Depends(require_auth)):
            try:
                body = await request.json()
                return (await cron_service.add_job(CronJobCreate(**body))).model_dump()
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=422)
        @app.get("/api/cron/jobs/{job_id}")
        async def cron_get_job(job_id: str, request: Request, _auth=Depends(require_auth)):
            job = cron_service.get_job(job_id)
            return job.model_dump() if job else JSONResponse({"error": "not found"}, status_code=404)
        @app.patch("/api/cron/jobs/{job_id}")
        async def cron_update_job(job_id: str, request: Request, _auth=Depends(require_auth)):
            try:
                job = await cron_service.update_job(job_id, CronJobPatch(**(await request.json())))
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=422)
            return job.model_dump() if job else JSONResponse({"error": "not found"}, status_code=404)
        @app.delete("/api/cron/jobs/{job_id}")
        async def cron_delete_job(job_id: str, request: Request, _auth=Depends(require_auth)):
            return {"removed": await cron_service.remove_job(job_id)}
        @app.post("/api/cron/jobs/{job_id}/run")
        async def cron_run_job(job_id: str, request: Request, _auth=Depends(require_auth)):
            result = await cron_service.run_job_now(job_id)
            return result.model_dump() if result else JSONResponse({"error": "not found"}, status_code=404)

    @app.post("/api/approval/{request_id}/resolve")
    async def resolve_approval(request_id: str, request: Request, _auth=Depends(require_auth)):
        from src.tools.approval import get_approval_manager
        body = await request.json()
        decision = body.get("decision", "deny")
        mgr = get_approval_manager()
        ok = mgr.resolve(request_id, decision)
        return {"resolved": True, "request_id": request_id, "decision": decision} if ok else JSONResponse({"error": "not found or already resolved"}, status_code=404)

    @app.get("/api/approval/pending")
    async def list_pending_approvals(request: Request, _auth=Depends(require_auth)):
        from src.tools.approval import get_approval_manager
        return [r.model_dump() for r in get_approval_manager().list_pending()]

    @app.get("/api/plugins")
    async def list_plugins(request: Request, _auth=Depends(require_auth)):
        try:
            from src.plugins.registry import get_plugin_registry
            return get_plugin_registry().list_plugins()
        except Exception:
            return []

    @app.get("/api/commands")
    async def list_commands(request: Request, _auth=Depends(require_auth)):
        try:
            from src.commands.dispatcher import get_dispatcher
            d = get_dispatcher()
            return {"commands": d.list_commands()} if d else {"commands": []}
        except Exception:
            return {"commands": []}

    @app.post("/api/pair")
    async def pair_device(request: Request, rbac=Depends(require_rbac)):
        try:
            body = await request.json()
            if not body.get("code") or not body.get("device_id"):
                return JSONResponse({"error": "code and device_id required"}, status_code=400)
            user = rbac.store.verify_pairing(body["code"], body["device_id"], platform=body.get("platform", "web"), name=body.get("name", ""))
            return {"paired": True, "user_id": user.user_id, "role": user.role.value} if user else JSONResponse({"error": "invalid or expired pairing code"}, status_code=404)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/users")
    async def list_users(rbac=Depends(require_rbac)):
        try:
            return [u.model_dump() for u in rbac.store.list_users()]
        except Exception:
            return []

    @app.patch("/api/users/{user_id}")
    async def update_user(user_id: str, request: Request, rbac=Depends(require_rbac)):
        try:
            body = await request.json()
            if body.get("role"):
                try:
                    from src.auth.models import UserRole
                    rbac.store.update_user_role(user_id, UserRole(body["role"]))
                except ValueError:
                    return JSONResponse({"error": f"invalid role: {body['role']}"}, status_code=400)
            if body.get("allowed_tools") is not None or body.get("denied_tools") is not None:
                rbac.store.update_user_tools(user_id, allowed_tools=body.get("allowed_tools"), denied_tools=body.get("denied_tools"))
            user = rbac.store.get_user(user_id)
            return user.model_dump() if user else JSONResponse({"error": "user not found"}, status_code=404)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.delete("/api/users/{user_id}")
    async def delete_user(user_id: str, rbac=Depends(require_rbac)):
        try:
            return {"removed": rbac.store.delete_user(user_id)}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/users/{user_id}/devices")
    async def list_user_devices(user_id: str, rbac=Depends(require_rbac)):
        try:
            return [d.model_dump() for d in rbac.store.list_user_devices(user_id)]
        except Exception:
            return []

    @app.delete("/api/devices/{device_id}")
    async def delete_device(device_id: str, rbac=Depends(require_rbac)):
        try:
            return {"removed": rbac.store.delete_device(device_id)}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/sessions/search")
    async def session_search_api(request: Request, _auth=Depends(require_auth)):
        from src.session_index.store import get_session_index
        store = get_session_index()
        if not store:
            return JSONResponse({"error": "session search not enabled"}, status_code=503)
        return JSONResponse(store.search(request.query_params.get("q", ""), limit=int(request.query_params.get("limit", "10"))))

    @app.get("/api/sessions")
    async def session_list_api(request: Request, _auth=Depends(require_auth)):
        from src.session_index.store import get_session_index
        store = get_session_index()
        if not store:
            return JSONResponse({"error": "session search not enabled"}, status_code=503)
        return JSONResponse(store.search("", limit=int(request.query_params.get("limit", "50"))))

    @app.get("/api/sessions/{thread_id}/messages")
    async def session_messages_api(thread_id: str, request: Request, _auth=Depends(require_auth)):
        from src.session_index.store import get_session_index
        store = get_session_index()
        if not store:
            return JSONResponse({"error": "session search not enabled"}, status_code=503)
        rows = store._db.execute("SELECT message_id, role, content, tool_name, timestamp FROM messages WHERE thread_id = ? ORDER BY timestamp ASC LIMIT ?", (thread_id, int(request.query_params.get("limit", "100")))).fetchall()
        return JSONResponse({"thread_id": thread_id, "messages": [dict(r) for r in rows]})

    @app.get("/api/config")
    async def get_config(request: Request):
        app = _get_app(request)
        if not app:
            raise HTTPException(503, "Application not ready")
        raw = app.config.model_dump()
        for sensitive_key in ("api_key", "app_secret", "client_secret", "auth_token"):
            for section in raw.values():
                if isinstance(section, dict) and sensitive_key in section:
                    section[sensitive_key] = "***"
        return raw

    @app.post("/api/config/reload")
    async def reload_config(request: Request):
        app = _get_app(request)
        if not app or not app._config_watcher:
            raise HTTPException(503, "Config watcher not active")
        await app._config_watcher._apply_reload()
        return {"status": "ok"}

    @app.patch("/api/config")
    async def patch_config(request: Request):
        import yaml as _yaml
        app = _get_app(request)
        if not app:
            raise HTTPException(503, "Application not ready")
        patch_data = await request.json()
        config_path = Path(app._config_path or str(Path.home() / ".flyclaw" / "config.yaml"))
        if config_path.exists():
            current = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            current = {}

        def _deep_merge(base, override):
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    _deep_merge(base[k], v)
                else:
                    base[k] = v

        _deep_merge(current, patch_data)
        config_path.write_text(
            _yaml.dump(current, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        return {"status": "written"}

    return app
