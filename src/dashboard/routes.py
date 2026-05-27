"""Dashboard routes for flyclaw web UI."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from collections import deque
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("flyclaw.dashboard")

# ── Log buffer for SSE streaming ──

_log_buffer: deque = deque(maxlen=200)
_log_subscribers: list[asyncio.Queue] = []


class _LogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            entry = {
                "ts": int(time.time()),
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
            }
            _log_buffer.append(entry)
            for q in _log_subscribers:
                try:
                    q.put_nowait(entry)
                except asyncio.QueueFull:
                    pass
        except Exception:
            pass


_log_handler = _LogHandler()
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))


def _install_log_handler():
    root = logging.getLogger("flyclaw")
    if _log_handler not in root.handlers:
        root.addHandler(_log_handler)


# ── Auth helper ──


def _check_auth(request: Request, app: FastAPI):
    from src.config import load_config

    cfg = load_config()
    if not cfg.gateway.auth_token:
        return
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    # Also accept token from query param for SSE
    if not token:
        token = request.query_params.get("token", "")
    if not hmac.compare_digest(token, cfg.gateway.auth_token):
        raise HTTPException(status_code=401, detail="unauthorized")


# ── Startup time tracking ──

_start_time = time.monotonic()


# ── App reference ──

_app_ref = None


def register_dashboard(app: FastAPI, application):
    global _app_ref
    _app_ref = application
    _install_log_handler()

    _template_path = Path(__file__).parent / "templates" / "dashboard.html"
    _html_template = _template_path.read_text(encoding="utf-8")

    router = APIRouter(tags=["dashboard"])

    @router.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page():
        cfg = _app_ref.config
        html = _html_template
        html = html.replace('"{{ auth_token }}"', json.dumps(cfg.gateway.auth_token or ""))
        html = html.replace('"{{ model_provider }}"', json.dumps(cfg.model.provider))
        html = html.replace('"{{ model_name }}"', json.dumps(cfg.model.name))
        return HTMLResponse(html)

    @router.get("/api/dashboard/status")
    async def dashboard_status(request: Request):
        _check_auth(request, app)
        cfg = _app_ref.config
        uptime = time.monotonic() - _start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)

        skills = _app_ref.skills_cache or []
        status = {
            "version": "0.1.0",
            "uptime": f"{hours}h {minutes}m {seconds}s",
            "uptime_seconds": int(uptime),
            "model": {
                "provider": cfg.model.provider,
                "name": cfg.model.name,
                "base_url": cfg.model.base_url or "default",
                "has_fallbacks": bool(cfg.model.fallbacks),
            },
            "gateway": f"{cfg.gateway.host}:{cfg.gateway.port}",
            "sessions": _app_ref.session_tracker.active_count if _app_ref.session_tracker else 0,
            "tools": len(_app_ref.agent_loop._tools) if _app_ref.agent_loop else 0,
            "skills": len(skills),
            "cron": {
                "enabled": cfg.cron.enabled,
                "jobs": 0,
            },
            "memory": getattr(cfg, "memory", None) and cfg.memory.enabled or False,
            "tts": getattr(cfg, "tts", None) and getattr(cfg.tts, "enabled", False) or False,
        }
        if _app_ref.cron_service:
            s = _app_ref.cron_service.status()
            status["cron"]["jobs"] = s.get("total_jobs", 0)
        return status

    @router.get("/api/dashboard/sessions")
    async def dashboard_sessions(request: Request):
        _check_auth(request, app)
        import re as _re
        sessions = []

        # Only show chat sessions (channel:user:* or channel:group:* or channel:sN:*)
        _chat_pattern = _re.compile(r'^(qq):(user|group|s\d+):')

        # Active sessions from tracker (with idle time)
        active_ids = set()
        if _app_ref.session_tracker:
            for s in _app_ref.session_tracker.get_sessions():
                tid = s["thread_id"]
                if _chat_pattern.match(tid):
                    active_ids.add(tid)
                    sessions.append({**s, "status": "active"})

        # Historical sessions from state store
        if _app_ref.state_store:
            try:
                for tid in _app_ref.state_store.list_threads():
                    if tid not in active_ids and _chat_pattern.match(tid):
                        state = await _app_ref.state_store.aload(tid)
                        msg_count = len(state.messages) if state else 0
                        sessions.append({
                            "thread_id": tid,
                            "last_active": None,
                            "status": "idle",
                            "checkpoint_count": msg_count,
                        })
            except Exception:
                pass

        return sessions

    @router.post("/api/dashboard/sessions/{thread_id}/reset")
    async def dashboard_reset_session(thread_id: str, request: Request):
        _check_auth(request, app)
        if not _app_ref.state_store:
            raise HTTPException(status_code=500, detail="State store not initialized")
        try:
            _app_ref.state_store.delete(thread_id)
            if _app_ref.session_tracker:
                _app_ref.session_tracker.remove(thread_id)
            if _app_ref.agent_loop:
                _app_ref.agent_loop.invalidate_memory_cache()
            return {"ok": True}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @router.get("/api/dashboard/cron")
    async def dashboard_cron(request: Request):
        _check_auth(request, app)
        if not _app_ref.cron_service:
            return {"enabled": False, "jobs": []}
        jobs = _app_ref.cron_service.list_jobs()
        return {
            "enabled": True,
            "status": _app_ref.cron_service.status(),
            "jobs": [j.model_dump() for j in jobs],
        }

    @router.get("/api/dashboard/tools")
    async def dashboard_tools(request: Request):
        _check_auth(request, app)
        tools_info = []
        # Gather tools from the registry
        try:
            from src.tools.registry import get_tool_registry

            registry = get_tool_registry()
            all_tools = registry.collect()
            for t in all_tools:
                tools_info.append(
                    {
                        "name": t.name,
                        "description": (t.description or "")[:120],
                    }
                )
        except Exception:
            pass
        return tools_info

    @router.get("/api/dashboard/skills")
    async def dashboard_skills(request: Request):
        _check_auth(request, app)
        skills = _app_ref.skills_cache or []
        config = _app_ref.config
        return [
            {
                "name": s.name,
                "description": s.description[:120],
                "source": s.source,
                "user_invocable": s.metadata.user_invocable,
                "disable_model_invocation": s.metadata.disable_model_invocation,
                "disabled": s.name in config.skills.disabled,
                "channel_disabled": {
                    ch: s.name in chans
                    for ch, chans in config.skills.channel_disabled.items()
                },
            }
            for s in skills
        ]

    @router.get("/api/dashboard/config")
    async def dashboard_config(request: Request):
        _check_auth(request, app)
        cfg = _app_ref.config
        return {
            "model": {
                "provider": cfg.model.provider,
                "name": cfg.model.name,
                "temperature": cfg.model.temperature,
                "base_url": cfg.model.base_url or "default",
                "fallbacks": len(cfg.model.fallbacks),
            },
            "agents": {
                "workspace": cfg.agents.workspace,
                "max_tool_rounds": cfg.agents.max_tool_rounds,
                "subagents": list(cfg.agents.subagents.keys()),
            },
            "gateway": {
                "host": cfg.gateway.host,
                "port": cfg.gateway.port,
                "has_auth": bool(cfg.gateway.auth_token),
            },
            "session": {
                "scope": cfg.session.scope,
                "idle_reset_minutes": cfg.session.idle_reset_minutes,
            },
            "tools": {
                "exec_enabled": cfg.tools.exec.enabled,
                "exec_approval": cfg.tools.exec.approval_mode,
                "exec_sandbox": cfg.tools.exec.sandbox_enabled,
                "web_fetch": cfg.tools.web_fetch.enabled,
                "media": cfg.tools.media_understanding.enabled,
            },
            "cron": {
                "enabled": cfg.cron.enabled,
                "max_concurrent": cfg.cron.max_concurrent_runs,
            },
            "memory_store": {
                "enabled": getattr(cfg.memory_store, "enabled", False),
                "db_path": cfg.memory_store.db_path,
                "memory_judge_model": cfg.memory_store.memory_judge_model,
            },
            "task": {
                "enabled": getattr(cfg.task, "enabled", False),
                "max_parallel": cfg.task.max_parallel,
                "default_timeout": cfg.task.default_timeout,
                "defer_minutes": cfg.task.defer_minutes,
            },
            "memory": {
                "enabled": getattr(cfg.memory, "enabled", False),
            },
            "tts": {
                "enabled": getattr(getattr(cfg, "tts", None), "enabled", False),
                "provider": getattr(getattr(cfg, "tts", None), "provider", ""),
            },
            "checkpointer": {
                "type": cfg.checkpointer.type,
            },
            "auth": {
                "enabled": getattr(cfg.auth, "enabled", False),
                "pairing_enabled": getattr(cfg.auth, "pairing_enabled", False),
                "default_role": getattr(cfg.auth, "default_role", "guest"),
            },
        }

    @router.get("/api/dashboard/logs")
    async def dashboard_logs(request: Request):
        _check_auth(request, app)
        return list(_log_buffer)

    @router.get("/api/dashboard/stream")
    async def dashboard_stream(request: Request):
        _check_auth(request, app)
        import json

        from starlette.responses import StreamingResponse

        async def event_generator():
            q: asyncio.Queue = asyncio.Queue(maxsize=50)
            _log_subscribers.append(q)
            try:
                # Send buffered logs first
                for entry in _log_buffer:
                    yield f"event: log\ndata: {json.dumps(entry, ensure_ascii=False)}\n\n"
                # Stream new logs
                while True:
                    try:
                        entry = await asyncio.wait_for(q.get(), timeout=30)
                        yield f"event: log\ndata: {json.dumps(entry, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                if q in _log_subscribers:
                    _log_subscribers.remove(q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/api/dashboard/state-stream")
    async def dashboard_state_stream(request: Request):
        """SSE endpoint pushing periodic state snapshots for real-time UI updates."""
        _check_auth(request, app)
        import json

        from starlette.responses import StreamingResponse

        async def state_generator():
            try:
                while True:
                    state = await _build_state_snapshot()
                    yield f"data: {json.dumps(state, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(3)
            except asyncio.CancelledError:
                pass

        return StreamingResponse(
            state_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/api/dashboard/events")
    async def dashboard_events(request: Request):
        """SSE endpoint pushing real-time events from the event bus."""
        _check_auth(request, app)
        import json

        from starlette.responses import StreamingResponse

        async def event_generator():
            q: asyncio.Queue = asyncio.Queue(maxsize=100)

            def _on_event(event, **ctx):
                try:
                    q.put_nowait({"event": event, "context": ctx, "ts": ctx.get("_ts", 0)})
                except asyncio.QueueFull:
                    pass  # Drop events if client is too slow

            from src.events import subscribe_async
            sub = subscribe_async("*", _on_event, priority=999)

            try:
                # Send initial state snapshot
                yield f"event: init\ndata: {json.dumps(await _build_state_snapshot(), ensure_ascii=False)}\n\n"

                # Stream events
                while True:
                    try:
                        evt = await asyncio.wait_for(q.get(), timeout=30)
                        yield f"event: {evt['event']}\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                from src.events import unsubscribe
                unsubscribe("*", _on_event)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _build_state_snapshot() -> dict:
        cfg = _app_ref.config
        uptime = time.monotonic() - _start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        snapshot = {
            "uptime": f"{hours}h {minutes}m {seconds}s",
            "uptime_seconds": int(uptime),
            "session_count": _app_ref.session_tracker.active_count if _app_ref.session_tracker else 0,
            "skill_count": len(_app_ref.skills_cache or []),
        }
        # Sessions (active + historical from checkpointer, chat only)
        import re as _re
        _chat_pattern = _re.compile(r'^(qq):(user|group|s\d+):')
        all_sessions = []
        active_ids = set()
        if _app_ref.session_tracker:
            for s in _app_ref.session_tracker.get_sessions():
                if _chat_pattern.match(s["thread_id"]):
                    active_ids.add(s["thread_id"])
                    all_sessions.append({**s, "status": "active"})
        if _app_ref.state_store:
            try:
                for tid in _app_ref.state_store.list_threads():
                    if tid not in active_ids and _chat_pattern.match(tid):
                        state = await _app_ref.state_store.aload(tid)
                        msg_count = len(state.messages) if state else 0
                        all_sessions.append({
                            "thread_id": tid,
                            "last_active": None,
                            "status": "idle",
                            "checkpoint_count": msg_count,
                        })
            except Exception:
                pass
        snapshot["sessions"] = all_sessions
        snapshot["session_count"] = len(active_ids)
        # Pending approvals
        try:
            from src.tools.approval import get_approval_manager
            snapshot["pending_approvals"] = [r.model_dump() for r in get_approval_manager().list_pending()]
        except Exception:
            snapshot["pending_approvals"] = []
        # Cron
        if _app_ref.cron_service:
            snapshot["cron_jobs"] = [j.model_dump() for j in _app_ref.cron_service.list_jobs()]
        # Auth users
        if getattr(cfg.auth, "enabled", False):
            try:
                from src.auth.rbac import get_rbac
                rbac = get_rbac()
                if rbac:
                    snapshot["auth_users"] = [
                        {**u.model_dump(), "device_count": len(rbac.store.list_user_devices(u.user_id)),
                         "trusted_devices": sum(1 for d in rbac.store.list_user_devices(u.user_id) if d.trusted)}
                        for u in rbac.store.list_users()
                    ]
            except Exception:
                snapshot["auth_users"] = []
        # Beads
        try:
            from src.tools.memory_tools import get_memory_store
            store = get_memory_store()
            snapshot["memories"] = await store.list_all()
        except Exception:
            snapshot["memories"] = {}
        return snapshot

    # ── Model switching ─────────────────────────────────────

    @router.get("/api/dashboard/models")
    async def dashboard_list_models(request: Request):
        _check_auth(request, app)
        cfg = _app_ref.config
        active_model = _app_ref.model_ref.model if _app_ref.model_ref else None
        from src.agent.client import FallbackChain
        if isinstance(active_model, FallbackChain):
            idx = active_model._active_idx
            available = []
            for i, c in enumerate(active_model._all):
                available.append({"name": c.model, "active": i == idx})
            current = available[idx] if idx < len(available) else available[0]
        else:
            current = {"provider": cfg.model.provider, "name": cfg.model.name}
            available = [{"provider": current["provider"], "name": current["name"], "active": True}]
        return {"current": current, "available": available}

    @router.post("/api/dashboard/models/switch")
    async def dashboard_switch_model(request: Request):
        _check_auth(request, app)
        body = await request.json()
        provider = body.get("provider", "").strip()
        name = body.get("name", "").strip()
        if not provider or not name:
            raise HTTPException(status_code=400, detail="provider and name required")
        if not _app_ref.model_ref:
            raise HTTPException(status_code=500, detail="model_ref not initialized")

        active_model = _app_ref.model_ref.model
        from src.agent.client import FallbackChain
        if isinstance(active_model, FallbackChain):
            all_clients = active_model._all
            old_idx = active_model._active_idx
            old_name = all_clients[old_idx].model
            target_idx = None
            for i, c in enumerate(all_clients):
                if c.model == name:
                    target_idx = i
                    break
            if target_idx is None:
                raise HTTPException(status_code=404, detail=f"Model {name} not found in chain")
            active_model.switch_to(target_idx)
            logger.info("Model switched from %s to %s", old_name, name)
            return {"ok": True, "previous": old_name, "current": name}
        else:
            # Single model (no chain) — create new model for switch
            from src.agent.client import create_client
            cfg = _app_ref.config
            fb_base_url = None
            fb_api_key = None
            for fb in cfg.model.fallbacks:
                if fb.provider == provider and fb.name == name:
                    fb_base_url = getattr(fb, "base_url", None)
                    fb_api_key = getattr(fb, "api_key", None)
                    break
            try:
                new_model = create_client(
                    provider, name,
                    cfg.model.temperature,
                    base_url=fb_base_url or cfg.model.base_url,
                    api_key=fb_api_key or cfg.model.api_key,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to create model: {e}")
            old_name = f"{cfg.model.provider}/{cfg.model.name}"
            _app_ref.model_ref.model = new_model
            logger.info("Model switched from %s to %s/%s", old_name, provider, name)
            return {"ok": True, "previous": old_name, "current": f"{provider}/{name}"}

    # ── Auth dashboard API ──────────────────────────────────

    @router.get("/api/dashboard/users")
    async def dashboard_users(request: Request):
        _check_auth(request, app)
        cfg = _app_ref.config
        if not getattr(cfg.auth, "enabled", False):
            return {"enabled": False, "users": []}
        try:
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is None:
                return {"enabled": True, "users": []}
            users = rbac.store.list_users()
            result = []
            for u in users:
                devices = rbac.store.list_user_devices(u.user_id)
                result.append(
                    {
                        **u.model_dump(),
                        "device_count": len(devices),
                        "trusted_devices": sum(1 for d in devices if d.trusted),
                    }
                )
            return {"enabled": True, "users": result}
        except Exception as e:
            return {"enabled": True, "users": [], "error": str(e)}

    @router.get("/api/dashboard/users/{user_id}/devices")
    async def dashboard_user_devices(user_id: str, request: Request):
        _check_auth(request, app)
        try:
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is None:
                return []
            return [d.model_dump() for d in rbac.store.list_user_devices(user_id)]
        except Exception:
            return []

    @router.patch("/api/dashboard/users/{user_id}/role")
    async def dashboard_update_role(user_id: str, request: Request):
        _check_auth(request, app)
        try:
            body = await request.json()
            from src.auth.models import UserRole
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is None:
                raise HTTPException(status_code=500, detail="RBAC not initialized")
            new_role = UserRole(body.get("role", "guest"))
            rbac.store.update_user_role(user_id, new_role)
            return {"ok": True, "user_id": user_id, "role": new_role.value}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/api/dashboard/users/{user_id}")
    async def dashboard_delete_user(user_id: str, request: Request):
        _check_auth(request, app)
        try:
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is None:
                raise HTTPException(status_code=500, detail="RBAC not initialized")
            removed = rbac.store.delete_user(user_id)
            return {"ok": removed}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/api/dashboard/devices/{device_id}")
    async def dashboard_delete_device(device_id: str, request: Request):
        _check_auth(request, app)
        try:
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is None:
                raise HTTPException(status_code=500, detail="RBAC not initialized")
            removed = rbac.store.delete_device(device_id)
            return {"ok": removed}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Memory API ──────────────────────────────────────

    @router.get("/api/dashboard/memory/memories")
    async def dashboard_memory_list(request: Request):
        _check_auth(request, app)
        try:
            from src.tools.memory_tools import get_memory_store
            store = get_memory_store()
            items = await store.list_all()
            grouped = {}
            for m in items:
                cat = m.get("category", "其他")
                grouped.setdefault(cat, []).append(m)
            return grouped
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/dashboard/memory/recall/{key:path}")
    async def dashboard_memory_recall(key: str, request: Request):
        _check_auth(request, app)
        try:
            from src.tools.memory_tools import get_memory_store
            import json
            store = get_memory_store()
            return json.loads(await store.recall(key))
        except Exception as e:
            return {"error": str(e)}

    @router.delete("/api/dashboard/memory/forget/{key:path}")
    async def dashboard_memory_forget(key: str, request: Request):
        _check_auth(request, app)
        try:
            from src.tools.memory_tools import get_memory_store
            import json
            store = get_memory_store()
            return json.loads(await store.forget(key))
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @router.post("/api/dashboard/memory/remember")
    async def dashboard_memory_save(request: Request):
        _check_auth(request, app)
        try:
            body = await request.json()
            content = body.get("content", "").strip()
            key = body.get("key", "").strip()
            category = body.get("category", "fact").strip()
            if not content:
                raise HTTPException(status_code=400, detail="content is required")
            from src.tools.memory_tools import save_memory
            import json
            result = json.loads(await save_memory(content, key, category=category))
            return result
        except HTTPException:
            raise
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Phase 1: Dashboard Enhancements ─────────────────────────

    @router.post("/api/dashboard/config/reload")
    async def dashboard_config_reload(request: Request):
        """Hot-reload configuration."""
        _check_auth(request, app)
        try:
            from src.config import load_config
            new_cfg = load_config()
            _app_ref.config = new_cfg
            if _app_ref.agent_loop:
                from src.agent.client import create_chain
                _app_ref.agent_loop._client = create_chain(new_cfg)
            return {"ok": True, "message": "Config reloaded"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @router.get("/api/dashboard/search")
    async def dashboard_session_search(request: Request, q: str = ""):
        """Search session history via FTS5."""
        _check_auth(request, app)
        if not q:
            return {"results": []}
        try:
            from src.session_index.store import get_session_index
            idx = get_session_index()
            if not idx:
                return {"results": [], "error": "Session index not initialized"}
            results = idx.search(q, limit=20)
            return {"results": results, "query": q}
        except Exception as e:
            return {"results": [], "error": str(e)}

    @router.get("/api/dashboard/sessions/{thread_id}/messages")
    async def dashboard_session_messages(thread_id: str, request: Request, limit: int = 50, offset: int = 0):
        """Get message history for a session."""
        _check_auth(request, app)
        try:
            if not _app_ref.state_store:
                raise HTTPException(status_code=500, detail="State store not initialized")
            state = await _app_ref.state_store.aload(thread_id)
            if not state:
                raise HTTPException(status_code=404, detail="Session not found")
            messages = state.messages[offset:offset+limit]
            total = len(state.messages)
            return {
                "thread_id": thread_id,
                "messages": messages,
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total,
            }
        except HTTPException:
            raise
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/dashboard/memory/stats")
    async def dashboard_memory_stats(request: Request):
        """Get memory system statistics."""
        _check_auth(request, app)
        cfg = _app_ref.config
        if not getattr(cfg.memory, "enabled", False):
            return {"enabled": False}
        try:
            from src.memory.search import get_memory_searcher
            searcher = get_memory_searcher()
            if not searcher:
                return {"enabled": True, "error": "Memory searcher not initialized"}
            stats = searcher.get_stats()
            return {"enabled": True, **stats}
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    @router.get("/api/dashboard/security/audit")
    async def dashboard_security_audit(request: Request):
        """Run and return security audit results."""
        _check_auth(request, app)
        try:
            from src.security.audit import run_security_audit
            results = run_security_audit(_app_ref.config)
            return {
                "pass": results.get("pass", 0),
                "warn": results.get("warn", 0),
                "info": results.get("info", 0),
                "issues": results.get("issues", []),
            }
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/dashboard/skills/curation")
    async def dashboard_skills_curation(request: Request):
        """Get skill curation status."""
        _check_auth(request, app)
        try:
            from src.skills.curator import SkillCurator
            from pathlib import Path
            curator = SkillCurator(Path.home() / ".flyclaw" / "skills")
            skills = _app_ref.skills_cache or []
            lifecycle_counts = {"active": 0, "stale": 0, "archived": 0}
            for s in skills:
                usage = curator.manager.get_usage(s.name)
                state = usage.get("state", "active") if usage else "active"
                if state in lifecycle_counts:
                    lifecycle_counts[state] += 1
            return {
                "last_review": curator.state.get("last_review"),
                "total_reviews": curator.state.get("total_reviews", 0),
                "days_since_review": curator.days_since_last_review(),
                "review_interval_days": curator.review_interval_days,
                "skills_reviewed": len(skills),
                "lifecycle_counts": lifecycle_counts,
            }
        except Exception as e:
            return {"error": str(e)}

    @router.post("/api/dashboard/skills/curation/review")
    async def dashboard_skills_curation_review(request: Request, dry_run: bool = False):
        """Trigger skill curation review."""
        _check_auth(request, app)
        try:
            from src.skills.curator import SkillCurator
            from pathlib import Path
            curator = SkillCurator(Path.home() / ".flyclaw" / "skills")
            result = await curator.review_skills(dry_run=dry_run)
            return result
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/dashboard/learning/status")
    async def dashboard_learning_status(request: Request):
        """Get learning loop status."""
        _check_auth(request, app)
        cfg = _app_ref.config
        return {
            "memory_store_enabled": getattr(cfg.memory_store, "enabled", False),
            "memory_judge_model": getattr(cfg.memory_store, "memory_judge_model", ""),
            "curated_memory_sync": True,
            "skill_curation": True,
            "session_end_extraction": getattr(cfg.memory_store, "enabled", False),
        }

    @router.post("/api/dashboard/learning/trigger")
    async def dashboard_learning_trigger(request: Request):
        """Manually trigger learning cycle."""
        _check_auth(request, app)
        try:
            from src.agent.learning import LearningLoop
            loop = LearningLoop(_app_ref.config)
            result = await loop.trigger_full_learning_cycle()
            return result
        except Exception as e:
            return {"error": str(e)}

    # ── Audit Log API ──────────────────────────────────────────

    @router.get("/api/dashboard/audit")
    async def dashboard_audit(
        request: Request,
        tool_name: str = "",
        sender_id: str = "",
        status: str = "",
        limit: int = 100,
        offset: int = 0,
    ):
        """Get audit log entries with filters."""
        _check_auth(request, app)
        try:
            from src.analytics.audit_store import get_audit_store
            store = get_audit_store()
            
            success_filter = None
            if status == "success":
                success_filter = True
            elif status == "error":
                success_filter = False

            entries = store.query(
                tool_name=tool_name or None,
                sender_id=sender_id or None,
                success=success_filter,
                limit=limit,
                offset=offset,
            )
            return {
                "entries": [
                    {
                        "id": e.id,
                        "thread_id": e.thread_id,
                        "tool_name": e.tool_name,
                        "sender_id": e.sender_id,
                        "channel": e.channel,
                        "success": e.success,
                        "duration_ms": round(e.duration_ms, 2),
                        "args_preview": e.args_preview,
                        "error": e.error,
                        "timestamp": e.timestamp,
                    }
                    for e in entries
                ],
                "total": len(entries),
                "offset": offset,
                "limit": limit,
            }
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/dashboard/audit/stats")
    async def dashboard_audit_stats(request: Request, days: int = 7):
        """Get audit statistics."""
        _check_auth(request, app)
        try:
            from src.analytics.audit_store import get_audit_store
            store = get_audit_store()
            return store.get_stats(days=days)
        except Exception as e:
            return {"error": str(e)}

    @router.post("/api/dashboard/config/update")
    async def dashboard_config_update(request: Request):
        """Update configuration values."""
        _check_auth(request, app)
        try:
            body = await request.json()
            cfg = _app_ref.config
            
            # Apply updates to config object
            for section, values in body.items():
                if hasattr(cfg, section):
                    section_obj = getattr(cfg, section)
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
            
            # Save config to file using the app's config path
            from pathlib import Path as _Path
            from src.config import save_config
            config_path = getattr(_app_ref, '_config_path', None) or str(Path.home() / ".flyclaw" / "config.yaml")
            # Ensure absolute path to avoid saving to wrong location
            config_path = str(_Path(config_path).resolve())
            save_config(cfg, config_path)
            
            # Reload config in app
            _app_ref.config = cfg
            
            # Hot-reload runtime components
            # 1. Update agent_loop client (model config changes)
            if _app_ref.agent_loop:
                try:
                    from src.agent.client import create_chain
                    _app_ref.agent_loop._client = create_chain(cfg)
                except Exception:
                    pass
            
            # 2. Reload skills (skill config changes)
            # _reload_skills() already updates agent_loop._skills_prompt and dispatcher
            try:
                _app_ref._reload_skills()
            except Exception:
                pass
            
            return {"ok": True, "message": "Configuration updated and saved"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    app.include_router(router)
    logger.info("Dashboard registered at /dashboard")
