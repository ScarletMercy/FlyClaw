"""Dashboard routes for MyClaw web UI."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from collections import deque
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("myclaw.dashboard")

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
    root = logging.getLogger("myclaw")
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

    from jinja2 import Template
    _template_path = Path(__file__).parent / "templates" / "dashboard.html"
    _jinja_template = Template(_template_path.read_text(encoding="utf-8"))

    router = APIRouter(tags=["dashboard"])

    @router.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page():
        cfg = _app_ref.config
        uptime = time.monotonic() - _start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        skills = _app_ref._skills_cache or []

        # Collect tools list for SSR
        tools = []
        try:
            from src.tools.registry import get_tool_registry
            for t in get_tool_registry().collect():
                tools.append({"name": t.name, "description": (t.description or "")[:120]})
        except Exception:
            pass

        # Collect sessions for SSR
        sessions = []
        if _app_ref.session_tracker:
            sessions = _app_ref.session_tracker.get_sessions()

        # Collect cron jobs for SSR
        cron_jobs = []
        if _app_ref.cron_service:
            for j in _app_ref.cron_service.list_jobs():
                cron_jobs.append(j.model_dump())

        html = _jinja_template.render(
            model_provider=cfg.model.provider,
            model_name=cfg.model.name,
            model_base_url=cfg.model.base_url or "default",
            has_fallbacks=bool(cfg.model.fallbacks),
            uptime=f"{hours}h {minutes}m {seconds}s",
            gateway=f"{cfg.gateway.host}:{cfg.gateway.port}",
            feishu_enabled=cfg.channels.feishu.enabled,
            feishu_domain=cfg.channels.feishu.domain,
            session_count=_app_ref.session_tracker.active_count if _app_ref.session_tracker else 0,
            tool_count=len(tools),
            skill_count=len(skills),
            cron_enabled=cfg.cron.enabled,
            cron_jobs=cron_jobs,
            memory_enabled=getattr(cfg.memory, "enabled", False),
            tts_enabled=getattr(cfg.tts, "enabled", False),
            tools=tools,
            skills=skills,
            sessions=sessions,
            config={
                "model_provider": cfg.model.provider,
                "model_name": cfg.model.name,
                "temperature": cfg.model.temperature,
                "base_url": cfg.model.base_url or "default",
                "fallbacks": len(cfg.model.fallbacks),
                "workspace": cfg.agents.workspace,
                "max_tool_rounds": cfg.agents.max_tool_rounds,
                "subagents": list(cfg.agents.subagents.keys()),
                "gateway": f"{cfg.gateway.host}:{cfg.gateway.port}" + (" (auth)" if cfg.gateway.auth_token else ""),
                "feishu_enabled": cfg.channels.feishu.enabled,
                "feishu_domain": cfg.channels.feishu.domain,
                "feishu_dm": cfg.channels.feishu.dm_policy,
                "feishu_group": cfg.channels.feishu.group_policy,
                "feishu_mention": cfg.channels.feishu.require_mention,
                "feishu_streaming": cfg.channels.feishu.streaming,
                "session_scope": cfg.session.scope,
                "idle_reset_minutes": cfg.session.idle_reset_minutes,
                "exec_enabled": cfg.tools.exec.enabled,
                "exec_approval": cfg.tools.exec.approval_mode,
                "exec_sandbox": cfg.tools.exec.sandbox_enabled,
                "web_search": cfg.tools.web_search.enabled,
                "web_fetch": cfg.tools.web_fetch.enabled,
                "media": cfg.tools.media_understanding.enabled,
                "cron_enabled": cfg.cron.enabled,
                "cron_concurrent": cfg.cron.max_concurrent_runs,
                "memory_enabled": getattr(cfg.memory, "enabled", False),
                "tts_enabled": getattr(cfg.tts, "enabled", False),
                "tts_provider": getattr(cfg.tts, "provider", ""),
                "checkpointer": cfg.checkpointer.type,
            },
        )
        return HTMLResponse(html)

    @router.get("/api/dashboard/status")
    async def dashboard_status(request: Request):
        _check_auth(request, app)
        cfg = _app_ref.config
        uptime = time.monotonic() - _start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)

        skills = _app_ref._skills_cache or []
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
            "feishu": {
                "enabled": cfg.channels.feishu.enabled,
                "domain": cfg.channels.feishu.domain,
            },
            "sessions": _app_ref.session_tracker.active_count if _app_ref.session_tracker else 0,
            "tools": len(_app_ref.compiled_graph.get_graph().nodes) - 2 if _app_ref.compiled_graph else 0,  # subtract agent + tools nodes
            "skills": len(skills),
            "cron": {
                "enabled": cfg.cron.enabled,
                "jobs": 0,
            },
            "memory": getattr(cfg, "memory", None) and cfg.memory.enabled or False,
            "tts": getattr(cfg, "tts", None) and cfg.tts.enabled or False,
        }
        if _app_ref.cron_service:
            s = _app_ref.cron_service.status()
            status["cron"]["jobs"] = s.get("total_jobs", 0)
        return status

    @router.get("/api/dashboard/sessions")
    async def dashboard_sessions(request: Request):
        _check_auth(request, app)
        if not _app_ref.session_tracker:
            return []
        sessions = _app_ref.session_tracker.get_sessions()
        for s in sessions:
            idle = s["last_active"]
            if idle < 60:
                s["idle_text"] = f"{int(idle)}s"
            elif idle < 3600:
                s["idle_text"] = f"{int(idle // 60)}m {int(idle % 60)}s"
            else:
                h, m = divmod(int(idle), 3600)
                m2, _ = divmod(m, 60)
                s["idle_text"] = f"{h}h {m2}m"
        return sessions

    @router.post("/api/dashboard/sessions/{thread_id}/reset")
    async def dashboard_reset_session(thread_id: str, request: Request):
        _check_auth(request, app)
        if not _app_ref.compiled_graph:
            raise HTTPException(status_code=500, detail="Graph not initialized")
        try:
            config = {"configurable": {"thread_id": thread_id}}
            await _app_ref.compiled_graph.aupdate_state(config, {"messages": []})
            if _app_ref.session_tracker:
                _app_ref.session_tracker.remove(thread_id)
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
                tools_info.append({
                    "name": t.name,
                    "description": (t.description or "")[:120],
                })
        except Exception:
            pass
        return tools_info

    @router.get("/api/dashboard/skills")
    async def dashboard_skills(request: Request):
        _check_auth(request, app)
        skills = _app_ref._skills_cache or []
        return [
            {
                "name": s.name,
                "description": s.description[:120],
                "source": s.source,
                "user_invocable": s.metadata.user_invocable,
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
            "feishu": {
                "enabled": cfg.channels.feishu.enabled,
                "domain": cfg.channels.feishu.domain,
                "dm_policy": cfg.channels.feishu.dm_policy,
                "group_policy": cfg.channels.feishu.group_policy,
                "require_mention": cfg.channels.feishu.require_mention,
                "streaming": cfg.channels.feishu.streaming,
            },
            "session": {
                "scope": cfg.session.scope,
                "idle_reset_minutes": cfg.session.idle_reset_minutes,
            },
            "tools": {
                "exec_enabled": cfg.tools.exec.enabled,
                "exec_approval": cfg.tools.exec.approval_mode,
                "exec_sandbox": cfg.tools.exec.sandbox_enabled,
                "web_search": cfg.tools.web_search.enabled,
                "web_fetch": cfg.tools.web_fetch.enabled,
                "media": cfg.tools.media_understanding.enabled,
            },
            "cron": {
                "enabled": cfg.cron.enabled,
                "max_concurrent": cfg.cron.max_concurrent_runs,
            },
            "memory": {
                "enabled": getattr(cfg.memory, "enabled", False),
            },
            "tts": {
                "enabled": getattr(cfg.tts, "enabled", False),
                "provider": getattr(cfg.tts, "provider", ""),
            },
            "checkpointer": {
                "type": cfg.checkpointer.type,
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
                    yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                # Stream new logs
                while True:
                    try:
                        entry = await asyncio.wait_for(q.get(), timeout=30)
                        yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
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

    app.include_router(router)
    logger.info("Dashboard registered at /dashboard")
