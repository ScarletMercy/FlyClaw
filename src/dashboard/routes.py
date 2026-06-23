"""Dashboard routes for flyclaw web UI."""

from __future__ import annotations

import asyncio
import collections
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from src.version import __version__

logger = logging.getLogger("flyclaw.dashboard")

# ── Log ring buffer ──

_LOG_MAX = 1000
_log_buffer: collections.deque[logging.LogRecord] = collections.deque(maxlen=_LOG_MAX)


class _LogBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        _log_buffer.append(record)


_log_handler: _LogBufferHandler | None = None

# ── Dashboard session cookie (HMAC-signed, stateless) ──
# Cookie value = "<expiry_epoch>.<hmac_sha256(auth_token, expiry)>".
# Stateless: no server-side session store; signing key reuses gateway.auth_token.

_SESSION_TTL_SECONDS = 7 * 86400  # 7 天
_SESSION_COOKIE = "fc_auth"


def _sign_session(auth_token: str, now: int) -> str:
    """生成 `<expiry>.<hmac>` 形式的无状态 session cookie 值。"""
    expiry = now + _SESSION_TTL_SECONDS
    payload = str(expiry)
    sig = hmac.new(auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_session(auth_token: str, cookie_val: str, now: int) -> bool:
    """校验签名（HMAC 用 compare_digest 恒定时间比较）+ 过期检查。任一环节失败返回 False。

    注：格式/类型的结构检查有早返回，并非全程恒定时间；但唯一秘密是 HMAC
    密钥，签名比较本身恒定时间，不泄露 key。
    """
    if not auth_token or not cookie_val or "." not in cookie_val:
        return False
    payload, _, sig = cookie_val.rpartition(".")
    if not payload or not sig:
        return False
    expected = hmac.new(auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False
    try:
        if int(payload) < now:
            return False  # 过期
    except ValueError:
        return False
    return True


# ── Auth helper ──


def _check_auth(request: Request):
    cfg = _app_ref.config
    if not cfg.gateway.auth_token:
        return  # auth 未启用，放行（与全项目空 token 语义一致）
    # 1) Bearer header（API 客户端）
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        # 2) 查询参数（SSE / EventSource 无法设 header）
        token = request.query_params.get("token", "")
    if token:
        # 给了 token 就必须精确匹配，不静默回退到 cookie
        if hmac.compare_digest(token, cfg.gateway.auth_token):
            return
        raise HTTPException(status_code=401, detail="unauthorized")
    # 3) 无 token —— 尝试签名 session cookie（浏览器登录流）
    cookie = request.cookies.get(_SESSION_COOKIE, "")
    if cookie and _verify_session(cfg.gateway.auth_token, cookie, int(time.time())):
        return
    raise HTTPException(status_code=401, detail="unauthorized")


# ── Startup time tracking ──

_start_time = time.monotonic()


# ── App reference ──

_app_ref = None


def register_dashboard(app: FastAPI, application):
    global _app_ref, _log_handler
    _app_ref = application
    from src.gateway import GATEWAY_HOST

    # Lazy-register log handler only when dashboard is actually mounted
    if _log_handler is None:
        _log_handler = _LogBufferHandler()
        logging.getLogger("flyclaw").addHandler(_log_handler)

    _template_path = Path(__file__).parent / "templates" / "dashboard.html"
    _html_template = _template_path.read_text(encoding="utf-8")

    router = APIRouter(tags=["dashboard"])

    _LOGIN_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlyClaw 登录</title>
<style>body{font-family:system-ui,sans-serif;max-width:340px;margin:80px auto;padding:0 16px;color:#222}
h2{margin-bottom:8px}input{width:100%;padding:10px;margin:8px 0;box-sizing:border-box;border:1px solid #ccc;border-radius:4px}
button{width:100%;padding:11px;background:#2563eb;color:#fff;border:0;border-radius:4px;cursor:pointer;font-size:15px}
.err{color:#b91c1c;margin:4px 0}.hint{color:#666;font-size:13px;margin:12px 0 0;line-height:1.5}
code{background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:12px}</style></head>
<body><h2>FlyClaw Dashboard</h2>
<!--ERR-->
<form method="post" action="/dashboard/login">
<input type="password" name="token" placeholder="Gateway Auth Token" autofocus required>
<button type="submit">登录</button>
</form>
<p class="hint">首次启动会自动生成令牌,见启动日志或文件 <code>~/.flyclaw/data/gateway_token</code><br>(若你在 config.yaml 手动设置过 <code>gateway.auth_token</code>,则用那个)。</p>
</body></html>"""

    @router.get("/dashboard/login", response_class=HTMLResponse)
    async def dashboard_login_page(request: Request):
        cfg = _app_ref.config
        if not cfg.gateway.auth_token:
            return RedirectResponse(url="/dashboard", status_code=303)
        return HTMLResponse(_LOGIN_HTML)

    @router.post("/dashboard/login")
    async def dashboard_login_submit(request: Request):
        cfg = _app_ref.config
        if not cfg.gateway.auth_token:
            return RedirectResponse(url="/dashboard", status_code=303)
        # 手动解析 form-urlencoded body，避免引入 python-multipart 依赖。
        raw = await request.body()
        if len(raw) > 4096:  # 登录表单极小，超大 body 直接拒（防 DoS 探测）
            return HTMLResponse(_LOGIN_HTML, status_code=401)
        body = raw.decode("utf-8", errors="replace")
        token = urllib.parse.parse_qs(body).get("token", [""])[0].strip()
        if not hmac.compare_digest(token, cfg.gateway.auth_token):
            return HTMLResponse(
                _LOGIN_HTML.replace("<!--ERR-->", '<p class="err">无效的 token</p>'),
                status_code=401,
            )
        cookie_val = _sign_session(cfg.gateway.auth_token, int(time.time()))
        resp = RedirectResponse(url="/dashboard", status_code=303)
        # httponly=True: JS 读不到 → XSS 偷不走。samesite=strict: CSRF 安全。
        # secure=False: gateway 在内网可能纯 HTTP；上 TLS 反代后改 True。
        resp.set_cookie(
            _SESSION_COOKIE,
            cookie_val,
            max_age=_SESSION_TTL_SECONDS,
            httponly=True,
            samesite="strict",
            secure=False,
        )
        return resp

    @router.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page(request: Request):
        cfg = _app_ref.config
        if cfg.gateway.auth_token:
            cookie = request.cookies.get(_SESSION_COOKIE, "")
            if not _verify_session(cfg.gateway.auth_token, cookie, int(time.time())):
                return RedirectResponse(url="/dashboard/login", status_code=303)
        html = _html_template
        # auth_token 故意不再嵌入 —— 鉴权改走 HttpOnly cookie
        html = html.replace('"{{ model_provider }}"', json.dumps(cfg.model.provider))
        html = html.replace('"{{ model_name }}"', json.dumps(cfg.model.name))
        return HTMLResponse(html)

    @router.get("/api/dashboard/status")
    async def dashboard_status(request: Request):
        _check_auth(request)
        cfg = _app_ref.config
        uptime = time.monotonic() - _start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)

        skills = _app_ref.skills_cache or []
        status = {
            "version": __version__,
            "uptime": f"{hours}h {minutes}m {seconds}s",
            "uptime_seconds": int(uptime),
            "model": {
                "provider": cfg.model.provider,
                "name": cfg.model.name,
                "base_url": cfg.model.base_url or "default",
                "has_fallbacks": bool(cfg.model.fallbacks),
            },
            "gateway": f"{GATEWAY_HOST}:{cfg.gateway.port}",
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
        _check_auth(request)
        import re as _re

        sessions = []

        # Only show chat sessions: channel:(user|group|sN):... 以及 DM 塌缩的 channel:dm
        # dm 无后续段，故用 (:|$) 兼容 "qq:dm" / "weixin:dm" 这种两段塌缩 key
        _chat_pattern = _re.compile(r"^(qq|weixin):(user|group|s\d+|dm)(:|$)")

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
                for tid in await _app_ref.state_store.list_threads():
                    if tid not in active_ids and _chat_pattern.match(tid):
                        state = await _app_ref.state_store.load(tid)
                        msg_count = len(state.messages) if state else 0
                        sessions.append(
                            {
                                "thread_id": tid,
                                "last_active": None,
                                "status": "idle",
                                "checkpoint_count": msg_count,
                            }
                        )
            except Exception:
                pass

        return sessions

    @router.post("/api/dashboard/sessions/{thread_id}/reset")
    async def dashboard_reset_session(thread_id: str, request: Request):
        _check_auth(request)
        if not _app_ref.state_store:
            raise HTTPException(status_code=500, detail="State store not initialized")
        try:
            await _app_ref.state_store.delete(thread_id)
            if _app_ref.session_tracker:
                _app_ref.session_tracker.remove(thread_id)
            if _app_ref.agent_loop:
                _app_ref.agent_loop.invalidate_memory_cache()
            # Clean up tool cache files for this thread
            from src.agent.tool_cache import clear_thread_cache

            clear_thread_cache(thread_id)
            return {"ok": True}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @router.get("/api/dashboard/cron")
    async def dashboard_cron(request: Request):
        _check_auth(request)
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
        _check_auth(request)
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
        _check_auth(request)
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
                "channel_disabled": {ch: s.name in chans for ch, chans in config.skills.channel_disabled.items()},
            }
            for s in skills
        ]

    @router.get("/api/dashboard/config")
    async def dashboard_config(request: Request):
        _check_auth(request)
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
                "host": GATEWAY_HOST,
                "port": cfg.gateway.port,
                "has_auth": bool(cfg.gateway.auth_token),
            },
            "session": {
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
            },
            "memory_store": {
                "enabled": getattr(cfg.memory_store, "enabled", False),
                "db_path": cfg.memory_store.db_path,
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
        _check_auth(request)
        return [
            {"ts": r.created, "level": r.levelname, "logger": r.name, "message": r.getMessage()} for r in _log_buffer
        ]

    # ── Model switching ─────────────────────────────────────

    @router.get("/api/dashboard/models")
    async def dashboard_list_models(request: Request):
        _check_auth(request)
        cfg = _app_ref.config
        from src.agent.client import FallbackChain

        active_client = _app_ref.agent_loop._client if _app_ref.agent_loop else None
        if isinstance(active_client, FallbackChain):
            idx = active_client._active_idx
            available = []
            for i, c in enumerate(active_client._all):
                available.append({"name": c.model, "active": i == idx})
            current = available[idx] if idx < len(available) else available[0]
        else:
            current = {"provider": cfg.model.provider, "name": cfg.model.name}
            available = [{"provider": current["provider"], "name": current["name"], "active": True}]
        return {"current": current, "available": available}

    @router.post("/api/dashboard/models/switch")
    async def dashboard_switch_model(request: Request):
        _check_auth(request)
        body = await request.json()
        provider = body.get("provider", "").strip()
        name = body.get("name", "").strip()
        if not provider or not name:
            raise HTTPException(status_code=400, detail="provider and name required")
        if not _app_ref.agent_loop:
            raise HTTPException(status_code=500, detail="agent_loop not initialized")

        active_client = _app_ref.agent_loop._client
        from src.agent.client import FallbackChain

        if isinstance(active_client, FallbackChain):
            all_clients = active_client._all
            old_idx = active_client._active_idx
            old_name = all_clients[old_idx].model
            target_idx = None
            for i, c in enumerate(all_clients):
                if c.model == name:
                    target_idx = i
                    break
            if target_idx is None:
                raise HTTPException(status_code=404, detail=f"Model {name} not found in chain")
            active_client.switch_to(target_idx)
            logger.info("Model switched from %s to %s", old_name, name)
            return {"ok": True, "previous": old_name, "current": name}
        else:
            # Single model (no chain) — create new model for switch
            from src.agent.client import create_client

            cfg = _app_ref.config
            fb_base_url = None
            fb_api_key = None
            fb_multimodal = None
            for fb in cfg.model.fallbacks:
                if fb.provider == provider and fb.name == name:
                    fb_base_url = getattr(fb, "base_url", None)
                    fb_api_key = getattr(fb, "api_key", None)
                    fb_multimodal = getattr(fb, "multimodal", False)
                    break
            # 切回 primary(无匹配 fb)用 cfg.model 的 multimodal;否则用匹配 fb 的
            target_multimodal = fb_multimodal if fb_multimodal is not None else getattr(cfg.model, "multimodal", False)
            try:
                new_model = create_client(
                    provider,
                    name,
                    cfg.model.temperature,
                    base_url=fb_base_url or cfg.model.base_url,
                    api_key=fb_api_key or cfg.model.api_key,
                    multimodal=target_multimodal,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to create model: {e}")
            old_name = f"{cfg.model.provider}/{cfg.model.name}"
            _app_ref.agent_loop.swap_client(new_model)
            logger.info("Model switched from %s to %s/%s", old_name, provider, name)
            return {"ok": True, "previous": old_name, "current": f"{provider}/{name}"}

    # ── Auth dashboard API ──────────────────────────────────

    @router.get("/api/dashboard/users")
    async def dashboard_users(request: Request):
        _check_auth(request)
        cfg = _app_ref.config
        if not getattr(cfg.auth, "enabled", False):
            return {"enabled": False, "users": []}
        try:
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is None:
                return {"enabled": True, "users": []}
            users = await rbac.store.list_users()
            result = []
            for u in users:
                devices = await rbac.store.list_user_devices(u.user_id)
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
        _check_auth(request)
        try:
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is None:
                return []
            return [d.model_dump() for d in await rbac.store.list_user_devices(user_id)]
        except Exception:
            return []

    @router.patch("/api/dashboard/users/{user_id}/role")
    async def dashboard_update_role(user_id: str, request: Request):
        _check_auth(request)
        try:
            body = await request.json()
            from src.auth.models import UserRole
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is None:
                raise HTTPException(status_code=500, detail="RBAC not initialized")
            new_role = UserRole(body.get("role", "guest"))
            await rbac.store.update_user_role(user_id, new_role)
            return {"ok": True, "user_id": user_id, "role": new_role.value}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/api/dashboard/users/{user_id}")
    async def dashboard_delete_user(user_id: str, request: Request):
        _check_auth(request)
        try:
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is None:
                raise HTTPException(status_code=500, detail="RBAC not initialized")
            removed = await rbac.store.delete_user(user_id)
            return {"ok": removed}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/api/dashboard/devices/{device_id}")
    async def dashboard_delete_device(device_id: str, request: Request):
        _check_auth(request)
        try:
            from src.auth.rbac import get_rbac

            rbac = get_rbac()
            if rbac is None:
                raise HTTPException(status_code=500, detail="RBAC not initialized")
            removed = await rbac.store.delete_device(device_id)
            return {"ok": removed}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Memory API ──────────────────────────────────────

    @router.get("/api/dashboard/memory/memories")
    async def dashboard_memory_list(request: Request):
        _check_auth(request)
        try:
            from src.tools.memory_tools import get_memory_store

            store = await get_memory_store()
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
        _check_auth(request)
        try:
            from src.tools.memory_tools import get_memory_store
            import json

            store = await get_memory_store()
            return json.loads(await store.recall(key))
        except Exception as e:
            return {"error": str(e)}

    @router.delete("/api/dashboard/memory/forget/{key:path}")
    async def dashboard_memory_forget(key: str, request: Request):
        _check_auth(request)
        try:
            from src.tools.memory_tools import get_memory_store
            import json

            store = await get_memory_store()
            result = json.loads(await store.forget(key))
            if _app_ref.agent_loop:
                _app_ref.agent_loop.invalidate_memory_cache()
            return result
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @router.post("/api/dashboard/memory/remember")
    async def dashboard_memory_save(request: Request):
        _check_auth(request)
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
            if _app_ref.agent_loop:
                _app_ref.agent_loop.invalidate_memory_cache()
            return result
        except HTTPException:
            raise
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Phase 1: Dashboard Enhancements ─────────────────────────

    @router.post("/api/dashboard/config/reload")
    async def dashboard_config_reload(request: Request):
        """Hot-reload configuration."""
        _check_auth(request)
        try:
            from src.config import load_config

            new_cfg = load_config()
            _app_ref.config = new_cfg
            if _app_ref.agent_loop:
                from src.agent.client import create_chain

                old_client = _app_ref.agent_loop._client
                _app_ref.agent_loop.swap_client(create_chain(new_cfg))
                if hasattr(old_client, "close"):
                    try:
                        await old_client.close()
                    except Exception:
                        pass
            return {"ok": True, "message": "Config reloaded"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @router.get("/api/dashboard/search")
    async def dashboard_session_search(request: Request, q: str = ""):
        """Search session history via FTS5."""
        _check_auth(request)
        if not q:
            return {"results": []}
        try:
            from src.session_index.store import get_session_index

            idx = get_session_index()
            if not idx:
                return {"results": [], "error": "Session index not initialized"}
            results = await idx.search(q, limit=20)
            return {"results": results, "query": q}
        except Exception as e:
            return {"results": [], "error": str(e)}

    @router.get("/api/dashboard/sessions/{thread_id}/messages")
    async def dashboard_session_messages(thread_id: str, request: Request, limit: int = 50, offset: int = 0):
        """Get message history for a session."""
        _check_auth(request)
        try:
            if not _app_ref.state_store:
                raise HTTPException(status_code=500, detail="State store not initialized")
            state = await _app_ref.state_store.load(thread_id)
            if not state:
                raise HTTPException(status_code=404, detail="Session not found")
            messages = state.messages[offset : offset + limit]
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
        _check_auth(request)
        cfg = _app_ref.config
        if not getattr(cfg.memory, "enabled", False):
            return {"enabled": False}
        try:
            searcher = _app_ref.memory_searcher
            if not searcher:
                return {"enabled": True, "error": "Memory searcher not initialized"}
            docs = await searcher.store.list_documents()
            stats = {
                "total_documents": len(docs),
            }
            return {"enabled": True, **stats}
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    @router.get("/api/dashboard/security/audit")
    async def dashboard_security_audit(request: Request):
        """Run and return security audit results."""
        _check_auth(request)
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
        _check_auth(request)
        try:
            from src.skills.curator import SkillCurator
            from pathlib import Path

            from src.instance import skills_dir

            curator = SkillCurator(skills_dir())
            skills = _app_ref.skills_cache or []
            lifecycle_counts = {"active": 0, "stale": 0, "archived": 0}
            for s in skills:
                usage = await curator.manager.get_usage(s.name)
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
        _check_auth(request)
        try:
            from src.skills.curator import SkillCurator
            from pathlib import Path

            from src.instance import skills_dir

            curator = SkillCurator(skills_dir())
            result = await curator.review_skills(dry_run=dry_run)
            return result
        except Exception as e:
            return {"error": str(e)}

    @router.get("/api/dashboard/learning/status")
    async def dashboard_learning_status(request: Request):
        """Get learning loop status."""
        _check_auth(request)
        cfg = _app_ref.config
        return {
            "memory_store_enabled": getattr(cfg.memory_store, "enabled", False),
            "skill_curation": True,
            "session_end_extraction": getattr(cfg.memory_store, "enabled", False),
        }

    @router.post("/api/dashboard/learning/trigger")
    async def dashboard_learning_trigger(request: Request):
        """Manually trigger learning cycle."""
        _check_auth(request)
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
        _check_auth(request)
        try:
            from src.analytics.audit_store import get_audit_store

            store = get_audit_store()

            success_filter = None
            if status == "success":
                success_filter = True
            elif status == "error":
                success_filter = False

            entries = await store.query(
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
        _check_auth(request)
        try:
            from src.analytics.audit_store import get_audit_store

            store = get_audit_store()
            return await store.get_stats(days=days)
        except Exception as e:
            return {"error": str(e)}

    @router.post("/api/dashboard/config/update")
    async def dashboard_config_update(request: Request):
        """Update configuration values."""
        _check_auth(request)
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

            config_path = getattr(_app_ref, "_config_path", None)
            if not config_path:
                from src.instance import config_path as _cp

                config_path = str(_cp())
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

                    old_client = _app_ref.agent_loop._client
                    _app_ref.agent_loop.swap_client(create_chain(cfg))
                    if hasattr(old_client, "close"):
                        try:
                            await old_client.close()
                        except Exception:
                            pass
                except Exception:
                    pass

            # 2. Reload skills (skill config changes)
            # _reload_skills() already updates agent_loop._skills_prompt and dispatcher
            try:
                await _app_ref._reload_skills()
            except Exception:
                pass

            return {"ok": True, "message": "Configuration updated and saved"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Agent Flow SSE ─────────────────────────────────────────

    _sse_semaphore = asyncio.Semaphore(10)  # max concurrent SSE connections

    @router.get("/api/dashboard/flow/events")
    async def flow_events(request: Request):
        """SSE endpoint: stream agent/tool/message events in real-time."""
        _check_auth(request)

        # Reject new connections when at capacity
        if _sse_semaphore.locked():
            return JSONResponse(
                {"error": "Too many SSE connections"},
                status_code=429,
            )

        await _sse_semaphore.acquire()

        try:
            from src.events import get_event_bus

            bus = get_event_bus()
            queue: asyncio.Queue = asyncio.Queue(maxsize=200)

            FLOW_EVENTS = {
                "message.received",
                "message.replied",
                "agent_loop.started",
                "agent_loop.completed",
                "agent_loop.assistant_message",
                "agent.error",
                "tool.exec_started",
                "tool.exec_completed",
                "tool.exec_failed",
                "tool.approval_pending",
            }

            async def _on_event(**ctx):
                """Async handler: push event into per-client queue (thread-safe)."""
                try:
                    queue.put_nowait(ctx)
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                        queue.put_nowait(ctx)
                    except Exception:
                        pass

            subscriptions = []
            for ev_name in FLOW_EVENTS:
                sub = bus.subscribe(ev_name, _on_event)
                subscriptions.append(sub)

            async def event_generator():
                try:
                    yield f"data: {json.dumps({'type': 'connected'})}\n\n"
                    while True:
                        try:
                            item = await asyncio.wait_for(queue.get(), timeout=30.0)
                            payload = json.dumps(item, ensure_ascii=False, default=str)
                            yield f"data: {payload}\n\n"
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                except asyncio.CancelledError:
                    pass
                finally:
                    for sub in subscriptions:
                        bus.unsubscribe(sub.event, sub.handler)
                    _sse_semaphore.release()

            from starlette.responses import StreamingResponse

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception:
            _sse_semaphore.release()
            raise

    app.include_router(router)
    logger.info("Dashboard registered at /dashboard")
