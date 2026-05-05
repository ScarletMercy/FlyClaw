from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, RootModel

logger = logging.getLogger("myclaw.approval")


class ApprovalRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    tool_name: str
    args_digest: str
    args_preview: str
    sender_id: str = ""
    chat_id: str = ""
    message_id: str = ""
    created_at: float = Field(default_factory=time.time)
    timeout_seconds: int = 300


class ApprovalData(RootModel[dict[str, list[str]]]):
    """Pydantic model for validating the approvals.json file structure."""


class _PendingApproval:
    def __init__(self, request: ApprovalRequest):
        self.request = request
        self.event = asyncio.Event()
        self.decision: str = ""


class ApprovalManager:
    def __init__(self, data_dir: str = "data"):
        self._data_dir = Path(data_dir)
        self._pending: dict[str, _PendingApproval] = {}
        self._durable: dict[str, list[str]] = {}
        self._durable_path = self._data_dir / "approvals.json"
        self._load_durable()

    def _load_durable(self):
        if self._durable_path.exists():
            try:
                raw_data = json.loads(self._durable_path.read_text(encoding="utf-8"))
                # Validate with Pydantic model
                validated = ApprovalData.model_validate(raw_data if isinstance(raw_data, dict) else {})
                self._durable = validated.root
            except Exception as e:
                logger.warning("Failed to load approvals.json: %s", e)
                self._durable = {}

    def _save_durable(self):
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            import tempfile

            fd, tmp_path = tempfile.mkstemp(dir=str(self._data_dir), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._durable, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(self._durable_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.error("Failed to save durable approvals: %s", e)

    @staticmethod
    def _make_digest(tool_name: str, args_preview: str) -> str:
        return hashlib.sha256(f"{tool_name}:{args_preview}".encode()).hexdigest()[:16]

    def needs_approval(self, tool_name: str, args: str, approval_mode: str, denylisted: bool) -> bool:
        if approval_mode == "off":
            return False
        if approval_mode == "always":
            return True
        if approval_mode == "on_denylist_miss":
            return denylisted
        return False

    def has_durable_approval(self, tool_name: str, args: str) -> bool:
        digest = self._make_digest(tool_name, args[:200])
        entries = self._durable.get(tool_name, [])
        return digest in entries

    def request_approval(
        self,
        tool_name: str,
        args: str,
        sender_id: str = "",
        chat_id: str = "",
        message_id: str = "",
        timeout_seconds: int = 300,
    ) -> ApprovalRequest:
        req = ApprovalRequest(
            tool_name=tool_name,
            args_digest=self._make_digest(tool_name, args[:200]),
            args_preview=args[:300],
            sender_id=sender_id,
            chat_id=chat_id,
            message_id=message_id,
            timeout_seconds=timeout_seconds,
        )
        self._pending[req.id] = _PendingApproval(req)
        logger.info("Approval requested: %s (id=%s)", tool_name, req.id)
        return req

    async def await_approval(self, request_id: str, timeout: Optional[int] = None) -> str:
        pending = self._pending.get(request_id)
        if pending is None:
            return "deny"
        effective_timeout = timeout or pending.request.timeout_seconds
        try:
            await asyncio.wait_for(pending.event.wait(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            if pending.event.is_set():
                decision = pending.decision
            else:
                logger.warning("Approval timed out: %s", request_id)
                self._pending.pop(request_id, None)
                return "timeout"
        decision = pending.decision
        self._pending.pop(request_id, None)

        if decision == "allow_always":
            tool_name = pending.request.tool_name
            digest = pending.request.args_digest
            if tool_name not in self._durable:
                self._durable[tool_name] = []
            if digest not in self._durable[tool_name]:
                self._durable[tool_name].append(digest)
                self._save_durable()
            logger.info("Durable approval saved for %s (%s)", tool_name, digest)

        return decision

    def resolve(self, request_id: str, decision: str) -> bool:
        if decision not in ("allow_once", "allow_always", "deny"):
            logger.warning("Invalid approval decision: %s", decision)
            return False
        pending = self._pending.get(request_id)
        if pending is None:
            logger.warning("Unknown approval request: %s", request_id)
            return False
        pending.decision = decision
        pending.event.set()
        logger.info("Approval resolved: %s -> %s", request_id, decision)
        return True

    def get_pending(self, request_id: str) -> Optional[ApprovalRequest]:
        pending = self._pending.get(request_id)
        return pending.request if pending else None

    def list_pending(self) -> list[ApprovalRequest]:
        return [p.request for p in self._pending.values()]


_approval_manager: Optional[ApprovalManager] = None


def get_approval_manager() -> ApprovalManager:
    global _approval_manager
    if _approval_manager is None:
        _approval_manager = ApprovalManager()
    return _approval_manager
