"""Learning loop orchestrator that ties together curated sync and skill curation.

Memory extraction and skill creation are handled by the daily consolidation
service (src/services/daily_consolidation.py), which runs at a configurable
time (default 3 AM) via the cron system. Per-turn regex extraction for obvious
personal info (name, email) still runs in message.py.

This module handles session-end housekeeping:
1. Curated memory sync
2. Skill curation trigger
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.config import AppConfig
from src.memory.memory_sync import sync_memories_to_curated_files
from src.skills.curator import SkillCurator
from src.skills.manager import SkillManager

logger = logging.getLogger("flyclaw.learning")


class LearningLoop:
    """学习循环编排器。"""

    def __init__(self, config: AppConfig):
        self.config = config
        # Use configured skills directory if available, otherwise default
        from pathlib import Path

        from src.instance import skills_dir as _skills_dir

        skills_dir = _skills_dir()
        extra_dirs = getattr(config.skills, "extra_dirs", [])
        if extra_dirs:
            skills_dir = Path(extra_dirs[0]).expanduser().resolve()

        self.skill_manager = SkillManager(skills_dir)
        self.curator = SkillCurator(skills_dir)

    async def on_session_end(self, messages: list[dict]) -> dict:
        """会话结束时的学习循环。

        记忆和技能的批量提取已由每日整合服务（daily_consolidation）处理。
        此处仅处理策展同步和技能审查。

        Args:
            messages: 完整的会话消息列表

        Returns:
            学习结果字典
        """
        result = {
            "curated": False,
        }

        # 1. 同步策展记忆文件
        try:
            workspace = Path(self.config.agents.workspace).expanduser().resolve()
            await sync_memories_to_curated_files(workspace)
            result["curated"] = True
        except Exception as e:
            logger.warning("Learning loop curated sync failed: %s", e)

        # 2. 检查是否需要触发策展
        if self.curator.days_since_last_review() >= self.curator.review_interval_days:
            try:
                await self.curator.review_skills()
                result["curated"] = True
                logger.info("Learning loop: triggered skill curation")
            except Exception as e:
                logger.warning("Learning loop curation failed: %s", e)

        return result

    async def trigger_full_learning_cycle(self) -> dict:
        """手动触发完整学习循环。

        Returns:
            循环结果字典
        """
        result = {
            "memories_synced": False,
            "skills_reviewed": False,
            "review_result": None,
        }

        # 1. 同步策展记忆
        try:
            workspace = Path(self.config.agents.workspace).expanduser().resolve()
            await sync_memories_to_curated_files(workspace)
            result["memories_synced"] = True
        except Exception as e:
            logger.warning("Full learning cycle memory sync failed: %s", e)

        # 2. 审查技能库
        try:
            review_result = await self.curator.review_skills()
            result["skills_reviewed"] = True
            result["review_result"] = review_result
        except Exception as e:
            logger.warning("Full learning cycle curation failed: %s", e)

        return result
