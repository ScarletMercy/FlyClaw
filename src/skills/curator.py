"""Background skill curator that reviews and maintains the skill library.

Runs periodically (triggered by idle or manual command) to:
- Transition skills through lifecycle states (active → stale → archived)
- Detect and propose merges for overlapping skills
- Clean up unused skills
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.skills.manager import SkillManager
from src.skills.types import Skill

logger = logging.getLogger("myclaw.skills.curator")

_DEFAULT_REVIEW_INTERVAL_DAYS = 7
_STALE_THRESHOLD_DAYS = 30
_ARCHIVE_THRESHOLD_DAYS = 90


class SkillCurator:
    """后台技能策展人。"""

    def __init__(
        self,
        skills_dir: Path = Path.home() / ".myclaw" / "skills",
        review_interval_days: int = _DEFAULT_REVIEW_INTERVAL_DAYS,
    ):
        self.skills_dir = skills_dir
        self.review_interval_days = review_interval_days
        self.state_file = skills_dir / ".curator_state"
        self.state = self._load_state()
        self.manager = SkillManager(skills_dir)

    def _load_state(self) -> dict:
        """加载策展状态。"""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "last_review": None,
            "total_reviews": 0,
            "skills": {},
        }

    def _save_state(self) -> None:
        """保存策展状态。"""
        self.state_file.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def days_since_last_review(self) -> int:
        """距离上次审查的天数。"""
        if not self.state.get("last_review"):
            return self.review_interval_days + 1  # Force first review
        last = datetime.fromisoformat(self.state["last_review"])
        return (datetime.now() - last).days

    async def review_skills(self, dry_run: bool = False) -> dict:
        """审查技能库，执行生命周期转换。
        
        Args:
            dry_run: 如果为 True，只报告变更不执行
        
        Returns:
            审查结果字典
        """
        skills = self.manager.list_skills()
        changes = []

        for skill in skills:
            usage = self.manager.get_usage(skill.name)
            if not usage:
                continue

            current_state = usage.get("state", "active")
            last_used = usage.get("last_used_at")
            pinned = usage.get("pinned", False)

            # 跳过的技能
            if pinned:
                continue

            # 活跃 → 陈旧（30 天未使用）
            if current_state == "active" and self._is_stale(last_used, days=_STALE_THRESHOLD_DAYS):
                if not dry_run:
                    self._update_skill_state(skill.name, "stale")
                changes.append({
                    "skill": skill.name,
                    "action": "marked_stale",
                    "reason": f"Not used for {_STALE_THRESHOLD_DAYS} days",
                })

            # 陈旧 → 归档（90 天未使用）
            elif current_state == "stale" and self._is_stale(last_used, days=_ARCHIVE_THRESHOLD_DAYS):
                if not dry_run:
                    self._update_skill_state(skill.name, "archived")
                changes.append({
                    "skill": skill.name,
                    "action": "archived",
                    "reason": f"Not used for {_ARCHIVE_THRESHOLD_DAYS} days",
                })

        # 更新审查时间
        self.state["last_review"] = datetime.now().isoformat()
        self.state["total_reviews"] = self.state.get("total_reviews", 0) + 1
        self._save_state()

        return {
            "reviewed_at": self.state["last_review"],
            "total_reviews": self.state["total_reviews"],
            "skills_reviewed": len(skills),
            "changes": changes,
            "dry_run": dry_run,
        }

    def _is_stale(self, last_used: Optional[str], days: int) -> bool:
        """检查技能是否过期。"""
        if not last_used:
            return True
        try:
            last = datetime.fromisoformat(last_used)
            return (datetime.now() - last) > timedelta(days=days)
        except Exception:
            return True

    def _update_skill_state(self, skill_name: str, new_state: str) -> None:
        """更新技能状态。"""
        usage_file = self.skills_dir / ".usage.json"
        if not usage_file.exists():
            return

        try:
            usage_data = json.loads(usage_file.read_text(encoding="utf-8"))
            if skill_name in usage_data:
                usage_data[skill_name]["state"] = new_state
                if new_state == "archived":
                    usage_data[skill_name]["archived_at"] = datetime.now().isoformat()
                usage_file.write_text(
                    json.dumps(usage_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as e:
            logger.warning("Failed to update skill state: %s", e)



