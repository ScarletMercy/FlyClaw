"""Background skill curator that reviews and maintains the skill library.

Runs periodically (triggered by idle or manual command) to:
- Transition skills through lifecycle states (active -> stale -> archived)
- Run LLM consolidation to merge overlapping skills into umbrellas
- Archive/restore skills
- Clean up unused agent-created skills
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from src.skills.manager import SkillManager
from src.skills.types import Skill
from src.utils.tz import now_iso

logger = logging.getLogger("flyclaw.skills.curator")

_STALE_THRESHOLD_DAYS = 30
_ARCHIVE_THRESHOLD_DAYS = 90

CURATOR_REVIEW_PROMPT = (
    "你是 flyclaw 的后台技能策展人。这是一次伞形构建整合审查，"
    "不是被动审计，也不是简单去重。\n\n"
    "技能库的目标是类级(CLASS-LEVEL)指令和经验知识的集合。"
    "数百个窄技能（每个捕获一个会话的特定 bug）是库的失败——不是特性。"
    "Agent 按描述匹配技能，不按精确名称；一个带有标记子节的宽泛伞形技能"
    "比五个窄的兄弟技能更易发现。\n\n"
    "硬性规则——不要违反：\n"
    "1. 不要修改非 agent 创建的技能（内置、hub 安装）。\n"
    "2. 不要删除任何技能。归档（移动到 .archive/）是最大破坏性操作。\n"
    "3. 不要修改 pinned=yes 的技能。\n"
    "4. 不要因为 use_count=0 就跳过整合。使用计数是新的，大多为零。"
    "按内容判断重叠。\n"
    "5. 不要因为'每个技能有不同的触发条件'就拒绝整合。"
    "正确标准是：'人类维护者会把它写成 N 个独立技能还是一个技能的 N 个标记子节？'\n\n"
    "如何工作——必须遵守：\n"
    "1. 扫描完整候选列表。识别前缀聚类（共享首词或领域关键词的技能）。"
    "例如可能找到：flyclaw-config-*、gateway-*、python-*、security-* 等。\n"
    "2. 对每个有 2+ 成员的聚类，问'这些技能服务的伞形类别是什么？"
    "维护者会为这个类别命名并写一个技能吗？'如果是，选择或创建伞形并吸收兄弟。\n"
    "3. 三种整合方式——每个聚类用合适的：\n"
    "   a. 合并到现有伞形——聚类中已有一个足够宽的技能。"
    '用 skill_manage(action="patch") 为每个兄弟添加标记子节，然后归档兄弟。\n'
    "   b. 创建新伞形 SKILL.md——没有现有成员足够宽。"
    '用 skill_manage(action="create") 写一个新的类级技能。归档被吸收的窄兄弟。\n'
    "   c. 降级为支撑文件——兄弟有窄但有价值的会话特定内容。"
    "移动到伞形的适当支撑目录（references/、templates/、scripts/）。"
    "然后归档旧兄弟。\n"
    "4. 也标记名称太窄的技能（包含 PR 号、功能代号、特定错误字符串、"
    "'audit'/'diagnosis'/'salvage' 会话产物）。"
    "这些几乎总是应作为类级伞形的子节或支撑文件。\n"
    "5. 迭代。完成一轮整合后，扫描剩余集合并寻找下一个伞形机会。"
    "不要在 3 次整合后就停止。\n\n"
    "完成后，输出你做了什么的结构化摘要。"
)


class SkillCurator:
    """后台技能策展人。"""

    def __init__(
        self,
        skills_dir: Path | None = None,
        review_interval_days: int = 7,
        stale_after_days: int = _STALE_THRESHOLD_DAYS,
        archive_after_days: int = _ARCHIVE_THRESHOLD_DAYS,
    ):
        if skills_dir is None:
            from src.instance import skills_dir as _sd

            skills_dir = _sd()
        self.skills_dir = skills_dir
        self.review_interval_days = review_interval_days
        self.stale_after_days = stale_after_days
        self.archive_after_days = archive_after_days
        self.state_file = skills_dir / ".curator_state"
        self.archive_dir = skills_dir / ".archive"
        self.reports_dir = skills_dir / ".curator_reports"
        self.state = self._load_state()
        self.manager = SkillManager(skills_dir)

    def _load_state(self) -> dict:
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
        self.state_file.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def days_since_last_review(self) -> int:
        if not self.state.get("last_review"):
            return self.review_interval_days + 1
        last = datetime.fromisoformat(self.state["last_review"])
        if last.tzinfo is None:
            last = last.astimezone()
        return (datetime.now().astimezone() - last).days

    async def review_skills(self, dry_run: bool = False) -> dict:
        """审查技能库，执行自动生命周期转换。"""
        skills = await self.manager.list_skills()
        changes = []

        for skill in skills:
            usage = await self.manager.get_usage(skill.name)
            if not usage:
                continue

            current_state = usage.get("state", "active")
            last_used = usage.get("last_used_at")
            last_viewed = usage.get("last_viewed_at")
            last_patched = usage.get("last_patched_at")
            pinned = usage.get("pinned", False)

            if pinned:
                continue

            last_activity = last_used or last_viewed or last_patched

            if current_state == "active" and self._is_stale(last_activity, days=self.stale_after_days):
                if not dry_run:
                    await self._update_skill_state(skill.name, "stale")
                changes.append(
                    {
                        "skill": skill.name,
                        "action": "marked_stale",
                        "reason": f"Not used for {self.stale_after_days} days",
                    }
                )

            elif current_state == "stale" and self._is_stale(last_activity, days=self.archive_after_days):
                if not dry_run:
                    self._archive_skill_dir(skill.name)
                    await self._update_skill_state(skill.name, "archived")
                changes.append(
                    {
                        "skill": skill.name,
                        "action": "archived",
                        "reason": f"Not used for {self.archive_after_days} days",
                    }
                )

            elif current_state == "stale" and not self._is_stale(last_activity, days=self.stale_after_days):
                if not dry_run:
                    await self._update_skill_state(skill.name, "active")
                changes.append(
                    {
                        "skill": skill.name,
                        "action": "reactivated",
                    }
                )

        self.state["last_review"] = now_iso()
        self.state["total_reviews"] = self.state.get("total_reviews", 0) + 1
        self._save_state()

        return {
            "reviewed_at": self.state["last_review"],
            "total_reviews": self.state["total_reviews"],
            "skills_reviewed": len(skills),
            "changes": changes,
            "dry_run": dry_run,
        }

    async def run_llm_consolidation(
        self,
        client: Any,
        tools: list,
        config: Any,
        dry_run: bool = False,
        max_rounds: int = 16,
    ) -> dict:
        """Spawn a background agent to consolidate agent-created skills into umbrellas."""
        agent_created = self._get_agent_created_skills()
        if not agent_created:
            return {"consolidation": "skipped", "reason": "no agent-created skills"}

        auto_result = await self.review_skills(dry_run=dry_run)

        candidate_names = [s for s in agent_created]
        report_prompt = (
            CURATOR_REVIEW_PROMPT
            + f"\n\n候选技能列表（仅 agent 创建）：\n"
            + "\n".join(f"  - {n}" for n in candidate_names)
            + f"\n\n总计：{len(candidate_names)} 个候选技能。"
        )

        from src.skills.review import spawn_background_review

        messages = [{"role": "user", "content": report_prompt}]

        summary = await spawn_background_review(
            client=client,
            tools=tools,
            config=config,
            messages_snapshot=messages,
            review_skills=True,
            review_memory=False,
            max_rounds=max_rounds,
        )

        self._write_report(auto_result, summary, dry_run)

        return {
            "consolidation": "completed",
            "auto_changes": auto_result.get("changes", []),
            "llm_summary": summary,
            "candidates": len(candidate_names),
            "dry_run": dry_run,
        }

    def _get_agent_created_skills(self) -> list[str]:
        usage_file = self.skills_dir / ".usage.json"
        if not usage_file.exists():
            return []
        try:
            data = json.loads(usage_file.read_text(encoding="utf-8"))
            return [
                name
                for name, record in data.items()
                if record.get("created_by") == "agent" and record.get("state") != "archived"
            ]
        except Exception:
            return []

    def _is_stale(self, last_used: Optional[str], days: int) -> bool:
        if not last_used:
            return True
        try:
            last = datetime.fromisoformat(last_used)
            if last.tzinfo is None:
                last = last.astimezone()
            return (datetime.now().astimezone() - last) > timedelta(days=days)
        except Exception:
            return True

    async def _update_skill_state(self, skill_name: str, new_state: str) -> None:
        def _set_state(record):
            record["state"] = new_state
            if new_state == "archived":
                record["archived_at"] = now_iso()

        await self.manager._mutate_usage(skill_name, _set_state)

    def _archive_skill_dir(self, skill_name: str) -> bool:
        """Move skill directory to .archive/."""
        skill_dir = self.skills_dir / skill_name
        if not skill_dir.exists():
            return False
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        dest = self.archive_dir / skill_name
        if dest.exists():
            dest = self.archive_dir / f"{skill_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            shutil.move(str(skill_dir), str(dest))
            return True
        except Exception as e:
            logger.warning("Failed to archive skill %s: %s", skill_name, e)
            return False

    def _write_report(self, auto_result: dict, llm_summary: str, dry_run: bool) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_dryrun" if dry_run else ""
        report_file = self.reports_dir / f"{ts}{suffix}.md"
        lines = [
            f"# Curator Report — {now_iso()}",
            f"Dry run: {dry_run}",
            "",
            "## Automatic Transitions",
        ]
        for change in auto_result.get("changes", []):
            lines.append(f"- {change.get('skill')}: {change.get('action')} ({change.get('reason', '')})")
        if llm_summary:
            lines.append("")
            lines.append("## LLM Consolidation Summary")
            lines.append(llm_summary)
        report_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Curator report written to %s", report_file)
