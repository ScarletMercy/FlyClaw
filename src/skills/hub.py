from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Union
from urllib.parse import urlparse

import httpx

from .guard import (
    TRUSTED_REPOS,
    content_hash,
)
from .types import ScanResult, SkillBundle, SkillMeta

logger = logging.getLogger("flyclaw.skills.hub")

SKILLS_DIR = Path.home() / ".flyclaw" / "skills"
HUB_DIR = SKILLS_DIR / ".hub"
QUARANTINE_DIR = HUB_DIR / "quarantine"
LOCK_FILE = HUB_DIR / "lock.json"
AUDIT_LOG = HUB_DIR / "audit.log"
INDEX_CACHE_DIR = HUB_DIR / "index-cache"
INDEX_CACHE_TTL = 3600

_REDIRECT_CODES = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 5


def _normalize_bundle_path(path_value: str, *, field_name: str, allow_nested: bool) -> str:
    if not isinstance(path_value, str):
        raise ValueError(f"Unsafe {field_name}: expected a string")
    raw = path_value.strip()
    if not raw:
        raise ValueError(f"Unsafe {field_name}: empty path")
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = [part for part in path.parts if part not in {"", "."}]
    if normalized.startswith("/") or path.is_absolute():
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if re.fullmatch(r"[A-Za-z]:", parts[0]):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if not allow_nested and len(parts) != 1:
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    return "/".join(parts)


def _validate_skill_name(name: str) -> str:
    return _normalize_bundle_path(name, field_name="skill name", allow_nested=False)


def _validate_bundle_rel_path(rel_path: str) -> str:
    return _normalize_bundle_path(rel_path, field_name="bundle file path", allow_nested=True)


def _guarded_http_get(url: str, *, timeout: int = 20) -> Optional[httpx.Response]:
    try:
        from src.security.url_safety import is_safe_url
        safe, _ = is_safe_url(url)
        if not safe:
            logger.warning("Blocked unsafe hub URL: %s", url)
            return None
    except ImportError:
        pass

    current_url = url
    for _ in range(_MAX_REDIRECTS + 1):
        try:
            resp = httpx.get(current_url, timeout=timeout, follow_redirects=False)
        except httpx.HTTPError:
            return None
        if resp.status_code in _REDIRECT_CODES:
            location = resp.headers.get("location")
            if not location:
                return None
            from urllib.parse import urljoin
            current_url = urljoin(current_url, location)
            try:
                from src.security.url_safety import is_safe_url
                safe, _ = is_safe_url(current_url)
                if not safe:
                    logger.warning("Blocked unsafe redirect target: %s", current_url)
                    return None
            except ImportError:
                pass
            continue
        return resp
    return None


def _read_index_cache(key: str) -> Optional[Any]:
    cache_file = INDEX_CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        stat = cache_file.stat()
        if time.time() - stat.st_mtime > INDEX_CACHE_TTL:
            return None
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_index_cache(key: str, data: Any) -> None:
    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = INDEX_CACHE_DIR / f"{key}.json"
    try:
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _skill_meta_to_dict(meta: SkillMeta) -> dict:
    return {
        "name": meta.name,
        "description": meta.description,
        "source": meta.source,
        "identifier": meta.identifier,
        "trust_level": meta.trust_level,
        "repo": meta.repo,
        "path": meta.path,
        "tags": meta.tags,
        "extra": meta.extra,
    }


class SkillSource(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[SkillMeta]: ...
    @abstractmethod
    def fetch(self, identifier: str) -> Optional[SkillBundle]: ...
    @abstractmethod
    def inspect(self, identifier: str) -> Optional[SkillMeta]: ...
    @abstractmethod
    def source_id(self) -> str: ...
    def trust_level_for(self, identifier: str) -> str:
        return "community"


class SkillsShSource(SkillSource):
    BASE_URL = "https://skills.sh"
    SEARCH_URL = f"{BASE_URL}/api/search"
    _SKILL_LINK_RE = re.compile(
        r'href=["\']/(?P<id>(?!agents/|_next/|api/)[^"\'/]+/[^"\'/]+/[^"\'/]+)["\']'
    )
    _INSTALL_CMD_RE = re.compile(
        r'npx\s+skills\s+add\s+(?P<repo>https?://github\.com/[^\s<]+|[^\s<]+)'
        r'(?:\s+--skill\s+(?P<skill>[^\s<]+))?',
        re.IGNORECASE,
    )
    _PAGE_H1_RE = re.compile(r'<h1[^>]*>(?P<title>.*?)</h1>', re.IGNORECASE | re.DOTALL)
    _PROSE_H1_RE = re.compile(
        r'<div[^>]*class=["\'][^"\']*prose[^"\']*["\'][^>]*>.*?<h1[^>]*>(?P<title>.*?)</h1>',
        re.IGNORECASE | re.DOTALL,
    )
    _PROSE_P_RE = re.compile(
        r'<div[^>]*class=["\'][^"\']*prose[^"\']*["\'][^>]*>.*?<p[^>]*>(?P<body>.*?)</p>',
        re.IGNORECASE | re.DOTALL,
    )

    def source_id(self) -> str:
        return "skills-sh"

    def trust_level_for(self, identifier: str) -> str:
        canonical = self._normalize_identifier(identifier)
        parts = canonical.split("/", 2)
        if len(parts) >= 2:
            repo = f"{parts[0]}/{parts[1]}"
            if repo in TRUSTED_REPOS:
                return "trusted"
        return "community"

    def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
        if not query.strip():
            return self._featured_skills(limit)

        cache_key = f"skills_sh_search_{hashlib.md5(f'{query}|{limit}'.encode()).hexdigest()}"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return [SkillMeta(**item) for item in cached][:limit]

        try:
            resp = httpx.get(
                self.SEARCH_URL, params={"q": query, "limit": limit}, timeout=20,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return []

        items = data.get("skills", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            return []

        results: list[SkillMeta] = []
        for item in items[:limit]:
            meta = self._meta_from_search_item(item)
            if meta:
                results.append(meta)

        _write_index_cache(cache_key, [_skill_meta_to_dict(m) for m in results])
        return results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        canonical = self._normalize_identifier(identifier)
        parts = canonical.split("/", 2)
        if len(parts) < 3:
            return None
        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2]

        detail = self._fetch_detail_page(canonical)
        if isinstance(detail, dict):
            repo = detail.get("repo", repo)

        for candidate_path in self._candidate_paths(skill_path):
            files = self._fetch_from_raw(repo, candidate_path)
            if files and "SKILL.md" in files:
                skill_name = candidate_path.rstrip("/").split("/")[-1]
                return SkillBundle(
                    name=skill_name,
                    files=files,
                    source="skills.sh",
                    identifier=self._wrap_identifier(canonical),
                    trust_level=self.trust_level_for(canonical),
                    metadata=self._detail_to_metadata(canonical, detail),
                )

        return None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        canonical = self._normalize_identifier(identifier)
        parts = canonical.split("/", 2)
        if len(parts) < 3:
            return None

        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2]
        detail = self._fetch_detail_page(canonical)
        if isinstance(detail, dict):
            repo = detail.get("repo", repo)
            body_summary = detail.get("body_summary")
        else:
            body_summary = None

        description = f"Indexed by skills.sh from {repo}"
        if body_summary:
            description = body_summary

        return SkillMeta(
            name=skill_path.split("/")[-1],
            description=description,
            source="skills.sh",
            identifier=self._wrap_identifier(canonical),
            trust_level=self.trust_level_for(canonical),
            repo=repo,
            path=skill_path,
            extra={"detail_url": f"{self.BASE_URL}/{canonical}"},
        )

    def _normalize_identifier(self, identifier: str) -> str:
        if identifier.startswith("skills-sh/"):
            return identifier[len("skills-sh/"):]
        if identifier.startswith("skills.sh/"):
            return identifier[len("skills.sh/"):]
        return identifier

    def _wrap_identifier(self, canonical: str) -> str:
        return f"skills-sh/{canonical}"

    def _candidate_paths(self, skill_path: str) -> list[str]:
        base_name = skill_path.split("/")[-1]
        paths: list[str] = [
            f"skills/{base_name}",
            f".agents/skills/{base_name}",
            f".claude/skills/{base_name}",
            skill_path,
        ]
        seen: set[str] = set()
        unique: list[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _fetch_from_raw(self, repo: str, skill_path: str) -> Optional[dict[str, str]]:
        files: dict[str, str] = {}
        for branch in ("main", "master"):
            skill_md_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{skill_path}/SKILL.md"
            resp = _guarded_http_get(skill_md_url, timeout=15)
            if resp and resp.status_code == 200:
                files["SKILL.md"] = resp.text
                for ref_candidate in self._try_ref_files(repo, branch, skill_path):
                    name, content = ref_candidate
                    files[f"references/{name}"] = content
                return files
        return None

    def _try_ref_files(self, repo: str, branch: str, skill_path: str) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        common_files = ["api.md", "guide.md", "examples.md"]
        for fname in common_files:
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{skill_path}/references/{fname}"
            resp = _guarded_http_get(url, timeout=10)
            if resp and resp.status_code == 200:
                results.append((fname, resp.text))
        return results

    def _featured_skills(self, limit: int) -> list[SkillMeta]:
        cache_key = "skills_sh_featured"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return [SkillMeta(**item) for item in cached][:limit]

        try:
            resp = httpx.get(self.BASE_URL, timeout=20)
            if resp.status_code != 200:
                return []
        except httpx.HTTPError:
            return []

        seen: set[str] = set()
        results: list[SkillMeta] = []
        for match in self._SKILL_LINK_RE.finditer(resp.text):
            canonical = match.group("id")
            if canonical in seen:
                continue
            seen.add(canonical)
            parts = canonical.split("/", 2)
            if len(parts) < 3:
                continue
            repo = f"{parts[0]}/{parts[1]}"
            skill_path = parts[2]
            results.append(SkillMeta(
                name=skill_path.split("/")[-1],
                description=f"Featured on skills.sh from {repo}",
                source="skills.sh",
                identifier=self._wrap_identifier(canonical),
                trust_level=self.trust_level_for(canonical),
                repo=repo,
                path=skill_path,
            ))
            if len(results) >= limit:
                break

        _write_index_cache(cache_key, [_skill_meta_to_dict(m) for m in results])
        return results

    def _meta_from_search_item(self, item: dict) -> Optional[SkillMeta]:
        if not isinstance(item, dict):
            return None
        canonical = item.get("id")
        repo = item.get("source")
        skill_path = item.get("skillId")
        if not isinstance(canonical, str) or canonical.count("/") < 2:
            if not (isinstance(repo, str) and isinstance(skill_path, str)):
                return None
            canonical = f"{repo}/{skill_path}"

        parts = canonical.split("/", 2)
        if len(parts) < 3:
            return None
        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2]
        installs = item.get("installs")
        installs_label = f" · {int(installs):,} installs" if isinstance(installs, int) else ""

        return SkillMeta(
            name=str(item.get("name") or skill_path.split("/")[-1]),
            description=f"Indexed by skills.sh from {repo}{installs_label}",
            source="skills.sh",
            identifier=self._wrap_identifier(canonical),
            trust_level=self.trust_level_for(canonical),
            repo=repo,
            path=skill_path,
            extra={"installs": installs, "detail_url": f"{self.BASE_URL}/{canonical}"},
        )

    def _fetch_detail_page(self, identifier: str) -> Optional[dict]:
        cache_key = f"skills_sh_detail_{hashlib.md5(identifier.encode()).hexdigest()}"
        cached = _read_index_cache(cache_key)
        if isinstance(cached, dict):
            return cached
        try:
            resp = httpx.get(f"{self.BASE_URL}/{identifier}", timeout=20)
            if resp.status_code != 200:
                return None
        except httpx.HTTPError:
            return None
        detail = self._parse_detail_page(identifier, resp.text)
        if detail:
            _write_index_cache(cache_key, detail)
        return detail

    def _parse_detail_page(self, identifier: str, html: str) -> Optional[dict]:
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return None
        default_repo = f"{parts[0]}/{parts[1]}"
        skill_token = parts[2]
        repo = default_repo
        install_skill = skill_token

        install_command = None
        install_match = self._INSTALL_CMD_RE.search(html)
        if install_match:
            install_command = install_match.group(0).strip()
            repo_value = (install_match.group("repo") or "").strip()
            install_skill = (install_match.group("skill") or install_skill).strip()
            repo = self._extract_repo_slug(repo_value) or repo

        page_title = self._extract_first_match(self._PAGE_H1_RE, html)
        body_title = self._extract_first_match(self._PROSE_H1_RE, html)
        body_summary = self._extract_first_match(self._PROSE_P_RE, html)

        return {
            "repo": repo,
            "install_skill": install_skill,
            "page_title": page_title,
            "body_title": body_title,
            "body_summary": body_summary,
            "install_command": install_command,
            "detail_url": f"{self.BASE_URL}/{identifier}",
        }

    @staticmethod
    def _extract_repo_slug(value: str) -> Optional[str]:
        if not value:
            return None
        if value.startswith("https://github.com/"):
            slug = value[len("https://github.com/"):].rstrip("/")
            if "/" in slug:
                return slug.split("/")[0] + "/" + slug.split("/")[1]
        if "/" in value:
            parts = value.split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
        return None

    @staticmethod
    def _extract_first_match(pattern: re.Pattern, text: str) -> Optional[str]:
        m = pattern.search(text)
        if m:
            raw = m.group(1) if m.lastindex else m.group(0)
            return re.sub(r'<[^>]+>', '', raw).strip()
        return None

    def _detail_to_metadata(self, identifier: str, detail: Optional[dict]) -> dict:
        if not isinstance(detail, dict):
            return {}
        return {
            "detail_url": detail.get("detail_url", ""),
            "page_title": detail.get("page_title"),
            "weekly_installs": detail.get("weekly_installs"),
        }


class ClawHubSource(SkillSource):
    BASE_URL = "https://clawhub.ai/api/v1"

    def source_id(self) -> str:
        return "clawhub"

    def trust_level_for(self, identifier: str) -> str:
        return "community"

    def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
        query = query.strip()
        if query:
            direct = self._exact_slug_meta(query)
            if direct:
                return [direct]

        cache_key = f"clawhub_search_{hashlib.md5(f'{query}|{limit}'.encode()).hexdigest()}"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return self._finalize_search_results(
                query, [SkillMeta(**s) for s in cached], limit,
            )

        try:
            resp = httpx.get(
                f"{self.BASE_URL}/skills",
                params={"search": query, "limit": limit},
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return []

        skills_data = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(skills_data, list):
            return []

        results: list[SkillMeta] = []
        for item in skills_data[:limit]:
            slug = item.get("slug")
            if not slug:
                continue
            display_name = item.get("displayName") or item.get("name") or slug
            summary = item.get("summary") or item.get("description") or ""
            tags = self._normalize_tags(item.get("tags", []))
            results.append(SkillMeta(
                name=display_name,
                description=summary,
                source="clawhub",
                identifier=slug,
                trust_level="community",
                tags=tags,
            ))

        final_results = self._finalize_search_results(query, results, limit)
        _write_index_cache(cache_key, [_skill_meta_to_dict(s) for s in final_results])
        return final_results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        slug = identifier.split("/")[-1]
        skill_data = self._get_json(f"{self.BASE_URL}/skills/{slug}")
        if not isinstance(skill_data, dict):
            return None

        latest_version = self._resolve_latest_version(slug, skill_data)
        if not latest_version:
            return None

        files = self._download_zip(slug, latest_version)
        if "SKILL.md" not in files:
            version_data = self._get_json(f"{self.BASE_URL}/skills/{slug}/versions/{latest_version}")
            if isinstance(version_data, dict):
                files = self._extract_files(version_data) or files
                if "SKILL.md" not in files:
                    nested = version_data.get("version", {})
                    if isinstance(nested, dict):
                        files = self._extract_files(nested) or files

        if "SKILL.md" not in files:
            return None

        return SkillBundle(
            name=slug,
            files=files,
            source="clawhub",
            identifier=slug,
            trust_level="community",
        )

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        slug = identifier.split("/")[-1]
        data = self._coerce_skill_payload(self._get_json(f"{self.BASE_URL}/skills/{slug}"))
        if not isinstance(data, dict):
            return None
        tags = self._normalize_tags(data.get("tags", []))
        return SkillMeta(
            name=data.get("displayName") or data.get("name") or data.get("slug") or slug,
            description=data.get("summary") or data.get("description") or "",
            source="clawhub",
            identifier=data.get("slug") or slug,
            trust_level="community",
            tags=tags,
        )

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, list):
            return [str(t) for t in tags]
        if isinstance(tags, dict):
            return [str(k) for k in tags if str(k) != "latest"]
        return []

    @staticmethod
    def _coerce_skill_payload(data: Any) -> Optional[dict]:
        if not isinstance(data, dict):
            return None
        nested = data.get("skill")
        if isinstance(nested, dict):
            merged = dict(nested)
            latest_version = data.get("latestVersion")
            if latest_version is not None and "latestVersion" not in merged:
                merged["latestVersion"] = latest_version
            return merged
        return data

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        return [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t]

    @classmethod
    def _search_score(cls, query: str, meta: SkillMeta) -> int:
        query_norm = query.strip().lower()
        if not query_norm:
            return 1
        identifier = (meta.identifier or "").lower()
        name = (meta.name or "").lower()
        description = (meta.description or "").lower()
        normalized_identifier = " ".join(cls._query_terms(identifier))
        normalized_name = " ".join(cls._query_terms(name))
        query_terms = cls._query_terms(query_norm)
        identifier_terms = cls._query_terms(identifier)
        name_terms = cls._query_terms(name)
        score = 0
        if query_norm == identifier:
            score += 140
        if query_norm == name:
            score += 130
        if normalized_identifier == query_norm:
            score += 125
        if normalized_name == query_norm:
            score += 120
        if normalized_identifier.startswith(query_norm):
            score += 95
        if normalized_name.startswith(query_norm):
            score += 90
        if query_norm in identifier:
            score += 40
        if query_norm in name:
            score += 35
        if query_norm in description:
            score += 10
        for term in query_terms:
            if term in identifier_terms:
                score += 15
            if term in name_terms:
                score += 12
            if term in description:
                score += 3
        return score

    @staticmethod
    def _dedupe_results(results: list[SkillMeta]) -> list[SkillMeta]:
        seen: set[str] = set()
        deduped: list[SkillMeta] = []
        for r in results:
            key = (r.identifier or r.name).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped

    def _exact_slug_meta(self, query: str) -> Optional[SkillMeta]:
        slug = query.strip().split("/")[-1]
        if slug and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", slug):
            return self.inspect(slug)
        return None

    def _finalize_search_results(
        self, query: str, results: list[SkillMeta], limit: int
    ) -> list[SkillMeta]:
        query_norm = query.strip()
        if not query_norm:
            return self._dedupe_results(results)[:limit]
        filtered = [m for m in results if self._search_score(query_norm, m) > 0]
        filtered.sort(key=lambda m: (-self._search_score(query_norm, m), m.name.lower()))
        filtered = self._dedupe_results(filtered)
        exact = self._exact_slug_meta(query_norm)
        if exact:
            filtered = [m for m in filtered if self._search_score(query_norm, m) >= 20]
            filtered = self._dedupe_results([exact] + filtered)
        if filtered:
            return filtered[:limit]
        return self._dedupe_results(results)[:limit]

    def _get_json(self, url: str, timeout: int = 20) -> Optional[Any]:
        try:
            resp = httpx.get(url, timeout=timeout)
            if resp.status_code != 200:
                return None
            return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return None

    def _resolve_latest_version(self, slug: str, skill_data: dict) -> Optional[str]:
        latest = skill_data.get("latestVersion")
        if isinstance(latest, dict):
            version = latest.get("version")
            if isinstance(version, str) and version:
                return version
        tags = skill_data.get("tags")
        if isinstance(tags, dict):
            latest_tag = tags.get("latest")
            if isinstance(latest_tag, str) and latest_tag:
                return latest_tag
        versions_data = self._get_json(f"{self.BASE_URL}/skills/{slug}/versions")
        if isinstance(versions_data, list) and versions_data:
            first = versions_data[0]
            if isinstance(first, dict):
                version = first.get("version")
                if isinstance(version, str) and version:
                    return version
        return None

    def _extract_files(self, version_data: dict) -> dict[str, str]:
        files: dict[str, str] = {}
        file_list = version_data.get("files")
        if isinstance(file_list, dict):
            return {k: v for k, v in file_list.items() if isinstance(v, str)}
        if not isinstance(file_list, list):
            return files
        for file_meta in file_list:
            if not isinstance(file_meta, dict):
                continue
            fname = file_meta.get("path") or file_meta.get("name")
            if not fname or not isinstance(fname, str):
                continue
            inline_content = file_meta.get("content")
            if isinstance(inline_content, str):
                files[fname] = inline_content
                continue
            raw_url = file_meta.get("rawUrl") or file_meta.get("downloadUrl") or file_meta.get("url")
            if isinstance(raw_url, str) and raw_url.startswith("http"):
                resp = _guarded_http_get(raw_url, timeout=20)
                if resp and resp.status_code == 200:
                    files[fname] = resp.text
        return files

    def _download_zip(self, slug: str, version: str) -> dict[str, str]:
        files: dict[str, str] = {}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = httpx.get(
                    f"{self.BASE_URL}/download",
                    params={"slug": slug, "version": version},
                    timeout=30,
                    follow_redirects=True,
                )
                if resp.status_code == 429:
                    try:
                        retry_after = int(resp.headers.get("retry-after", "5"))
                    except (ValueError, TypeError):
                        retry_after = 5
                    retry_after = min(retry_after, 15)
                    logger.debug("ClawHub rate-limited for %s, retry %ds", slug, retry_after)
                    time.sleep(retry_after)
                    continue
                if resp.status_code != 200:
                    return files
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        try:
                            name = _validate_bundle_rel_path(info.filename)
                        except ValueError:
                            continue
                        if info.file_size > 500_000:
                            continue
                        try:
                            raw = zf.read(info.filename)
                            files[name] = raw.decode("utf-8")
                        except (UnicodeDecodeError, KeyError):
                            continue
                return files
            except zipfile.BadZipFile:
                return files
            except httpx.HTTPError:
                return files
        return files


class UrlSource(SkillSource):
    _VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")

    def source_id(self) -> str:
        return "url"

    def trust_level_for(self, identifier: str) -> str:
        return "community"

    def search(self, query: str, limit: int = 10) -> list[SkillMeta]:
        return []

    def _matches(self, identifier: str) -> bool:
        if not isinstance(identifier, str):
            return False
        ident = identifier.strip()
        if not ident.lower().startswith(("http://", "https://")):
            return False
        try:
            path = urlparse(ident).path
        except ValueError:
            return False
        return path.lower().endswith(".md")

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        if not self._matches(identifier):
            return None
        text = self._fetch_text(identifier.strip())
        if text is None:
            return None
        fm = _parse_frontmatter_quick(text)
        name = self._resolve_skill_name(fm, identifier.strip())
        description = str(fm.get("description") or "")
        return SkillMeta(
            name=name or "",
            description=description,
            source="url",
            identifier=identifier.strip(),
            trust_level="community",
            path=name or "",
        )

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        if not self._matches(identifier):
            return None
        url = identifier.strip()
        text = self._fetch_text(url)
        if text is None:
            return None
        fm = _parse_frontmatter_quick(text)
        name = self._resolve_skill_name(fm, url)
        skill_name = ""
        if name:
            try:
                skill_name = _validate_skill_name(name)
            except ValueError:
                return None
        if not skill_name:
            return None
        return SkillBundle(
            name=skill_name,
            files={"SKILL.md": text},
            source="url",
            identifier=url,
            trust_level="community",
        )

    @staticmethod
    def _fetch_text(url: str) -> Optional[str]:
        resp = _guarded_http_get(url, timeout=20)
        if resp and resp.status_code == 200:
            return resp.text
        return None

    @classmethod
    def _is_valid_skill_name(cls, name: Optional[str]) -> bool:
        if not isinstance(name, str):
            return False
        candidate = name.strip().lower()
        if not candidate or candidate in {"skill", "readme", "index", "unnamed-skill"}:
            return False
        return bool(cls._VALID_NAME_RE.match(candidate))

    @classmethod
    def _resolve_skill_name(cls, fm: dict, url: str) -> Optional[str]:
        fm_name = fm.get("name") if isinstance(fm, dict) else None
        if isinstance(fm_name, str) and cls._is_valid_skill_name(fm_name):
            return fm_name.strip()
        try:
            path = urlparse(url).path
        except ValueError:
            return None
        parts = [p for p in path.split("/") if p]
        if parts and parts[-1].lower() == "skill.md" and len(parts) >= 2:
            candidate = parts[-2]
            if cls._is_valid_skill_name(candidate):
                return candidate
        if parts:
            candidate = re.sub(r"\.md$", "", parts[-1], flags=re.IGNORECASE)
            if cls._is_valid_skill_name(candidate):
                return candidate
        return None


def _parse_frontmatter_quick(content: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not m:
        return {}
    try:
        import yaml
        data = yaml.safe_load(m.group(1).strip())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class HubLockFile:
    def __init__(self, path: Path = LOCK_FILE):
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "installed": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "installed": {}}

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(str(tmp), str(self.path))

    def record_install(
        self,
        name: str,
        source: str,
        identifier: str,
        trust_level: str,
        scan_verdict: str,
        skill_hash: str,
        install_path: str,
        files: list[str],
        metadata: Optional[dict] = None,
    ) -> None:
        data = self.load()
        data["installed"][name] = {
            "source": source,
            "identifier": identifier,
            "trust_level": trust_level,
            "scan_verdict": scan_verdict,
            "content_hash": skill_hash,
            "install_path": install_path,
            "files": files,
            "metadata": metadata or {},
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save(data)

    def record_uninstall(self, name: str) -> None:
        data = self.load()
        data["installed"].pop(name, None)
        self.save(data)

    def get_installed(self, name: str) -> Optional[dict]:
        return self.load()["installed"].get(name)

    def list_installed(self) -> list[dict]:
        data = self.load()
        return [{"name": n, **e} for n, e in data["installed"].items()]


def append_audit_log(
    action: str, skill_name: str, source: str,
    trust_level: str, verdict: str, extra: str = "",
) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [timestamp, action, skill_name, f"{source}:{trust_level}", verdict]
    if extra:
        parts.append(extra)
    line = " ".join(parts) + "\n"
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def ensure_hub_dirs() -> None:
    HUB_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(exist_ok=True)
    INDEX_CACHE_DIR.mkdir(exist_ok=True)
    if not LOCK_FILE.exists():
        LOCK_FILE.write_text('{"version": 1, "installed": {}}\n', encoding="utf-8")
    if not AUDIT_LOG.exists():
        AUDIT_LOG.touch()
    if not (HUB_DIR / "taps.json").exists():
        (HUB_DIR / "taps.json").write_text('{"taps": []}\n', encoding="utf-8")


def quarantine_bundle(bundle: SkillBundle) -> Path:
    ensure_hub_dirs()
    skill_name = _validate_skill_name(bundle.name)
    validated_files: list[tuple[str, Union[str, bytes]]] = []
    for rel_path, file_content in bundle.files.items():
        safe_rel_path = _validate_bundle_rel_path(rel_path)
        validated_files.append((safe_rel_path, file_content))

    dest = QUARANTINE_DIR / skill_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for rel_path, file_content in validated_files:
        file_dest = dest.joinpath(*rel_path.split("/"))
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(file_content, bytes):
            file_dest.write_bytes(file_content)
        else:
            file_dest.write_text(file_content, encoding="utf-8")

    return dest


def install_from_quarantine(
    quarantine_path: Path,
    skill_name: str,
    bundle: SkillBundle,
    scan_result: ScanResult,
) -> Path:
    safe_skill_name = _validate_skill_name(skill_name)
    quarantine_resolved = quarantine_path.resolve()
    quarantine_root = QUARANTINE_DIR.resolve()
    if not quarantine_resolved.is_relative_to(quarantine_root):
        raise ValueError(f"Unsafe quarantine path: {quarantine_path}")

    install_dir = SKILLS_DIR / safe_skill_name
    install_dir.parent.mkdir(parents=True, exist_ok=True)

    skill_hash_val = content_hash(quarantine_path)

    if install_dir.exists():
        shutil.rmtree(install_dir)

    shutil.move(str(quarantine_path), str(install_dir))

    lock = HubLockFile()
    lock.record_install(
        name=safe_skill_name,
        source=bundle.source,
        identifier=bundle.identifier,
        trust_level=bundle.trust_level,
        scan_verdict=scan_result.verdict,
        skill_hash=skill_hash_val,
        install_path=str(install_dir.relative_to(SKILLS_DIR)),
        files=list(bundle.files.keys()),
        metadata=bundle.metadata,
    )

    append_audit_log(
        "INSTALL", safe_skill_name, bundle.source,
        bundle.trust_level, scan_result.verdict,
        skill_hash_val,
    )

    return install_dir


def create_sources() -> list[SkillSource]:
    return [SkillsShSource(), ClawHubSource(), UrlSource()]


def _search_sources(
    sources: list[SkillSource],
    query: str,
    limit: int,
    timeout: float,
) -> list[SkillMeta]:
    if not sources:
        return []
    all_results: list[SkillMeta] = []
    pool = ThreadPoolExecutor(max_workers=min(len(sources), 4))
    try:
        futures = {}
        for src in sources:
            fut = pool.submit(src.search, query, limit)
            futures[fut] = src.source_id()
        try:
            for fut in as_completed(futures, timeout=timeout):
                try:
                    results = fut.result(timeout=0)
                    all_results.extend(results)
                except Exception:
                    pass
        except TimeoutError:
            pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return all_results


def _dedup_results(results: list[SkillMeta]) -> list[SkillMeta]:
    _SOURCE_RANK = {"clawhub": 2, "skills.sh": 1, "url": 0}
    seen: dict[str, SkillMeta] = {}
    for r in results:
        key = r.name
        if key not in seen:
            seen[key] = r
        elif _SOURCE_RANK.get(r.source, 0) > _SOURCE_RANK.get(seen[key].source, 0):
            seen[key] = r
    return list(seen.values())


def parallel_search(
    sources: list[SkillSource],
    query: str,
    limit: int = 10,
    source_filter: str = "all",
    timeout: float = 30,
) -> list[SkillMeta]:
    active = []
    for src in sources:
        if source_filter != "all" and src.source_id() != source_filter:
            continue
        active.append(src)

    if not active:
        return []

    primary = [s for s in active if s.source_id() in ("clawhub", "url")]
    fallback = [s for s in active if s.source_id() not in ("clawhub", "url")]

    primary_results = _search_sources(primary, query, limit, timeout)
    if primary_results:
        return _dedup_results(primary_results)[:limit]

    fallback_results = _search_sources(fallback, query, limit, timeout)
    return _dedup_results(fallback_results)[:limit]


def resolve_source(
    identifier: str, sources: list[SkillSource],
) -> Optional[SkillSource]:
    if identifier.startswith("skills-sh/") or identifier.startswith("skills.sh/"):
        for s in sources:
            if s.source_id() == "skills-sh":
                return s
        return None
    if identifier.startswith("http://") or identifier.startswith("https://"):
        if identifier.lower().endswith(".md"):
            for s in sources:
                if s.source_id() == "url":
                    return s
        return None
    for s in sources:
        if s.source_id() == "clawhub":
            return s
    return None
