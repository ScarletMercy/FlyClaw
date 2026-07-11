# FlyClaw

轻量级多渠道 AI 助手框架。运行时仅 ~100MB 内存，39 个工具（含浏览器/画布/桌面控制等可选扩展），支持 QQ/微信渠道接入。事件驱动架构，全异步设计。

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-ScarletMercy%2FFlyClaw-blue?logo=github)](https://github.com/ScarletMercy/FlyClaw)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/ScarletMercy/FlyClaw/blob/main/LICENSE)

---

## 简介

FlyClaw 是一个自包含的 AI 助手框架，直接运行在你的设备上。它通过 QQ 或微信与你对话，能执行命令、读写文件、搜索网页、控制浏览器、处理图片/音频/视频，还能定时执行任务。支持配置热重载、斜杠命令、上下文自动压缩，内置事件驱动钩子系统。借鉴了 [hermes-agent](https://github.com/nousresearch/hermes-agent) 和 [openclaw](https://github.com/openclaw/openclaw) 的设计理念。

## 功能特性

- **多渠道接入** — QQ Bot（官方 API，C2C 私聊）、微信（iLink Bot API，私聊）
- **39 个工具**（核心 + 可选扩展）— 命令执行、文件操作、Web 搜索、浏览器自动化（可选）、媒体理解、TTS、记忆、定时任务、Windows 桌面控制（可选）等
- **AgentLoop 引擎** — 工具并行执行、主动上下文压缩、中断/排队/steer 三种忙碌输入模式、工具参数自动修复、Guardrails 风暴检测
- **子代理委派** — research、coder、reviewer 三个内置角色，支持批量并行任务
- **记忆系统** — KV 记忆（SQLite + FTS5 全文搜索 + 可选向量嵌入），自动记忆提取，Markdown 感知分块
- **Skill / Plugin 扩展** — SKILL.md 热加载 + plugin.json 插件系统 + 技能安全扫描 + 自动策展
- **Gateway** — OpenAI 兼容 API（`/v1/chat/completions`）、WebSocket（ACP 协议）、REST 管理接口、Web Dashboard
- **安全体系** — 命令审批、工具策略、RBAC 角色控制、SSRF 防护、凭证脱敏、注入检测、设备配对认证
- **事件系统** — 异步事件总线 + 用户自定义钩子回调
- **斜杠命令** — /reset、/skills 等内置命令
- **配置热重载** — 运行时自动检测配置文件变更并智能重载
- **会话管理** — 私聊塌缩为单一会话(不按 openid 分,根治身份漂移)、空闲重置、自动修剪
- **引导文件** — AGENTS.md / IDENTITY.md / USER.md / HEARTBEAT.md 自动注入上下文
- **链接理解** — 消息中 URL 自动抓取预览
- **轻量** — 运行时 ~100MB 内存

## 快速开始

```bash
# 从 PyPI 安装（稳定版）
uv tool install flyclaw
pip install flyclaw

# 从 GitHub 安装（最新版）
uv tool install git+https://github.com/ScarletMercy/FlyClaw.git
pip install git+https://github.com/ScarletMercy/FlyClaw.git

# 交互式配置 + 启动
flyclaw-setup
flyclaw
```

### CLI 命令

| 命令 | 说明 |
|------|------|
| `flyclaw` | 启动主服务（Gateway + 渠道 + AgentLoop） |
| `flyclaw-setup` | 交互式配置向导（模型、API Key、渠道） |
| `flyclaw-daemon` | 守护进程管理（install / uninstall / status） |
| `flyclaw-acp` | Agent Control Protocol 服务（JSON-RPC WebSocket，供外部程序控制代理） |

> 网关默认绑定 `127.0.0.1:18080`（仅本地访问，host 已固化为代码常量）。主要端点：`GET /healthz`、`POST /v1/chat/completions`、`GET /dashboard`。

## 配置

首次运行 `flyclaw-setup` 会自动生成 `~/.flyclaw/config.yaml`，也可手动编辑。配置文件支持 `${ENV_VAR}` 语法引用环境变量（未设置的变量会被替换为空串并告警）。

### 配置示例

核心配置段（完整字段见 `src/config.py`，另有 voice / delegation / consolidation / task / auth / hooks 等段）：

```yaml
model:
  provider: "openai"
  name: "deepseek-chat"
  base_url: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}"

gateway:
  port: 18080
  auth_token: "${GATEWAY_AUTH_TOKEN}"   # 留空则启动时自动生成并持久化

channels:
  qq:
    enabled: true
    app_id: "${QQ_APP_ID}"
    client_secret: "${QQ_CLIENT_SECRET}"
    dm_policy: "open"            # open | allowlist
    group_policy: "allowlist"    # open | allowlist | disabled

session:
  idle_reset_minutes: 120

tools:
  exec:
    enabled: true
    approval_mode: "off"         # off | ask | on_denylist_miss | always
    sandbox_enabled: true
    audit_log: true

memory:
  enabled: true
  backend: "sqlite"              # sqlite | lancedb
```

### 环境变量

```bash
export DEEPSEEK_API_KEY="sk-..."        # LLM API Key（必填）
export GATEWAY_AUTH_TOKEN="your-secret"  # 留空则自动生成；公网部署务必显式设置
export QQ_APP_ID="..."                   # QQ 渠道
export QQ_CLIENT_SECRET="..."
export TAVILY_API_KEY="tvly-..."         # 网页搜索（无 Key 自动回退 Bing）
```

## 支持的模型

FlyClaw 兼容所有提供 OpenAI API 接口的模型服务，只需配置 `base_url`、`api_key`、`model` 即可接入。常见兼容服务：

- **DeepSeek** — `base_url: https://api.deepseek.com/v1`
- **OpenAI** — `base_url: https://api.openai.com/v1`（默认，可省略）
- **Groq** — `base_url: https://api.groq.com/openai/v1`
- **Ollama（本地）** — `base_url: http://localhost:11434/v1`
- **智谱 GLM** — `base_url: https://open.bigmodel.cn/api/paas/v4`
- **Moonshot** — `base_url: https://api.moonshot.cn/v1`
- **通义千问** — `base_url: https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Together AI** — `base_url: https://api.together.xyz/v1`
- **任何 OpenAI 兼容服务** — 自定义 `base_url`

支持模型回退链（`model.fallbacks`），主模型失败时自动切换备用模型。

## 渠道

### QQ

基于 QQ 官方 Bot API（WebSocket + HTTP）：

1. 登录 [QQ 开放平台](https://q.qq.com/) 创建机器人应用，获取 `App ID` 和 `Client Secret`
2. 配置 `channels.qq`（字段见上方完整示例）
3. 支持 C2C 私聊/私信、文本/Markdown/图片/文件/语音、流式回复、自动 Token 刷新

WebSocket 断线**永不放弃重连**（backoff 1→2→5→10→30→60s 封顶，无限重试，自动从任何时长故障自愈），并根据 close code（4004 鉴权失败清 token、4007/4009 会话失效清 session）智能选择 RESUME 或重新 IDENTIFY。

### 微信

基于腾讯 iLink Bot API（长轮询 + HTTP）：

1. 获取微信 iLink Bot 账号
2. 配置 `channels.weixin`（字段见上方完整示例，含 `base_url`、`cdn_base_url`、`send_retry_count` 等）
3. 支持私聊、文本/图片/视频/文件/语音，媒体走 AES-128-ECB 加密的 CDN

## 内置工具

39 个工具，分核心（始终可用）和可选（按需启用）两类：

### 核心工具

**命令与进程**

- `exec_command` — 异步 shell 命令执行，超时控制、沙箱模式、审批拦截、审计日志
- `process_status` — 后台进程管理（poll/wait/kill/log）

**文件操作**

- `read_file` / `write_file` / `edit_file` — 文件读写编辑
- `list_dir` / `grep` / `glob` — 目录列表、内容搜索、模式匹配

**Web**

- `web_search` — Tavily API 搜索（无 Key 时回退 Bing）
- `web_fetch` — HTTP 抓取 + HTML→Markdown 转换

**媒体**

- `describe_media` — 图片描述、音频转录、视频理解（URL/本地路径/base64）
- `send_file` — 统一文件发送（自动识别图片/语音/文件，支持 `force_file` 强制以文件形式发送）
- `text_to_speech` — edge-tts 语音合成，支持自定义音色

**记忆**

- `memory` — KV 记忆存储（save/get/list/delete），SQLite + FTS5 全文搜索，自动检测用户记忆意图

**定时任务**

- `cron_list` / `cron_create` / `cron_delete` / `cron_toggle` / `cron_run` — 定时任务管理，支持 cron 表达式、间隔、一次性定时

**任务管理**

- `task_manage` — 自主任务模式（plan/status/advance/cancel），支持多步骤计划和定时触发

**子代理**

- `delegate_task` — 委派任务给 research/coder/reviewer 代理，支持批量并行

**会话**

- `session_search` — FTS5 关键词搜索历史会话

**技能管理**

- `skill_view` / `skill_manage` / `skill_hub` — 技能查看、编辑、市场搜索/安装

**AI**

- `subagent_status` — 查看子代理运行状态

### 可选工具

**浏览器自动化**（`tools.browser.enabled`，基于 Playwright）

- `browser_navigate` / `browser_snapshot` / `browser_click` / `browser_type` / `browser_scroll` / `browser_back` / `browser_press` / `browser_screenshot` / `browser_console` / `browser_close`
- Accessibility Tree 快照 + 元素引用（@e1, @e2...），反检测注入

**Canvas 画布**（`canvas.enabled`）

- `canvas_render` / `canvas_navigate` / `canvas_eval` — AI 生成 HTML 并实时预览

**Windows 桌面控制**（仅 Windows，`tools.windows_use.enabled`）

- `windows_screenshot` / `windows_press` / `windows_hotkey` — 截图、按键、组合键

## Skill & Plugin

### Skill

在 `skills/` 目录下创建 `SKILL.md`，自动热加载。支持 `skill_manage` 工具在线创建/编辑，`skill_hub` 从市场安装。安装前自动进行安全扫描（恶意代码检测、Unicode 混淆防护）。内置策展人自动审查和归档过期技能。Skill 目录优先级（高→低）：`extra_dirs` > `~/.flyclaw/skills` > `workspace/skills` > `.agents/skills`。

### Plugin

在 `plugins/` 目录下创建插件（含 `plugin.json` 清单和 Python 工具模块），支持自定义事件钩子（hooks），可在工具执行前后注入逻辑。

## 记忆系统

记忆系统采用混合架构，同时支持关键词搜索和语义搜索：

- **FTS5 全文搜索** — 基于 SQLite FTS5（unicode61 分词），开箱即用
- **向量语义搜索**（可选） — 基于 LanceDB 后端
- **自动记忆提取** — 通过正则模式匹配识别对话中的记忆意图，自动提取并存储值得记忆的事实片段
- **Markdown 感知分块** — 按 Markdown 结构智能分块，保留语义完整性
- **自动索引** — 配置 `memory.extra_paths` 可自动索引外部文件

## 安全体系

多层安全防护：

- **命令审批** — 敏感命令需用户确认，支持 durable/session 级别审批记忆
- **沙箱模式** — 限制命令执行的工作目录和环境变量
- **RBAC** — owner / admin / user / guest 四级角色控制
- **SSRF 防护** — 阻止访问私有 IP、云元数据端点
- **凭证脱敏** — 日志和输出中自动隐藏 API Key 等敏感信息
- **注入检测** — NFD Unicode 归一化，防止组合标记绕过
- **工具策略** — allow/deny/owner_only 精细控制工具可用性
- **Guardrails** — 工具调用风暴检测、重复失败自动阻断
- **设备配对** — 新设备通过配对码认证接入
- **技能安全扫描** — 安装前检测恶意代码、路径穿越、混淆攻击

## 生产部署

直接运行 `flyclaw`，或使用 `flyclaw-daemon install` 安装为系统服务。仓库自带多阶段 `Dockerfile`（非 root 用户 + HEALTHCHECK）：

```bash
docker build -t flyclaw .
docker run -d --name flyclaw -p 18080:18080 -v flyclaw-data:/home/flyclaw/.flyclaw/data flyclaw
```

## API 接入

Gateway 提供 OpenAI 兼容接口，任何支持 OpenAI API 的客户端均可直接接入（`base_url` 指向 `http://<host>:18080/v1`）。另有 ACP WebSocket（`/ws/acp`，JSON-RPC 协议）可供外部程序编程控制代理，详见 `src/acp/`。

## 运维

### 内置斜杠命令

在聊天中发送：

| 命令 | 说明 |
|------|------|
| `/help` | 列出所有可用命令 |
| `/status` | 查看运行状态 |
| `/reset` | 重置当前会话 |
| `/skills` | 列出已加载的 Skill |
| `/search` | FTS5 关键词搜索历史会话 |
| `/new` | 新建会话 |
| `/old` | 查看历史会话 |
| `/re <id>` | 切换会话 |
| `/prune` | 清理旧会话 |
| `/sandbox` | 沙箱开关 |
| `/model` | 查看/切换模型、设置温度 |
| `/voice` | 语音模式开关与设置 |
| `/pair` | 生成设备配对码 |
| `/whoami` | 查看当前身份 |
| `/role` | 修改用户角色 |
| `/agents` | 查看子代理列表 |
| `/ps` | 查看后台进程 |
| `/auto` | 自动任务模式 |
| `/compress` | 手动压缩上下文 |
| `/rounds` | 查看当前轮次 |

### 数据目录

配置、数据库、技能、插件等数据存放在 `~/.flyclaw/`。

## 常见问题

**启动报 `ModuleNotFoundError`**：`pip install -e .`（或 `uv pip install -e .`）以可编辑模式安装。

**启动报 `模型 API 密钥未配置`**：运行 `flyclaw-setup` 配置 API Key，或在 `~/.flyclaw/config.yaml` 设置 `model.api_key`。

**QQ 收不到消息**：① 检查 `channels.qq.enabled: true`；② 确认 `app_id`/`client_secret` 正确；③ 确认机器人应用已上线；④ 查日志是否有 WebSocket 连接成功字样。

## 安全检查清单

- [ ] `GATEWAY_AUTH_TOKEN` 已设置（或接受自动生成）
- [ ] QQ/微信白名单已配置（`allow_from` / `allowed_users`）
- [ ] `tools.exec.deny_patterns` 已配置危险命令
- [ ] `tools.exec.approval_mode` 非 `off`（生产环境建议）
- [ ] `tools.exec.audit_log: true` 已开启
- [ ] Gateway 不直接暴露公网（用 Nginx 反代 + TLS）
- [ ] config.yaml 中无明文密钥（全部使用 `${ENV_VAR}`）

## 开发

```bash
uv run ruff check      # 检查
uv run ruff format     # 格式化
uv run pytest          # 测试
```

## 致谢

本项目受以下项目启发：

- [hermes-agent](https://github.com/nousresearch/hermes-agent) — Nous Research 出品的自进化 AI 代理
- [openclaw](https://github.com/openclaw/openclaw) — 个人 AI 助手框架

## License

[MIT](https://github.com/ScarletMercy/FlyClaw/blob/main/LICENSE)
