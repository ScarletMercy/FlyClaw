# FlyClaw

轻量级多渠道 AI 助手框架。运行时仅 ~100MB 内存，40+ 内置工具，支持 QQ/微信渠道接入。事件驱动架构，全异步设计。

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 简介

FlyClaw 是一个自包含的 AI 助手框架，直接运行在你的设备上。它通过 QQ 或微信与你对话，能执行命令、读写文件、搜索网页、控制浏览器、处理图片/音频/视频，还能定时执行任务。支持配置热重载、斜杠命令、上下文自动压缩，内置事件驱动钩子系统。借鉴了 [hermes-agent](https://github.com/nousresearch/hermes-agent) 和 [openclaw](https://github.com/openclaw/openclaw) 的设计理念。

## 功能特性

- **多渠道接入** — QQ Bot（官方 API，C2C 私聊）、微信（iLink Bot API，私聊）
- **40+ 内置工具** — 命令执行、文件操作、Web 搜索、浏览器自动化、媒体理解、TTS、记忆、定时任务、Windows 桌面控制等
- **AgentLoop 引擎** — 工具并行执行、主动上下文压缩、中断/排队/steer 三种忙碌输入模式、工具参数自动修复、Guardrails 风暴检测
- **子代理委派** — research、coder、reviewer 三个内置角色，支持批量并行任务
- **记忆系统** — KV 记忆（SQLite + FTS5 全文搜索 + 可选向量嵌入），LLM 记忆评判，Markdown 感知分块
- **Skill / Plugin 扩展** — SKILL.md 热加载 + plugin.json 插件系统 + 技能安全扫描 + 自动策展
- **Gateway** — OpenAI 兼容 API（`/v1/chat/completions`）、WebSocket（ACP 协议）、REST 管理接口、Web Dashboard
- **安全体系** — 命令审批、工具策略、RBAC 角色控制、SSRF 防护、凭证脱敏、注入检测、设备配对认证
- **事件系统** — 异步事件总线 + 用户自定义钩子回调
- **斜杠命令** — /reset、/rollback、/skills 等内置命令
- **配置热重载** — 运行时自动检测配置文件变更并智能重载
- **会话管理** — per_sender / global 两种模式，空闲重置，自动修剪
- **引导文件** — AGENTS.md / IDENTITY.md / USER.md 自动注入上下文
- **链接理解** — 消息中 URL 自动抓取预览
- **轻量** — 运行时 ~100MB 内存

## 快速开始

```bash
# 推荐 uv
uv tool install flyclaw

# 或 pip
pip install flyclaw

# 交互式配置向导
flyclaw-setup

# 启动
flyclaw
```

### CLI 命令

| 命令 | 说明 |
|------|------|
| `flyclaw` | 启动主服务（Gateway + 渠道 + AgentLoop） |
| `flyclaw-setup` | 交互式配置向导（模型、API Key、渠道） |
| `flyclaw-daemon` | 守护进程管理（install / uninstall / restart / status） |
| `flyclaw-acp` | Agent Control Protocol 服务（JSON-RPC WebSocket，供外部程序控制代理） |

### 最小配置示例

首次运行 `flyclaw-setup` 会自动生成 `~/.flyclaw/config.yaml`，也可手动创建：

```yaml
model:
  name: "deepseek-chat"
  temperature: 0.0
  base_url: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}"
  context_window: 64000   # 模型上下文窗口大小（token 数）

channels:
  qq:
    enabled: true
    app_id: "你的APPID"
    client_secret: "你的密钥"
```

## 支持的模型

FlyClaw 兼容所有提供 OpenAI API 接口的模型服务，只需配置 `base_url`、`api_key`、`model` 即可接入：

```yaml
model:
  name: "deepseek-chat"
  temperature: 0.0
  base_url: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}"
```

常见兼容服务：

- **DeepSeek** — `base_url: https://api.deepseek.com/v1`
- **OpenAI** — `base_url: https://api.openai.com/v1`（默认，可省略）
- **Groq** — `base_url: https://api.groq.com/openai/v1`
- **Ollama（本地）** — `base_url: http://localhost:11434/v1`
- **智谱 GLM** — `base_url: https://open.bigmodel.cn/api/paas/v4`
- **Moonshot** — `base_url: https://api.moonshot.cn/v1`
- **通义千问** — `base_url: https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Together AI** — `base_url: https://api.together.xyz/v1`
- **任何 OpenAI 兼容服务** — 自定义 `base_url`

支持模型回退链，主模型失败时自动切换备用模型：

```yaml
model:
  name: "deepseek-chat"
  base_url: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}"
  fallbacks:
    - name: "qwen-plus"
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
      api_key: "${DASHSCOPE_API_KEY}"
```

## 渠道

### QQ

基于 QQ 官方 Bot API（WebSocket + HTTP），支持：

- C2C 私聊、私信
- 文本/Markdown 消息、图片、文件、语音消息
- 流式回复、自动 Token 刷新、WebSocket 断线重连（最多 100 次）

### 微信

基于腾讯 iLink Bot API（长轮询 + HTTP），支持：

- 私聊
- 文本、图片、视频、文件、语音收发
- AES-128-ECB CDN 媒体传输

## 内置工具

40+ 个工具，分核心（始终可用）和可选（按需启用）两类：

### 核心工具

**命令与进程**

- `exec_command` — 异步 shell 命令执行，超时控制、沙箱模式、审批拦截、审计日志
- `process_status` — 后台进程管理（poll/wait/kill/log）

**文件操作**

- `read_file` / `write_file` / `edit_file` — 文件读写编辑
- `list_dir` / `grep` / `glob` — 目录列表、内容搜索、模式匹配
- 快照机制：每次写操作前自动创建 shadow git 快照，支持 `/rollback` 回滚

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

- `skills_list` / `skill_view` / `skill_manage` / `skill_hub` — 技能浏览、查看、编辑、市场搜索/安装

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

在 `skills/` 目录下创建 `SKILL.md`，自动热加载：

```
skills/my-skill/SKILL.md
```

支持 `skill_manage` 工具在线创建/编辑，`skill_hub` 从市场安装。安装前自动进行安全扫描（恶意代码检测、Unicode 混淆防护）。内置策展人自动审查和归档过期技能。

### Plugin

在 `plugins/` 目录下创建插件，包含 `plugin.json` 清单和 Python 工具模块：

```
plugins/my-plugin/
├── plugin.json
└── tools.py
```

插件支持自定义事件钩子（hooks），可在工具执行前后注入逻辑。

## 记忆系统

记忆系统采用混合架构，同时支持关键词搜索和语义搜索：

- **FTS5 全文搜索** — 基于 SQLite FTS5（unicode61 分词），开箱即用
- **向量语义搜索**（可选） — 支持 sqlite-vec 和 LanceDB 两种后端
- **LLM 记忆评判** — 自动判断对话内容是否值得记忆，避免存入无用信息
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

## 部署

三种方式，详见 [DEPLOY.md](https://github.com/ScarletMercy/FlyClaw/blob/main/DEPLOY.md)：

**直接运行**

```bash
flyclaw
```

**Docker**

```bash
docker build -t flyclaw .
docker run -d -p 18080:18080 \
  -e API_KEY="your-api-key" \
  -e GATEWAY_AUTH_TOKEN="your-secret" \
  -v flyclaw-data:/app/data \
  flyclaw
```

**Daemon 服务**（systemd / launchd / schtasks）

```bash
flyclaw-daemon install
```

## 开发

```bash
make lint       # ruff 代码检查
make format     # ruff 格式化
make test       # pytest 测试
make test-cov   # 测试覆盖率
```

## 致谢

本项目受以下项目启发：

- [hermes-agent](https://github.com/nousresearch/hermes-agent) — Nous Research 出品的自进化 AI 代理
- [openclaw](https://github.com/openclaw/openclaw) — 个人 AI 助手框架

## License

[MIT](https://github.com/ScarletMercy/FlyClaw/blob/main/LICENSE)
