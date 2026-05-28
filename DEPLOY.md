# flyclaw-Py 从开发到生产部署全流程

## 前置条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- 飞书开放平台应用（如需飞书集成）
- LLM API Key（Anthropic 或 OpenAI）

---

## 第 1 步：环境准备

```bash
# 进入项目目录
cd flyclaw-py

# 使用 uv 创建虚拟环境（推荐）
uv venv
source .venv/bin/activate    # Linux/macOS
# 或
.venv\Scripts\activate       # Windows

# 安装依赖
uv pip install -e .
# 或使用 pip
pip install -e .
```

依赖说明（`pyproject.toml` 已声明）：

| 依赖 | 用途 |
|------|------|
| `openai` | OpenAI API client (LLM calls) |
| `httpx` | HTTP client (channels, tools, MCP) |
| `lark-oapi` | 飞书 SDK |
| `fastapi` + `uvicorn` | HTTP Gateway |
| `aiosqlite` | SQLite 异步驱动（会话持久化） |
| `apscheduler` | 定时任务调度 |
| `tavily-python` | 网页搜索 + 网页内容提取 |
| `watchfiles` | Skill 热重载 |

---

## 第 2 步：配置

### 2.1 配置向导（推荐）

运行交互式配置向导，自动生成 `config.yaml`：

```bash
flyclaw-setup
```

向导会引导你完成 5 步配置：
1. 选择模型提供商（Anthropic / OpenAI / DeepSeek / Groq / Ollama / 智谱 / 通义千问 / Moonshot / 自定义）
2. 配置 Gateway（端口、认证）
3. 配置飞书渠道（可选）
4. 配置网页搜索（可选）
5. 确认并保存

### 2.2 手动配置

直接编辑项目根目录的 `config.yaml`。

### 2.2 必填环境变量

```bash
# LLM API Key（至少配置一个）
export ANTHROPIC_API_KEY="sk-ant-..."          # Anthropic
export OPENAI_API_KEY="sk-..."                 # OpenAI

# Gateway 认证（生产环境必须设置）
export GATEWAY_AUTH_TOKEN="your-secret-token"

# 飞书应用（如需飞书渠道）
export FEISHU_APP_ID="cli_xxx"
export FEISHU_APP_SECRET="your-secret"

# Tavily API Key（网页搜索 + 网页提取共用）
export TAVILY_API_KEY="tvly-xxx"
```

或在 `.env` 文件中设置（需自行加载，如用 `python-dotenv`）。

### 2.3 配置项详解

```yaml
# config.yaml 完整配置项

gateway:
  host: "127.0.0.1"          # 生产环境改为 0.0.0.0
  port: 18080
  auth_token: "${GATEWAY_AUTH_TOKEN}"  # 生产必须设置

model:
  provider: "anthropic"     # "anthropic" | "openai"（或任何 OpenAI 兼容服务）
  name: "claude-sonnet-4-6"
  temperature: 0.0
  base_url: ""              # 自定义 API 端点（见下方支持的模型列表）
  api_key: "${...}"         # 可直接设置或用 ${ENV_VAR}
  fallbacks:                # 模型回退链（可选）
    - provider: "openai"
      name: "gpt-4o"

agents:
  system_prompt: |
    你是一个运行在 flyclaw 中的 AI 助手。
  workspace: "."           # 工作目录，exec_command 在此执行
  max_tool_rounds: 15      # 单次对话最大工具调用轮数

channels:
  feishu:
    enabled: true
    domain: "lark"         # "feishu"（国内）| "lark"（海外）
    dm_policy: "open"      # "open" | "allowlist" | "pairing"
    group_policy: "allowlist"
    allow_from: ["ou_xxx"] # 私聊白名单
    group_allow_from: ["oc_xxx"]  # 群白名单
    require_mention: true  # 群聊需 @机器人
    streaming: true        # 流式输出卡片
    typing_indicator: true # "正在输入"提示

session:
  scope: "per_sender"      # "per_sender" | "global"
  idle_reset_minutes: 120  # 空闲后重置会话

tools:
  exec:
    enabled: true
    timeout_seconds: 30
    approval_mode: "ask"   # "off" | "ask" | "on_denylist_miss" | "always"
    deny_patterns: []
    audit_log: true        # 工具调用审计日志
  web_search:
    enabled: false         # 改为 true 启用搜索
    api_key: "${TAVILY_API_KEY}"
  web_fetch:
    enabled: true          # 网页内容提取（使用同一个 TAVILY_API_KEY）
  feishu:
    doc: true              # 飞书文档读取
    chat: true             # 飞书群信息查询

checkpointer:
  type: "sqlite"           # "sqlite" | "memory"
  path: "data/checkpoints.db"

cron:
  enabled: true
  store_path: "data/cron.db"

skills:
  enabled: true
  extra_dirs: []
  budget_chars: 30000
  watch: true              # 自动热重载 SKILL.md 变更

# 安全审计（启动时自动检查）
security:
  enabled: true
  audit_on_startup: true   # 启动时输出安全检查报告

# 链接预览（消息中 URL 自动提取标题和摘要）
link_understanding:
  enabled: true
  max_previews: 3          # 每条消息最多预览几个链接

timeouts:
  tool_short: 30           # 短超时工具
  tool_long: 600           # 长超时工具
  session_idle: 3600       # 会话空闲超时
```

### 2.4 支持的模型

通过 `provider: "openai"` + `base_url` 可接入所有 OpenAI 兼容的模型服务：

| 提供商 | provider | model | base_url | 环境变量 |
|--------|----------|-------|----------|----------|
| Anthropic Claude | `anthropic` | `claude-sonnet-4-6` | — | `ANTHROPIC_API_KEY` |
| OpenAI GPT | `openai` | `gpt-4o` | — | `OPENAI_API_KEY` |
| DeepSeek | `openai` | `deepseek-chat` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| Groq | `openai` | `llama-3.3-70b-versatile` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| Together AI | `openai` | `meta-llama/Llama-3-70b-chat-hf` | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` |
| Ollama (本地) | `openai` | `llama3` | `http://localhost:11434/v1` | — |
| 智谱 GLM | `openai` | `glm-4-plus` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` |
| Moonshot (Kimi) | `openai` | `moonshot-v1-128k` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` |
| 通义千问 | `openai` | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| 自定义 | `openai` | 自定义 | 自定义 | 自定义 |

配置示例（DeepSeek）：

```yaml
model:
  provider: "openai"
  name: "deepseek-chat"
  temperature: 0.0
  base_url: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}"
```

配置示例（Ollama 本地模型）：

```yaml
model:
  provider: "openai"
  name: "llama3"
  temperature: 0.0
  base_url: "http://localhost:11434/v1"
```

配置模型回退链（主模型失败时自动切换）：

```yaml
model:
  provider: "anthropic"
  name: "claude-sonnet-4-6"
  fallbacks:
    - provider: "openai"
      name: "deepseek-chat"
      base_url: "https://api.deepseek.com/v1"
      api_key: "${DEEPSEEK_API_KEY}"
    - provider: "openai"
      name: "gpt-4o"
      api_key: "${OPENAI_API_KEY}"
```

---

## 第 3 步：本地开发运行

```bash
# 启动
python -m src.main
# 或
flyclaw
```

启动后日志输出：

```
14:00:00 [flyclaw.security] PASS gateway-auth
14:00:00 [flyclaw.security] INFO exec-approval
14:00:00 [flyclaw.security] PASS feishu-dm
14:00:00 [flyclaw.security] PASS data-dir
14:00:00 [flyclaw.security] PASS secrets
14:00:00 [flyclaw.security] Audit complete: 4 passed, 0 warnings
14:00:00 [flyclaw] flyclaw 0.1.0 starting...
14:00:00 [flyclaw] Model: anthropic/claude-sonnet-4-6
14:00:00 [flyclaw] Skills loaded: 1 active, 1 total
14:00:00 [flyclaw] Graph compiled with 6 tools, 1 skills
14:00:01 [flyclaw] SQLite checkpointer initialized
14:00:01 [flyclaw] Gateway ready: http://127.0.0.1:18080
14:00:01 [flyclaw] OpenAI compat: POST /v1/chat/completions
14:00:01 [flyclaw] WebSocket:     ws://127.0.0.1:18080/ws
14:00:01 [flyclaw] Health:        GET /healthz
14:00:01 [flyclaw] Cron API:      GET /api/cron/status
```

### 安全审计说明

启动时自动执行 5 项安全检查：

| 检查项 | 级别 | 说明 |
|--------|------|------|
| gateway-auth | WARN | host=0.0.0.0 但未设置 auth_token |
| feishu-dm | WARN | dm_policy=open 且白名单为空 |
| exec-approval | INFO | approval_mode=off（仅提示） |
| data-dir | WARN | 无法创建 data/ 目录 |
| secrets | WARN | config.yaml 中发现疑似明文密钥 |

### 验证服务

```bash
# 健康检查
curl http://127.0.0.1:18080/healthz

# OpenAI 兼容 API 测试
curl -X POST http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'

# 流式测试
curl -X POST http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d '{"messages":[{"role":"user","content":"hello"}],"stream":true}'
```

---

## 第 4 步：飞书渠道接入

### 4.1 创建飞书应用

1. 登录 [飞书开放平台](https://open.feishu.cn/)
2. 创建企业自建应用
3. 获取 `App ID` 和 `App Secret`
4. 在「事件与回调」中启用 **WebSocket 模式**（推荐）
5. 添加以下权限：
   - `im:message` — 接收消息
   - `im:message:send_as_bot` — 发送消息
   - `im:chat` — 获取群信息
   - `im:resource` — 获取消息中的资源文件
   - `docx:document` — 读取文档（如需文档工具）
6. 发布应用版本

### 4.2 配置飞书

```yaml
channels:
  feishu:
    enabled: true
    app_id: "${FEISHU_APP_ID}"
    app_secret: "${FEISHU_APP_SECRET}"
    domain: "feishu"       # 国内用 feishu，海外用 lark
    dm_policy: "allowlist"
    allow_from: ["ou_xxx"]  # 允许的用户 open_id
```

### 4.3 获取用户 ID

在飞书中 @机器人 发送 `/status`，查看日志中的 `sender_id` 即为 `open_id`。

---

## 第 5 步：扩展 Skill

在 `skills/` 目录下创建 SKILL.md 文件即可自动加载：

```bash
mkdir -p skills/my-skill
```

```markdown
# skills/my-skill/SKILL.md
---
name: my-skill
description: Describe what this skill does
user-invocable: true
---

# My Skill

When the user asks about X, do Y.

## Instructions

1. Step one
2. Step two

## Example

```bash
echo "hello"
```
```

Skill 会自动热重载（需安装 `watchfiles`，修改 SKILL.md 后立即生效）。

---

## 第 6 步：扩展 Plugin

在 `plugins/` 目录下创建插件：

```
plugins/my-plugin/
├── plugin.json      # 插件清单
└── tools.py         # 工具实现
```

```json
{
    "id": "my-plugin",
    "name": "My Plugin",
    "version": "1.0.0",
    "description": "A custom plugin",
    "tools": ["tools.py"],
    "hooks": {}
}
```

```python
# tools.py
from src.agent.tooldef import ToolDef

async def my_tool(query: str) -> str:
    """Your tool description."""
    return f"Result for: {query}"

def get_tools() -> list[ToolDef]:
    return [ToolDef.from_function(my_tool)]
```

---

## 第 7 步：生产部署

### 7.1 使用 Daemon 管理（推荐）

flyclaw 内置跨平台服务管理工具，支持 systemd（Linux）、launchd（macOS）、schtasks（Windows）。

```bash
# 安装为系统服务
flyclaw-daemon install

# 查看状态
flyclaw-daemon status

# 卸载服务
flyclaw-daemon uninstall
```

**Linux (systemd)**：会打印需要执行的 sudo 命令，包括生成 service 文件、daemon-reload、enable。

**macOS (launchd)**：会生成 plist 文件到 `~/Library/LaunchAgents/`，支持开机自启和崩溃自动重启。

**Windows (schtasks)**：直接创建开机自启的计划任务。

### 7.2 手动 systemd（Linux）

```ini
# /etc/systemd/system/flyclaw.service
[Unit]
Description=flyclaw AI Assistant
After=network.target

[Service]
Type=simple
User=flyclaw
WorkingDirectory=/opt/flyclaw-py
EnvironmentFile=/opt/flyclaw-py/.env
ExecStart=/opt/flyclaw-py/.venv/bin/python -m src.main
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable flyclaw
sudo systemctl start flyclaw
sudo journalctl -u flyclaw -f
```

### 7.3 使用 Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY skills/ skills/
COPY plugins/ plugins/
COPY config.yaml .

RUN mkdir -p data

EXPOSE 18080

CMD ["python", "-m", "src.main"]
```

```bash
docker build -t flyclaw .
docker run -d \
  --name flyclaw \
  -p 18080:18080 \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e GATEWAY_AUTH_TOKEN="your-secret" \
  -e FEISHU_APP_ID="cli_xxx" \
  -e FEISHU_APP_SECRET="xxx" \
  -e TAVILY_API_KEY="tvly-xxx" \
  -v flyclaw-data:/app/data \
  flyclaw
```

### 7.4 反向代理（Nginx）

```nginx
server {
    listen 443 ssl;
    server_name flyclaw.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:18080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;          # SSE 流式需要
        proxy_cache off;
        proxy_read_timeout 300s;      # LLM 响应较慢
    }
}
```

---

## 第 8 步：API 接入

部署完成后，可从任何支持 OpenAI API 的客户端接入：

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://flyclaw.example.com/v1",
    api_key="your-gateway-token",
)

response = client.chat.completions.create(
    model="flyclaw",
    messages=[{"role": "user", "content": "hello"}],
)
print(response.choices[0].message.content)
```

### cURL

```bash
curl -X POST https://flyclaw.example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-gateway-token" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

### WebSocket

```javascript
const ws = new WebSocket("wss://flyclaw.example.com/ws");

// 1. HMAC 认证（如果设置了 auth_token）
// 2. 发送请求
ws.send(JSON.stringify({
  type: "req",
  id: "1",
  method: "chat.send",
  params: { text: "hello", thread_id: "my-session" }
}));

// 3. 接收响应
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg);
};
```

---

## 第 9 步：运维

### 工具与审计日志

所有工具调用自动记录审计日志（可通过 `tools.exec.audit_log` 开关）：

```
[flyclaw.audit] [command-audit] tool=exec_command sender=ou_xxx args="ls -la" ok dur=0.32s
[flyclaw.audit] [command-audit] tool=web_search sender=ou_xxx args="query=\"AI news\"" ok dur=1.52s
[flyclaw.audit] [command-audit] tool=web_fetch sender=ou_xxx args="url=\"https://...\"" ok dur=2.11s
```

敏感参数（api_key、secret、token 等）自动脱敏为 `***REDACTED***`。

### 链接预览

当用户消息中包含 URL 时，自动提取网页标题和摘要并追加到回复末尾：

```
📎 **OpenClaw - Multi-channel AI Gateway**
   OpenClaw is a personal AI assistant framework...
   🔗 https://github.com/example/openclaw
```

可通过 `link_understanding.enabled: false` 关闭，`max_previews` 控制预览数量。

### 定时任务管理

```bash
# 查看状态
curl -H "Authorization: Bearer token" http://localhost:18080/api/cron/status

# 列出任务
curl -H "Authorization: Bearer token" http://localhost:18080/api/cron/jobs

# 创建定时任务
curl -X POST -H "Authorization: Bearer token" -H "Content-Type: application/json" \
  http://localhost:18080/api/cron/jobs \
  -d '{
    "name": "daily summary",
    "schedule": {"kind": "cron", "expr": "0 9 * * *"},
    "payload": {"kind": "agent_turn", "message": "生成今日工作总结"}
  }'

# 立即执行
curl -X POST -H "Authorization: Bearer token" \
  http://localhost:18080/api/cron/jobs/{job_id}/run
```

### 内置命令（飞书中发送）

| 命令 | 说明 |
|------|------|
| `/help` | 列出所有可用命令 |
| `/status` | 查看运行状态 |
| `/reset` | 重置当前会话 |
| `/skills` | 列出已加载的 Skill |

### 数据目录

```
data/
├── checkpoints.db       # 会话持久化（SQLite）
├── cron.db              # 定时任务存储（SQLite）
└── approvals.json       # 持久化审批记录
```

备份：`cp -r data/ data-backup-$(date +%Y%m%d)/`

### Daemon 服务管理

```bash
flyclaw-daemon install    # 安装为系统服务
flyclaw-daemon uninstall  # 卸载服务
flyclaw-daemon status     # 查看服务状态
```

---

## 常见问题

### Q: 启动报 `ModuleNotFoundError`

```bash
pip install -e .
```

### Q: 飞书收不到消息

1. 检查 `channels.feishu.enabled: true`
2. 检查飞书应用是否已发布
3. 检查 WebSocket 权限是否开启
4. 查看日志中是否有 `WS client connected` 字样

### Q: 工具调用报 `ApprovalNeededError`

配置 `tools.exec.approval_mode: "off"` 跳过审批，或通过飞书卡片审批。

### Q: 流式输出不工作

确保 Nginx 配置了 `proxy_buffering off`，并且飞书开启了 `streaming: true`。

### Q: web_fetch 返回 `[error] Tavily API key not configured`

需要设置 `TAVILY_API_KEY` 环境变量，或在 `config.yaml` 中配置 `tools.web_search.api_key`。web_fetch 和 web_search 共用同一个 API Key。

### Q: 安全审计报 WARN

根据提示修复配置问题，常见的有：
- 未设置 `GATEWAY_AUTH_TOKEN`（公网部署必须设置）
- 飞书 `allow_from` 白名单为空
- config.yaml 中有明文密钥（应改为 `${ENV_VAR}` 格式）

---

## 安全检查清单

- [ ] `GATEWAY_AUTH_TOKEN` 已设置且足够复杂
- [ ] 飞书 `allow_from` 白名单已配置
- [ ] 群聊 `require_mention: true` 已开启
- [ ] `tools.exec.deny_patterns` 已配置
- [ ] `tools.exec.approval_mode` 非 `off`（生产环境）
- [ ] `tools.exec.audit_log: true` 已开启（审计日志）
- [ ] `security.audit_on_startup: true` 已开启（启动审计）
- [ ] Gateway 不直接暴露到公网（使用 Nginx 反代 + TLS）
- [ ] `data/` 目录权限受限（`chmod 700 data/`）
- [ ] config.yaml 中无明文密钥（全部使用 `${ENV_VAR}`）
