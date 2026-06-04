# FlyClaw 从开发到生产部署全流程

## 前置条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- LLM API Key（DeepSeek / OpenAI / 智谱 / 通义千问 / Ollama 等任一）
- QQ Bot 应用（如需 QQ 渠道）或微信 iLink Bot（如需微信渠道）

---

## 第 1 步：环境准备

### 从 PyPI 安装（推荐）

```bash
# 推荐 uv
uv tool install flyclaw

# 或 pip
pip install flyclaw
```

### 从源码开发

```bash
git clone https://github.com/ScarletMercy/FlyClaw.git
cd FlyClaw

# 使用 uv（推荐）
uv venv
source .venv/bin/activate    # Linux/macOS
# 或
.venv\Scripts\activate       # Windows

uv pip install -e .
# 或使用 pip
pip install -e .
```

依赖说明（`pyproject.toml` 已声明）：

| 依赖 | 用途 |
|------|------|
| `openai` | OpenAI API client（LLM 调用） |
| `fastapi` + `uvicorn` | HTTP Gateway |
| `httpx` | HTTP client（渠道、工具） |
| `aiosqlite` | SQLite 异步驱动（会话持久化） |
| `apscheduler` | 定时任务调度 |
| `tavily-python` | 网页搜索 |
| `watchfiles` | Skill 热重载 + 配置热重载 |
| `websockets` | QQ Bot WebSocket 连接 |
| `playwright` | 浏览器自动化 |
| `edge-tts` | 语音合成 |
| `cryptography` | 微信 CDN 媒体解密 |
| `aiohttp` | 微信 iLink API |

可选依赖：

| 依赖 | 用途 | 安装 |
|------|------|------|
| `sqlite-vec` | 向量语义搜索（记忆系统） | `pip install flyclaw[memory]` |
| `lancedb` + `pyarrow` | LanceDB 向量后端 | `pip install flyclaw[memory-lancedb]` |

---

## 第 2 步：配置

### 2.1 配置向导（推荐）

运行交互式配置向导，自动生成 `~/.flyclaw/config.yaml`：

```bash
flyclaw-setup
```

### 2.2 手动配置

直接编辑 `~/.flyclaw/config.yaml`：

```yaml
model:
  name: "deepseek-chat"
  temperature: 0.0
  base_url: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}"
  context_window: 64000

gateway:
  host: "127.0.0.1"
  port: 18080
  auth_token: "${GATEWAY_AUTH_TOKEN}"

channels:
  qq:
    enabled: true
    app_id: "${QQ_APP_ID}"
    client_secret: "${QQ_CLIENT_SECRET}"
    dm_policy: "open"
    markdown_support: false
    approval_keyboard: true

  weixin:
    enabled: false
    account_id: ""
    token: ""

session:
  scope: "per_sender"
  idle_reset_minutes: 120

agents:
  system_prompt: "You are a helpful AI assistant."
  workspace: "~/.flyclaw/workspace"
  max_tool_rounds: 100
  busy_input_mode: "interrupt"
  timezone: "Asia/Shanghai"
  language: "zh"

tools:
  exec:
    enabled: true
    timeout_seconds: 30
    approval_mode: "off"
    sandbox_enabled: true
    audit_log: true
  web_search:
    api_key: "${TAVILY_API_KEY}"
  browser:
    enabled: true
    headless: true
    stealth: true
  windows_use:
    enabled: true   # 仅 Windows 生效

memory:
  enabled: true
  backend: "sqlite"
  db_path: "~/.flyclaw/data/memory.db"

cron:
  enabled: true
  store_path: "~/.flyclaw/data/cron.db"

skills:
  enabled: true

security:
  enabled: true
  audit_on_startup: true

compression:
  enabled: true
  threshold_percent: 0.6
```

### 2.3 环境变量

```bash
# LLM API Key（必填）
export DEEPSEEK_API_KEY="sk-..."

# Gateway 认证（生产环境必须设置）
export GATEWAY_AUTH_TOKEN="your-secret-token"

# QQ Bot（如需 QQ 渠道）
export QQ_APP_ID="..."
export QQ_CLIENT_SECRET="..."

# 网页搜索（可选）
export TAVILY_API_KEY="tvly-..."
```

或在项目根目录的 `.env` 文件中设置（`python-dotenv` 自动加载）。

### 2.4 支持的模型

通过 `base_url` 可接入所有 OpenAI 兼容的模型服务：

| 提供商 | model | base_url | 环境变量 |
|--------|-------|----------|----------|
| DeepSeek | `deepseek-chat` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| OpenAI | `gpt-4o` | `https://api.openai.com/v1`（可省略） | `OPENAI_API_KEY` |
| Groq | `llama-3.3-70b-versatile` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| Ollama（本地） | `llama3` | `http://localhost:11434/v1` | — |
| 智谱 GLM | `glm-4-plus` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` |
| Moonshot | `moonshot-v1-128k` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` |
| 通义千问 | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| Together AI | `meta-llama/Llama-3-70b-chat-hf` | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` |

配置模型回退链（主模型失败时自动切换）：

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

---

## 第 3 步：本地运行

```bash
# 方式一：CLI 命令（安装后可用）
flyclaw

# 方式二：从源码运行
python -m src.main
```

启动后日志输出：

```
15:00:00 [flyclaw.security] PASS gateway-auth
15:00:00 [flyclaw.security] INFO exec-approval
15:00:00 [flyclaw.security] Audit complete: 4 passed, 0 warnings
15:00:00 [flyclaw] flyclaw starting...
15:00:00 [flyclaw] Model: openai/deepseek-chat
15:00:00 [flyclaw] Skills loaded: 2 active, 2 total
15:00:00 [flyclaw] AgentLoop 已创建: 30 个工具, 2 个技能
15:00:00 [flyclaw] RBAC 已初始化 (默认角色=owner, 配对=True)
15:00:01 [flyclaw] Gateway ready: http://127.0.0.1:18080
15:00:01 [flyclaw] OpenAI compat: POST /v1/chat/completions
15:00:01 [flyclaw] WebSocket ACP: ws://127.0.0.1:18080/ws/acp
15:00:01 [flyclaw] Health:        GET /healthz
15:00:01 [flyclaw] Dashboard:     GET /dashboard
```

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

### 安全审计说明

启动时自动执行安全检查：

| 检查项 | 级别 | 说明 |
|--------|------|------|
| gateway-auth | WARN | host=0.0.0.0 但未设置 auth_token |
| exec-approval | INFO | approval_mode=off（仅提示） |
| data-dir | WARN | 无法创建 data/ 目录 |
| secrets | WARN | config.yaml 中发现疑似明文密钥 |

---

## 第 4 步：渠道接入

### 4.1 QQ 渠道

1. 登录 [QQ 开放平台](https://q.qq.com/)
2. 创建机器人应用
3. 获取 `App ID` 和 `Client Secret`
4. 配置：

```yaml
channels:
  qq:
    enabled: true
    app_id: "${QQ_APP_ID}"
    client_secret: "${QQ_CLIENT_SECRET}"
    dm_policy: "open"           # "open" | "allowlist"
    allow_from: []              # 私聊白名单（dm_policy=allowlist 时生效）
    markdown_support: false     # Markdown 消息格式
    approval_keyboard: true     # QQ 原生按钮审批（需 markdown_support=true）
```

QQ 渠道基于 WebSocket + HTTP，支持自动 Token 刷新和断线重连（最多 100 次）。

### 4.2 微信渠道

1. 获取微信 iLink Bot 账号
2. 配置：

```yaml
channels:
  weixin:
    enabled: true
    account_id: "your-account-id"
    token: "your-token"
    base_url: "https://ilinkai.weixin.qq.com"
    dm_policy: "open"           # "open" | "allowlist" | "disabled"
    allowed_users: []           # 白名单
    split_multiline_messages: false  # 多行消息拆分发送
    send_retry_count: 4         # 发送重试次数
```

微信渠道基于长轮询 + HTTP，媒体传输使用 AES-128-ECB 加密的 CDN。

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
```

Skill 会自动热重载（修改 SKILL.md 后立即生效）。也可通过 `skill_manage` 工具在线创建/编辑，`skill_hub` 从市场安装。

Skill 目录优先级（高→低）：`extra_dirs` > `~/.flyclaw/skills` > `workspace/skills` > `.agents/skills`

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

插件支持事件钩子（hooks），可在工具执行前后注入自定义逻辑。

---

## 第 7 步：生产部署

### 7.1 使用 Daemon 管理（推荐）

flyclaw 内置跨平台服务管理工具：

```bash
# 安装为系统服务
flyclaw-daemon install

# 查看状态
flyclaw-daemon status

# 卸载服务
flyclaw-daemon uninstall
```

- **Linux (systemd)** — 生成 service 文件，`daemon-reload` + `enable`
- **macOS (launchd)** — 生成 plist 到 `~/Library/LaunchAgents/`，支持开机自启和崩溃重启
- **Windows (schtasks)** — 创建开机自启的计划任务

### 7.2 手动 systemd（Linux）

```ini
# /etc/systemd/system/flyclaw.service
[Unit]
Description=FlyClaw AI Assistant
After=network.target

[Service]
Type=simple
User=flyclaw
WorkingDirectory=/opt/flyclaw
EnvironmentFile=/opt/flyclaw/.env
ExecStart=/opt/flyclaw/.venv/bin/flyclaw
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
RUN pip install --no-cache-dir .

COPY src/ src/
COPY skills/ skills/
COPY plugins/ plugins/

RUN mkdir -p /app/data

EXPOSE 18080

CMD ["flyclaw"]
```

```bash
docker build -t flyclaw .
docker run -d \
  --name flyclaw \
  -p 18080:18080 \
  -e DEEPSEEK_API_KEY="sk-..." \
  -e GATEWAY_AUTH_TOKEN="your-secret" \
  -e QQ_APP_ID="..." \
  -e QQ_CLIENT_SECRET="..." \
  -v flyclaw-data:/root/.flyclaw/data \
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

### ACP WebSocket

通过 JSON-RPC WebSocket 协议进行程序化控制：

```javascript
const ws = new WebSocket("wss://flyclaw.example.com/ws/acp");

// 初始化
ws.send(JSON.stringify({
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {}
}));

// 发送消息
ws.send(JSON.stringify({
  jsonrpc: "2.0",
  id: 2,
  method: "message/send",
  params: { message: "hello", session_id: "my-session" }
}));

// 接收响应
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg);
};
```

---

## 第 9 步：运维

### 内置斜杠命令

在聊天中发送以下命令：

| 命令 | 说明 |
|------|------|
| `/help` | 列出所有可用命令 |
| `/status` | 查看运行状态 |
| `/reset` | 重置当前会话 |
| `/rollback` | 回滚到最近的文件快照 |
| `/skills` | 列出已加载的 Skill |

### 工具审计日志

所有工具调用可自动记录审计日志（`tools.exec.audit_log: true`）：

```
[flyclaw.audit] tool=exec_command sender=ou_xxx args="ls -la" ok dur=0.32s
[flyclaw.audit] tool=web_search sender=ou_xxx args="query=\"AI news\"" ok dur=1.52s
```

敏感参数（api_key、secret、token 等）自动脱敏。

### 定时任务管理

```bash
# 查看状态
curl -H "Authorization: Bearer token" http://localhost:18080/api/cron/status

# 列出任务
curl -H "Authorization: Bearer token" http://localhost:18080/api/cron/jobs
```

也可通过 `cron_create` / `cron_list` / `cron_delete` 工具在对话中管理。

### Dashboard

访问 `http://localhost:18080/dashboard` 可查看：
- 系统状态（运行时间、模型信息、会话数）
- 会话列表
- 已加载技能
- 当前配置

### 数据目录

```
~/.flyclaw/
├── config.yaml           # 主配置文件
├── data/
│   ├── checkpoints.db    # 会话持久化（SQLite）
│   ├── cron.db           # 定时任务存储（SQLite）
│   ├── memory.db         # 记忆系统（SQLite + FTS5）
│   ├── auth.db           # 认证数据（SQLite）
│   ├── session_index.db  # 会话搜索索引（FTS5）
│   └── approvals.json    # 持久化审批记录
├── skills/               # 用户级 Skill
├── workspace/            # 工作目录（exec_command 默认路径）
└── plugins/              # 插件目录
```

备份：`cp -r ~/.flyclaw/data/ ~/.flyclaw/data-backup-$(date +%Y%m%d)/`

---

## 常见问题

### Q: 启动报 `ModuleNotFoundError`

```bash
pip install -e .
# 或
uv pip install -e .
```

### Q: 启动报 `[错误] 模型 API 密钥未配置`

运行 `flyclaw-setup` 配置 API Key，或在 `~/.flyclaw/config.yaml` 中设置 `model.api_key`。

### Q: QQ 收不到消息

1. 检查 `channels.qq.enabled: true`
2. 检查 `app_id` 和 `client_secret` 是否正确
3. 确认 QQ 机器人应用已上线
4. 查看日志中是否有 WebSocket 连接成功字样

### Q: 工具调用报 `ApprovalPending`

配置 `tools.exec.approval_mode` 控制审批行为：
- `"off"` — 不审批
- `"ask"` — 每次询问
- `"on_denylist_miss"` — 仅 denylist 中的命令需审批
- `"always"` — 所有命令都需审批

### Q: 流式输出不工作

确保 Nginx 配置了 `proxy_buffering off`，并检查 `proxy_read_timeout` 是否足够（LLM 响应可能较慢）。

### Q: web_search 返回 API key 错误

需要设置 `TAVILY_API_KEY` 环境变量，或在 `config.yaml` 中配置 `tools.web_search.api_key`。无 Key 时会自动回退到 Bing 搜索。

### Q: 安全审计报 WARN

根据提示修复配置问题：
- 未设置 `GATEWAY_AUTH_TOKEN`（公网部署必须设置）
- QQ/微信 `allow_from` 白名单为空
- config.yaml 中有明文密钥（应改为 `${ENV_VAR}` 格式）

---

## 安全检查清单

- [ ] `GATEWAY_AUTH_TOKEN` 已设置且足够复杂
- [ ] QQ/微信白名单已配置（`allow_from` / `allowed_users`）
- [ ] QQ `allow_from` 白名单已配置（如使用 allowlist 模式）
- [ ] `tools.exec.deny_patterns` 已配置危险命令
- [ ] `tools.exec.approval_mode` 非 `off`（生产环境建议）
- [ ] `tools.exec.audit_log: true` 已开启
- [ ] `security.audit_on_startup: true` 已开启
- [ ] Gateway 不直接暴露到公网（使用 Nginx 反代 + TLS）
- [ ] `~/.flyclaw/data/` 目录权限受限（`chmod 700`）
- [ ] config.yaml 中无明文密钥（全部使用 `${ENV_VAR}`）
- [ ] `sandbox_enabled: true` 已开启（限制命令执行范围）
