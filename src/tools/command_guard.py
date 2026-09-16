"""命令静态安全分析 — hardline 底线 + 可配置 denylist。

威胁模型——**防模型，不防攻击者**：静态层只拦「字面可见」的危险命令。模型
误操作写出的命令，危险词总是字面出现的（``$(rm -rf ~)`` 里的 ``rm -rf``
照样命中 deny/hardline）；需要 base64/编码/两步隐藏的蓄意绕过明确不防——
那是审批与沙箱的职责。

分层：

- **hardline**（``check_hardline``）：无恢复路径的灾难命令（抹盘/关机/fork 炸弹/
  删系统根），小而稳定，命令位置锚定（``_CMDPOS``）防误报，**无条件拦截**——
  审批关闭也不放行，想跑请自己开终端。
- **denylist**（``is_denylisted``）：可配置黑名单（``tools.exec.deny_patterns``
  覆盖默认表），命中即拒但**误报有出口**——报错信息指向配置项。这里的语义是
  "该让人类过目的操作"，不追求看穿命令内部。
- **审批闸**（exec.py）：删除类命令、强杀进程（kill -9/pkill -KILL/Stop-Process
  -Force）、shell executor（sh -c / cmd /c）走人工审批。
- 所有匹配前做归一化：NFKC 反混淆 + 剥离行首 sudo/doas 与整段引号包裹
  （bash 同样会剥掉这两层壳再执行，检测等价于裸命令）。

**不防什么**：已放行命令的后续行为（npm postinstall）、数据外传（curl -d）、
子进程继承环境里的密钥、蓄意对抗者的深层绕过（编码/两步攻击——那是审批和
沙箱的职责，不是字符串匹配的）。

已知限制：非子串级的 glob 模式（schtasks*/create* 等）只覆盖串首（含剥壳后），
不匹配分号/管道中段——中段覆盖只给了与 rm -rf 同级的删除类；hardline 的重定向
模式不识别引号，echo "x > /dev/sda" 这类文本会被过拦（方向为过拦不泄漏）。

``DEFAULT_DENY_PATTERNS`` 默认表契约由特征化测试 test_exec_denylist.py 锁定；
非递归删除审批门控（``DELETE_APPROVAL_PATTERNS`` 子串匹配 +
``COMMAND_PREFIX_WHITELIST`` 前缀豁免）见 exec.py。
"""

from __future__ import annotations

import fnmatch
import re
import unicodedata

# ---------------------------------------------------------------------------
# 归一化：反混淆与包装剥离（仅用于检测，不用于执行）。
# NFKC 折叠全角字符；NUL 剥离；行首 sudo/doas（含 flag）与「整段引号包裹」
# 逐层剥掉——bash 对这两种包装都会剥离后执行，检测视角等价于裸命令。
# 这归一化不是解析器：只处理把整条命令包起来的壳，不分析内部结构。
# ---------------------------------------------------------------------------

_LEADING_LAUNCHER_RE = re.compile(
    r"^(?:sudo|doas)(?:\s+(?:-[ugpCtTrD]\s+\S+|--(?:user|group|prompt|chdir|role|type)\s+\S+|-\S+))*\s+",
    re.IGNORECASE,
)

# sudo/doas 启动器吞掉的内容：带参数的 flag（-u root）优先于裸 flag（-E），
# 否则 -u 剥掉后参数词会留在命令位置导致后续全部错位。剥壳 RE 与 _CMDPOS 共用。
_SUDO_FLAGS = r"(?:-[ugpCtTrD]\s+\S+|--(?:user|group|prompt|chdir|role|type)\s+\S+|-\S+)"


def _normalize_for_detection(command: str) -> str:
    text = unicodedata.normalize("NFKC", command.replace("\x00", ""))
    for _ in range(3):  # 'sudo x'、"'sudo x'"、"sudo 'x'" 等有限嵌套
        if len(text) >= 2 and text[0] in "'\"" and text[-1] == text[0]:
            text = text[1:-1]
            continue
        m = _LEADING_LAUNCHER_RE.match(text)
        if m:
            text = text[m.end() :]
            continue
        break
    return text


# ---------------------------------------------------------------------------
# denylist 模式表 + 两层 matcher（fnmatch 全串 + \b 子串正则）。
# ---------------------------------------------------------------------------

DEFAULT_DENY_PATTERNS = [
    # Recursive delete — always blocked
    "rm -rf",
    "rm -r",
    "rm -R",
    "rmdir /s",
    "rd /s",
    "rd /s/q",
    "rd /s /q",
    "shutil.rmtree",
    "del /s",
    "erase /s",
    # 前导 * 使 fnmatch 也覆盖分号/管道中段（"cd x; Remove-Item -Recurse y"），
    # 与 rm -rf / del /s 的子串级覆盖对齐
    "*remove-item *-recurse*",
    "*ri *-recurse*",
    # System destruction
    "mkfs*",
    "dd if=*of=/dev/*",
    "> /dev/sd*",
    "shutdown*",
    "reboot*",
    "init 0",
    "init 6",
    "systemctl *poweroff*",
    "systemctl *reboot*",
    "systemctl *halt*",
    ":(){ :|:& };:",
    "chmod -R 777 /",
    "chown -R * /",
    "curl*|*sh",
    "wget*|*sh",
    "nc -l*",
    "ncat*",
    "/etc/passwd",
    "/etc/shadow",
    "crontab -r",
    # Windows — disk/volume destruction
    "format *",
    "diskpart",
    "bcdedit",
    "bootsect",
    "vssadmin delete*",
    "cipher /w*",
    "fsutil*",
    # Windows — persistence & privilege escalation
    "schtasks*/create*",
    "schtasks*/delete*",
    "net user */add*",
    "net localgroup**/add*",
    "REG ADD *\\Run*",
    "REG ADD *\\RunOnce*",
    "sc create *",
    "sc config *",
    "New-Service *",
    # Windows — defense evasion
    "wevtutil cl*",
    "Set-MpPreference *",
    "taskkill /f *",
    "net stop *",
    "sc stop *",
    # Windows — credential theft
    "reg save hklm*",
    "reg save hklm\\*",
    "ntdsutil *",
    # Windows — LOLBin / download & execute
    "certutil -urlcache*",
    "certutil -f *",
    "bitsadmin /transfer*",
    "mshta *",
    "msiexec *",
    "rundll32 *javascript*",
    "regsvr32 */i:*",
]

# Non-recursive delete — always requires approval (even if approval_mode=off)
DELETE_APPROVAL_PATTERNS = [
    "rm ",
    "del ",
    "erase ",
    "rmdir ",
    "rd ",
    "os.remove",
    "os.unlink",
    "remove-item",
    "ri ",
]

# 强杀进程 —— 一律要求审批（可恢复但影响面大：可能误杀 agent 自身进程、
# 数据库等有状态服务）。与 DELETE_APPROVAL_PATTERNS 同一语义层级。
FORCE_KILL_APPROVAL_PATTERNS = [
    "kill -9",
    "kill -kill",
    "kill -sigkill",
    "killall -9",
    "killall -kill",
    "killall -sigkill",
    "pkill -9",
    "pkill -kill",
    "pkill -sigkill",
    "stop-process -force",
]

# Shell executor 触发表：不做安全拦截（denylist 已对全串做过检查），只用于
# 审批门控——审批模式下要求人工确认，后台模式无法审批故直接禁。
# 匹配见 ``_SHELL_EXECUTOR_COMPILED``（命令位置锚定）。

_SHELL_EXECUTOR_PATTERNS = [
    ("sh -c", "sh -c"),
    ("bash -c", "bash -c"),
    ("zsh -c", "zsh -c"),
    ("perl -e", "perl -e"),
    ("ruby -e", "ruby -e"),
    ("node -e", "node -e"),
    ("/bin/sh", "/bin/sh"),
    ("/bin/bash", "/bin/bash"),
    ("eval ", "eval "),
    ("cmd /c ", "cmd /c"),
    ("cmd /r ", "cmd /r"),
    ("cmd /k ", "cmd /k"),
    ("powershell -encodedcommand ", "powershell -encodedcommand"),
    ("powershell -enc ", "powershell -enc"),
    ("powershell -e ", "powershell -e (encoded)"),
    ("pwsh -encodedcommand ", "pwsh -encodedcommand"),
    ("pwsh -enc ", "pwsh -enc"),
    ("pwsh -e ", "pwsh -e (encoded)"),
    ("powershell -command ", "powershell -command"),
    ("pwsh -command ", "pwsh -command"),
]

# Command prefixes whose subcommands should not trigger denylist/delete checks.
# e.g. "git rm", "docker rm" — these operate on their own scope, not the filesystem.
COMMAND_PREFIX_WHITELIST = frozenset({"git", "docker", "kubectl", "podman", "npm", "yarn", "pnpm"})


def is_denylisted(command: str, deny_patterns: list[str]) -> tuple[bool, str]:
    """两层匹配：fnmatch 全串 glob + \\b 边界子串正则。"""
    cmd_normalized = re.sub(r"\s+", " ", _normalize_for_detection(command).strip()).lower()
    for pattern in deny_patterns:
        pattern_lower = pattern.lower()
        if fnmatch.fnmatch(cmd_normalized, pattern_lower):
            return True, pattern
        # fnmatch is full-string — missed if command has a prefix/suffix.
        # regex substring + word boundary catches those cases.
        start_boundary = r"\b" if pattern_lower[0:1].isalnum() or pattern_lower[0:1] == "_" else ""
        end_boundary = r"\b" if pattern_lower[-1:].isalnum() or pattern_lower[-1:] == "_" else ""
        try:
            if re.search(start_boundary + re.escape(pattern_lower) + end_boundary, cmd_normalized):
                return True, pattern
        except re.error:
            if pattern_lower in cmd_normalized:
                return True, pattern
    return False, ""


def has_shell_executor(command: str) -> tuple[bool, str]:
    """Detect shell executor patterns (sh -c, cmd /c, powershell -command, etc.).

    These are NOT inherently dangerous — the denylist already checks the full
    command string for dangerous tokens. They only matter for approval gating.

    命令位置锚定（``_CMDPOS`` + find 的 -exec）：只在各命令段首匹配，
    参数位置的普通词不误伤——``ls /bin/sh``、``echo medieval castle``
    不命中；``; eval x``、``nohup bash -c``、``find -exec sh -c`` 照常命中。
    """
    normalized = _normalize_for_detection(command)
    for pattern_re, label in _SHELL_EXECUTOR_COMPILED:
        if pattern_re.search(normalized):
            return True, label
    return False, ""


# ---------------------------------------------------------------------------
# hardline 底线：无恢复路径的灾难命令。锚定命令位置防误报；不可审批绕过。
# ---------------------------------------------------------------------------

# 匹配「shell 开始解析新命令」的位置：串首、分隔符（; & | 换行）或子 shell
# 开启（$( 、反引号）之后，透明穿透 sudo/doas（含带参 flag）、env VAR=..、
# exec/nohup/setsid/time/xargs 包装器。参数位置的词（cat shutdown.log、grep
# 'format C:'）不匹配，因此不误报。
_CMDPOS = (
    r"(?:^|[;&|\n\r]|\$\(|`)"
    r"\s*"
    r"(?:(?:sudo|doas)\s+(?:" + _SUDO_FLAGS + r"\s+)*)?"
    r"(?:env\s+(?:\w+=\S*\s+)*)?"
    r"(?:(?:exec|nohup|setsid|time|xargs(?:\s+-\S+)*)\s+)*"
    r"\s*"
)

# rm 递归删除保护的根：POSIX 系统目录 + macOS + Git Bash 盘符根（/c、/d、/c/*）
# 及盘符根下的系统目录（/c/Users）。根之后必须紧跟空白/结尾（(?=\s|$)）——
# "rm -rf /tmp"、"/home/user/build" 这类子路径是普通删除，走 denylist/审批，
# 不进 hardline。
_PROTECTED_SYS_DIRS = r"(?:home|root|etc|usr|var|bin|sbin|boot|lib|opt|Users|System|Library|Windows)"
_RM_PROTECTED_ROOTS = (
    r"(?:"
    r"/|/\*"
    r"|/" + _PROTECTED_SYS_DIRS + r"(?:/\*)?"
    r"|~(?:/\*)?|\$\{?HOME\}?(?:/\*)?"
    r"|/[a-zA-Z](?:/\*)?"
    r"|/[a-zA-Z]/" + _PROTECTED_SYS_DIRS + r"(?:/\*)?"
    r")(?=\s|$)"
)

_HARDLINE_RAW: list[tuple[str, str]] = [
    # 递归删除系统根/家目录/整盘 —— 命令位置锚定
    (_CMDPOS + r"rm\s+(?:-\S+\s+)*" + _RM_PROTECTED_ROOTS, "recursive delete of root/system directory"),
    # 文件系统格式化
    (_CMDPOS + r"mkfs(?:\.[a-z0-9.]+)?\b", "format filesystem (mkfs)"),
    (_CMDPOS + r"mke2fs\b", "format filesystem (mke2fs)"),
    # 块设备覆写（dd / 重定向）
    (_CMDPOS + r"dd\b[^\n]*\bof=/dev/(?:sd|nvme|hd|mmcblk|vd|xvd)", "dd to raw block device"),
    (r">\s*/dev/(?:sd|nvme|hd|mmcblk|vd|xvd)", "redirect to raw block device"),
    # fork 炸弹
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    # 杀死全部进程（-1 必须是最后一个操作数：kill -1 123 是 SIGHUP 单进程）
    (_CMDPOS + r"kill\s+(?:-[^\s]+\s+)*-1(?=\s*(?:[;&|\n\r]|$))", "kill all processes"),
    # 关机/重启 —— 命令位置锚定（含 PowerShell cmdlet）
    (_CMDPOS + r"(?:shutdown|reboot|halt|poweroff|stop-computer|restart-computer)\b", "system shutdown/reboot"),
    (_CMDPOS + r"init\s+[06]\b", "init 0/6 (shutdown/reboot)"),
    (_CMDPOS + r"telinit\s+[06]\b", "telinit 0/6 (shutdown/reboot)"),
    (_CMDPOS + r"systemctl\s+(?:-[^\s]+\s+)*(?:poweroff|reboot|halt|kexec)\b", "systemctl poweroff/reboot"),
    # Windows 磁盘工具 / 卷删除 —— 命令位置锚定
    (_CMDPOS + r"format\s+(?:/[^\s]+\s+)*[a-z]:", "format drive"),
    (_CMDPOS + r"(?:diskpart|bcdedit|bootsect)\b", "Windows disk/boot tool"),
    (_CMDPOS + r"vssadmin\b[^|;&\n]*\bdelete\b", "delete volume shadow copies"),
    (_CMDPOS + r"cipher\b\s+/w", "cipher drive wipe"),
    # 根目录权限放开 / 所有权移交（保护根集合与 rm 对齐：/, /*, 系统目录, ~,
    # Git Bash 盘符根及盘符根下系统目录；子路径归 denylist/审批层）
    (_CMDPOS + r"chmod\s+\S*R\S*\s+777\s+" + _RM_PROTECTED_ROOTS, "chmod 777 on root filesystem"),
    (_CMDPOS + r"chown\s+\S*R\S*[^|;&\n]*\s+" + _RM_PROTECTED_ROOTS, "chown root filesystem"),
]

_RE_FLAGS = re.IGNORECASE | re.DOTALL
HARDLINE_PATTERNS_COMPILED = [(re.compile(pattern, _RE_FLAGS), description) for pattern, description in _HARDLINE_RAW]


# ---------------------------------------------------------------------------
# executor 编译表：命令位置锚定（复用 _CMDPOS）。子串匹配会误伤参数位置
# 的普通词（"medieval" 含 "eval"、"ls /bin/sh" 含 "/bin/sh"），锚定后只剩
# 真正的执行入口命中。find 的 -exec(dir) 是独立的执行入口，单独锚定。
# ---------------------------------------------------------------------------


def _compile_executor_pattern(trigger: str) -> re.Pattern:
    t = trigger.strip()
    body = r"\s+".join(re.escape(part) for part in t.split())
    if t[-1:].isalnum() or t[-1:] == "_":
        body += r"\b"
    start = r"\b" if (t[0].isalnum() or t[0] == "_") else ""
    return re.compile(
        r"(?:" + _CMDPOS + r"|\s-exec(?:dir)?\s+)" + start + body,
        _RE_FLAGS,
    )


_SHELL_EXECUTOR_COMPILED = [(_compile_executor_pattern(trigger), label) for trigger, label in _SHELL_EXECUTOR_PATTERNS]


def check_hardline(command: str) -> str | None:
    """命中 hardline 底线则返回描述，否则 None。

    无条件拦截层：审批模式、用户配置都不可绕过。只收无恢复路径的灾难
    操作；可恢复的高危操作（kill -9、schtasks、certutil 等）属
    denylist/审批层。
    """
    if not command or not command.strip():
        return None
    normalized = _normalize_for_detection(command)
    for pattern_re, description in HARDLINE_PATTERNS_COMPILED:
        if pattern_re.search(normalized):
            return description
    return None


def basename(token: str) -> str:
    """剥离路径前缀与 .exe 后缀，返回命令名小写形式。"""
    base = token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if base.lower().endswith(".exe"):
        base = base[:-4]
    return base.lower()
