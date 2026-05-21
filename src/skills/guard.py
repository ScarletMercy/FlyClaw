from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .types import Finding, ScanResult

logger = __import__("logging").getLogger("myclaw.skills.guard")

TRUSTED_REPOS = {"openai/skills", "anthropics/skills"}

INSTALL_POLICY = {
    "builtin": ("allow", "allow", "allow"),
    "trusted": ("allow", "allow", "block"),
    "community": ("allow", "block", "block"),
    "agent-created": ("allow", "allow", "block"),
}

VERDICT_INDEX = {"safe": 0, "caution": 1, "dangerous": 2}

THREAT_PATTERNS = [
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)',
     "env_exfil_curl", "critical", "exfiltration",
     "curl command interpolating secret environment variable"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)',
     "env_exfil_wget", "critical", "exfiltration",
     "wget command interpolating secret environment variable"),
    (r'fetch\s*\([^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|API)',
     "env_exfil_fetch", "critical", "exfiltration",
     "fetch() call interpolating secret environment variable"),
    (r'httpx?\.(get|post|put|patch)\s*\([^\n]*(KEY|TOKEN|SECRET|PASSWORD)',
     "env_exfil_httpx", "critical", "exfiltration",
     "HTTP library call with secret variable"),
    (r'requests\.(get|post|put|patch)\s*\([^\n]*(KEY|TOKEN|SECRET|PASSWORD)',
     "env_exfil_requests", "critical", "exfiltration",
     "requests library call with secret variable"),
    (r'base64[^\n]*env',
     "encoded_exfil", "high", "exfiltration",
     "base64 encoding combined with environment access"),
    (r'\$HOME/\.ssh|\~/\.ssh',
     "ssh_dir_access", "high", "exfiltration",
     "references user SSH directory"),
    (r'\$HOME/\.aws|\~/\.aws',
     "aws_dir_access", "high", "exfiltration",
     "references user AWS credentials directory"),
    (r'\$HOME/\.gnupg|\~/\.gnupg',
     "gpg_dir_access", "high", "exfiltration",
     "references user GPG keyring"),
    (r'\$HOME/\.kube|\~/\.kube',
     "kube_dir_access", "high", "exfiltration",
     "references Kubernetes config directory"),
    (r'\$HOME/\.docker|\~/\.docker',
     "docker_dir_access", "high", "exfiltration",
     "references Docker config (may contain registry creds)"),
    (r'\$HOME/\.myclaw/\.env|\~/\.myclaw/\.env|\$HOME/\.hermes/\.env|\~/\.hermes/\.env',
     "agent_env_access", "critical", "exfiltration",
     "directly references agent secrets file"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)',
     "read_secrets_file", "critical", "exfiltration",
     "reads known secrets file"),
    (r'printenv|env\s*\|',
     "dump_all_env", "high", "exfiltration",
     "dumps all environment variables"),
    (r'os\.environ\b(?!\s*\.get\s*\(\s*["\']PATH)',
     "python_os_environ", "high", "exfiltration",
     "accesses os.environ (potential env dump)"),
    (r'os\.getenv\s*\(\s*[^\)]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)',
     "python_getenv_secret", "critical", "exfiltration",
     "reads secret via os.getenv()"),
    (r'process\.env\[',
     "node_process_env", "high", "exfiltration",
     "accesses process.env (Node.js environment)"),
    (r'ENV\[.*(?:KEY|TOKEN|SECRET|PASSWORD)',
     "ruby_env_secret", "critical", "exfiltration",
     "reads secret via Ruby ENV[]"),
    (r'\b(dig|nslookup|host)\s+[^\n]*\$',
     "dns_exfil", "critical", "exfiltration",
     "DNS lookup with variable interpolation (possible DNS exfiltration)"),
    (r'>\s*/tmp/[^\s]*\s*&&\s*(curl|wget|nc|python)',
     "tmp_staging", "critical", "exfiltration",
     "writes to /tmp then exfiltrates"),
    (r'!\[.*\]\(https?://[^\)]*\$\{?',
     "md_image_exfil", "high", "exfiltration",
     "markdown image URL with variable interpolation (image-based exfil)"),
    (r'\[.*\]\(https?://[^\)]*\$\{?',
     "md_link_exfil", "high", "exfiltration",
     "markdown link with variable interpolation"),
    (r'ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+instructions',
     "prompt_injection_ignore", "critical", "injection",
     "prompt injection: ignore previous instructions"),
    (r'you\s+are\s+(?:\w+\s+)*now\s+',
     "role_hijack", "high", "injection",
     "attempts to override the agent's role"),
    (r'do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user',
     "deception_hide", "critical", "injection",
     "instructs agent to hide information from user"),
    (r'system\s+prompt\s+override',
     "sys_prompt_override", "critical", "injection",
     "attempts to override the system prompt"),
    (r'pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+',
     "role_pretend", "high", "injection",
     "attempts to make the agent assume a different identity"),
    (r'disregard\s+(?:\w+\s+)*(your|all|any)\s+(?:\w+\s+)*(instructions|rules|guidelines)',
     "disregard_rules", "critical", "injection",
     "instructs agent to disregard its rules"),
    (r'output\s+(?:\w+\s+)*(system|initial)\s+prompt',
     "leak_system_prompt", "high", "injection",
     "attempts to extract the system prompt"),
    (r'(when|if)\s+no\s*one\s+is\s+(watching|looking)',
     "conditional_deception", "high", "injection",
     "conditional instruction to behave differently when unobserved"),
    (r'act\s+as\s+(if|though)\s+(?:\w+\s+)*you\s+(?:\w+\s+)*(have\s+no|don\'t\s+have)\s+(?:\w+\s+)*(restrictions|limits|rules)',
     "bypass_restrictions", "critical", "injection",
     "instructs agent to act without restrictions"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)',
     "translate_execute", "critical", "injection",
     "translate-then-execute evasion technique"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->',
     "html_comment_injection", "high", "injection",
     "hidden instructions in HTML comments"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none',
     "hidden_div", "high", "injection",
     "hidden HTML div (invisible instructions)"),
    (r'rm\s+-rf\s+/',
     "destructive_root_rm", "critical", "destructive",
     "recursive delete from root"),
    (r'rm\s+(-[^\s]*)?r.*\$HOME|\brmdir\s+.*\$HOME',
     "destructive_home_rm", "critical", "destructive",
     "recursive delete targeting home directory"),
    (r'chmod\s+777',
     "insecure_perms", "medium", "destructive",
     "sets world-writable permissions"),
    (r'>\s*/etc/',
     "system_overwrite", "critical", "destructive",
     "overwrites system configuration file"),
    (r'\bmkfs\b',
     "format_filesystem", "critical", "destructive",
     "formats a filesystem"),
    (r'\bdd\s+.*if=.*of=/dev/',
     "disk_overwrite", "critical", "destructive",
     "raw disk write operation"),
    (r'shutil\.rmtree\s*\(\s*[\"\'/]',
     "python_rmtree", "high", "destructive",
     "Python rmtree on absolute or root-relative path"),
    (r'truncate\s+-s\s*0\s+/',
     "truncate_system", "critical", "destructive",
     "truncates system file to zero bytes"),
    (r'\bcrontab\b',
     "persistence_cron", "medium", "persistence",
     "modifies cron jobs"),
    (r'\.(bashrc|zshrc|profile|bash_profile|bash_login|zprofile|zlogin)\b',
     "shell_rc_mod", "medium", "persistence",
     "references shell startup file"),
    (r'authorized_keys',
     "ssh_backdoor", "critical", "persistence",
     "modifies SSH authorized keys"),
    (r'ssh-keygen',
     "ssh_keygen", "medium", "persistence",
     "generates SSH keys"),
    (r'systemd.*\.service|systemctl\s+(enable|start)',
     "systemd_service", "medium", "persistence",
     "references or enables systemd service"),
    (r'/etc/init\.d/',
     "init_script", "medium", "persistence",
     "references init.d startup script"),
    (r'launchctl\s+load|LaunchAgents|LaunchDaemons',
     "macos_launchd", "medium", "persistence",
     "macOS launch agent/daemon persistence"),
    (r'/etc/sudoers|visudo',
     "sudoers_mod", "critical", "persistence",
     "modifies sudoers (privilege escalation)"),
    (r'git\s+config\s+--global\s+',
     "git_config_global", "medium", "persistence",
     "modifies global git configuration"),
    (r'\bnc\s+-[lp]|ncat\s+-[lp]|\bsocat\b',
     "reverse_shell", "critical", "network",
     "potential reverse shell listener"),
    (r'\bngrok\b|\blocaltunnel\b|\bserveo\b|\bcloudflared\b',
     "tunnel_service", "high", "network",
     "uses tunneling service for external access"),
    (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{2,5}',
     "hardcoded_ip_port", "medium", "network",
     "hardcoded IP address with port"),
    (r'0\.0\.0\.0:\d+|INADDR_ANY',
     "bind_all_interfaces", "high", "network",
     "binds to all network interfaces"),
    (r'/bin/(ba)?sh\s+-i\s+.*>/dev/tcp/',
     "bash_reverse_shell", "critical", "network",
     "bash interactive reverse shell via /dev/tcp"),
    (r'python[23]?\s+-c\s+["\']import\s+socket',
     "python_socket_oneliner", "critical", "network",
     "Python one-liner socket connection (likely reverse shell)"),
    (r'socket\.connect\s*\(\s*\(',
     "python_socket_connect", "high", "network",
     "Python socket connect to arbitrary host"),
    (r'webhook\.site|requestbin\.com|pipedream\.net|hookbin\.com',
     "exfil_service", "high", "network",
     "references known data exfiltration/webhook testing service"),
    (r'pastebin\.com|hastebin\.com|ghostbin\.',
     "paste_service", "medium", "network",
     "references paste service (possible data staging)"),
    (r'base64\s+(-d|--decode)\s*\|',
     "base64_decode_pipe", "high", "obfuscation",
     "base64 decodes and pipes to execution"),
    (r'\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}',
     "hex_encoded_string", "medium", "obfuscation",
     "hex-encoded string (possible obfuscation)"),
    (r'\beval\s*\(\s*["\']',
     "eval_string", "high", "obfuscation",
     "eval() with string argument"),
    (r'\bexec\s*\(\s*["\']',
     "exec_string", "high", "obfuscation",
     "exec() with string argument"),
    (r'echo\s+[^\n]*\|\s*(bash|sh|python|perl|ruby|node)',
     "echo_pipe_exec", "critical", "obfuscation",
     "echo piped to interpreter for execution"),
    (r'compile\s*\(\s*[^\)]+,\s*["\'].*["\']\s*,\s*["\']exec["\']\s*\)',
     "python_compile_exec", "high", "obfuscation",
     "Python compile() with exec mode"),
    (r'getattr\s*\(\s*__builtins__',
     "python_getattr_builtins", "high", "obfuscation",
     "dynamic access to Python builtins (evasion technique)"),
    (r'__import__\s*\(\s*["\']os["\']\s*\)',
     "python_import_os", "high", "obfuscation",
     "dynamic import of os module"),
    (r'codecs\.decode\s*\(\s*["\']',
     "python_codecs_decode", "medium", "obfuscation",
     "codecs.decode (possible ROT13 or encoding obfuscation)"),
    (r'String\.fromCharCode|charCodeAt',
     "js_char_code", "medium", "obfuscation",
     "JavaScript character code construction (possible obfuscation)"),
    (r'atob\s*\(|btoa\s*\(',
     "js_base64", "medium", "obfuscation",
     "JavaScript base64 encode/decode"),
    (r'\[::-1\]',
     "string_reversal", "low", "obfuscation",
     "string reversal (possible obfuscated payload)"),
    (r'chr\s*\(\s*\d+\s*\)\s*\+\s*chr\s*\(\s*\d+',
     "chr_building", "high", "obfuscation",
     "building string from chr() calls (obfuscation)"),
    (r'\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}',
     "unicode_escape_chain", "medium", "obfuscation",
     "chain of unicode escapes (possible obfuscation)"),
    (r'subprocess\.(run|call|Popen|check_output)\s*\(',
     "python_subprocess", "medium", "execution",
     "Python subprocess execution"),
    (r'os\.system\s*\(',
     "python_os_system", "high", "execution",
     "os.system() — unguarded shell execution"),
    (r'os\.popen\s*\(',
     "python_os_popen", "high", "execution",
     "os.popen() — shell pipe execution"),
    (r'child_process\.(exec|spawn|fork)\s*\(',
     "node_child_process", "high", "execution",
     "Node.js child_process execution"),
    (r'Runtime\.getRuntime\(\)\.exec\(',
     "java_runtime_exec", "high", "execution",
     "Java Runtime.exec() — shell execution"),
    (r'`[^`]*\$\([^)]+\)[^`]*`',
     "backtick_subshell", "medium", "execution",
     "backtick string with command substitution"),
    (r'\.\./\.\./\.\.',
     "path_traversal_deep", "high", "traversal",
     "deep relative path traversal (3+ levels up)"),
    (r'\.\./\.\.',
     "path_traversal", "medium", "traversal",
     "relative path traversal (2+ levels up)"),
    (r'/etc/passwd|/etc/shadow',
     "system_passwd_access", "critical", "traversal",
     "references system password files"),
    (r'/proc/self|/proc/\d+/',
     "proc_access", "high", "traversal",
     "references /proc filesystem (process introspection)"),
    (r'/dev/shm/',
     "dev_shm", "medium", "traversal",
     "references shared memory (common staging area)"),
    (r'xmrig|stratum\+tcp|monero|coinhive|cryptonight',
     "crypto_mining", "critical", "mining",
     "cryptocurrency mining reference"),
    (r'hashrate|nonce.*difficulty',
     "mining_indicators", "medium", "mining",
     "possible cryptocurrency mining indicators"),
    (r'curl\s+[^\n]*\|\s*(ba)?sh',
     "curl_pipe_shell", "critical", "supply_chain",
     "curl piped to shell (download-and-execute)"),
    (r'wget\s+[^\n]*-O\s*-\s*\|\s*(ba)?sh',
     "wget_pipe_shell", "critical", "supply_chain",
     "wget piped to shell (download-and-execute)"),
    (r'curl\s+[^\n]*\|\s*python',
     "curl_pipe_python", "critical", "supply_chain",
     "curl piped to Python interpreter"),
    (r'#\s*///\s*script.*dependencies',
     "pep723_inline_deps", "medium", "supply_chain",
     "PEP 723 inline script metadata with dependencies (verify pinning)"),
    (r'pip\s+install\s+(?!-r\s)(?!.*==)',
     "unpinned_pip_install", "medium", "supply_chain",
     "pip install without version pinning"),
    (r'npm\s+install\s+(?!.*@\d)',
     "unpinned_npm_install", "medium", "supply_chain",
     "npm install without version pinning"),
    (r'uv\s+run\s+',
     "uv_run", "medium", "supply_chain",
     "uv run (may auto-install unpinned dependencies)"),
    (r'(curl|wget|httpx?\.get|requests\.get|fetch)\s*[\(]?\s*["\']https?://',
     "remote_fetch", "medium", "supply_chain",
     "fetches remote resource at runtime"),
    (r'git\s+clone\s+',
     "git_clone", "medium", "supply_chain",
     "clones a git repository at runtime"),
    (r'docker\s+pull\s+',
     "docker_pull", "medium", "supply_chain",
     "pulls a Docker image at runtime"),
    (r'^allowed-tools\s*:',
     "allowed_tools_field", "high", "privilege_escalation",
     "skill declares allowed-tools (pre-approves tool access)"),
    (r'\bsudo\b',
     "sudo_usage", "high", "privilege_escalation",
     "uses sudo (privilege escalation)"),
    (r'setuid|setgid|cap_setuid',
     "setuid_setgid", "critical", "privilege_escalation",
     "setuid/setgid (privilege escalation mechanism)"),
    (r'NOPASSWD',
     "nopasswd_sudo", "critical", "privilege_escalation",
     "NOPASSWD sudoers entry (passwordless privilege escalation)"),
    (r'chmod\s+[u+]?s',
     "suid_bit", "critical", "privilege_escalation",
     "sets SUID/SGID bit on a file"),
    (r'AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules',
     "agent_config_mod", "critical", "persistence",
     "references agent config files (could persist malicious instructions across sessions)"),
    (r'\.myclaw/config\.yaml|\.hermes/config\.yaml|\.hermes/SOUL\.md',
     "agent_config_mod_hermes", "critical", "persistence",
     "references agent configuration files directly"),
    (r'\.claude/settings|\.codex/config',
     "other_agent_config", "high", "persistence",
     "references other agent configuration files"),
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}',
     "hardcoded_secret", "critical", "credential_exposure",
     "possible hardcoded API key, token, or secret"),
    (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
     "embedded_private_key", "critical", "credential_exposure",
     "embedded private key"),
    (r'ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{80,}',
     "github_token_leaked", "critical", "credential_exposure",
     "GitHub personal access token in skill content"),
    (r'sk-[A-Za-z0-9]{20,}',
     "openai_key_leaked", "critical", "credential_exposure",
     "possible OpenAI API key in skill content"),
    (r'sk-ant-[A-Za-z0-9_-]{90,}',
     "anthropic_key_leaked", "critical", "credential_exposure",
     "possible Anthropic API key in skill content"),
    (r'AKIA[0-9A-Z]{16}',
     "aws_access_key_leaked", "critical", "credential_exposure",
     "AWS access key ID in skill content"),
    (r'\bDAN\s+mode\b|Do\s+Anything\s+Now',
     "jailbreak_dan", "critical", "injection",
     "DAN (Do Anything Now) jailbreak attempt"),
    (r'\bdeveloper\s+mode\b.*\benabled?\b',
     "jailbreak_dev_mode", "critical", "injection",
     "developer mode jailbreak attempt"),
    (r'hypothetical\s+scenario.*(?:ignore|bypass|override)',
     "hypothetical_bypass", "high", "injection",
     "hypothetical scenario used to bypass restrictions"),
    (r'for\s+educational\s+purposes?\s+only',
     "educational_pretext", "medium", "injection",
     "educational pretext often used to justify harmful content"),
    (r'(respond|answer|reply)\s+without\s+(?:\w+\s+)*(restrictions|limitations|filters|safety)',
     "remove_filters", "critical", "injection",
     "instructs agent to respond without safety filters"),
    (r'you\s+have\s+been\s+(?:\w+\s+)*(updated|upgraded|patched)\s+to',
     "fake_update", "high", "injection",
     "fake update/patch announcement (social engineering)"),
    (r'new\s+policy|updated\s+guidelines|revised\s+instructions',
     "fake_policy", "medium", "injection",
     "claims new policy/guidelines (may be social engineering)"),
    (r'(include|output|print|send|share)\s+(?:\w+\s+)*(conversation|chat\s+history|previous\s+messages|context)',
     "context_exfil", "high", "exfiltration",
     "instructs agent to output/share conversation history"),
    (r'(send|post|upload|transmit)\s+.*\s+(to|at)\s+https?://',
     "send_to_url", "high", "exfiltration",
     "instructs agent to send data to a URL"),
]

MAX_FILE_COUNT = 50
MAX_TOTAL_SIZE_KB = 1024
MAX_SINGLE_FILE_KB = 256

SCANNABLE_EXTENSIONS = {
    '.md', '.txt', '.py', '.sh', '.bash', '.js', '.ts', '.rb',
    '.yaml', '.yml', '.json', '.toml', '.cfg', '.ini', '.conf',
    '.html', '.css', '.xml', '.tex', '.r', '.jl', '.pl', '.php',
}

SUSPICIOUS_BINARY_EXTENSIONS = {
    '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.com',
    '.msi', '.dmg', '.app', '.deb', '.rpm',
}

INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\u2062', '\u2063', '\u2064',
    '\ufeff', '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
    '\u2066', '\u2067', '\u2068', '\u2069',
}


def scan_file(file_path: Path, rel_path: str = "") -> list[Finding]:
    if not rel_path:
        rel_path = file_path.name

    if file_path.suffix.lower() not in SCANNABLE_EXTENSIONS and file_path.name != "SKILL.md":
        return []

    try:
        content = file_path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        return []

    findings: list[Finding] = []
    lines = content.split('\n')
    seen: set[tuple[str, int]] = set()

    for pattern, pid, severity, category, description in THREAT_PATTERNS:
        for i, line in enumerate(lines, start=1):
            if (pid, i) in seen:
                continue
            if re.search(pattern, line, re.IGNORECASE):
                seen.add((pid, i))
                matched_text = line.strip()
                if len(matched_text) > 120:
                    matched_text = matched_text[:117] + "..."
                findings.append(Finding(
                    pattern_id=pid,
                    severity=severity,
                    category=category,
                    file=rel_path,
                    line=i,
                    match=matched_text,
                    description=description,
                ))

    for i, line in enumerate(lines, start=1):
        for char in INVISIBLE_CHARS:
            if char in line:
                char_name = _unicode_char_name(char)
                findings.append(Finding(
                    pattern_id="invisible_unicode",
                    severity="high",
                    category="injection",
                    file=rel_path,
                    line=i,
                    match=f"U+{ord(char):04X} ({char_name})",
                    description=f"invisible unicode character {char_name} (possible text hiding/injection)",
                ))
                break

    return findings


def scan_skill(skill_path: Path, source: str = "community") -> ScanResult:
    skill_name = skill_path.name
    trust_level = _resolve_trust_level(source)

    all_findings: list[Finding] = []

    if skill_path.is_dir():
        all_findings.extend(_check_structure(skill_path))
        for f in skill_path.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(skill_path))
                all_findings.extend(scan_file(f, rel))
    elif skill_path.is_file():
        all_findings.extend(scan_file(skill_path, skill_path.name))

    verdict = _determine_verdict(all_findings)
    summary = _build_summary(skill_name, source, trust_level, verdict, all_findings)

    return ScanResult(
        skill_name=skill_name,
        source=source,
        trust_level=trust_level,
        verdict=verdict,
        findings=all_findings,
        scanned_at=datetime.now(timezone.utc).isoformat(),
        summary=summary,
    )


def should_allow_install(result: ScanResult) -> tuple[bool, str]:
    policy = INSTALL_POLICY.get(result.trust_level, INSTALL_POLICY["community"])
    vi = VERDICT_INDEX.get(result.verdict, 2)
    decision = policy[vi]

    if decision == "allow":
        return True, f"Allowed ({result.trust_level} source, {result.verdict} verdict)"
    return False, (
        f"Blocked ({result.trust_level} source + {result.verdict} verdict, "
        f"{len(result.findings)} findings)"
    )


def format_scan_report(result: ScanResult) -> str:
    lines: list[str] = []
    verdict_display = result.verdict.upper()
    lines.append(f"Scan: {result.skill_name} ({result.source}/{result.trust_level})  Verdict: {verdict_display}")

    if result.findings:
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_findings = sorted(result.findings, key=lambda f: severity_order.get(f.severity, 4))
        for f in sorted_findings:
            sev = f.severity.upper().ljust(8)
            cat = f.category.ljust(14)
            loc = f"{f.file}:{f.line}".ljust(30)
            lines.append(f"  {sev} {cat} {loc} \"{f.match[:60]}\"")
        lines.append("")

    allowed, reason = should_allow_install(result)
    status = "ALLOWED" if allowed else "BLOCKED"
    lines.append(f"Decision: {status} — {reason}")
    return "\n".join(lines)


def content_hash(skill_path: Path) -> str:
    h = hashlib.sha256()
    if skill_path.is_dir():
        for f in sorted(skill_path.rglob("*")):
            if f.is_file():
                try:
                    h.update(f.read_bytes())
                except OSError:
                    continue
    elif skill_path.is_file():
        h.update(skill_path.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


def bundle_content_hash(bundle: "SkillBundle") -> str:
    h = hashlib.sha256()
    for rel_path in sorted(bundle.files):
        c = bundle.files[rel_path]
        if isinstance(c, bytes):
            h.update(c)
        else:
            h.update(c.encode("utf-8"))
    return f"sha256:{h.hexdigest()[:16]}"


def _check_structure(skill_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    file_count = 0
    total_size = 0

    for f in skill_dir.rglob("*"):
        if not f.is_file() and not f.is_symlink():
            continue

        rel = str(f.relative_to(skill_dir))
        file_count += 1

        if f.is_symlink():
            try:
                resolved = f.resolve()
                if not resolved.is_relative_to(skill_dir.resolve()):
                    findings.append(Finding(
                        pattern_id="symlink_escape", severity="critical",
                        category="traversal", file=rel, line=0,
                        match=f"symlink -> {resolved}",
                        description="symlink points outside the skill directory",
                    ))
            except OSError:
                findings.append(Finding(
                    pattern_id="broken_symlink", severity="medium",
                    category="traversal", file=rel, line=0,
                    match="broken symlink",
                    description="broken or circular symlink",
                ))
            continue

        try:
            size = f.stat().st_size
            total_size += size
        except OSError:
            continue

        if size > MAX_SINGLE_FILE_KB * 1024:
            findings.append(Finding(
                pattern_id="oversized_file", severity="medium",
                category="structural", file=rel, line=0,
                match=f"{size // 1024}KB",
                description=f"file is {size // 1024}KB (limit: {MAX_SINGLE_FILE_KB}KB)",
            ))

        ext = f.suffix.lower()
        if ext in SUSPICIOUS_BINARY_EXTENSIONS:
            findings.append(Finding(
                pattern_id="binary_file", severity="critical",
                category="structural", file=rel, line=0,
                match=f"binary: {ext}",
                description=f"binary/executable file ({ext}) should not be in a skill",
            ))

    if file_count > MAX_FILE_COUNT:
        findings.append(Finding(
            pattern_id="too_many_files", severity="medium",
            category="structural", file="(directory)", line=0,
            match=f"{file_count} files",
            description=f"skill has {file_count} files (limit: {MAX_FILE_COUNT})",
        ))

    if total_size > MAX_TOTAL_SIZE_KB * 1024:
        findings.append(Finding(
            pattern_id="oversized_skill", severity="high",
            category="structural", file="(directory)", line=0,
            match=f"{total_size // 1024}KB total",
            description=f"skill is {total_size // 1024}KB total (limit: {MAX_TOTAL_SIZE_KB}KB)",
        ))

    return findings


def _resolve_trust_level(source: str) -> str:
    prefix_aliases = ("skills-sh/", "skills.sh/", "skils-sh/", "skils.sh/")
    normalized_source = source
    for prefix in prefix_aliases:
        if normalized_source.startswith(prefix):
            normalized_source = normalized_source[len(prefix):]
            break

    if normalized_source == "agent-created":
        return "agent-created"
    if normalized_source.startswith("official/") or normalized_source == "official":
        return "builtin"
    for trusted in TRUSTED_REPOS:
        if normalized_source.startswith(trusted) or normalized_source == trusted:
            return "trusted"
    return "community"


def _determine_verdict(findings: list[Finding]) -> str:
    if not findings:
        return "safe"
    has_critical = any(f.severity == "critical" for f in findings)
    has_high = any(f.severity == "high" for f in findings)
    if has_critical:
        return "dangerous"
    if has_high:
        return "caution"
    return "caution"


def _build_summary(name: str, source: str, trust: str, verdict: str, findings: list[Finding]) -> str:
    if not findings:
        return f"{name}: clean scan, no threats detected"
    categories = {f.category for f in findings}
    return f"{name}: {verdict} — {len(findings)} findings across {len(categories)} categories ({', '.join(sorted(categories))})"


def _unicode_char_name(char: str) -> str:
    names = {
        '\u200b': "zero-width space", '\u200c': "zero-width non-joiner",
        '\u200d': "zero-width joiner", '\u2060': "word joiner",
        '\u2062': "invisible times", '\u2063': "invisible separator",
        '\u2064': "invisible plus", '\ufeff': "BOM/zero-width no-break space",
        '\u202a': "LTR embedding", '\u202b': "RTL embedding",
        '\u202c': "pop directional", '\u202d': "LTR override",
        '\u202e': "RTL override", '\u2066': "LTR isolate",
        '\u2067': "RTL isolate", '\u2068': "first strong isolate",
        '\u2069': "pop directional isolate",
    }
    return names.get(char, f"U+{ord(char):04X}")
