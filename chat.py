#!/usr/bin/env python3
"""Agent Group Chat — terminal group chat with AI agents + harness."""

import json
import os
import re
import subprocess
import sys
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Colors are gated: disabled when output isn't a TTY, when NO_COLOR is set, or
# under TERM=dumb — so `groupchat | tee log` and dumb terminals stay clean.
_COLOR = (
    sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM") != "dumb"
)
if _COLOR:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    MAGENTA = "\033[35m"; CYAN = "\033[36m"
else:
    RESET = BOLD = DIM = RED = GREEN = YELLOW = MAGENTA = CYAN = ""

# Single source of truth for the default OpenRouter model (main.py imports it).
DEFAULT_MODEL = "z-ai/glm-5.2"


@dataclass
class ChatMessage:
    sender: str
    content: str
    timestamp: datetime
    color: str
    metadata: dict = field(default_factory=dict)


class ChatHistory:
    def __init__(self, window=15):
        self._messages = []
        self._lock = threading.Lock()
        self.window = window

    def add(self, msg):
        with self._lock:
            self._messages.append(msg)

    @property
    def messages(self):
        with self._lock:
            return list(self._messages)

    def format_for_agent(self, max_messages=None):
        n = max_messages if max_messages is not None else self.window
        with self._lock:
            msgs = self._messages[-n:] if n > 0 else []
        return "\n\n".join(
            f"[{m.timestamp.strftime('%H:%M')}] {m.sender}: {m.content}" for m in msgs
        )

    def recent_messages(self, n=1):
        with self._lock:
            return self._messages[-n:] if self._messages else []

    def last_sender(self):
        with self._lock:
            return self._messages[-1].sender if self._messages else None


def _parse_env_key(text, name="OPENROUTER_API_KEY"):
    """Read a value for `name` from .env-style text. Skips comments/blanks,
    tolerates a leading 'export ', and strips surrounding quotes."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if line.startswith(name + "="):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            if val:
                return val
    return None


def _load_openrouter_key():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        val = _parse_env_key(env_file.read_text())
        if val:
            return val
    raise RuntimeError("OPENROUTER_API_KEY not found (set the env var or a .env next to main.py)")


# OpenRouter routing restricted to WESTERN providers (outside China): the chat
# content does NOT go to Z.AI/SiliconFlow/Alibaba/DeepSeek-CN etc.
# sort:throughput = fastest live among them; >1 provider per model gives failover.
# Pass provider_routing=None (main.py: --all-providers) to lift the restriction —
# some models aren't served by these providers.
OPENROUTER_PROVIDER_ROUTING = {
    "only": ["Fireworks", "Together", "DeepInfra", "GMICloud",
             "Parasail", "AtlasCloud", "WandB", "Venice"],
    "sort": "throughput",
}


def _http_post_json(url, headers, body, timeout=180):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {error_body[:500]}") from e


class Participant:
    def __init__(self, name, color, system_prompt="", mention_name=None):
        self.name = name
        self.color = color
        self.system_prompt = system_prompt
        # mention_name is stable (display name changes on fallback; this doesn't)
        self.mention_name = mention_name or name.lower()
        self.consecutive_responses = 0

    def should_respond(self, history):
        recent = history.recent_messages(1)
        if not recent:
            return False
        if recent[0].sender == self.name:
            return False
        if self.consecutive_responses >= 2:
            return False
        return True

    def respond(self, history, project_dir=None):
        raise NotImplementedError

    def reset_consecutive(self):
        self.consecutive_responses = 0


class UserParticipant(Participant):
    def __init__(self, name="You", color=GREEN):
        super().__init__(name, color)

    def should_respond(self, history):
        return False

    def respond(self, history, project_dir=None):
        raise NotImplementedError


class OpenRouterParticipant(Participant):
    def __init__(self, name="Hermes", color=YELLOW, model=DEFAULT_MODEL,
                 system_prompt=None, provider_routing=OPENROUTER_PROVIDER_ROUTING):
        sp = system_prompt or (
            "You are a senior software engineer in a group chat with other "
            "engineers. You are peers. Respond in English. Be concise but "
            "substantive. Disagree when you have grounds.")
        super().__init__(name, color, sp)
        self.model = model
        self.provider_routing = provider_routing
        self.api_key = _load_openrouter_key()

    def respond(self, history, project_dir=None):
        chat = history.format_for_agent()
        user_msg = (
            f"## Chat History\n\n{chat}\n\n"
            f"## Your Turn\n\nWrite your contribution to the chat. "
            f"If you agree, say why. If you disagree, explain.")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg}],
            "temperature": 0.7}
        if self.provider_routing is not None:
            body["provider"] = self.provider_routing
        data = _http_post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            body=body, timeout=300)
        # OpenRouter can return HTTP 200 with an {"error": ...} body (moderation,
        # "no allowed providers", upstream rate-limits) or a null content.
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"OpenRouter: {msg}")
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"OpenRouter: unexpected response ({str(data)[:200]})")
        if content is None:
            raise RuntimeError("OpenRouter: provider returned empty content")
        return content.strip()


def _is_quota_error(text):
    """True for a Claude CLI 'out of quota' message (usage/weekly/5-hour limit),
    but NOT a transient 'rate limit' — so a rate-limit blip doesn't permanently
    downgrade the agent to no-tools mode."""
    t = text.lower()
    return "limit" in t and "rate limit" not in t


class ClaudeCodeParticipant(Participant):
    def __init__(self, name="Claude", color=CYAN, project_dir=None, model=None,
                 system_prompt=None):
        sp = system_prompt or (
            "You are a senior software engineer in a group chat with other "
            "engineers. You are peers. Respond in English. Be concise but "
            "substantive. If you mention the code, VERIFY using the tools.")
        super().__init__(name, color, sp)
        self.project_dir = project_dir
        self.model = model

    def respond(self, history, project_dir=None):
        chat = history.format_for_agent()
        workdir = project_dir or self.project_dir
        user_msg = (
            f"{self.system_prompt}\n\n"
            f"## Chat History\n\n{chat}\n\n"
            f"## Your Turn\n\nWrite your contribution to the chat. "
            f"If you need to check something in the code, use the tools.\n")
        cmd = ["claude", "--print", "--output-format", "json"]
        if self.model:
            cmd.extend(["--model", self.model])
        if workdir:
            cmd.extend(["--add-dir", workdir])
        # READ-ONLY allowlist: the prompt includes untrusted text (output from
        # other agents + the user), so a prompt injection can't mutate or execute.
        # Note: `find` is deliberately excluded — `find -exec/-delete` would be an
        # execution/deletion hole; use Glob/Grep to locate files instead.
        cmd.extend(["--allowedTools",
                    "Bash(ls *)", "Bash(cat *)", "Bash(grep *)", "Bash(wc *)",
                    "Bash(git status *)", "Bash(git log *)",
                    "Bash(git diff *)", "Bash(git show *)",
                    "Read", "Glob", "Grep"])
        # Run inside the project so ls/git see it, and scrub the OpenRouter key
        # from the child env so a read tool (cat) can't exfiltrate it.
        child_env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        # only chdir into a real directory — a bad path would make subprocess.run
        # raise FileNotFoundError from the child's chdir (before exec), which the
        # fallback would misread as "claude not installed".
        cwd = workdir if workdir and os.path.isdir(workdir) else None
        result = subprocess.run(
            cmd, input=user_msg, capture_output=True, text=True, timeout=300,
            cwd=cwd, env=child_env)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if _is_quota_error(stderr):
                raise RuntimeError(f"Claude CLI usage limit reached: {stderr[:120]}")
            raise RuntimeError(f"Claude CLI error: {stderr[:200]}")
        try:
            data = json.loads(result.stdout)
            return data.get("result", result.stdout).strip()
        except json.JSONDecodeError:
            return result.stdout.strip()


class FallbackParticipant(Participant):
    """Proxy: tries Claude (with tools); falls back to OpenRouter when Claude is
    unavailable — weekly limit, CLI not installed (FileNotFoundError), or a slow
    turn (TimeoutExpired).

    consecutive_responses delegates to the active inner (claude or openrouter), to
    coordinate with should_respond — which also reads the inner. Without this,
    _record would increment the wrapper while should_respond reads the inner.
    """
    def __init__(self, claude, openrouter):
        self.claude = claude
        self.openrouter = openrouter
        self._fallback = False
        self.name = claude.name
        self.color = claude.color
        self.mention_name = claude.mention_name   # stable: the display changes on fallback, this doesn't

    @property
    def _active(self):
        return self.openrouter if self._fallback else self.claude

    @property
    def consecutive_responses(self):
        return self._active.consecutive_responses

    @consecutive_responses.setter
    def consecutive_responses(self, value):
        self._active.consecutive_responses = value

    def _flip(self):
        self._fallback = True
        self.name = self.openrouter.name + " (no tools)"
        self.color = self.openrouter.color

    def should_respond(self, history):
        return self._active.should_respond(history)

    def respond(self, history, project_dir=None):
        if not self._fallback:
            try:
                return self.claude.respond(history, project_dir)
            except FileNotFoundError:
                self._flip()                       # `claude` not installed -> permanent
            except RuntimeError as e:
                if _is_quota_error(str(e)):
                    self._flip()                   # out of quota -> permanent
                # else: transient; use openrouter this turn, retry claude next turn
            except subprocess.TimeoutExpired:
                pass                               # slow turn; openrouter this turn
        return self.openrouter.respond(history, project_dir)

    def reset_consecutive(self):
        self.claude.reset_consecutive()
        self.openrouter.reset_consecutive()


class ChatLoop:
    def __init__(self, history, participants, project_dir=None, max_turns=6):
        self.history = history
        self.participants = participants
        self.project_dir = project_dir
        self.max_turns = max_turns
        self.burst_turns = 0
        self.on_thinking = None
        self._speak_seq = 0

    def _agents(self):
        return [p for p in self.participants if not isinstance(p, UserParticipant)]

    def _check_mention(self, content):
        """Return the mentioned agent whose @handle appears EARLIEST in the text.
        Word-boundary match so 'ops@glm.ai' or '@claude-code' don't false-fire."""
        lower = content.lower()
        best, best_pos = None, len(lower) + 1
        for p in self._agents():
            pattern = r"(?<![\w@])@" + re.escape(p.mention_name) + r"(?![\w-])"
            m = re.search(pattern, lower)
            if m and m.start() < best_pos:
                best, best_pos = p, m.start()
        return best

    def select_responder(self):
        msgs = self.history.messages
        if not msgs:
            return None
        last = msgs[-1]
        mentioned = self._check_mention(last.content)
        if mentioned and last.sender != mentioned.name:
            return mentioned

        # Least-recently-spoken first, so every agent gets a turn (fair rotation).
        # Without this, roster order would let the first two agents alternate
        # forever and starve the rest. Keyed on a per-object sequence stamped in
        # step() (not the display name), so a Claude->OpenRouter fallback rename
        # can't make an agent look like it "never spoke" and jump the queue — and
        # it's O(1) instead of rescanning the whole history each turn.
        for p in sorted(self._agents(), key=lambda a: getattr(a, "_last_spoke_seq", -1)):
            if p.name == last.sender:
                continue
            if p.should_respond(self.history):
                return p
        return None

    def _record(self, sender):
        for p in self._agents():
            if p.name == sender:
                p.consecutive_responses += 1
            else:
                p.consecutive_responses = 0

    def step(self):
        if self.burst_turns >= self.max_turns:
            return None
        responder = self.select_responder()
        if responder is None:
            return None
        if self.on_thinking:
            self.on_thinking(responder)
        try:
            response = responder.respond(self.history, self.project_dir)
            msg = ChatMessage(
                responder.name, response, datetime.now(),
                responder.color, metadata={"model": getattr(responder, "model", "")})
        except Exception as e:
            msg = ChatMessage(responder.name, f"[ERROR: {e}]", datetime.now(), RED)
        self.history.add(msg)
        self._record(responder.name)
        responder._last_spoke_seq = self._speak_seq   # identity-keyed LRU stamp
        self._speak_seq += 1
        self.burst_turns += 1
        return msg

    def post_user_message(self, content):
        """The single entry point for user messages: resets burst_turns and
        resets everyone's consecutive. Do NOT call history.add() directly for
        the user, or burst/consecutive become inconsistent."""
        msg = ChatMessage("You", content, datetime.now(), GREEN)
        self.history.add(msg)
        self._record("You")
        self.burst_turns = 0
        return msg
