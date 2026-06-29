#!/usr/bin/env python3
"""Agent Group Chat — terminal group chat with AI agents + harness."""

import json
import os
import subprocess
import threading
import time
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"
BLUE="\033[34m"; MAGENTA="\033[35m"; CYAN="\033[36m"; GRAY="\033[90m"

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
    def format_for_agent(self, agent_name, max_messages=None):
        n = max_messages if max_messages is not None else self.window
        with self._lock:
            msgs = self._messages[-n:]
        return "\n\n".join(
            f"[{m.timestamp.strftime('%H:%M')}] {m.sender}: {m.content}" for m in msgs
        )
    def recent_messages(self, n=1):
        with self._lock:
            return self._messages[-n:] if self._messages else []
    def last_sender(self):
        with self._lock:
            return self._messages[-1].sender if self._messages else None
    def last_n_senders(self, n=3):
        with self._lock:
            return [m.sender for m in self._messages[-n:]]

def _load_openrouter_key():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found (env or .env)")

# OpenRouter routing restricted to WESTERN providers (outside China): the chat
# content does NOT go to Z.AI/SiliconFlow/Alibaba/DeepSeek-CN etc.
# sort:throughput = fastest live among them; >1 provider per model gives failover.
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
    def __init__(self, name="Hermes", color=YELLOW, model="z-ai/glm-5.2", system_prompt=None):
        sp = system_prompt or (
            "You are a senior software engineer in a group chat with other "
            "engineers. You are peers. Respond in English. Be concise but "
            "substantive. Disagree when you have grounds.")
        super().__init__(name, color, sp)
        self.model = model
        self.api_key = _load_openrouter_key()
    def respond(self, history, project_dir=None):
        chat = history.format_for_agent(self.name)
        user_msg = (
            f"## Chat History\n\n{chat}\n\n"
            f"## Your Turn\n\nWrite your contribution to the chat. "
            f"If you agree, say why. If you disagree, explain.")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg}],
            "temperature": 0.7,
            "provider": OPENROUTER_PROVIDER_ROUTING}
        data = _http_post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            body=body, timeout=300)
        return data["choices"][0]["message"]["content"].strip()

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
        chat = history.format_for_agent(self.name)
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
        # other agents + the user), so prompt-injection can't mutate/execute.
        # No Bash(npm run *) (arbitrary exec) nor broad Bash(git *) (push/reset).
        cmd.extend(["--allowedTools",
                     "Bash(ls *)", "Bash(cat *)", "Bash(grep *)",
                     "Bash(find *)", "Bash(wc *)",
                     "Bash(git status *)", "Bash(git log *)",
                     "Bash(git diff *)", "Bash(git show *)",
                     "Read", "Glob", "Grep"])
        result = subprocess.run(
            cmd, input=user_msg, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "limit" in stderr.lower():
                raise RuntimeError("Claude CLI weekly limit reached")
            raise RuntimeError(f"Claude CLI error: {stderr[:200]}")
        try:
            data = json.loads(result.stdout)
            return data.get("result", result.stdout).strip()
        except json.JSONDecodeError:
            return result.stdout.strip()

class FallbackParticipant(Participant):
    """Proxy: tries Claude (with tools); falls back to OpenRouter on the weekly limit.

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

    def should_respond(self, history):
        return self._active.should_respond(history)

    def respond(self, history, project_dir=None):
        if not self._fallback:
            try:
                return self.claude.respond(history, project_dir)
            except RuntimeError as e:
                if "limit" in str(e).lower():
                    self._fallback = True
                    self.name = self.openrouter.name + " (no tools)"
                    self.color = self.openrouter.color
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

    def _check_mention(self, content):
        lower = content.lower()
        for p in self.participants:
            if isinstance(p, UserParticipant):
                continue
            if f"@{p.mention_name}" in lower:
                return p
        return None

    def select_responder(self):
        msgs = self.history.messages
        if not msgs:
            return None
        last = msgs[-1]
        mentioned = self._check_mention(last.content)
        if mentioned and last.sender != mentioned.name:
            return mentioned
        for p in self.participants:
            if isinstance(p, UserParticipant):
                continue
            if p.name == last.sender:
                continue
            if p.should_respond(self.history):
                return p
        return None

    def _record(self, sender):
        for p in self.participants:
            if isinstance(p, UserParticipant):
                continue
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
