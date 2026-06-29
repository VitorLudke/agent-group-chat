#!/usr/bin/env python3
"""Agent Group Chat — fluid terminal group chat, WhatsApp-style.

You type at any moment (even while an agent is "thinking").
The agents debate in short bursts and go quiet when the burst is spent.

Usage:
  groupchat                              # opens; your 1st message becomes the topic
  groupchat is it worth caching here?    # bare topic, no quotes
  groupchat --topic "..." --no-claude --project-dir ~/code/my-project
@Claude / @Hermes / @GLM to mention. "exit"/Ctrl-D/Ctrl-C save and quit.
"""

import argparse
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

from chat import (
    ChatHistory, ChatLoop,
    UserParticipant, OpenRouterParticipant, ClaudeCodeParticipant, FallbackParticipant,
    GREEN, CYAN, YELLOW, MAGENTA, RESET, DIM, BOLD,
)

_EOF = object()
EXIT_WORDS = ("exit", "quit", "/q")


class UserInputThread(threading.Thread):
    """Reads stdin in the background (full lines) without blocking the agent loop."""
    def __init__(self):
        super().__init__(daemon=True)
        self._queue = []
        self._lock = threading.Lock()
        self._running = True
        self._buf = ""

    def run(self):
        import select
        fd = sys.stdin.fileno()
        while self._running:
            try:
                if select.select([fd], [], [], 0.1)[0]:
                    # raw fd: select() + os.read() compose correctly. Do NOT use
                    # sys.stdin.read(1) (buffered) — over-read leaves chars stuck
                    # in Python's buffer, invisible to the next select().
                    data = os.read(fd, 4096)
                    if not data:  # EOF (Ctrl-D / closed pipe)
                        with self._lock:
                            self._queue.append(_EOF)
                        self._running = False
                        break
                    self._buf += data.decode("utf-8", errors="replace")
                    while "\n" in self._buf:
                        line, self._buf = self._buf.split("\n", 1)
                        with self._lock:
                            self._queue.append(line)
            except Exception:
                break

    def get_input(self):
        with self._lock:
            return self._queue.pop(0) if self._queue else None

    def wait_for_input(self):
        while self._running or self._queue:
            item = self.get_input()
            if item is not None:
                return item
            time.sleep(0.05)
        return _EOF

    def stop(self):
        self._running = False


def print_msg(msg):
    t = msg.timestamp.strftime("%H:%M")
    print(f"\n[{t}] {msg.color}{BOLD}{msg.sender}{RESET}:")
    for line in msg.content.split("\n"):
        while len(line) > 80:
            print(f"  {line[:80]}")
            line = line[80:]
        print(f"  {line}")


def print_thinking(responder):
    print(f"\n  {DIM}{responder.color}{responder.name}{RESET} is thinking...{RESET}",
          end="", flush=True)


def print_done():
    print("\r" + " " * 60 + "\r", end="", flush=True)


def make_banner(project_name, agent_names):
    names = ", ".join([f"{GREEN}You{RESET}"] + agent_names)
    return (
        f"{BOLD}{'=' * 60}{RESET}\n"
        f"  {BOLD}Group Chat{RESET} - {CYAN}{project_name}{RESET}\n"
        f"  Participants: {names}\n"
        f"  {DIM}Type + Enter | @name mentions | Ctrl+C/Ctrl+D to quit{RESET}\n"
        f"{BOLD}{'=' * 60}{RESET}"
    )


def save_chat(history, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# Group Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for msg in history.messages:
            t = msg.timestamp.strftime("%H:%M:%S")
            f.write(f"**[{t}] {msg.sender}:**\n{msg.content}\n\n")
    print(f"\n{DIM}Chat saved to: {path}{RESET}")


def resolve_topic(topic_flag, topic_words):
    """Join the topic from --topic and/or the bare words (no quotes).
    Returns None when nothing was passed (opens and waits for the 1st message)."""
    topic = topic_flag
    if topic_words:
        joined = " ".join(topic_words)
        topic = f"{topic} {joined}".strip() if topic else joined
    return topic


# Personas (debate lenses). Each agent has its own -> viewpoint diversity, not just
# a label. Claude's persona travels with it on fallback (survives the weekly limit).
PERSONA_ARCHITECT = (
    "You are the ARCHITECT of the group: you think about structure, coupling, "
    "maintenance and the long term. When you mention the code, VERIFY with the "
    "tools. Respond in English, concise, as a peer. Disagree with grounds.")
PERSONA_PRAGMATIC = (
    "You are the PRAGMATIST of the group: you focus on the smallest increment that "
    "delivers value; you're skeptical of over-engineering and premature abstraction. "
    "Respond in English, concise, as a peer. Disagree when it's needless complexity.")
PERSONA_SKEPTIC = (
    "You are the SKEPTIC of the group: you look for what breaks — risks, cost, "
    "security, edge cases, what the group is ignoring. Respond in English, concise, "
    "as a peer. Disagree by pointing at the hole.")


def build_participants(args, project_dir):
    """Build the roster with distinct personas. Returns (participants, colored_names)."""
    participants = [UserParticipant()]
    names = []
    if not args.no_claude:
        claude = ClaudeCodeParticipant(name="Claude", color=CYAN, project_dir=project_dir,
                                       model=args.model_claude, system_prompt=PERSONA_ARCHITECT)
        # the fallback carries the SAME persona -> architect survives Claude's limit
        or_fb = OpenRouterParticipant(name="Claude", color=CYAN,
                                      model=args.model_openrouter, system_prompt=PERSONA_ARCHITECT)
        participants.append(FallbackParticipant(claude, or_fb))
        names.append(f"{CYAN}Claude{RESET}")
    participants.append(OpenRouterParticipant(name="Hermes", color=YELLOW,
                                              model=args.model_openrouter, system_prompt=PERSONA_PRAGMATIC))
    names.append(f"{YELLOW}Hermes{RESET}")
    participants.append(OpenRouterParticipant(name="GLM", color=MAGENTA,
                                              model=args.model_openrouter, system_prompt=PERSONA_SKEPTIC))
    names.append(f"{MAGENTA}GLM{RESET}")
    return participants, names


def main():
    parser = argparse.ArgumentParser(description="Fluid terminal group chat with AI agents")
    parser.add_argument("topic_words", nargs="*",
                        help="Initial topic (optional, no quotes). Without it, just start typing.")
    parser.add_argument("--topic", default=None, help="Initial topic (alternative to the positional)")
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--no-claude", action="store_true")
    parser.add_argument("--model-openrouter", default="z-ai/glm-5.2")
    parser.add_argument("--model-claude", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--context-window", type=int, default=15)
    args = parser.parse_args()

    # topic: "--topic phrase", bare positional "groupchat talk about X", or nothing
    topic = resolve_topic(args.topic, args.topic_words)

    project_dir = os.path.expanduser(args.project_dir) if args.project_dir else None
    project_name = Path(project_dir).name if project_dir else "general"

    history = ChatHistory(window=args.context_window)
    participants, agent_names = build_participants(args, project_dir)
    loop = ChatLoop(history, participants, project_dir=project_dir, max_turns=args.max_turns)
    loop.on_thinking = print_thinking

    print(make_banner(project_name, agent_names))
    if topic:
        loop.post_user_message(topic)
        print_msg(history.messages[-1])
    else:
        print(f"\n  {DIM}Start typing your first message and press Enter...{RESET}")

    inp = UserInputThread()
    inp.start()

    def handle_user(item):
        """Handle a user line. Returns False if we should quit."""
        if item is _EOF:
            return False
        if item.strip().lower() in EXIT_WORDS:
            return False
        if item.strip():
            loop.post_user_message(item)
            print_msg(history.messages[-1])
        return True

    try:
        while True:
            # 1. drain the user's input BEFORE the next agent turn
            item = inp.get_input()
            if item is not None:
                if not handle_user(item):
                    break
                continue
            # 2. one agent turn, if the burst isn't spent
            msg = loop.step()
            if msg is not None:
                print_done()
                print_msg(msg)
                time.sleep(args.delay)
                continue
            # 3. idle: wait for the user
            if not handle_user(inp.wait_for_input()):
                break
    except KeyboardInterrupt:
        pass
    finally:
        inp.stop()
        save_path = args.save or f"chats/chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        save_chat(history, save_path)


if __name__ == "__main__":
    main()
