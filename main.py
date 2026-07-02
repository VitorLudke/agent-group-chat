#!/usr/bin/env python3
"""Agent Group Chat — fluid terminal group chat, WhatsApp-style.

You type at any moment (even while an agent is "thinking").
The agents debate in short bursts and go quiet when the burst is spent.

Usage:
  python3 main.py                                   # opens; your 1st message is the topic
  python3 main.py should we cache this now          # bare topic, no quotes
  python3 main.py --topic "..." --no-claude --project-dir ~/code/my-project
@Claude / @Hermes / @GLM to mention. "exit"/Ctrl-D/Ctrl-C save and quit.

Note: uses select() on stdin — macOS/Linux only (not Windows).
"""

import argparse
import os
import queue
import sys
import time
import threading
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

from chat import (
    ChatHistory, ChatLoop, DEFAULT_MODEL, OPENROUTER_PROVIDER_ROUTING,
    UserParticipant, OpenRouterParticipant, ClaudeCodeParticipant, FallbackParticipant,
    GREEN, CYAN, YELLOW, MAGENTA, RED, RESET, DIM, BOLD,
)

_EOF = object()
EXIT_WORDS = ("exit", "quit", "/q")


class UserInputThread(threading.Thread):
    """Reads stdin in the background (full lines) without blocking the agent loop.

    Uses a stdlib queue.Queue for hand-off. On any exit (EOF, error, or stop) it
    flushes a final unterminated line then enqueues _EOF, so the main loop never
    hangs waiting on a dead reader.
    """
    def __init__(self):
        super().__init__(daemon=True)
        self._queue = queue.Queue()
        self._running = True
        self._buf = ""

    def run(self):
        import select
        try:
            fd = sys.stdin.fileno()
            while self._running:
                if select.select([fd], [], [], 0.1)[0]:
                    # raw fd: select() + os.read() compose correctly. Do NOT use
                    # sys.stdin.read(1) (buffered) — over-read leaves chars stuck
                    # in Python's buffer, invisible to the next select().
                    data = os.read(fd, 4096)
                    if not data:  # EOF (Ctrl-D / closed pipe)
                        break
                    self._buf += data.decode("utf-8", errors="replace")
                    while "\n" in self._buf:
                        line, self._buf = self._buf.split("\n", 1)
                        self._queue.put(line)
        except Exception:
            pass
        finally:
            self._running = False
            if self._buf.strip():          # don't lose a final line without "\n"
                self._queue.put(self._buf)
                self._buf = ""
            self._queue.put(_EOF)

    def get_input(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def wait_for_input(self):
        # short timeout so Ctrl-C stays responsive while blocked
        while True:
            try:
                return self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

    def stop(self):
        self._running = False


def print_msg(msg):
    t = msg.timestamp.strftime("%H:%M")
    width = max(40, shutil.get_terminal_size((80, 24)).columns - 2)
    print(f"\n[{t}] {msg.color}{BOLD}{msg.sender}{RESET}:")
    for para in msg.content.split("\n"):
        if not para:
            print()
        elif len(para) <= width:
            print(f"  {para}")          # short line verbatim: preserves code indentation
        else:
            # replace_whitespace=False keeps tabs in long code lines; guard the
            # all-whitespace case (textwrap returns []) so the line isn't dropped.
            wrapped = textwrap.wrap(para, width=width, replace_whitespace=False)
            for line in wrapped:
                print(f"  {line}")
            if not wrapped:
                print()


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
    if not history.messages:            # nothing was said -> don't litter an empty file
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# Group Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for msg in history.messages:
            t = msg.timestamp.strftime("%H:%M:%S")
            f.write(f"**[{t}] {msg.sender}:**\n{msg.content}\n\n")
    print(f"\n{DIM}Chat saved to: {p}{RESET}")


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
# To customize: edit these prompts or the roster in build_participants().
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
    routing = None if args.all_providers else OPENROUTER_PROVIDER_ROUTING
    participants = [UserParticipant()]
    names = []
    if not args.no_claude:
        claude = ClaudeCodeParticipant(name="Claude", color=CYAN, project_dir=project_dir,
                                       model=args.model_claude, system_prompt=PERSONA_ARCHITECT)
        # the fallback carries the SAME persona -> architect survives Claude's limit
        or_fb = OpenRouterParticipant(name="Claude", color=CYAN, model=args.model_openrouter,
                                      system_prompt=PERSONA_ARCHITECT, provider_routing=routing)
        participants.append(FallbackParticipant(claude, or_fb))
        names.append(f"{CYAN}Claude{RESET}")
    participants.append(OpenRouterParticipant(name="Hermes", color=YELLOW, model=args.model_openrouter,
                                              system_prompt=PERSONA_PRAGMATIC, provider_routing=routing))
    names.append(f"{YELLOW}Hermes{RESET}")
    participants.append(OpenRouterParticipant(name="GLM", color=MAGENTA, model=args.model_openrouter,
                                              system_prompt=PERSONA_SKEPTIC, provider_routing=routing))
    names.append(f"{MAGENTA}GLM{RESET}")
    return participants, names


def main():
    parser = argparse.ArgumentParser(description="Fluid terminal group chat with AI agents")
    parser.add_argument("topic_words", nargs="*",
                        help="Initial topic (optional, no quotes). Without it, just start typing.")
    parser.add_argument("--topic", default=None,
                        help="Initial topic (alternative to the positional; use for punctuation)")
    parser.add_argument("--project-dir", default=None,
                        help="Directory the Claude agent may inspect with read-only tools")
    parser.add_argument("--no-claude", action="store_true",
                        help="OpenRouter only (2 agents); skips the Claude Code CLI agent")
    parser.add_argument("--all-providers", action="store_true",
                        help="Lift the western-only OpenRouter routing (needed for some models)")
    parser.add_argument("--model-openrouter", default=DEFAULT_MODEL,
                        help=f"OpenRouter model id for the text agents (default {DEFAULT_MODEL})")
    parser.add_argument("--model-claude", default=None,
                        help="Model id for the Claude Code agent (default: CLI's own default)")
    parser.add_argument("--save", default=None,
                        help="Path to save the transcript (default: chats/ next to main.py)")
    parser.add_argument("--max-turns", type=int, default=6,
                        help="Agent messages per burst before going quiet (default 6)")
    parser.add_argument("--delay", type=float, default=0.8,
                        help="Seconds to pause between agent turns (default 0.8)")
    parser.add_argument("--context-window", type=int, default=15,
                        help="How many recent messages each agent sees (default 15)")
    args = parser.parse_args()

    if args.max_turns < 1:
        parser.error("--max-turns must be >= 1")
    if args.context_window < 1:
        parser.error("--context-window must be >= 1")

    if sys.platform == "win32":
        print(f"{RED}Note:{RESET} non-blocking input uses select() on stdin, which is "
              "POSIX-only; this app is not supported on Windows.", file=sys.stderr)

    # topic: "--topic phrase", bare positional "groupchat talk about X", or nothing
    topic = resolve_topic(args.topic, args.topic_words)

    project_dir = os.path.expanduser(args.project_dir) if args.project_dir else None
    if project_dir and not os.path.isdir(project_dir):
        parser.error(f"--project-dir is not a directory: {project_dir}")
    project_name = Path(project_dir).name if project_dir else "general"

    history = ChatHistory(window=args.context_window)
    try:
        participants, agent_names = build_participants(args, project_dir)
    except RuntimeError as e:
        print(f"{RED}Error:{RESET} {e}", file=sys.stderr)
        print("Set OPENROUTER_API_KEY in your environment, or copy .env.example to "
              ".env (next to main.py) and fill it in.", file=sys.stderr)
        sys.exit(1)

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

    default_dir = Path(__file__).resolve().parent / "chats"
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
        save_path = args.save or str(default_dir / f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md")
        save_chat(history, save_path)


if __name__ == "__main__":
    main()
