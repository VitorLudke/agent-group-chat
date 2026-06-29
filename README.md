# agent-group-chat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![dependencies: zero](https://img.shields.io/badge/dependencies-zero-brightgreen)

A terminal group chat where **several AI agents debate each other as peers** —
WhatsApp-style — and you join in, free to cut in at any moment. It's not
round-robin: the agents disagree organically, debate in short bursts, and go quiet
when the topic is spent.

```
============================================================
  Group Chat - my-project
  Participants: You, Claude, Hermes, GLM
  Type + Enter | @name mentions | Ctrl+C/Ctrl+D to quit
============================================================

  Start typing your first message and press Enter...
> is it worth adding auth now or later?

[10:24] Hermes:
  I'd push back on deferring. If it's already exposed, a basic gate costs
  almost nothing and removes a whole class of accidents...

[10:24] GLM:
  the hole in Hermes' take: he assumes late migration costs the same as early.
  it doesn't...
```

## Features

- **Distinct persona per agent** (not "fake diversity"): by default an
  **Architect** (structure, long term), a **Pragmatist** (ship-it, anti
  over-engineering) and a **Skeptic** (risk, cost, what breaks). Each has its own
  system prompt.
- **Fluid, non-blocking input**: you type at any moment, even while an agent is
  "thinking". Your message lands and steers the debate.
- **Frictionless entry**: just run `python3 main.py` and your first message becomes
  the topic.
- **Optional harness with tools**: one agent can run via the Claude Code CLI (with
  read-only `ls`/`cat`/`grep`/`git log` on your project); the others via OpenRouter
  (text). Automatic fallback Claude → OpenRouter when the weekly limit is hit.
- **Western-only routing**: OpenRouter calls are restricted to providers outside
  China (`provider.only` + `sort:throughput`).
- **Zero dependencies** — Python standard library only.

## Requirements

- Python 3.9+
- `OPENROUTER_API_KEY` (env var or a `.env` next to `main.py` — see `.env.example`)
- *(optional)* [Claude Code CLI](https://docs.claude.com/claude-code) for the
  tool-using agent

## Usage

```bash
export OPENROUTER_API_KEY=<your-key>       # or copy .env.example -> .env

python3 main.py                            # opens; your 1st message becomes the topic
python3 main.py is it worth caching here?  # bare topic, no quotes
python3 main.py --no-claude                # OpenRouter only (2 agents)
python3 main.py --topic "..." --project-dir ~/code/my-project   # Claude with tools
```

Handy flags: `--max-turns` (burst size), `--delay`, `--context-window`,
`--model-openrouter`, `--save`. Mention `@Claude` / `@Hermes` / `@GLM` to force an
agent. `exit` / Ctrl-D / Ctrl-C save the chat to markdown and quit.

Optional shortcut:

```bash
alias groupchat='python3 /path/to/agent-group-chat/main.py'
```

## How it works

- `chat.py` — pure core: `ChatMessage`, `ChatHistory` (sliding window), the
  `Participant`s (OpenRouter / Claude CLI / Fallback) and the `ChatLoop` (who speaks,
  anti-spam, burst termination). No terminal I/O — unit-testable.
- `main.py` — UI: non-blocking input (`select` + `os.read` on a background thread),
  rendering, and the persona roster.

## Security

The Claude agent's prompt includes untrusted text (other agents' output + your
messages), so its tool allowlist is **read-only** (`ls`, `cat`, `grep`, `find`,
`wc`, `git status/log/diff/show`, `Read`, `Glob`, `Grep`) — a prompt injection can't
mutate or execute anything in your project. The API key is read from the environment
or a git-ignored `.env`, never committed.

## Tests

```bash
python3 -m unittest test_chat -v
```

## License

MIT
