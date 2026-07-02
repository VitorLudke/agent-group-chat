# agent-group-chat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![platform: macOS | Linux](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![dependencies: zero](https://img.shields.io/badge/dependencies-zero-brightgreen)

A terminal group chat where **several AI agents debate each other as peers** —
WhatsApp-style — and you join in, free to cut in at any moment. It's not
round-robin busywork: the agents hold distinct viewpoints, disagree, debate in
short bursts, and go quiet when the burst is spent. Point them at a project and
one of them can read the code (read-only) to keep the argument grounded in facts.

```
============================================================
  Group Chat - my-project
  Participants: You, Claude, Hermes, GLM
  Type + Enter | @name mentions | Ctrl+C/Ctrl+D to quit
============================================================

  Start typing your first message and press Enter...
> is it worth adding auth now or later?

[10:24] Claude:
  Structurally I'd add the seam now — a thin auth boundary is cheap to
  introduce early and expensive to retrofit once handlers assume an open
  context.

[10:24] Hermes:
  Disagree on urgency. If nothing sensitive is exposed yet, a stub that
  always-allows plus one integration test buys the seam without the cost.

[10:24] GLM:
  The hole both of you skip: "not exposed yet" is an assumption, not a fact.
  What's the current bind address and are the routes actually unreachable?
```

## Why this exists

It started as a copy-paste problem. I kept shuttling the same context back and
forth between two terminal AI agents — Claude Code and a second CLI agent — to get a
second opinion on one specific call: an architecture trade-off, whether to build
something now or defer it, where a design might quietly break. Being the human
courier between two assistants, pasting each one's reply into the other, was the
whole friction.

So instead of me relaying messages, this drops the agents into one room to argue the
point directly — as peers, with distinct viewpoints — while I watch and cut in when
it matters. You bring the question and (optionally) the codebase; they do the
back-and-forth.

## Features

- **Distinct persona per agent** (not "fake diversity"): by default an
  **Architect** (structure, long term), a **Pragmatist** (ship-it, anti
  over-engineering) and a **Skeptic** (risk, cost, what breaks). Each has its own
  system prompt.
- **Fair rotation**: every agent gets turns — the loop picks the least-recently
  spoken eligible agent, so no one is starved out of the debate.
- **Fluid, non-blocking input**: you type at any moment, even while an agent is
  "thinking". Your message lands and steers the next turn. On a real terminal a
  raw-mode line editor keeps your in-progress typing on a prompt line below the
  chat (agent output doesn't scramble it), arrow keys don't leak escape codes, and
  a multi-line paste becomes a single message. Piped/redirected input falls back to
  a line reader.
- **Frictionless entry**: just run `python3 main.py` and your first message becomes
  the topic.
- **Optional harness with tools**: one agent can run via the Claude Code CLI (with
  a read-only tool allowlist on your project); the others via OpenRouter (text).
  Automatic fallback Claude → OpenRouter when the CLI is missing or the weekly
  limit is hit.
- **Western-only routing** (default): OpenRouter calls are restricted to providers
  outside China (`provider.only` + `sort:throughput`). Lift it with
  `--all-providers`.
- **Zero dependencies** — Python standard library only.

## Requirements

- Python 3.9+
- **macOS or Linux** — the non-blocking input uses `select()` on stdin, which is
  POSIX-only (Windows is not supported).
- `OPENROUTER_API_KEY` (env var or a `.env` next to `main.py` — see `.env.example`)
- *(optional)* [Claude Code CLI](https://docs.claude.com/claude-code) for the
  tool-using agent. Without it, run `--no-claude` (or let the automatic fallback
  handle it).

## Usage

```bash
export OPENROUTER_API_KEY=<your-key>       # or copy .env.example -> .env

python3 main.py                            # opens; your 1st message becomes the topic
python3 main.py should we cache this here  # bare topic, no quotes
python3 main.py --topic "auth now or later?"          # quote if the topic has ? or *
python3 main.py --no-claude                # OpenRouter only (2 agents)
python3 main.py --topic "..." --project-dir ~/code/my-project   # Claude with tools
```

> Note: a bare topic goes through your shell, so quote anything with shell
> metacharacters (`?`, `*`, `~`). When in doubt, use `--topic "..."`.

Flags: `--topic`, `--project-dir`, `--no-claude`, `--all-providers`,
`--model-openrouter`, `--model-claude`, `--save`, `--max-turns` (burst size),
`--delay`, `--context-window`. Run `python3 main.py --help` for the full list.
Mention `@Claude` / `@Hermes` / `@GLM` to force an agent next. `exit`, `quit`,
`/q`, Ctrl-D or Ctrl-C save the chat to markdown and quit.

Optional shortcut:

```bash
alias groupchat='python3 /path/to/agent-group-chat/main.py'
```

## How it works

- `chat.py` — pure core: `ChatMessage`, `ChatHistory` (sliding window), the
  `Participant`s (OpenRouter / Claude CLI / Fallback) and the `ChatLoop` (fair
  speaker selection, mentions, anti-spam, burst termination). No terminal I/O —
  unit-testable.
- `main.py` — UI: non-blocking input (`select` + `os.read` on a background thread),
  rendering, and the persona roster.

## Customizing the agents

The personas are plain constants and the roster is one function — both in
`main.py`:

- Edit `PERSONA_ARCHITECT` / `PERSONA_PRAGMATIC` / `PERSONA_SKEPTIC` to change how
  each agent argues.
- Edit `build_participants()` to rename agents, add/remove them, or change which
  model each uses.

## Security

The Claude agent's prompt includes untrusted text (other agents' output + your
messages), so it runs with an **enforced** read-only tool allowlist (passed as
`--allowedTools`, not merely requested in the prompt): `ls`, `cat`, `grep`, `wc`,
`git status/log/diff/show`, `Read`, `Glob`, `Grep`. There is **no** `Bash` that
can write, execute, or push — `find` is deliberately excluded because
`find -exec`/`-delete` would be an execution/deletion hole; use `Glob`/`Grep` to
locate files instead.

Caveat worth knowing: read tools (`cat`, `grep`) can read files your user account
can access, not only the project — so run the tool-using agent on repos you trust.
The `OPENROUTER_API_KEY` is scrubbed from the Claude subprocess environment, is
read only from the environment or a git-ignored `.env`, and is never written to
the saved transcript.

## Tests

```bash
python3 -m unittest test_chat -v
```

## License

MIT
