import os
import subprocess
import sys
import unittest
from datetime import datetime
from unittest import mock
import chat
from chat import (ChatHistory, ChatMessage, ChatLoop, Participant,
                  UserParticipant, FallbackParticipant, OpenRouterParticipant,
                  ClaudeCodeParticipant)


def mk_history(*pairs):
    h = ChatHistory()
    for sender, content in pairs:
        h.add(ChatMessage(sender, content, datetime.now(), ""))
    return h


class FakeAgent(Participant):
    def __init__(self, name, reply="response"):
        super().__init__(name, "")
        self.reply = reply
    def respond(self, history, project_dir=None):
        return self.reply


class TestChatHistoryWindow(unittest.TestCase):
    def test_format_window_defaults_to_15(self):
        h = ChatHistory()
        for i in range(20):
            h.add(ChatMessage("A", f"msg{i}", datetime.now(), ""))
        out = h.format_for_agent()
        self.assertEqual(len(out.split("\n\n")), 15)

    def test_format_window_is_configurable(self):
        h = ChatHistory(window=3)
        for i in range(10):
            h.add(ChatMessage("A", f"msg{i}", datetime.now(), ""))
        out = h.format_for_agent()
        self.assertEqual(len(out.split("\n\n")), 3)

    def test_window_zero_sends_no_history(self):
        h = ChatHistory(window=0)
        for i in range(5):
            h.add(ChatMessage("A", f"msg{i}", datetime.now(), ""))
        self.assertEqual(h.format_for_agent(), "")


class TestChatLoopSelection(unittest.TestCase):
    def _loop(self, history):
        a, b = FakeAgent("A"), FakeAgent("B")
        return ChatLoop(history, [UserParticipant(), a, b]), a, b

    def test_select_skips_last_sender(self):
        loop, a, b = self._loop(mk_history(("You", "hi"), ("A", "spoke")))
        self.assertIs(loop.select_responder(), b)

    def test_mention_forces_agent(self):
        loop, a, b = self._loop(mk_history(("You", "hey @B what do you think")))
        self.assertIs(loop.select_responder(), b)

    def test_consecutive_cap_blocks_third(self):
        loop, a, b = self._loop(mk_history(("You", "hi")))
        a.consecutive_responses = 2
        self.assertFalse(a.should_respond(loop.history))

    def test_record_resets_other_agents(self):
        loop, a, b = self._loop(ChatHistory())
        a.consecutive_responses = 1
        loop._record("B")
        self.assertEqual(b.consecutive_responses, 1)
        self.assertEqual(a.consecutive_responses, 0)

    def test_step_idle_when_burst_exhausted(self):
        h = mk_history(("You", "topic"))
        a, b = FakeAgent("A"), FakeAgent("B")
        loop = ChatLoop(h, [UserParticipant(), a, b], max_turns=2)
        self.assertIsNotNone(loop.step())
        self.assertIsNotNone(loop.step())
        self.assertIsNone(loop.step())

    def test_user_message_resets_burst_and_revives(self):
        h = mk_history(("You", "topic"))
        a, b = FakeAgent("A"), FakeAgent("B")
        loop = ChatLoop(h, [UserParticipant(), a, b], max_turns=2)
        loop.step(); loop.step()
        self.assertIsNone(loop.step())
        loop.post_user_message("new question")
        self.assertEqual(loop.burst_turns, 0)
        self.assertIsNotNone(loop.step())

    def test_every_agent_gets_a_turn_no_starvation(self):
        # Regression: with roster order, the first two agents alternated forever
        # and the third (GLM) never spoke. Fair rotation must reach all of them.
        h = mk_history(("You", "topic"))
        roster = [UserParticipant(), FakeAgent("Claude"), FakeAgent("Hermes"), FakeAgent("GLM")]
        loop = ChatLoop(h, roster, max_turns=6)
        speakers = []
        for _ in range(6):
            m = loop.step()
            speakers.append(m.sender)
        self.assertEqual(set(speakers), {"Claude", "Hermes", "GLM"})

    def test_renamed_agent_does_not_queue_jump(self):
        # Rotation is keyed on a per-object seq, not the display name — so a
        # fallback rename (Claude -> "Claude (no tools)") can't make an agent that
        # already spoke look brand new and cut the line ahead of one that hasn't.
        a, b, c = FakeAgent("A"), FakeAgent("B"), FakeAgent("C")
        loop = ChatLoop(mk_history(("You", "t")), [UserParticipant(), a, b, c], max_turns=10)
        loop.step()                 # A speaks
        loop.step()                 # B speaks; C has never spoken
        a.name = "A (no tools)"     # simulate the fallback rename after A spoke
        self.assertIs(loop.select_responder(), c)   # C (never spoke), not renamed A


class TestMentionSelection(unittest.TestCase):
    def _loop(self, *agents):
        roster = [UserParticipant()] + [FakeAgent(n) for n in agents]
        return ChatLoop(ChatHistory(), roster), roster[1:]

    def test_earliest_mention_wins_over_roster_order(self):
        loop, (claude, hermes) = self._loop("Claude", "Hermes")
        # Hermes is mentioned first in the text but Claude is first in the roster
        self.assertIs(loop._check_mention("@hermes first, then @claude maybe"), hermes)

    def test_no_false_positive_inside_email(self):
        loop, (glm,) = self._loop("GLM")
        self.assertIsNone(loop._check_mention("ping ops@glm.ai about it"))

    def test_no_false_positive_on_hyphen_suffix(self):
        loop, (claude,) = self._loop("Claude")
        self.assertIsNone(loop._check_mention("see the @claude-code docs"))

    def test_trailing_punctuation_still_matches(self):
        loop, (glm,) = self._loop("GLM")
        self.assertIs(loop._check_mention("what about it, @GLM?"), glm)


class TestFallbackProxy(unittest.TestCase):
    def test_consecutive_proxies_to_active_inner(self):
        claude = FakeAgent("Claude")
        openr = FakeAgent("Claude")
        fb = FallbackParticipant(claude, openr)
        fb.consecutive_responses = 2          # routes to the active inner (claude)
        self.assertEqual(claude.consecutive_responses, 2)
        self.assertEqual(fb.consecutive_responses, 2)
        fb._fallback = True                   # now reflects the openrouter inner
        self.assertEqual(fb.consecutive_responses, 0)


class TestResolveTopic(unittest.TestCase):
    def test_resolve(self):
        from main import resolve_topic
        self.assertIsNone(resolve_topic(None, []))                       # just `groupchat` -> waits
        self.assertEqual(resolve_topic(None, ["talk", "about", "X"]),    # bare positional
                         "talk about X")
        self.assertEqual(resolve_topic("quoted phrase", []),             # --topic "..."
                         "quoted phrase")
        self.assertEqual(resolve_topic("let's", ["talk", "about", "how"]),  # the case that broke
                         "let's talk about how")


class TestMentionName(unittest.TestCase):
    def test_mention_name_defaults_to_lower(self):
        self.assertEqual(FakeAgent("GLM").mention_name, "glm")

    def test_mention_survives_fallback_rename(self):
        fb = FallbackParticipant(FakeAgent("Claude"), FakeAgent("Claude"))
        loop = ChatLoop(ChatHistory(), [UserParticipant(), fb, FakeAgent("Hermes")])
        fb._fallback = True
        fb.name = fb.openrouter.name + " (no tools)"     # simulates the rename in respond()
        self.assertEqual(fb.mention_name, "claude")       # stable mention
        self.assertIs(loop._check_mention("hey @claude, agree?"), fb)


class TestEmptyHistoryIdle(unittest.TestCase):
    def test_empty_history_is_idle_until_first_message(self):
        a, b = FakeAgent("A"), FakeAgent("B")
        loop = ChatLoop(ChatHistory(), [UserParticipant(), a, b])
        # no topic: empty history -> nobody responds -> idle (waits for the user)
        self.assertIsNone(loop.step())
        # the user's 1st message becomes the topic; agents start responding
        loop.post_user_message("first message")
        self.assertIsNotNone(loop.step())


class TestFallback(unittest.TestCase):
    def test_fallback_switches_on_weekly_limit(self):
        claude = FakeAgent("Claude", reply="with tools")
        def boom(history, project_dir=None):
            raise RuntimeError("Claude CLI usage limit reached")
        claude.respond = boom
        openr = FakeAgent("Claude", reply="no tools")
        fb = FallbackParticipant(claude, openr)

        out = fb.respond(mk_history(("You", "hi")))

        self.assertEqual(out, "no tools")
        self.assertTrue(fb._fallback)
        self.assertIn("no tools", fb.name)

    def test_fallback_when_claude_cli_missing(self):
        # `claude` not installed -> subprocess.run raises FileNotFoundError, which
        # must trigger a permanent fallback (not escape as a red [ERROR]).
        claude = FakeAgent("Claude")
        def missing(history, project_dir=None):
            raise FileNotFoundError(2, "No such file or directory", "claude")
        claude.respond = missing
        fb = FallbackParticipant(claude, FakeAgent("Claude", reply="no tools"))
        self.assertEqual(fb.respond(mk_history(("You", "hi"))), "no tools")
        self.assertTrue(fb._fallback)

    def test_rate_limit_is_transient_not_a_permanent_downgrade(self):
        claude = FakeAgent("Claude", reply="with tools")
        calls = {"n": 0}
        def flaky(history, project_dir=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("HTTP 429: rate limit exceeded, retry shortly")
            return "with tools"
        claude.respond = flaky
        fb = FallbackParticipant(claude, FakeAgent("Claude", reply="no tools"))
        self.assertEqual(fb.respond(mk_history(("You", "hi"))), "no tools")   # 1st via openrouter
        self.assertFalse(fb._fallback)                                        # NOT permanent
        self.assertEqual(fb.respond(mk_history(("You", "hi"))), "with tools") # retries claude

    def test_quota_helper_distinguishes_rate_limit(self):
        self.assertTrue(chat._is_quota_error("Claude usage limit reached"))
        self.assertTrue(chat._is_quota_error("Weekly limit reached"))
        self.assertFalse(chat._is_quota_error("rate limit exceeded"))
        self.assertFalse(chat._is_quota_error("some other error"))

    def test_transient_timeout_does_not_permanently_downgrade(self):
        claude = FakeAgent("Claude", reply="with tools")
        calls = {"n": 0}
        def flaky(history, project_dir=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise subprocess.TimeoutExpired("claude", 300)
            return "with tools"
        claude.respond = flaky
        fb = FallbackParticipant(claude, FakeAgent("Claude", reply="no tools"))
        self.assertEqual(fb.respond(mk_history(("You", "hi"))), "no tools")  # 1st: timeout -> openrouter
        self.assertFalse(fb._fallback)                                       # but not permanent
        self.assertEqual(fb.respond(mk_history(("You", "hi"))), "with tools")  # 2nd: claude again


class TestOpenRouterResponse(unittest.TestCase):
    def setUp(self):
        self._patch = mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def _respond_with(self, payload):
        with mock.patch.object(chat, "_http_post_json", return_value=payload):
            p = OpenRouterParticipant(name="Hermes", system_prompt="persona")
            return p.respond(mk_history(("You", "hi")))

    def test_error_body_raises_clear_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._respond_with({"error": {"message": "no allowed providers"}})
        self.assertIn("no allowed providers", str(ctx.exception))

    def test_null_content_raises(self):
        with self.assertRaises(RuntimeError):
            self._respond_with({"choices": [{"message": {"content": None}}]})

    def test_missing_choices_raises_clean(self):
        with self.assertRaises(RuntimeError):
            self._respond_with({})


class TestClaudeSubprocess(unittest.TestCase):
    class _Result:
        returncode = 0
        stdout = '{"result": "ok"}'
        stderr = ""

    def _run_capturing(self, project_dir):
        captured = {}
        def fake_run(cmd, **kw):
            captured.update(kw)
            return self._Result()
        with mock.patch.object(chat.subprocess, "run", fake_run), \
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "secret"}):
            p = ClaudeCodeParticipant(project_dir=project_dir)
            out = p.respond(mk_history(("You", "hi")))
        return out, captured

    def test_bad_project_dir_is_not_used_as_cwd(self):
        # Regression guard: a non-existent dir must NOT become cwd (would raise
        # FileNotFoundError from the child chdir and be misread as missing CLI).
        out, captured = self._run_capturing("/no/such/dir/xyz")
        self.assertEqual(out, "ok")
        self.assertIsNone(captured.get("cwd"))

    def test_valid_project_dir_becomes_cwd(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            _, captured = self._run_capturing(d)
            self.assertEqual(captured.get("cwd"), d)

    def test_openrouter_key_scrubbed_from_child_env(self):
        _, captured = self._run_capturing(None)
        self.assertNotIn("OPENROUTER_API_KEY", captured.get("env", {}))
        self.assertIn("PATH", captured.get("env", {}))   # rest of env preserved

    def test_find_not_in_allowlist(self):
        captured = {}
        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return self._Result()
        with mock.patch.object(chat.subprocess, "run", fake_run):
            ClaudeCodeParticipant().respond(mk_history(("You", "hi")))
        joined = " ".join(captured["cmd"])
        self.assertNotIn("Bash(find", joined)     # find -exec/-delete hole removed
        self.assertIn("Glob", captured["cmd"])


class TestRender(unittest.TestCase):
    def test_whitespace_only_long_line_is_not_dropped(self):
        import io
        import main
        from contextlib import redirect_stdout
        msg = ChatMessage("A", " " * 200, datetime.now(), "")
        buf = io.StringIO()
        with redirect_stdout(buf):
            main.print_msg(msg)
        # header + a (blank) body line — the whitespace line must not vanish silently
        self.assertGreaterEqual(len([ln for ln in buf.getvalue().split("\n")]), 3)


class TestEgressRouting(unittest.TestCase):
    def setUp(self):
        self._patch = mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def _capture_body(self, **kwargs):
        captured = {}
        def fake(url, headers, body, timeout=180):
            captured.update(body)
            return {"choices": [{"message": {"content": "ok"}}]}
        with mock.patch.object(chat, "_http_post_json", fake):
            p = OpenRouterParticipant(name="Hermes", system_prompt="persona", **kwargs)
            p.respond(mk_history(("You", "hi")))
        return captured

    def test_openrouter_body_restricts_to_western_providers(self):
        captured = self._capture_body()
        self.assertEqual(captured["provider"]["sort"], "throughput")
        self.assertIn("Fireworks", captured["provider"]["only"])
        self.assertNotIn("Novita", captured["provider"]["only"])       # SG, excluded
        self.assertNotIn("SiliconFlow", captured["provider"]["only"])  # CN, excluded

    def test_all_providers_omits_routing(self):
        captured = self._capture_body(provider_routing=None)
        self.assertNotIn("provider", captured)


class TestEnvParsing(unittest.TestCase):
    def test_quoted_and_exported_values_are_stripped(self):
        self.assertEqual(chat._parse_env_key("OPENROUTER_API_KEY=sk-plain"), "sk-plain")
        self.assertEqual(chat._parse_env_key('OPENROUTER_API_KEY="sk-quoted"'), "sk-quoted")
        self.assertEqual(chat._parse_env_key("export OPENROUTER_API_KEY='sk-exp'"), "sk-exp")

    def test_comments_and_blanks_are_skipped(self):
        text = "# OPENROUTER_API_KEY=old-commented\n\nOPENROUTER_API_KEY=sk-real\n"
        self.assertEqual(chat._parse_env_key(text), "sk-real")

    def test_missing_key_returns_none(self):
        self.assertIsNone(chat._parse_env_key("SOMETHING_ELSE=1"))


class TestInputThread(unittest.TestCase):
    def test_eof_flushes_unterminated_line_then_signals_eof(self):
        import main
        r, w = os.pipe()
        os.write(w, b"hello\nworld")   # 'world' has no trailing newline
        os.close(w)

        class FakeStdin:
            def fileno(self_inner):
                return r
        old = sys.stdin
        sys.stdin = FakeStdin()
        try:
            t = main.UserInputThread()
            t.start()
            t.join(timeout=2)
            self.assertFalse(t.is_alive())
            drained = []
            while True:
                item = t.get_input()
                if item is None:
                    break
                drained.append(item)
        finally:
            sys.stdin = old
            os.close(r)
        self.assertEqual(drained, ["hello", "world", main._EOF])
        self.assertFalse(t._running)


if __name__ == "__main__":
    unittest.main()
