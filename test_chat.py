import os
import unittest
from datetime import datetime
import chat
from chat import (ChatHistory, ChatMessage, ChatLoop, Participant,
                  UserParticipant, FallbackParticipant, OpenRouterParticipant)


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
        out = h.format_for_agent("B")
        self.assertEqual(len(out.split("\n\n")), 15)

    def test_format_window_is_configurable(self):
        h = ChatHistory(window=3)
        for i in range(10):
            h.add(ChatMessage("A", f"msg{i}", datetime.now(), ""))
        out = h.format_for_agent("B")
        self.assertEqual(len(out.split("\n\n")), 3)


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
            raise RuntimeError("Claude CLI weekly limit reached")
        claude.respond = boom
        openr = FakeAgent("Claude", reply="no tools")
        fb = FallbackParticipant(claude, openr)

        out = fb.respond(mk_history(("You", "hi")))

        self.assertEqual(out, "no tools")
        self.assertTrue(fb._fallback)
        self.assertIn("no tools", fb.name)


class TestEgressRouting(unittest.TestCase):
    def test_openrouter_body_restricts_to_western_providers(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key"   # avoids reading .env
        captured = {}
        orig = chat._http_post_json
        def fake(url, headers, body, timeout=180):
            captured.update(body)
            return {"choices": [{"message": {"content": "ok"}}]}
        chat._http_post_json = fake
        try:
            p = OpenRouterParticipant(name="Hermes", system_prompt="persona")
            p.respond(mk_history(("You", "hi")))
        finally:
            chat._http_post_json = orig
        self.assertEqual(captured["provider"]["sort"], "throughput")
        self.assertIn("Fireworks", captured["provider"]["only"])
        self.assertNotIn("Novita", captured["provider"]["only"])       # SG, excluded
        self.assertNotIn("SiliconFlow", captured["provider"]["only"])  # CN, excluded


if __name__ == "__main__":
    unittest.main()
