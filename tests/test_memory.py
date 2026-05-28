from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant.conversation import ConversationManager
from me_engineering_assistant.graph import ECUAgent
from me_engineering_assistant.memory import MemoryStore


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["ME_USE_LLM_PLANNER"] = "false"
        os.environ["ME_USE_LLM_ANSWER"] = "false"
        os.environ["ME_FORCE_LLM"] = "false"

    def test_long_term_memory_is_persisted_and_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.sqlite"
            store = MemoryStore(path)

            store.add_memory(
                "User preference: use standard RAG instead of hardcoded question-specific handlers.",
                scope="ecu-project",
                kind="preference",
                importance=0.9,
            )
            matches = store.search_memories("standard RAG architecture", scope="ecu-project", limit=3)

            self.assertEqual(len(matches), 1)
            self.assertIn("standard RAG", matches[0].content)

    def test_reflection_updates_summary_and_redacts_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(Path(temp_dir) / "memory.sqlite")
            manager = ConversationManager(store, scope="ecu-project")

            result = manager.record_turn(
                session_id="demo",
                query="请记住：不要使用 hardcoded synthesizer，要使用 standard RAG。",
                answer="Understood.",
                confidence=0.9,
                sources=[],
                previous_summary="",
            )
            secret_result = manager.record_turn(
                session_id="demo",
                query="remember this api key: sk-1234567890abcdef",
                answer="I will not store secrets.",
                confidence=0.4,
                sources=[],
                previous_summary=store.get_session_summary(session_id="demo", scope="ecu-project"),
            )

            summary = store.get_session_summary(session_id="demo", scope="ecu-project")
            turns = store.recent_turns(session_id="demo", scope="ecu-project", limit=5)

            self.assertTrue(result.stored_memory_ids)
            self.assertFalse(secret_result.stored_memory_ids)
            self.assertIn("standard RAG", summary)
            self.assertIn("[REDACTED_SECRET]", summary)
            self.assertNotIn("sk-1234567890abcdef", summary)
            self.assertNotIn("sk-1234567890abcdef", " ".join(turn.user_message for turn in turns))

    def test_agent_uses_recent_turns_for_follow_up_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(Path(temp_dir) / "memory.sqlite")
            agent = ECUAgent(docs_dir=ROOT, memory_enabled=True, memory_store=store, memory_scope="ecu-project")

            first = agent.answer("How much RAM does the ECU-850 have?", session_id="demo")
            second = agent.answer("What about the plus version?", session_id="demo", include_trace=True)

            step_names = [step["name"] for step in second.trace or []]
            self.assertIn("2 GB", first.answer)
            self.assertIn("4 GB", second.answer)
            self.assertIn("load_memory", step_names)
            self.assertIn("reflect_memory", step_names)


if __name__ == "__main__":
    unittest.main()
