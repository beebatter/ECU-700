from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant.documents import chunk_documents, load_source_documents
from me_engineering_assistant.retriever import InMemoryECURetriever
from me_engineering_assistant.tools import ECUToolbox


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        docs = load_source_documents(base_path=ROOT)
        cls.toolbox = ECUToolbox(InMemoryECURetriever(chunk_documents(docs)))

    def test_tool_manifest_exposes_function_calls(self) -> None:
        names = {tool["name"] for tool in self.toolbox.manifest()}

        self.assertEqual(names, {"search_documents", "list_sources"})

    def test_search_documents_returns_grounded_source_chunks(self) -> None:
        result = self.toolbox.search_documents("How much RAM does the ECU-850 have?", models=["ECU-850"])

        self.assertIn("2 GB", str(result.result))
        self.assertIn("ECU-800_Series_Base.md", result.sources)

    def test_list_sources_returns_indexed_documents(self) -> None:
        result = self.toolbox.list_sources()

        self.assertEqual(len(result.result), 3)
        self.assertIn("ECU-700_Series_Manual.md", result.sources)


if __name__ == "__main__":
    unittest.main()
