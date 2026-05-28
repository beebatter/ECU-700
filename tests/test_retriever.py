from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant.documents import chunk_documents, load_source_documents
from me_engineering_assistant.retriever import InMemoryECURetriever


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = InMemoryECURetriever(chunk_documents(load_source_documents(base_path=ROOT)))

    def test_hybrid_retrieval_returns_all_storage_sources(self) -> None:
        results = self.retriever.retrieve(
            "storage capacity across all ECU models",
            filters={"models": ["ECU-750", "ECU-850", "ECU-850b"], "fields": ["storage"]},
            top_k=6,
        )

        self.assertEqual(
            {result.metadata["source"] for result in results},
            {"ECU-700_Series_Manual.md", "ECU-800_Series_Base.md", "ECU-800_Series_Plus.md"},
        )
        self.assertTrue(all(result.metadata["retrieval_method"] == "hybrid" for result in results))

    def test_bm25_keyword_signal_finds_malformed_table_row(self) -> None:
        results = self.retriever.retrieve("Internal Flash", filters={"models": ["ECU-750"], "fields": ["storage"]})

        self.assertIn("2 MB Internal Flash", results[0].content)
        self.assertIn("bm25_score", results[0].metadata)


if __name__ == "__main__":
    unittest.main()
