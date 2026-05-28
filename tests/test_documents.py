from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant.documents import chunk_documents, load_source_documents


class DocumentTests(unittest.TestCase):
    def test_loads_expected_documents_with_metadata(self) -> None:
        documents = load_source_documents(base_path=ROOT)

        self.assertEqual(len(documents), 3)
        self.assertEqual({doc.metadata["model"] for doc in documents}, {"ECU-750", "ECU-850", "ECU-850b"})
        self.assertEqual({doc.metadata["series"] for doc in documents}, {"ECU-700", "ECU-800"})

    def test_chunks_do_not_require_strict_markdown_tables(self) -> None:
        documents = load_source_documents(base_path=ROOT)
        chunks = chunk_documents(documents)

        self.assertGreaterEqual(len(chunks), 8)
        self.assertTrue(any("CAN Interface" in chunk.content for chunk in chunks))

    def test_table_text_remains_available_for_rag_retrieval(self) -> None:
        chunks = chunk_documents(load_source_documents(base_path=ROOT))
        combined = "\n".join(chunk.content for chunk in chunks)

        self.assertIn("1 Mbps", combined)
        self.assertIn("2 GB", combined)
        self.assertIn("5 TOPS AI Accelerator", combined)


if __name__ == "__main__":
    unittest.main()
