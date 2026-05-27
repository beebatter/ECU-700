from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from me_engineering_assistant.documents import chunk_documents, load_source_documents
from me_engineering_assistant.knowledge import extract_specs


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

    def test_extracts_specs_from_malformed_can_row(self) -> None:
        specs = extract_specs(load_source_documents(base_path=ROOT))

        self.assertIn("1 Mbps", specs["ECU-750"].can_interface or "")
        self.assertEqual(specs["ECU-850"].memory_ram, "2 GB LPDDR4")
        self.assertEqual(specs["ECU-850b"].npu, "5 TOPS AI Accelerator")


if __name__ == "__main__":
    unittest.main()

