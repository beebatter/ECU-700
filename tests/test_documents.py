from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant.documents import (
    build_document_catalog,
    build_model_field_table,
    chunk_documents,
    load_source_documents,
)


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

    def test_table_rows_create_field_chunks_for_coverage(self) -> None:
        chunks = chunk_documents(load_source_documents(base_path=ROOT))
        storage_chunks = [
            chunk for chunk in chunks
            if chunk.metadata.get("chunk_type") == "field" and chunk.metadata.get("field") == "storage"
        ]

        self.assertEqual({chunk.metadata["model"] for chunk in storage_chunks}, {"ECU-750", "ECU-850", "ECU-850b"})
        self.assertTrue(any("2 MB Internal Flash" in chunk.content for chunk in storage_chunks))

    def test_catalog_and_model_field_table_are_built(self) -> None:
        documents = load_source_documents(base_path=ROOT)
        chunks = chunk_documents(documents)
        catalog = build_document_catalog(documents)
        field_table = build_model_field_table(chunks)
        fields = {row.field for row in field_table}

        self.assertEqual(len(catalog), 3)
        self.assertIn("storage", fields)
        self.assertIn("memory_ram", fields)
        self.assertIn("can_interface", fields)
        self.assertTrue({"operating_temperature", "operating_temp"} & fields)


if __name__ == "__main__":
    unittest.main()
