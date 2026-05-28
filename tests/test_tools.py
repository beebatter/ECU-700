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
from me_engineering_assistant.retriever import InMemoryECURetriever
from me_engineering_assistant.tools import ECUToolbox


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        docs = load_source_documents(base_path=ROOT)
        chunks = chunk_documents(docs)
        cls.toolbox = ECUToolbox(
            InMemoryECURetriever(chunks),
            catalog=build_document_catalog(docs),
            field_table=build_model_field_table(chunks),
        )

    def test_tool_manifest_exposes_function_calls(self) -> None:
        names = {tool["name"] for tool in self.toolbox.manifest()}

        self.assertEqual(
            names,
            {"search_documents", "list_sources", "get_document_catalog", "get_model_field_evidence", "check_evidence_coverage"},
        )

    def test_search_documents_returns_grounded_source_chunks(self) -> None:
        result = self.toolbox.search_documents("How much RAM does the ECU-850 have?", models=["ECU-850"])

        self.assertIn("2 GB", str(result.result))
        self.assertIn("ECU-800_Series_Base.md", result.sources)

    def test_search_documents_accepts_series_filter(self) -> None:
        result = self.toolbox.search_documents("OTA update capability", series=["ECU-800"], top_k=3)

        self.assertTrue(result.result)
        self.assertEqual(result.arguments["series"], ["ECU-800"])
        self.assertNotIn("ECU-700_Series_Manual.md", result.sources)

    def test_search_documents_accepts_field_filter(self) -> None:
        result = self.toolbox.search_documents("ECU-750 storage specification", models=["ECU-750"], field="storage")

        self.assertIn("2 MB Internal Flash", str(result.result))
        self.assertEqual(result.sources, ["ECU-700_Series_Manual.md"])

    def test_model_field_evidence_and_coverage_tools(self) -> None:
        evidence = self.toolbox.get_model_field_evidence(
            models=["ECU-750", "ECU-850", "ECU-850b"],
            field="storage",
        )
        coverage = self.toolbox.check_evidence_coverage(
            models=["ECU-750", "ECU-850", "ECU-850b"],
            field="storage",
        )

        self.assertEqual(len(evidence.result), 3)
        self.assertTrue(coverage.result["complete"])

    def test_list_sources_returns_indexed_documents(self) -> None:
        result = self.toolbox.list_sources()

        self.assertEqual(len(result.result), 3)
        self.assertIn("ECU-700_Series_Manual.md", result.sources)


if __name__ == "__main__":
    unittest.main()
