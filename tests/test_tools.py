from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from me_engineering_assistant.documents import chunk_documents, load_source_documents
from me_engineering_assistant.knowledge import extract_specs
from me_engineering_assistant.retriever import InMemoryECURetriever
from me_engineering_assistant.tools import ECUToolbox


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        docs = load_source_documents(base_path=ROOT)
        cls.toolbox = ECUToolbox(
            extract_specs(docs),
            InMemoryECURetriever(chunk_documents(docs)),
        )

    def test_tool_manifest_exposes_function_calls(self) -> None:
        names = {tool["name"] for tool in self.toolbox.manifest()}

        self.assertEqual(
            names,
            {"search_documents", "read_model_spec", "compare_model_specs", "list_models"},
        )

    def test_read_model_spec_function_returns_grounded_source(self) -> None:
        result = self.toolbox.read_model_spec("ECU-850")

        self.assertEqual(result.result["memory_ram"], "2 GB LPDDR4")
        self.assertEqual(result.sources, ["ECU-800_Series_Base.md"])

    def test_compare_model_specs_function_returns_grounded_sources(self) -> None:
        result = self.toolbox.compare_model_specs(["ECU-750", "ECU-850"], fields=["can"])

        self.assertIn("1 Mbps", str(result.result))
        self.assertIn("2 Mbps", str(result.result))
        self.assertEqual(result.sources, ["ECU-700_Series_Manual.md", "ECU-800_Series_Base.md"])


if __name__ == "__main__":
    unittest.main()
