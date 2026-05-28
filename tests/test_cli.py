from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant.cli import format_response


class CLITests(unittest.TestCase):
    def test_format_response_is_plain_text_not_raw_json(self) -> None:
        rendered = format_response(
            {
                "answer": "The answer text.",
                "sources": ["ECU-700_Series_Manual.md"],
                "confidence": 0.9,
                "needs_review": False,
            }
        )

        self.assertIn("Answer", rendered)
        self.assertIn("Sources", rendered)
        self.assertIn("- ECU-700_Series_Manual.md", rendered)
        self.assertIn("Confidence", rendered)
        self.assertNotIn('"answer"', rendered)


if __name__ == "__main__":
    unittest.main()
