from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


class MCPServerTests(unittest.TestCase):
    @unittest.skipIf(ClientSession is None, "mcp SDK is not installed")
    def test_stdio_server_lists_and_calls_tools(self) -> None:
        asyncio.run(self._assert_stdio_server())

    async def _assert_stdio_server(self) -> None:
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "ME_USE_LLM_PLANNER": "false",
            "ME_USE_LLM_ANSWER": "false",
            "ME_FORCE_LLM": "false",
        }
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "me_engineering_assistant.mcp_server",
                "--docs-dir",
                str(ROOT),
            ],
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = {tool.name for tool in tools.tools}
                result = await session.call_tool("search_documents", {"query": "How much RAM does ECU-850 have?"})

        self.assertIn("search_documents", tool_names)
        self.assertIn("list_sources", tool_names)
        self.assertFalse(result.isError)
        self.assertIn("2 GB", str(result.structuredContent["result"]))
        self.assertIn("ECU-800_Series_Base.md", result.structuredContent["sources"])


if __name__ == "__main__":
    unittest.main()
