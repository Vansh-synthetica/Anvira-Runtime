"""Minimal stdio MCP server fixture for Orcha integration tests.

Deliberately uses only the low-level MCP SDK server API (no FastMCP layer)
so the integration test exercises the real transport machinery both sides.

Run:  python mcp_echo_server.py
"""
import sys

from mcp.server.mcpserver.server import MCPServer

server = MCPServer(name="echo-server")


@server.tool(
    description="Echo a message back, uppercased.",
)
def echo(message: str) -> str:
    return f"echo: {message.upper()}"


@server.tool(
    description="Add two integers.",
)
def add(a: int, b: int) -> int:
    return a + b


@server.tool(
    description="Always fails — exercises the server-side error path.",
)
def fail() -> str:
    raise ValueError("deliberate server failure")


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
