"""MCP (Model Context Protocol) adapter for Tramontane.

Real MCP client connecting via stdio or SSE transport.  Translates
MCP tool definitions into Mistral function-calling format so any
MCP server plugs into Tramontane pipelines with one line.

Reference: ScaleMCP (hf.co/papers/2505.06416) — dynamic tool selection.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
from typing import Any

from pydantic import BaseModel

from tramontane.tools.registry import ToolCategory, TramontaneTool, registry

logger = logging.getLogger(__name__)


class MCPTransport(enum.Enum):
    """Transport protocol for MCP server communication."""

    STDIO = "stdio"
    SSE = "sse"


class MCPServerConfig(BaseModel):
    """Configuration for connecting to an MCP server."""

    server_id: str
    transport: MCPTransport
    command: list[str] | None = None
    url: str | None = None
    env: dict[str, str] = {}
    description: str = ""


class MCPAdapter:
    """Connects to MCP servers and registers their tools on the Tramontane registry.

    Supports stdio (subprocess) and SSE (HTTP) transports.  Translates
    MCP tool definitions to Mistral function-calling format automatically.
    """

    def __init__(
        self, tool_registry: TramontaneTool | None = None,
    ) -> None:
        self._registry = registry
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._server_tools: dict[str, list[str]] = {}

    # -- Connect -----------------------------------------------------------

    async def connect(
        self, config: MCPServerConfig,
    ) -> list[TramontaneTool]:
        """Connect to an MCP server and register its tools.

        Returns the list of TramontaneTools that were registered.
        """
        if config.transport == MCPTransport.STDIO:
            return await self._connect_stdio(config)
        return await self._connect_sse(config)

    async def _connect_stdio(
        self, config: MCPServerConfig,
    ) -> list[TramontaneTool]:
        """Connect via stdio transport (subprocess)."""
        if not config.command:
            logger.error("STDIO transport requires 'command' in config")
            return []

        env = {**os.environ, **config.env}
        try:
            proc = await asyncio.create_subprocess_exec(
                *config.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self._processes[config.server_id] = proc

            # Send initialize
            init_msg = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "tramontane",
                        "version": "0.1.2",
                    },
                },
                "id": 1,
            }
            await self._send_jsonrpc(proc, init_msg)
            await self._read_jsonrpc(proc)

            # Send initialized notification
            await self._send_jsonrpc(proc, {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            })

            # List tools
            tools_msg = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2,
            }
            await self._send_jsonrpc(proc, tools_msg)
            response = await self._read_jsonrpc(proc)

            return self._register_mcp_tools(config, response)

        except Exception:
            logger.warning(
                "Failed to connect to MCP server '%s' via stdio",
                config.server_id,
                exc_info=True,
            )
            return []

    async def _connect_sse(
        self, config: MCPServerConfig,
    ) -> list[TramontaneTool]:
        """Connect via SSE transport (HTTP)."""
        if not config.url:
            logger.error("SSE transport requires 'url' in config")
            return []

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    config.url.rstrip("/") + "/tools",
                    headers={"Accept": "application/json"},
                )
                if response.status_code == 200:
                    data = response.json()
                    tools_data: dict[str, Any] = {
                        "result": {"tools": data.get("tools", data)}
                    }
                    return self._register_mcp_tools(config, tools_data)

        except Exception:
            logger.warning(
                "Failed to connect to MCP server '%s' via SSE at %s",
                config.server_id, config.url,
                exc_info=True,
            )
        return []

    # -- Call tool ----------------------------------------------------------

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> Any:
        """Route a tool call back through the correct MCP server."""
        tool = self._registry.get(tool_name)
        server_id = tool.mcp_server_id
        if not server_id:
            msg = f"Tool '{tool_name}' is not an MCP tool"
            raise ValueError(msg)

        proc = self._processes.get(server_id)
        if proc is None:
            msg = f"No active connection to MCP server '{server_id}'"
            raise ConnectionError(msg)

        call_msg = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": 3,
        }
        await self._send_jsonrpc(proc, call_msg)
        response = await self._read_jsonrpc(proc)
        return response.get("result", response)

    def connect_sync(
        self, config: MCPServerConfig,
    ) -> list[TramontaneTool]:
        """Synchronous wrapper for connect(). Do not call from async context."""
        from tramontane.core._sync import run_sync

        return run_sync(self.connect(config))

    # -- Known server configs ----------------------------------------------

    @staticmethod
    def context7() -> MCPServerConfig:
        """Config for Context7 — real-time library documentation."""
        return MCPServerConfig(
            server_id="context7",
            transport=MCPTransport.SSE,
            url="https://mcp.context7.com/sse",
            description="Real-time library documentation",
        )

    @staticmethod
    def github(token: str | None = None) -> MCPServerConfig:
        """Config for GitHub MCP server."""
        return MCPServerConfig(
            server_id="github",
            transport=MCPTransport.STDIO,
            command=["npx", "-y", "@modelcontextprotocol/server-github"],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": token
                or os.environ.get("GITHUB_TOKEN", ""),
            },
            description="GitHub repos, issues, PRs",
        )

    @staticmethod
    def nymbo_tools() -> MCPServerConfig:
        """Config for Nymbo Tools — general purpose HuggingFace Spaces tools."""
        return MCPServerConfig(
            server_id="nymbo-tools",
            transport=MCPTransport.SSE,
            url="https://nymbo-tools.hf.space/mcp",
            description="General purpose tools from HuggingFace Spaces",
        )

    # -- Internal helpers --------------------------------------------------

    def _register_mcp_tools(
        self,
        config: MCPServerConfig,
        response: dict[str, Any],
    ) -> list[TramontaneTool]:
        """Parse MCP tools/list response and register as TramontaneTools."""
        result = response.get("result", {})
        raw_tools = result.get("tools", []) if isinstance(result, dict) else []
        registered: list[TramontaneTool] = []

        for raw in raw_tools:
            tool = TramontaneTool(
                name=raw.get("name", "unknown"),
                description=raw.get("description", ""),
                category=ToolCategory.MCP,
                json_schema=raw.get("inputSchema"),
                mcp_server_id=config.server_id,
            )
            self._registry.register(tool)
            registered.append(tool)

        self._server_tools[config.server_id] = [t.name for t in registered]

        logger.info(
            "Connected to %s: %d tools registered",
            config.server_id, len(registered),
        )
        return registered

    @staticmethod
    async def _send_jsonrpc(
        proc: asyncio.subprocess.Process,
        message: dict[str, Any],
    ) -> None:
        """Send a JSON-RPC message via stdin."""
        if proc.stdin is None:
            return
        data = json.dumps(message) + "\n"
        proc.stdin.write(data.encode())
        await proc.stdin.drain()

    @staticmethod
    async def _read_jsonrpc(
        proc: asyncio.subprocess.Process,
    ) -> dict[str, Any]:
        """Read a JSON-RPC response from stdout."""
        if proc.stdout is None:
            return {}
        try:
            line = await asyncio.wait_for(
                proc.stdout.readline(), timeout=10.0,
            )
            if line:
                result: dict[str, Any] = json.loads(line.decode())
                return result
        except (asyncio.TimeoutError, json.JSONDecodeError):
            pass
        return {}
