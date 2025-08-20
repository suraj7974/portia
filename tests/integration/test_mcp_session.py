"""MCP Session Tests."""

import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import mcp
import pytest

from portia.mcp_session import (
    SseMcpClientConfig,
    StdioMcpClientConfig,
    StreamableHttpMcpClientConfig,
    get_mcp_session,
)

SERVER_FILE_PATH = Path(__file__).parent / "mcp_server.py"


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.mark.asyncio
async def test_mcp_session_stdio() -> None:
    """Test the MCP session with stdio."""
    async with get_mcp_session(
        StdioMcpClientConfig(
            server_name="test_server",
            command="uv",
            args=["run", "python", str(SERVER_FILE_PATH.absolute()), "stdio"],
        ),
    ) as session:
        tools = await session.list_tools()
        assert isinstance(tools, mcp.ListToolsResult)
        assert len(tools.tools) == 1


@pytest.fixture
def sse_background_server() -> Iterator[None]:
    """Start the MCP server in the background."""
    process = subprocess.Popen(["uv", "run", "python", str(SERVER_FILE_PATH.absolute()), "sse"])  # noqa: S607, S603
    try:
        # Wait for server to start
        time.sleep(3)

        # Check if process is still running
        if process.poll() is not None:
            raise Exception(f"Server process exited with code {process.poll()}")  # noqa: TRY002

        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.mark.asyncio
@pytest.mark.usefixtures("sse_background_server")
async def test_mcp_session_sse() -> None:
    """Test the MCP session with SSE."""
    async with get_mcp_session(
        SseMcpClientConfig(
            server_name="test_server",
            url="http://localhost:11385/sse",
            sse_read_timeout=5,
            timeout=5,
        ),
    ) as session:
        tools = await session.list_tools()
        assert isinstance(tools, mcp.ListToolsResult)
        assert len(tools.tools) == 1


@pytest.fixture
def streamable_http_background_server() -> Iterator[None]:
    """Start the MCP server in StreamableHTTP mode in the background."""
    process = subprocess.Popen(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "python",
            str(SERVER_FILE_PATH.absolute()),
            "streamable-http",
        ]
    )
    try:
        # Wait for server to start
        time.sleep(3)

        # Check if process is still running
        if process.poll() is not None:
            raise Exception(f"Server process exited with code {process.poll()}")  # noqa: TRY002

        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.mark.asyncio
@pytest.mark.usefixtures("streamable_http_background_server")
async def test_mcp_session_streamable_http() -> None:
    """Test the MCP session with StreamableHTTP."""
    async with get_mcp_session(
        StreamableHttpMcpClientConfig(
            server_name="test_server",
            url="http://localhost:11385/mcp",
            timeout=5,
            sse_read_timeout=5,
        ),
    ) as session:
        tools = await session.list_tools()
        assert isinstance(tools, mcp.ListToolsResult)
        assert len(tools.tools) == 1
