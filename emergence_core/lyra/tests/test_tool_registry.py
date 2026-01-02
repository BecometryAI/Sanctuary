"""
Tests for Tool Registry
"""
import pytest
import asyncio
from lyra.cognitive_core.tool_registry import (
    ToolRegistry,
    ToolResult,
    ToolStatus,
    ToolDefinition,
    create_default_registry
)


@pytest.mark.asyncio
async def test_tool_registration():
    """Test registering tools."""
    registry = ToolRegistry()
    
    async def dummy_tool(x: int) -> int:
        return x * 2
    
    registry.register_tool(
        name="double",
        handler=dummy_tool,
        description="Double a number",
        timeout=5.0
    )
    
    assert registry.is_tool_registered("double")
    assert len(registry.get_available_tools()) == 1


@pytest.mark.asyncio
async def test_tool_execution():
    """Test executing a registered tool."""
    registry = ToolRegistry()
    
    async def add_tool(a: int, b: int) -> int:
        await asyncio.sleep(0.01)  # Simulate work
        return a + b
    
    registry.register_tool("add", add_tool, "Add two numbers")
    
    result = await registry.execute_tool("add", a=5, b=3)
    
    assert result.status == ToolStatus.SUCCESS
    assert result.result == 8
    assert result.execution_time > 0


@pytest.mark.asyncio
async def test_tool_timeout():
    """Test tool execution timeout."""
    registry = ToolRegistry()
    
    async def slow_tool():
        await asyncio.sleep(10)  # Takes too long
        return "done"
    
    registry.register_tool("slow", slow_tool, "Slow tool", timeout=0.1)
    
    result = await registry.execute_tool("slow")
    
    assert result.status == ToolStatus.TIMEOUT
    assert result.error is not None


@pytest.mark.asyncio
async def test_tool_error():
    """Test tool execution error handling."""
    registry = ToolRegistry()
    
    async def error_tool():
        raise ValueError("Something went wrong")
    
    registry.register_tool("error", error_tool, "Error tool")
    
    result = await registry.execute_tool("error")
    
    assert result.status == ToolStatus.FAILURE
    assert "Something went wrong" in result.error


@pytest.mark.asyncio
async def test_unregistered_tool():
    """Test executing unregistered tool."""
    registry = ToolRegistry()
    
    result = await registry.execute_tool("nonexistent")
    
    assert result.status == ToolStatus.ERROR
    assert "not registered" in result.error


@pytest.mark.asyncio
async def test_tool_stats():
    """Test tool execution statistics."""
    registry = ToolRegistry()
    
    async def test_tool(x: int) -> int:
        return x + 1
    
    registry.register_tool("test", test_tool, "Test tool")
    
    # Execute multiple times
    await registry.execute_tool("test", x=1)
    await registry.execute_tool("test", x=2)
    await registry.execute_tool("test", x=3)
    
    stats = registry.get_tool_stats("test")
    
    assert stats["test"]["total_calls"] == 3
    assert stats["test"]["successes"] == 3
    assert stats["test"]["failures"] == 0


@pytest.mark.asyncio
async def test_default_registry():
    """Test creating default registry with example tools."""
    registry = create_default_registry()
    
    tools = registry.get_available_tools()
    
    assert len(tools) > 0
    
    # Check that expected tools are registered
    tool_names = [tool["name"] for tool in tools]
    assert "web_search" in tool_names
    assert "compute" in tool_names
    assert "arxiv_search" in tool_names


@pytest.mark.asyncio
async def test_tool_unregistration():
    """Test unregistering a tool."""
    registry = ToolRegistry()
    
    async def temp_tool():
        return "temp"
    
    registry.register_tool("temp", temp_tool, "Temporary tool")
    assert registry.is_tool_registered("temp")
    
    result = registry.unregister_tool("temp")
    assert result is True
    assert not registry.is_tool_registered("temp")


@pytest.mark.asyncio
async def test_execution_history():
    """Test execution history tracking."""
    registry = ToolRegistry()
    
    async def test_tool():
        return "result"
    
    registry.register_tool("test", test_tool, "Test tool")
    
    # Execute tool
    await registry.execute_tool("test")
    
    assert len(registry.execution_history) == 1
    assert registry.execution_history[0].tool_name == "test"


@pytest.mark.asyncio
async def test_clear_history():
    """Test clearing execution history."""
    registry = ToolRegistry()
    
    async def test_tool():
        return "result"
    
    registry.register_tool("test", test_tool, "Test tool")
    
    await registry.execute_tool("test")
    assert len(registry.execution_history) > 0
    
    registry.clear_history()
    assert len(registry.execution_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
