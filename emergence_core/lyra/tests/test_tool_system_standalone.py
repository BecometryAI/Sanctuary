#!/usr/bin/env python3
"""
Standalone test for tool registry and cache functionality.
"""

import sys
import asyncio
from pathlib import Path

# Direct import by modifying sys.path
sys.path.insert(0, str(Path("emergence_core/lyra/cognitive_core")))

from tool_registry import ToolRegistry, ToolStatus, create_default_registry
from tool_cache import ToolCache


async def test_tool_registry():
    """Test tool registry functionality."""
    print("=" * 70)
    print("TEST: Tool Registry")
    print("=" * 70)
    
    # Create registry
    registry = ToolRegistry()
    
    # Register test tool
    async def multiply_tool(x: int, y: int) -> int:
        await asyncio.sleep(0.01)
        return x * y
    
    registry.register_tool(
        name="multiply",
        handler=multiply_tool,
        description="Multiply two numbers",
        timeout=5.0
    )
    
    print(f"\n✅ Registered tool: multiply")
    
    # Execute tool
    result = await registry.execute_tool("multiply", x=7, y=6)
    
    assert result.status == ToolStatus.SUCCESS
    assert result.result == 42
    print(f"✅ Tool execution successful: 7 * 6 = {result.result}")
    
    # Test timeout
    async def slow_tool():
        await asyncio.sleep(10)
        return "done"
    
    registry.register_tool("slow", slow_tool, "Slow tool", timeout=0.1)
    result = await registry.execute_tool("slow")
    
    assert result.status == ToolStatus.TIMEOUT
    print(f"✅ Timeout handling works")
    
    # Test error handling
    async def error_tool():
        raise ValueError("Test error")
    
    registry.register_tool("error", error_tool, "Error tool")
    result = await registry.execute_tool("error")
    
    assert result.status == ToolStatus.FAILURE
    print(f"✅ Error handling works")
    
    # Get stats
    stats = registry.get_tool_stats()
    print(f"\n📊 Tool statistics:")
    for tool_name, tool_stats in stats.items():
        print(f"  {tool_name}: {tool_stats['total_calls']} calls, "
              f"{tool_stats['successes']} successes")
    
    return True


async def test_default_registry():
    """Test default registry with example tools."""
    print("\n" + "=" * 70)
    print("TEST: Default Registry")
    print("=" * 70)
    
    registry = create_default_registry()
    
    tools = registry.get_available_tools()
    print(f"\n✅ Created default registry with {len(tools)} tools")
    
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")
    
    # Test web search
    result = await registry.execute_tool("web_search", query="Python", num_results=3)
    assert result.status == ToolStatus.SUCCESS
    print(f"\n✅ web_search executed successfully")
    print(f"  Query: Python")
    print(f"  Results: {len(result.result['results'])} results")
    
    # Test compute
    result = await registry.execute_tool("compute", expression="2 + 2")
    assert result.status == ToolStatus.SUCCESS
    print(f"\n✅ compute executed successfully")
    print(f"  Expression: 2 + 2")
    print(f"  Result: {result.result['result']}")
    
    return True


def test_tool_cache():
    """Test tool cache functionality."""
    print("\n" + "=" * 70)
    print("TEST: Tool Cache")
    print("=" * 70)
    
    cache = ToolCache(max_size=10, default_ttl=60.0)
    
    # Test set and get
    tool_name = "test_tool"
    inputs = {"query": "test"}
    result = {"data": "cached result"}
    
    cache.set(tool_name, inputs, result)
    cached = cache.get(tool_name, inputs)
    
    assert cached == result
    print(f"\n✅ Cache set/get works")
    
    # Test cache miss
    missed = cache.get("nonexistent", {"query": "missing"})
    assert missed is None
    print(f"✅ Cache miss detection works")
    
    # Test different inputs
    cache.set(tool_name, {"query": "test1"}, "result1")
    cache.set(tool_name, {"query": "test2"}, "result2")
    
    assert cache.get(tool_name, {"query": "test1"}) == "result1"
    assert cache.get(tool_name, {"query": "test2"}) == "result2"
    print(f"✅ Different inputs cached separately")
    
    # Test LRU eviction
    cache2 = ToolCache(max_size=3)
    cache2.set("tool", {"id": 1}, "r1")
    cache2.set("tool", {"id": 2}, "r2")
    cache2.set("tool", {"id": 3}, "r3")
    cache2.set("tool", {"id": 4}, "r4")  # Should evict id=1
    
    assert cache2.get("tool", {"id": 1}) is None
    assert cache2.get("tool", {"id": 4}) == "r4"
    print(f"✅ LRU eviction works")
    
    # Test stats
    stats = cache.get_stats()
    print(f"\n📊 Cache statistics:")
    print(f"  Size: {stats['size']}/{stats['max_size']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    
    return True


async def test_integration():
    """Test integration of registry and cache."""
    print("\n" + "=" * 70)
    print("TEST: Registry + Cache Integration")
    print("=" * 70)
    
    registry = ToolRegistry()
    cache = ToolCache()
    
    # Register tool
    async def add_tool(a: int, b: int) -> int:
        await asyncio.sleep(0.05)  # Simulate work
        return a + b
    
    registry.register_tool("add", add_tool, "Add numbers")
    
    tool_name = "add"
    inputs = {"a": 10, "b": 20}
    
    # First call - not cached
    import time
    start = time.time()
    
    cached_result = cache.get(tool_name, inputs)
    if cached_result is None:
        print(f"\n🔍 Cache miss - executing tool")
        result = await registry.execute_tool(tool_name, **inputs)
        if result.status == ToolStatus.SUCCESS:
            cache.set(tool_name, inputs, result.result)
            cached_result = result.result
    
    first_call_time = time.time() - start
    print(f"  First call: {first_call_time:.4f}s")
    print(f"  Result: {cached_result}")
    
    # Second call - should be cached
    start = time.time()
    cached_result = cache.get(tool_name, inputs)
    second_call_time = time.time() - start
    
    print(f"\n🎯 Cache hit")
    print(f"  Second call: {second_call_time:.4f}s")
    print(f"  Result: {cached_result}")
    print(f"  Speedup: {first_call_time / second_call_time:.1f}x faster")
    
    assert second_call_time < first_call_time
    print(f"\n✅ Cache provides performance improvement")
    
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TOOL SYSTEM TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Tool Registry", test_tool_registry()),
        ("Default Registry", test_default_registry()),
        ("Tool Cache", test_tool_cache()),
        ("Integration", test_integration()),
    ]
    
    results = []
    for name, test_coro in tests:
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
