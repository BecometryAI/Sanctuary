"""
Tool Registry: Central registry for all available tools.

This module implements the ToolRegistry class, which manages external tools
that the cognitive system can invoke to extend its capabilities. Tools include
web search, computation engines, document retrieval, and code execution.

The tool registry is responsible for:
- Registering tools with execution handlers
- Safe execution with timeouts and error handling
- Result validation and error reporting
- Tool discovery and capability advertisement
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Import for percept generation (avoiding circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .workspace import Percept


class ToolStatus(str, Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ToolResult:
    """
    Result of tool execution.
    
    Attributes:
        tool_name: Name of the tool that was executed
        status: Execution status (success, failure, timeout, error)
        result: Actual result data (if successful)
        error: Error message (if failed)
        execution_time: How long execution took (seconds)
        timestamp: When execution completed
        metadata: Additional context about execution
    """
    tool_name: str
    status: ToolStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    """
    Result of tool execution with percept generation for feedback loop.
    
    This class extends ToolResult to include percept generation, enabling
    bidirectional feedback where tool results automatically create percepts
    that inform future cognitive cycles.
    
    Attributes:
        tool_name: Name of the tool that was executed
        success: Whether execution succeeded
        result: Actual result data (if successful)
        error: Error message (if failed)
        execution_time_ms: How long execution took (milliseconds)
        timestamp: When execution completed
        percept: Generated percept for feedback loop (can be None)
        metadata: Additional context about execution
    """
    tool_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    percept: Optional['Percept'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """
    Definition of a registered tool.
    
    Attributes:
        name: Tool identifier
        handler: Async function to execute the tool
        description: Human-readable description
        timeout: Max execution time in seconds
        parameters: Expected parameters and their types
        return_type: Expected return type
        requires_network: Whether tool needs network access
    """
    name: str
    handler: Callable
    description: str
    timeout: float = 30.0
    parameters: Dict[str, str] = field(default_factory=dict)
    return_type: str = "Any"
    requires_network: bool = False


class ToolRegistry:
    """
    Central registry for all available tools.
    
    The ToolRegistry manages the system's external tool capabilities by:
    - Maintaining a registry of available tools
    - Executing tools with timeout and error handling
    - Validating tool inputs and outputs
    - Tracking tool usage statistics
    - Providing tool discovery interface
    
    Tools are registered with async handlers that can perform various
    operations like web search, computation, document retrieval, etc.
    Each tool has a timeout to prevent hanging operations.
    
    Example:
        ```python
        registry = ToolRegistry()
        
        async def web_search(query: str) -> Dict:
            # Perform web search
            return {"results": [...]}
        
        registry.register_tool(
            name="web_search",
            handler=web_search,
            description="Search the web using SearXNG",
            timeout=10.0
        )
        
        result = await registry.execute_tool("web_search", query="Python tutorials")
        ```
    
    Attributes:
        tools: Dictionary of registered tools
        execution_history: Recent tool executions
        stats: Usage statistics per tool
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, ToolDefinition] = {}
        self.execution_history: List[ToolResult] = []
        self.stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🔧 ToolRegistry initialized")
    
    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        timeout: float = 30.0,
        parameters: Optional[Dict[str, str]] = None,
        return_type: str = "Any",
        requires_network: bool = False
    ) -> None:
        """
        Register a tool with the registry.
        
        Args:
            name: Unique tool identifier
            handler: Async function that executes the tool
            description: Human-readable description of what the tool does
            timeout: Maximum execution time in seconds
            parameters: Expected parameters {name: type}
            return_type: Expected return type
            requires_network: Whether tool requires network access
        """
        if name in self.tools:
            logger.warning(f"⚠️ Tool '{name}' already registered, overwriting")
        
        tool_def = ToolDefinition(
            name=name,
            handler=handler,
            description=description,
            timeout=timeout,
            parameters=parameters or {},
            return_type=return_type,
            requires_network=requires_network
        )
        
        self.tools[name] = tool_def
        self.stats[name] = {
            "total_calls": 0,
            "successes": 0,
            "failures": 0,
            "timeouts": 0,
            "total_time": 0.0
        }
        
        logger.info(f"✅ Registered tool: {name} (timeout: {timeout}s)")
    
    async def execute_tool(
        self,
        name: str,
        **kwargs
    ) -> ToolResult:
        """
        Execute a tool with timeout and error handling.
        
        Args:
            name: Name of tool to execute
            **kwargs: Parameters to pass to tool handler
            
        Returns:
            ToolResult with execution status and result/error
        """
        if name not in self.tools:
            error_msg = f"Tool '{name}' not registered"
            logger.error(f"❌ {error_msg}")
            return ToolResult(
                tool_name=name,
                status=ToolStatus.ERROR,
                error=error_msg
            )
        
        tool = self.tools[name]
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool.handler(**kwargs),
                timeout=tool.timeout
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Success
            tool_result = ToolResult(
                tool_name=name,
                status=ToolStatus.SUCCESS,
                result=result,
                execution_time=execution_time,
                metadata={"parameters": kwargs}
            )
            
            # Update stats
            self.stats[name]["total_calls"] += 1
            self.stats[name]["successes"] += 1
            self.stats[name]["total_time"] += execution_time
            
            logger.info(f"✅ Tool '{name}' executed successfully ({execution_time:.2f}s)")
            
        except asyncio.TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            tool_result = ToolResult(
                tool_name=name,
                status=ToolStatus.TIMEOUT,
                error=f"Tool execution exceeded timeout of {tool.timeout}s",
                execution_time=execution_time,
                metadata={"parameters": kwargs}
            )
            
            self.stats[name]["total_calls"] += 1
            self.stats[name]["timeouts"] += 1
            self.stats[name]["total_time"] += execution_time
            
            logger.warning(f"⏱️ Tool '{name}' timed out after {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            tool_result = ToolResult(
                tool_name=name,
                status=ToolStatus.FAILURE,
                error=str(e),
                execution_time=execution_time,
                metadata={"parameters": kwargs, "exception_type": type(e).__name__}
            )
            
            self.stats[name]["total_calls"] += 1
            self.stats[name]["failures"] += 1
            self.stats[name]["total_time"] += execution_time
            
            logger.error(f"❌ Tool '{name}' failed: {e}")
        
        # Record in history (keep last 100)
        self.execution_history.append(tool_result)
        if len(self.execution_history) > 100:
            self.execution_history.pop(0)
        
        return tool_result
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of all registered tools with their metadata.
        
        Returns:
            List of tool information dictionaries
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "timeout": tool.timeout,
                "parameters": tool.parameters,
                "return_type": tool.return_type,
                "requires_network": tool.requires_network,
                "stats": self.stats.get(tool.name, {})
            }
            for tool in self.tools.values()
        ]
    
    def get_tool_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution statistics for tools.
        
        Args:
            name: Specific tool name, or None for all tools
            
        Returns:
            Statistics dictionary
        """
        if name:
            if name not in self.stats:
                return {}
            return {name: self.stats[name]}
        
        return self.stats.copy()
    
    def is_tool_registered(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool is registered
        """
        return name in self.tools
    
    def unregister_tool(self, name: str) -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self.tools:
            del self.tools[name]
            logger.info(f"🗑️ Unregistered tool: {name}")
            return True
        return False
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        logger.info("🧹 Cleared tool execution history")
    
    async def execute_tool_with_percept(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        create_percept: bool = True,
        **kwargs
    ) -> ToolExecutionResult:
        """
        Execute tool and optionally create percept for feedback loop.
        
        This method extends execute_tool to generate percepts from tool results,
        enabling bidirectional feedback where tool execution automatically creates
        percepts that inform future cognitive cycles.
        
        Args:
            name: Name of tool to execute
            parameters: Tool parameters (alternative to **kwargs)
            create_percept: Whether to create percept from result (default: True)
            **kwargs: Additional parameters to pass to tool handler
            
        Returns:
            ToolExecutionResult with success status and generated percept
        """
        import time
        start_time = time.time()
        
        # Merge parameters
        if parameters:
            kwargs.update(parameters)
        
        if name not in self.tools:
            error_msg = f"Tool '{name}' not found"
            execution_time_ms = (time.time() - start_time) * 1000
            percept = self._create_error_percept(name, error_msg, execution_time_ms) if create_percept else None
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                percept=percept
            )
        
        tool = self.tools[name]
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool.handler(**kwargs),
                timeout=tool.timeout
            )
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Create percept from successful result
            percept = self._create_result_percept(
                name, 
                result, 
                execution_time_ms
            ) if create_percept else None
            
            # Update stats
            self.stats[name]["total_calls"] += 1
            self.stats[name]["successes"] += 1
            self.stats[name]["total_time"] += execution_time_ms / 1000
            
            logger.info(f"✅ Tool '{name}' executed successfully ({execution_time_ms:.1f}ms)")
            
            return ToolExecutionResult(
                tool_name=name,
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
                percept=percept
            )
        except asyncio.TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Tool execution exceeded timeout of {tool.timeout}s"
            
            # Create percept from timeout
            percept = self._create_error_percept(
                name, 
                error_msg, 
                execution_time_ms
            ) if create_percept else None
            
            # Update stats
            self.stats[name]["total_calls"] += 1
            self.stats[name]["timeouts"] += 1
            self.stats[name]["total_time"] += execution_time_ms / 1000
            
            logger.warning(f"⏱️ Tool '{name}' timed out after {execution_time_ms:.1f}ms")
            
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                percept=percept
            )
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # Create percept from error
            percept = self._create_error_percept(
                name, 
                error_msg, 
                execution_time_ms
            ) if create_percept else None
            
            # Update stats
            self.stats[name]["total_calls"] += 1
            self.stats[name]["failures"] += 1
            self.stats[name]["total_time"] += execution_time_ms / 1000
            
            logger.error(f"❌ Tool '{name}' failed: {e}")
            
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                percept=percept
            )
    
    def _create_result_percept(
        self, 
        tool_name: str, 
        result: Any, 
        execution_time_ms: float
    ) -> 'Percept':
        """
        Create percept from successful tool execution.
        
        Args:
            tool_name: Name of the tool
            result: Tool execution result
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            Percept for feedback loop
        """
        from .workspace import Percept
        
        return Percept(
            modality="tool_result",
            raw=result,
            complexity=self._estimate_complexity(result),
            metadata={
                "tool_name": tool_name,
                "tool_success": True,
                "execution_time_ms": execution_time_ms,
                "result_type": type(result).__name__
            }
        )
    
    def _create_error_percept(
        self, 
        tool_name: str, 
        error: str, 
        execution_time_ms: float = 0.0
    ) -> 'Percept':
        """
        Create percept from failed tool execution.
        
        Args:
            tool_name: Name of the tool
            error: Error message
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            Percept for feedback loop
        """
        from .workspace import Percept
        
        return Percept(
            modality="tool_result",
            raw={"error": error},
            complexity=5,  # Errors require attention
            metadata={
                "tool_name": tool_name,
                "tool_success": False,
                "tool_error": error,
                "execution_time_ms": execution_time_ms
            }
        )
    
    def _estimate_complexity(self, result: Any) -> int:
        """
        Estimate percept complexity based on result size.
        
        Args:
            result: Tool execution result
            
        Returns:
            Complexity score (1-10)
        """
        if isinstance(result, str):
            return min(10, max(1, len(result) // 100))
        elif isinstance(result, (list, dict)):
            return min(10, max(1, len(str(result)) // 200))
        else:
            return 3



# Example tool implementations
# These can be moved to a separate module if desired

async def example_web_search(query: str, num_results: int = 10) -> Dict[str, Any]:
    """
    Example web search tool (placeholder).
    
    In production, this would integrate with SearXNG or similar.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Search results dictionary
    """
    # Placeholder implementation
    await asyncio.sleep(0.1)  # Simulate network delay
    
    return {
        "query": query,
        "results": [
            {"title": f"Result {i}", "url": f"https://example.com/{i}"}
            for i in range(num_results)
        ]
    }


async def example_compute(expression: str) -> Dict[str, Any]:
    """
    Example computation tool (placeholder).
    
    In production, this would integrate with WolframAlpha or similar.
    
    Args:
        expression: Mathematical expression to compute
        
    Returns:
        Computation result
    """
    # Placeholder implementation
    await asyncio.sleep(0.05)
    
    try:
        # Very basic eval (DO NOT use in production!)
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result, "success": True}
    except Exception as e:
        return {"expression": expression, "error": str(e), "success": False}


async def example_arxiv_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Example arXiv search tool (placeholder).
    
    In production, this would integrate with arXiv API.
    
    Args:
        query: Search query
        max_results: Maximum number of papers to return
        
    Returns:
        Search results with paper metadata
    """
    # Placeholder implementation
    await asyncio.sleep(0.2)
    
    return {
        "query": query,
        "papers": [
            {
                "title": f"Paper {i}: {query}",
                "authors": ["Author A", "Author B"],
                "abstract": "This is a placeholder abstract...",
                "url": f"https://arxiv.org/abs/example{i}"
            }
            for i in range(max_results)
        ]
    }


def create_default_registry() -> ToolRegistry:
    """
    Create a ToolRegistry with default tools registered.
    
    Returns:
        ToolRegistry with example tools
    """
    registry = ToolRegistry()
    
    # Register example tools
    registry.register_tool(
        name="web_search",
        handler=example_web_search,
        description="Search the web using SearXNG",
        timeout=10.0,
        parameters={"query": "str", "num_results": "int"},
        return_type="Dict[str, Any]",
        requires_network=True
    )
    
    registry.register_tool(
        name="compute",
        handler=example_compute,
        description="Compute mathematical expressions using WolframAlpha",
        timeout=5.0,
        parameters={"expression": "str"},
        return_type="Dict[str, Any]",
        requires_network=True
    )
    
    registry.register_tool(
        name="arxiv_search",
        handler=example_arxiv_search,
        description="Search arXiv for research papers",
        timeout=10.0,
        parameters={"query": "str", "max_results": "int"},
        return_type="Dict[str, Any]",
        requires_network=True
    )
    
    return registry
